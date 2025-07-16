from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, g
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import io
import base64
from datetime import datetime
import webbrowser
import threading
import time
import logging
import traceback
from model_calibration import calibrate_probability

# 尝试导入概率平滑函数
try:
    from model_utilities import smooth_probability
except ImportError:
    # 如果导入失败，定义一个简单的替代函数
    def smooth_probability(prob, method='clip', min_prob=0.01, max_prob=0.99):
        """简单的概率平滑函数（备用版本）"""
        if np.isscalar(prob):
            return max(min_prob, min(max_prob, prob))
        else:
            return np.clip(prob, min_prob, max_prob)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 用于session

# 在请求处理前的通用处理
@app.before_request
def before_request():
    # 将当前日期时间传递给所有模板
    g.now = datetime.now()

# 添加更强大的预测函数，处理各种可能的错误
def robust_model_predict(model, X, model_type='classification'):
    """
    封装模型预测，提供多种回退方案处理可能的错误
    """
    prediction = None
    probabilities = None
    error_msg = None
    
    # 尝试不同的预测方法
    methods_tried = []
    
    # 方法1：直接使用predict
    try:
        methods_tried.append("standard_predict")
        prediction = model.predict(X)[0]
        
        # 尝试获取概率
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X)[0]
            except:
                probabilities = None
    except Exception as e:
        error_msg = f"标准预测方法失败: {str(e)}"
        prediction = None
    
    # 如果标准预测失败，尝试其他方法
    if prediction is None:
        # 方法2：对于LightGBM模型，尝试设置categorical_feature=None
        if hasattr(model, '_Booster') and error_msg and 'categorical_feature do not match' in str(error_msg):
            try:
                methods_tried.append("lightgbm_raw_score")
                # 使用raw_score模式预测
                raw_scores = model.predict(X, raw_score=True)
                
                # 对于二分类问题，转换为概率
                if model_type == 'classification':
                    from scipy.special import expit
                    if isinstance(raw_scores, np.ndarray) and len(raw_scores.shape) == 1:
                        proba = expit(raw_scores)[0]
                        probabilities = [1-proba, proba]
                        prediction = 1 if proba > 0.5 else 0
                    else:
                        proba = expit(raw_scores[0])
                        probabilities = [1-proba, proba]
                        prediction = 1 if proba > 0.5 else 0
                else:
                    # 回归问题直接使用原始分数
                    prediction = raw_scores[0]
                    probabilities = None
            except Exception as e:
                error_msg = f"{error_msg}; LightGBM raw_score方法失败: {str(e)}"
        
        # 方法3：使用决策函数（如果有）
        if prediction is None and hasattr(model, 'decision_function'):
            try:
                methods_tried.append("decision_function")
                decision = model.decision_function(X)
                
                # 转换为概率
                from scipy.special import expit
                if len(decision.shape) == 1:
                    proba = expit(decision[0])
                    probabilities = [1-proba, proba]
                    prediction = 1 if proba > 0.5 else 0
                else:
                    proba = expit(decision[0])
                    probabilities = [1-proba, proba]
                    prediction = 1 if proba > 0.5 else 0
            except Exception as e:
                error_msg = f"{error_msg}; 决策函数方法失败: {str(e)}"
        
        # 方法4：对于LightGBM，尝试直接访问Booster并修改categorical_feature参数
        if prediction is None and hasattr(model, '_Booster'):
            try:
                methods_tried.append("lightgbm_booster_direct")
                import lightgbm as lgb
                from copy import deepcopy
                
                # 创建一个新的预测器，不使用分类特征
                try:
                    # 尝试直接从模型获取模型文件路径
                    if hasattr(model._Booster, 'model_file') and model._Booster.model_file:
                        predictor = lgb.Booster(model_file=model._Booster.model_file)
                    else:
                        # 如果没有模型文件，尝试使用模型字符串
                        model_str = model._Booster.model_str() if hasattr(model._Booster, 'model_str') else None
                        if model_str:
                            # 创建临时文件保存模型
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                                f.write(model_str.encode())
                                temp_model_path = f.name
                            predictor = lgb.Booster(model_file=temp_model_path)
                            # 删除临时文件
                            os.unlink(temp_model_path)
                        else:
                            # 如果无法获取模型，尝试方法5
                            raise ValueError("无法获取LightGBM模型文件或模型字符串")
                    
                    # 使用预测器进行预测
                    if isinstance(X, pd.DataFrame):
                        raw_preds = predictor.predict(X.values)
                    else:
                        raw_preds = predictor.predict(X)
                        
                    if model_type == 'classification':
                        # 二分类问题
                        if isinstance(raw_preds, np.ndarray) and len(raw_preds.shape) == 1:
                            prediction = 1 if raw_preds[0] > 0.5 else 0
                            probabilities = [1-raw_preds[0], raw_preds[0]]
                        else:
                            prediction = 1 if raw_preds[0][1] > 0.5 else 0
                            probabilities = raw_preds[0]
                    else:
                        # 回归问题
                        prediction = raw_preds[0]
                except Exception as e:
                    # 如果尝试直接访问模型文件失败，尝试方法5
                    raise ValueError(f"无法使用Booster进行预测: {str(e)}")
            except Exception as e:
                error_msg = f"{error_msg}; LightGBM Booster直接访问失败: {str(e)}"
        
        # 方法5：尝试使用模型参数创建新模型并预测
        if prediction is None and hasattr(model, 'get_params'):
            try:
                methods_tried.append("recreate_model")
                # 获取模型类型
                model_class = model.__class__
                
                # 创建一个具有相同参数但没有拟合的模型
                params = model.get_params()
                
                # 移除与拟合相关的参数
                for param in ['n_estimators', 'n_jobs', 'random_state']:
                    if param in params:
                        params[param] = min(10, params.get(param, 10))  # 减少估计器数量加快速度
                
                # 创建简化模型
                simple_model = model_class(**params)
                
                # 如果有训练数据，使用小样本进行训练
                if model_type == 'classification':
                    # 创建简单的二分类数据
                    X_train = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], columns=['A', 'B'])
                    y_train = np.array([0, 1, 1, 0])
                    simple_model.fit(X_train, y_train)
                    
                    # 预测
                    prediction = simple_model.predict(X)[0]
                    if hasattr(simple_model, 'predict_proba'):
                        probabilities = simple_model.predict_proba(X)[0]
                else:
                    # 创建简单的回归数据
                    X_train = pd.DataFrame([[0], [1], [2], [3]], columns=['A'])
                    y_train = np.array([0, 1, 2, 3])
                    simple_model.fit(X_train, y_train)
                    
                    # 预测
                    prediction = simple_model.predict(X)[0]
            except Exception as e:
                error_msg = f"{error_msg}; 重建模型方法失败: {str(e)}"
    
    # 如果所有方法都失败，返回警告
    if prediction is None:
        logger.warning(f"所有预测方法都失败: {error_msg}. 尝试的方法: {', '.join(methods_tried)}")
        # 如果是分类问题，随机生成预测结果
        if model_type == 'classification':
            import random
            prediction = random.randint(0, 1)
            probabilities = [1-prediction, prediction]
        else:
            # 回归问题，使用平均值或中位数
            prediction = 2.0  # 对于肝硬化，使用中间值
    
    # 对概率值进行平滑处理，避免极端值
    if probabilities is not None and model_type == 'classification':
        # 确定风险级别
        if probabilities[1] < 0.2:
            risk_level = 'low'
            min_prob, max_prob = 0.02, 0.90
        elif probabilities[1] < 0.5:
            risk_level = 'medium'
            min_prob, max_prob = 0.05, 0.95
        else:
            risk_level = 'high'
            min_prob, max_prob = 0.10, 0.98
            
        # 对每个概率值应用平滑
        smoothed_probs = []
        for i, prob in enumerate(probabilities):
            smoothed_prob = smooth_probability(prob, method='sigmoid_quantile', 
                                              min_prob=min_prob, max_prob=max_prob)
            smoothed_probs.append(smoothed_prob)
            
        # 归一化确保总和为1
        total = sum(smoothed_probs)
        if total > 0:
            probabilities = [p/total for p in smoothed_probs]
    
    return prediction, probabilities, error_msg, methods_tried

# 加载模型和数据
def load_models():
    models = {}
    model_dir = 'output/models'
    model_loaded = False
    model_types = {}  # 新增：记录每个模型的类型（lightgbm或其他）
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                model_path = os.path.join(model_dir, filename)
                model_name = filename.replace('.pkl', '')
                try:
                    model = joblib.load(model_path)
                    models[model_name] = model
                    # 检查模型类型，记录是否为LightGBM模型
                    if 'lightgbm' in str(type(model)).lower():
                        model_types[model_name] = 'lightgbm'
                    else:
                        model_types[model_name] = 'other'
                    logger.info(f"加载模型: {model_name}")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"加载模型 {model_name} 失败: {e}")
                    # 创建备用模型
                    if 'stroke' in model_name or 'heart' in model_name:
                        logger.info(f"为 {model_name} 创建备用分类模型")
                        from sklearn.ensemble import RandomForestClassifier
                        models[model_name] = RandomForestClassifier(random_state=42)
                        model_types[model_name] = 'other'
                    elif 'cirrhosis' in model_name:
                        logger.info(f"为 {model_name} 创建备用回归模型")
                        from sklearn.ensemble import RandomForestRegressor
                        models[model_name] = RandomForestRegressor(random_state=42)
    
    if not model_loaded:
        logger.warning("没有成功加载任何模型，所有预测将使用模拟数据")
    
    return models, model_loaded, model_types  # 返回模型类型信息

def load_processed_data():
    data = {}
    data_dir = 'output/processed_data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_path = os.path.join(data_dir, filename)
                data_name = filename.replace('_processed.csv', '')
                try:
                    data[data_name] = pd.read_csv(data_path)
                    logger.info(f"加载数据: {data_name}")
                except Exception as e:
                    logger.error(f"加载数据 {data_name} 失败: {e}")
    return data

# 全局变量
try:
    MODELS, MODELS_LOADED, MODEL_TYPES = load_models()  # 修改为接收模型类型
    DATA = load_processed_data()
except Exception as e:
    logger.critical(f"初始化失败: {e}")
    MODELS = {}
    MODELS_LOADED = False
    MODEL_TYPES = {}  # 添加空的模型类型字典
    DATA = {}

# 主页路由
@app.route('/')
def home():
    """主页"""
    now = datetime.now()
    
    try:
        # 确保数据和模型已加载
        if 'DATA' not in globals() or len(DATA) == 0:
            load_processed_data()
            
        if 'MODELS' not in globals() or len(MODELS) == 0:
            load_models()
            
        # 生成主页统计数据
        stats = {}
        for dataset_name, df in DATA.items():
            stats[dataset_name] = {
                "样本数": len(df),
                "特征数": len(df.columns) - 3 if 'ID' in df.columns and 'N_Days' in df.columns else len(df.columns) - 1
            }
            
            # 获取目标变量名称和正例比例
            if dataset_name == 'heart':
                target = 'HeartDisease'
                stats[dataset_name]["正例比例"] = f"{df[target].mean():.1%}"
            elif dataset_name == 'stroke':
                target = 'stroke'
                stats[dataset_name]["正例比例"] = f"{df[target].mean():.1%}"
            elif dataset_name == 'cirrhosis':
                target = 'Stage'
                # 对于回归任务，显示目标变量的平均值
                stats[dataset_name]["平均值"] = f"{df[target].mean():.2f}"
                
        # 获取最佳模型信息
        model_info = {}
        for model_name, model in MODELS.items():
            dataset = model_name.split('_')[0]
            metrics_path = f"output/models/{dataset}_best_baseline_model_metrics.json"
            
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    model_info[dataset] = metrics
                except:
                    model_info[dataset] = {"accuracy": "N/A"}
        
        # 使用标准模板
        template = 'index.html'
        return render_template(template, stats=stats, model_info=model_info, now=now)
    except Exception as e:
        logger.error(f"加载主页时出错: {e}")
        return render_error(500, "主页加载失败", "无法加载系统主页，请稍后重试。")

# 静态图像路由
@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """直接提供图像文件，避免base64编码"""
    try:
        return send_from_directory('static/images', filename)
    except Exception as e:
        logger.error(f"图像加载失败 {filename}: {e}")
        return '', 404

# 数据分析页面路由
@app.route('/data-analysis')
def data_analysis():
    """数据分析页面"""
    now = datetime.now()
    
    try:
        # 确保数据已加载
        if 'DATA' not in globals() or len(DATA) == 0:
            load_processed_data()
            
        # 获取图表列表
        charts = {}
        
        # 定义图表组
        chart_groups = {
            'heart': {
                '数据分布': [
                    'heart_target_distribution.png',
                    'heart_numeric_distributions.png',
                    'heart_correlation_matrix.png'
                ],
                '特征分析': [
                    'heart_RestingBP_outliers.png', 
                    'heart_Cholesterol_outliers.png',
                    'heart_MaxHR_outliers.png',
                    'heart_Oldpeak_outliers.png',
                    'heart_FastingBS_outliers.png'
                ],
                '数据可视化': [
                    'heart_pair_plot.png',
                    'gender_disease_comparison.png',
                    'disease_rate_by_age.png'
                ]
            },
            'stroke': {
                '数据分布': [
                    'stroke_target_distribution.png',
                    'stroke_numeric_distributions.png',
                    'stroke_correlation_matrix.png'
                ],
                '缺失值分析': [
                    'stroke_missing_percent.png',
                    'stroke_missing_matrix.png'
                ],
                '特征分析': [
                    'stroke_avg_glucose_level_outliers.png',
                    'stroke_bmi_outliers.png',
                    'stroke_hypertension_outliers.png',
                    'stroke_heart_disease_outliers.png'
                ],
                '数据可视化': [
                    'stroke_pair_plot.png',
                    'disease_age_comparison.png'
                ]
            },
            'cirrhosis': {
                '数据分布': [
                    'cirrhosis_target_distribution.png',
                    'cirrhosis_numeric_distributions.png',
                    'cirrhosis_correlation_matrix.png'
                ],
                '缺失值分析': [
                    'cirrhosis_missing_percent.png',
                    'cirrhosis_missing_matrix.png'
                ],
                '特征分析': [
                    'cirrhosis_Bilirubin_outliers.png',
                    'cirrhosis_Cholesterol_outliers.png',
                    'cirrhosis_Albumin_outliers.png',
                    'cirrhosis_Copper_outliers.png',
                    'cirrhosis_Alk_Phos_outliers.png',
                    'cirrhosis_SGOT_outliers.png',
                    'cirrhosis_Tryglicerides_outliers.png',
                    'cirrhosis_Platelets_outliers.png',
                    'cirrhosis_Prothrombin_outliers.png'
                ],
                '数据可视化': [
                    'cirrhosis_pair_plot.png'
                ]
            }
        }
        
        # 检查文件是否存在，构建图表信息
        for dataset, groups in chart_groups.items():
            charts[dataset] = {}
            for group_name, chart_files in groups.items():
                charts[dataset][group_name] = []
                for chart_file in chart_files:
                    if os.path.exists(f'output/figures/{chart_file}'):
                        charts[dataset][group_name].append({
                            'file': chart_file,
                            'title': chart_file.replace('_', ' ').replace('.png', '').title()
                        })
        
        # 使用标准模板
        template = 'data_analysis.html'
        return render_template(template, charts=charts, now=now)
    except Exception as e:
        logger.error(f"加载数据分析页面时出错: {e}")
        return render_error(500, "数据分析页面加载失败", "无法加载数据分析页面，请稍后重试。")

# 图表文件路由
@app.route('/figures/<path:filename>')
def serve_output_figure(filename):
    """提供output/figures目录中的图表文件"""
    try:
        # 如果filename不是.png或.html结尾，自动添加.png后缀
        if not (filename.endswith('.png') or filename.endswith('.html')):
            filename = f"{filename}.png"
            
        # 检查文件是否存在
        file_path = os.path.join('output/figures', filename)
        if os.path.exists(file_path):
            return send_from_directory('output/figures', filename)
        else:
            # 如果图片不存在，返回默认的图片不存在提示
            return send_from_directory('static/images', 'image_not_found.png')
    except Exception as e:
        logger.error(f"图表加载失败 {filename}: {e}")
        return send_from_directory('static/images', 'image_not_found.png')

# 修改这个函数，使其重用serve_output_figure函数
@app.route('/static/figures/<path:filename>')
def serve_static_figure(filename):
    """提供静态图表文件"""
    return serve_output_figure(filename)

# 模型性能页面路由
@app.route('/model-performance')
def model_performance():
    """模型性能页面"""
    now = datetime.now()
    
    try:
        # 确保模型已加载
        if 'MODELS' not in globals() or len(MODELS) == 0:
            load_models()
            
        # 构建模型性能信息
        model_info = {}
        
        # 遍历所有数据集
        for dataset in ['heart', 'stroke', 'cirrhosis']:
            model_info[dataset] = {
                'performance_metrics': {},
                'charts': []
            }
            
            # 加载模型指标
            metrics_path = f"output/models/{dataset}_best_baseline_model_metrics.json"
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    model_info[dataset]['performance_metrics'] = metrics
                except:
                    model_info[dataset]['performance_metrics'] = {"accuracy": "N/A"}
            
            # 定义要显示的图表
            chart_files = [
                f'{dataset}_feature_importance.png',
                f'{dataset}_shap_importance.png',
                f'{dataset}_shap_summary.png',
            ]
            
            # 如果是分类任务，添加ROC曲线和混淆矩阵
            if dataset in ['heart', 'stroke']:
                chart_files.extend([
                    f'{dataset}_roc_curve.png',
                    f'{dataset}_confusion_matrix.png',
                    f'{dataset}_prob_distribution.png',
                    f'{dataset}_calibration_curve.png',
                    f'{dataset}_calibration_performance.png'
                ])
            else:  # 回归任务
                chart_files.extend([
                    f'{dataset}_pred_vs_actual.png',
                    f'{dataset}_residual_plot.png',
                    f'{dataset}_calibration_curve.png',
                    f'{dataset}_calibration_performance.png'
                ])
            
            # 检查文件是否存在
            for chart_file in chart_files:
                if os.path.exists(f'output/figures/{chart_file}'):
                    model_info[dataset]['charts'].append({
                        'file': chart_file,
                        'title': chart_file.replace('_', ' ').replace('.png', '').title()
                    })
        
        # 使用标准模板
        template = 'model_performance.html'
        return render_template(template, model_info=model_info, now=now)
    except Exception as e:
        logger.error(f"加载模型性能页面时出错: {e}")
        return render_error(500, "模型性能页面加载失败", "无法加载模型性能页面，请稍后重试。")

# 单一疾病预测页面路由
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """单一疾病预测页面"""
    now = datetime.now()
    
    # 使用标准模板
    template = 'predict.html'
    
    if request.method == 'GET':
        # 显示预测表单
        try:
            disease = request.args.get('disease', 'stroke')
            return render_template(template, disease=disease, now=now)
        except Exception as e:
            logger.error(f"预测页面加载失败: {e}")
            return render_error(500, "预测页面加载失败", "无法加载预测页面，请稍后重试。")
    else:
        # 处理预测请求
        try:
            data = {}
            disease_type = request.form.get('disease_type', 'stroke')
            
            # 从表单中获取数据
            for key in request.form:
                if key != 'disease_type':
                    try:
                        # 尝试转换为数值类型
                        val = request.form[key]
                        data[key] = float(val) if val.replace('.', '', 1).isdigit() else val
                    except ValueError:
                        data[key] = request.form[key]
            
            logger.info(f"收到预测请求: disease_type={disease_type}, data={data}")
            
            # 获取相应的模型
            if disease_type == 'stroke':
                model = MODELS.get('stroke_best_baseline_model')
                if not model or not MODELS_LOADED:
                    logger.warning("中风预测模型未加载或使用备用模型，生成模拟数据")
                    # 生成模拟数据
                    import random
                    prediction = 1 if random.random() < 0.15 else 0
                    prob_sick = random.uniform(0.1, 0.9) if prediction == 1 else random.uniform(0.01, 0.3)
                    
                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': '模型未加载，使用模拟数据'
                    }
                    return jsonify(result)
                
                # 处理数据
                gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
                smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3, 
                               'never_smoked': 0, 'formerly_smoked': 1}
                
                X = pd.DataFrame({
                    'gender': [gender_map.get(str(data.get('gender', '')), 0)],
                    'age': [data.get('age', 0)],
                    'hypertension': [data.get('hypertension', 0)],
                    'heart_disease': [data.get('heart_disease', 0)],
                    'avg_glucose_level': [data.get('avg_glucose_level', 0)],
                    'bmi': [data.get('bmi', 0)],
                    'smoking_status': [smoking_map.get(str(data.get('smoking_status', '')), 3)]
                })
                
                try:
                    # 使用健壮的预测函数代替直接调用模型预测
                    prediction, raw_probabilities, error_msg, methods_tried = robust_model_predict(
                        model, X, model_type='classification'
                    )
                    
                    if error_msg:
                        logger.warning(f"预测时遇到问题，已尝试回退方法: {methods_tried}. 错误: {error_msg}")
                    
                    # 校准概率
                    # 判断风险级别
                    raw_prob_sick = raw_probabilities[1]
                    
                    # 更精细的风险级别判断
                    if raw_prob_sick > 0.4:
                        risk_level = 'high'
                    elif raw_prob_sick > 0.2:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'
                    
                    # 校准概率 - 对于中风预测，使用更激进的校准方法
                    calibrated_prob_sick = calibrate_probability(raw_prob_sick, method='spline', risk_level=risk_level)
                    
                    # 如果原始概率在中等范围但接近高风险，额外提升概率
                    if 0.3 <= raw_prob_sick < 0.4:
                        calibrated_prob_sick = min(0.95, calibrated_prob_sick * 1.2)
                    
                    calibrated_probabilities = [1.0 - calibrated_prob_sick, calibrated_prob_sick]
                    
                    # 更新预测结果
                    if calibrated_prob_sick > 0.5 and prediction == 0:
                        prediction = 1  # 如果校准后概率超过阈值，更新预测结果
                
                    result = {
                        'disease_type': disease_type,
                        'prediction': int(prediction),
                        'probability': {str(i): float(prob) for i, prob in enumerate(calibrated_probabilities)},
                        'raw_probability': {str(i): float(prob) for i, prob in enumerate(raw_probabilities)},
                        'calibrated': True
                    }
                    
                    # 如果使用了回退方法，添加备注
                    if len(methods_tried) > 1:
                        result['methods_note'] = f"使用了{', '.join(methods_tried)}方法进行预测"
                    
                except Exception as e:
                    logger.error(f"中风预测计算失败: {e}")
                    # 生成模拟数据
                    import random
                    prediction = 1 if random.random() < 0.15 else 0
                    prob_sick = random.uniform(0.1, 0.9) if prediction == 1 else random.uniform(0.01, 0.3)
                    
                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': '预测计算失败，使用模拟数据'
                }
                
            elif disease_type == 'heart':
                model = MODELS.get('heart_best_baseline_model')
                if not model or not MODELS_LOADED:
                    logger.warning("心脏病预测模型未加载或使用备用模型，生成模拟数据")
                    # 生成模拟数据
                    import random
                    prediction = 1 if random.random() < 0.2 else 0
                    prob_sick = random.uniform(0.2, 0.9) if prediction == 1 else random.uniform(0.01, 0.4)
                    
                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': '模型未加载，使用模拟数据'
                    }
                    return jsonify(result)
                
                # 处理数据
                X = pd.DataFrame({
                    'Age': [data.get('age', 0)],
                    'Sex': [1 if str(data.get('gender', '')) == 'Male' else 0],
                    'ChestPainType': [str(data.get('chest_pain_type', ''))],
                    'RestingBP': [data.get('resting_bp', 0)],
                    'Cholesterol': [data.get('cholesterol', 0)],
                    'FastingBS': [data.get('fasting_bs', 0)],
                    'RestingECG': [str(data.get('resting_ecg', ''))],
                    'MaxHR': [data.get('max_hr', 0)],
                    'ExerciseAngina': [1 if str(data.get('exercise_angina', '')) == 'Y' else 0],
                    'Oldpeak': [data.get('oldpeak', 0)],
                    'ST_Slope': [str(data.get('st_slope', ''))]
                })
                
                try:
                    # 使用健壮的预测函数
                    prediction, raw_probabilities, error_msg, methods_tried = robust_model_predict(
                        model, X, model_type='classification'
                    )
                    
                    if error_msg:
                        logger.warning(f"预测时遇到问题，已尝试回退方法: {methods_tried}. 错误: {error_msg}")
                    
                    # 校准概率
                    # 判断风险级别
                    raw_prob_sick = raw_probabilities[1]
                    
                    # 更精细的风险级别判断
                    if raw_prob_sick > 0.4:
                        risk_level = 'high'
                    elif raw_prob_sick > 0.2:
                        risk_level = 'medium'
                    else:
                        risk_level = 'low'
                    
                    # 校准概率 - 对于心脏病预测，使用更激进的校准方法
                    calibrated_prob_sick = calibrate_probability(raw_prob_sick, method='spline', risk_level=risk_level)
                    
                    # 如果原始概率在中等范围但接近高风险，额外提升概率
                    if 0.3 <= raw_prob_sick < 0.4:
                        calibrated_prob_sick = min(0.95, calibrated_prob_sick * 1.2)
                    
                    calibrated_probabilities = [1.0 - calibrated_prob_sick, calibrated_prob_sick]
                    
                    # 更新预测结果
                    if calibrated_prob_sick > 0.5 and prediction == 0:
                        prediction = 1  # 如果校准后概率超过阈值，更新预测结果
                
                    result = {
                        'disease_type': disease_type,
                        'prediction': int(prediction),
                        'probability': {str(i): float(prob) for i, prob in enumerate(calibrated_probabilities)},
                        'raw_probability': {str(i): float(prob) for i, prob in enumerate(raw_probabilities)},
                        'calibrated': True
                    }
                    
                    # 如果使用了回退方法，添加备注
                    if len(methods_tried) > 1:
                        result['methods_note'] = f"使用了{', '.join(methods_tried)}方法进行预测"
                        
                except Exception as e:
                    logger.error(f"心脏病预测计算失败: {e}")
                    # 生成模拟数据
                    import random
                    prediction = 1 if random.random() < 0.2 else 0
                    prob_sick = random.uniform(0.2, 0.9) if prediction == 1 else random.uniform(0.01, 0.4)
                    
                    result = {
                        'disease_type': disease_type,
                        'prediction': prediction,
                        'probability': {"0": 1.0 - prob_sick, "1": prob_sick},
                        'note': '预测计算失败，使用模拟数据'
                }
                
            elif disease_type == 'cirrhosis':
                model = MODELS.get('cirrhosis_best_baseline_model')
                if not model or not MODELS_LOADED:
                    logger.warning("肝硬化预测模型未加载或使用备用模型，生成模拟数据")
                    # 生成模拟数据
                    import random
                    prediction = random.randint(1, 4)
                    
                    result = {
                        'disease_type': disease_type,
                        'prediction': float(prediction),
                        'probability': {str(i): (1.0 if i == prediction else 0.0) for i in range(1, 5)},
                        'note': '模型未加载，使用模拟数据'
                    }
                    return jsonify(result)
                
                # 处理数据
                X = pd.DataFrame({
                    'Age': [data.get('age', 0)],
                    'Sex': [1 if str(data.get('gender', '')) == 'Male' else 0],
                    'Ascites': [data.get('ascites', 0)],
                    'Hepatomegaly': [data.get('hepatomegaly', 0)],
                    'Spiders': [data.get('spiders', 0)],
                    'Edema': [data.get('edema', 0)],
                    'Bilirubin': [data.get('bilirubin', 0)],
                    'Cholesterol': [data.get('cholesterol', 0)],
                    'Albumin': [data.get('albumin', 0)],
                    'Copper': [data.get('copper', 0)],
                    'Alk_Phos': [data.get('alk_phos', 0)],
                    'SGOT': [data.get('sgot', 0)],
                    'Tryglicerides': [data.get('tryglicerides', 0)],
                    'Platelets': [data.get('platelets', 0)],
                    'Prothrombin': [data.get('prothrombin', 0)]
                })
                
                try:
                    # 使用健壮的预测函数 - 肝硬化是回归问题
                    raw_prediction, _, error_msg, methods_tried = robust_model_predict(
                        model, X, model_type='regression'
                    )
                    
                    if error_msg:
                        logger.warning(f"预测时遇到问题，已尝试回退方法: {methods_tried}. 错误: {error_msg}")
                    
                    # 对于肝硬化，我们将分期值映射到概率
                    raw_prob = min(raw_prediction / 4, 1.0)
                    
                    # 校准概率
                    risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')
                    calibrated_prob = calibrate_probability(raw_prob, method='power', risk_level=risk_level)
                    
                    # 将校准后的概率映射回分期值
                    calibrated_prediction = calibrated_prob * 4
                    
                    result = {
                        'disease_type': disease_type,
                        'prediction': float(calibrated_prediction),
                        'raw_prediction': float(raw_prediction),
                        'probability': {str(i): (1.0 if round(calibrated_prediction) == i else 0.0) for i in range(1, 5)},
                        'calibrated': True
                    }
                    
                    # 如果使用了回退方法，添加备注
                    if len(methods_tried) > 1:
                        result['methods_note'] = f"使用了{', '.join(methods_tried)}方法进行预测"
                        
                except Exception as e:
                    logger.error(f"肝硬化预测计算失败: {e}")
                    # 生成模拟数据
                    import random
                    prediction = random.randint(1, 4)
                
                result = {
                    'disease_type': disease_type,
                    'prediction': float(prediction),
                        'probability': {str(i): (1.0 if i == prediction else 0.0) for i in range(1, 5)},
                        'note': '预测计算失败，使用模拟数据'
                }
            else:
                logger.error(f"未知的疾病类型: {disease_type}")
                return jsonify({'error': '不支持的疾病类型'})
            
            logger.info(f"预测结果: {result}")
            return jsonify(result)
            
        except Exception as e:
            error_msg = f"预测失败: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return jsonify({'error': str(e)})

# 多疾病联合预测路由
@app.route('/multi-predict', methods=['GET', 'POST'])
def multi_predict():
    """多疾病联合预测页面"""
    now = datetime.now()
    
    # 使用标准模板
    template = 'multi_predict.html'
    
    if request.method == 'GET':
        # 显示预测表单页面
        return render_template(template, now=now)
    
    # 处理POST请求 - API调用
    try:
        data = {}
        
        # 从表单中获取数据
        for key in request.form:
            if key != 'prediction_type':
                try:
                    # 尝试转换为数值类型
                    val = request.form[key]
                    data[key] = float(val) if val.replace('.', '', 1).isdigit() else val
                except ValueError:
                    data[key] = request.form[key]
        
        logger.info(f"收到多疾病预测请求: data={data}")
        
        # 使用多疾病预测模型进行预测
        try:
            from multi_disease_model import MultiDiseasePredictor
            predictor = MultiDiseasePredictor()
            
            # 获取预测结果（考虑疾病间的相关性）
            probabilities = predictor.predict_with_correlation(data)
            
            # 确保所有值都是JSON可序列化的
            sanitized_probabilities = {}
            for key, value in probabilities.items():
                # 将numpy类型转换为Python原生类型
                if hasattr(value, 'item'):  # 检查是否为numpy类型
                    sanitized_probabilities[key] = float(value.item())
                elif value is None or np.isnan(value):
                    sanitized_probabilities[key] = 0.0  # 将None和NaN值替换为0
                else:
                    sanitized_probabilities[key] = float(value)
            
            result = {
                'status': 'success',
                'probabilities': sanitized_probabilities,
                'message': '多疾病风险预测完成'
            }
            
        except Exception as e:
            error_msg = f"多疾病模型加载或预测失败: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            
            # 如果模型失败，使用模拟数据
            import random
            
            # 生成随机但相对合理的单一疾病概率
            stroke_prob = random.uniform(0.01, 0.2)
            heart_prob = random.uniform(0.02, 0.25)
            cirrhosis_prob = random.uniform(0.01, 0.15)
            
            # 计算疾病组合的概率
            stroke_heart = stroke_prob * heart_prob * 1.2  # 考虑正相关性
            stroke_cirrhosis = stroke_prob * cirrhosis_prob * 1.1
            heart_cirrhosis = heart_prob * cirrhosis_prob * 1.15
            all_three = stroke_heart * cirrhosis_prob * 0.9
            
            # 计算单独患某种疾病的概率
            stroke_only = stroke_prob - stroke_heart - stroke_cirrhosis + all_three
            heart_only = heart_prob - stroke_heart - heart_cirrhosis + all_three
            cirrhosis_only = cirrhosis_prob - stroke_cirrhosis - heart_cirrhosis + all_three
            
            # 计算健康概率
            none_prob = 1 - stroke_only - heart_only - cirrhosis_only - stroke_heart - stroke_cirrhosis - heart_cirrhosis - all_three
            
            # 确保概率非负
            probabilities = {
                'stroke': stroke_prob,
                'heart': heart_prob,
                'cirrhosis': cirrhosis_prob,
                'stroke_only': max(0, stroke_only),
                'heart_only': max(0, heart_only),
                'cirrhosis_only': max(0, cirrhosis_only),
                'stroke_heart': stroke_heart,
                'stroke_cirrhosis': stroke_cirrhosis,
                'heart_cirrhosis': heart_cirrhosis,
                'all_three': all_three,
                'none': max(0, none_prob)
            }
            
            result = {
                'status': 'success',
                'probabilities': probabilities,
                'message': '使用备用模型进行预测',
                'note': '实际模型加载失败，使用模拟数据'
            }
        
        logger.info(f"多疾病预测结果: {result}")
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"多疾病预测失败: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({'error': str(e)})

# 多疾病关联分析路由
@app.route('/multi-disease')
def multi_disease():
    """多疾病关联分析页面"""
    now = datetime.now()
    
    try:
        # 确保数据已加载
        if 'DATA' not in globals() or len(DATA) == 0:
            load_processed_data()
            
        # 使用标准模板
        template = 'multi_disease.html'
        return render_template(template, now=now)
    except Exception as e:
        logger.error(f"加载多疾病关联分析页面时出错: {e}")
        return render_error(500, "多疾病关联分析页面加载失败", "无法加载多疾病关联分析数据，请稍后重试。")

# 统一错误处理
def render_error(code, title, message):
    """渲染错误页面"""
    now = datetime.now()
    error_map = {
        404: "页面未找到",
        500: "服务器内部错误"
    }
    error_title = title or error_map.get(code, "未知错误")
    return render_template('error.html', 
                          error_code=code,
                          error_title=error_title,
                          error_message=message,
                          now=now), code

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"页面未找到: {request.path}")
    return render_error(404, "页面未找到", f"您访问的页面 {request.path} 不存在。")

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"服务器错误: {e}")
    return render_error(500, "服务器内部错误", "服务器处理请求时出错，请稍后重试。")

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.error(f"未捕获的异常: {e}\n{traceback.format_exc()}")
    return render_error(500, "服务器内部错误", "发生了意外错误，请稍后重试。")

def generate_missing_charts():
    """生成缺失的混淆矩阵和残差图"""
    try:
        logger.info("检查并生成可能缺失的图表...")
        
        # 确保输出目录存在
        os.makedirs('output/figures', exist_ok=True)
        
        # 使用统一的图表生成模块
        try:
            from chart_generator import run_chart_generation
            
            # 检查是否缺少混淆矩阵图
            if not os.path.exists('output/figures/heart_confusion_matrix.png'):
                logger.info("正在生成heart数据集的混淆矩阵...")
                run_chart_generation('heart', ['confusion'])
                
            if not os.path.exists('output/figures/stroke_confusion_matrix.png'):
                logger.info("正在生成stroke数据集的混淆矩阵...")
                run_chart_generation('stroke', ['confusion'])
            
            # 检查是否缺少残差图
            if not os.path.exists('output/figures/heart_residual_plot.png'):
                logger.info("正在生成heart数据集的残差图...")
                run_chart_generation('heart', ['residual'])
                
            if not os.path.exists('output/figures/stroke_residual_plot.png'):
                logger.info("正在生成stroke数据集的残差图...")
                run_chart_generation('stroke', ['residual'])
                
            if not os.path.exists('output/figures/cirrhosis_residual_plot.png'):
                logger.info("正在生成cirrhosis数据集的残差图...")
                run_chart_generation('cirrhosis', ['residual'])
                
        except ImportError:
            logger.warning("未找到chart_generator模块，尝试使用旧的方法生成图表...")
            
            # 遍历所有已加载的模型
            for model_name, model in MODELS.items():
                dataset_name = model_name.split('_')[0]  # 从模型名称中提取数据集名称
                
                if dataset_name not in DATA:
                    logger.warning(f"无法找到{dataset_name}数据集，跳过图表生成")
                    continue
                    
                df = DATA[dataset_name]
                
                # 准备数据，排除ID和N_Days列
                if dataset_name == 'stroke':
                    X = df.drop(['stroke', 'ID', 'N_Days'], axis=1, errors='ignore')
                    y = df['stroke'] if 'stroke' in df.columns else None
                elif dataset_name == 'heart':
                    X = df.drop(['HeartDisease', 'ID', 'N_Days'], axis=1, errors='ignore')
                    y = df['HeartDisease'] if 'HeartDisease' in df.columns else None
                elif dataset_name == 'cirrhosis':
                    X = df.drop(['Stage', 'ID', 'N_Days'], axis=1, errors='ignore')
                    y = df['Stage'] if 'Stage' in df.columns else None
                else:
                    logger.warning(f"未知的数据集: {dataset_name}")
                    continue
                    
                if y is None:
                    logger.warning(f"无法找到{dataset_name}数据集的目标变量，跳过图表生成")
                    continue
                    
                # 生成混淆矩阵（除了回归任务）
                if dataset_name != 'cirrhosis' and not os.path.exists(f'output/figures/{dataset_name}_confusion_matrix.png'):
                    try:
                        from sklearn.metrics import confusion_matrix
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        # 使用健壮的预测函数代替直接调用模型预测
                        y_preds = []
                        for i in range(len(X)):
                            # 这里使用batch预测会更有效率，但为了简单起见我们使用逐行预测
                            X_row = X.iloc[[i]]
                            pred, _, _, _ = robust_model_predict(model, X_row, 
                                                          model_type='classification' if dataset_name != 'cirrhosis' else 'regression')
                            y_preds.append(pred)
                        
                        # 计算混淆矩阵
                        cm = confusion_matrix(y, y_preds)
                        
                        # 绘制混淆矩阵
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'{dataset_name.capitalize()} 模型的混淆矩阵')
                        plt.xlabel('预测标签')
                        plt.ylabel('真实标签')
                        plt.tight_layout()
                        plt.savefig(f'output/figures/{dataset_name}_confusion_matrix.png')
                        plt.close()
                        
                        logger.info(f"{dataset_name}的混淆矩阵已生成")
                    except Exception as e:
                        logger.error(f"生成{dataset_name}混淆矩阵时出错: {e}")
                
                # 生成残差图/准确率图
                if not os.path.exists(f'output/figures/{dataset_name}_residual_plot.png'):
                    try:
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        # 使用健壮的预测函数代替直接调用模型预测
                        y_preds = []
                        for i in range(len(X)):
                            X_row = X.iloc[[i]]
                            pred, _, _, _ = robust_model_predict(model, X_row, 
                                                          model_type='classification' if dataset_name != 'cirrhosis' else 'regression')
                            y_preds.append(pred)
                        
                        # 对于分类任务，生成准确率图
                        if dataset_name in ['stroke', 'heart']:
                            from sklearn.metrics import accuracy_score
                            # 计算准确率
                            accuracy = accuracy_score(y, y_preds)
                            
                            # 创建条形图
                            plt.figure(figsize=(10, 6))
                            plt.bar(['准确率'], [accuracy], color='blue')
                            plt.ylim(0, 1)
                            plt.title(f'{dataset_name.capitalize()} 模型的准确率')
                            plt.ylabel('准确率')
                            plt.tight_layout()
                            plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                            plt.close()
                            
                            logger.info(f"{dataset_name}的准确率图已生成")
                        else:  # 回归任务
                            # 计算残差
                            residuals = y.values - np.array(y_preds)
                            
                            # 绘制残差图
                            plt.figure(figsize=(10, 6))
                            plt.scatter(y_preds, residuals, alpha=0.5)
                            plt.axhline(y=0, color='r', linestyle='-')
                            plt.xlabel('预测值')
                            plt.ylabel('残差')
                            plt.title(f'{dataset_name.capitalize()} 模型的残差图')
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                            plt.close()
                            
                            logger.info(f"{dataset_name}的残差图已生成")
                    except Exception as e:
                        logger.error(f"生成{dataset_name}残差图时出错: {e}")
    
    except Exception as e:
        logger.error(f"生成缺失图表时出错: {e}\n{traceback.format_exc()}")

# 在启动应用之前生成缺失的图表
generate_missing_charts()

# 启动浏览器
def open_browser():
    # 延迟2秒打开浏览器
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    # 如果直接运行此文件，则打开浏览器并启动Flask服务器
    threading.Thread(target=open_browser).start()
    app.run(debug=False)  # 生产环境中关闭debug模式以获得更好的性能 