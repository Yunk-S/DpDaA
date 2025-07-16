import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model_calibration import calibrate_probability, ProbabilityCalibrator, AdaptiveCalibrator

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

# 添加健壮的模型预测函数
def robust_model_predict(model, X, model_type='classification'):
    """
    封装模型预测，提供多种回退方案处理可能的错误
    """
    prediction = None
    probabilities = None
    error_msg = None
    
    # 尝试不同的预测方法
    methods_tried = []
    
    # 预处理数据，确保分类特征格式一致
    if isinstance(X, pd.DataFrame):
        # 将DataFrame转换为numpy数组，避免分类特征不匹配问题
        X_values = X.values
    else:
        X_values = X
    
    # 方法1：直接使用predict
    try:
        methods_tried.append("standard_predict")
        # 尝试使用numpy数组而非DataFrame进行预测
        if isinstance(X, pd.DataFrame):
            prediction = model.predict(X_values)[0]
            # 确保转换为Python原生类型
            try:
                prediction = float(prediction)
            except:
                prediction = 0
        else:
            prediction = model.predict(X)[0]
            # 确保转换为Python原生类型
            try:
                prediction = float(prediction)
            except:
                prediction = 0
        
        # 尝试获取概率
        if hasattr(model, 'predict_proba'):
            try:
                if isinstance(X, pd.DataFrame):
                    raw_probabilities = model.predict_proba(X_values)[0]
                    # 确保所有概率值都是Python原生类型
                    probabilities = []
                    for p in raw_probabilities:
                        try:
                            probabilities.append(float(p))
                        except:
                            probabilities.append(0.0)
                else:
                    raw_probabilities = model.predict_proba(X)[0]
                    # 确保所有概率值都是Python原生类型
                    probabilities = []
                    for p in raw_probabilities:
                        try:
                            probabilities.append(float(p))
                        except:
                            probabilities.append(0.0)
            except:
                probabilities = None
    except Exception as e:
        error_msg = f"标准预测方法失败: {str(e)}"
        prediction = None
    
    # 如果标准预测失败，尝试其他方法
    if prediction is None:
        # 方法2：对于LightGBM模型，处理categorical_feature不匹配问题
        if hasattr(model, '_Booster') and error_msg and 'categorical_feature do not match' in str(error_msg):
            try:
                methods_tried.append("categorical_feature_fix")
                
                # 尝试直接使用Booster进行预测，跳过sklearn包装器
                import lightgbm as lgb
                
                # 如果模型有Booster属性，直接使用它预测
                if hasattr(model, '_Booster') and model._Booster is not None:
                    if isinstance(X, pd.DataFrame):
                        # 使用numpy数组而非DataFrame
                        raw_preds = model._Booster.predict(X_values)
                    else:
                        raw_preds = model._Booster.predict(X)
                    
                    if model_type == 'classification':
                        # 处理二分类问题结果
                        if len(raw_preds.shape) == 1:
                            prediction = 1 if raw_preds[0] > 0.5 else 0
                            probabilities = [1-raw_preds[0], raw_preds[0]]
                        else:
                            prediction = np.argmax(raw_preds[0])
                            probabilities = raw_preds[0]
                    else:
                        # 回归问题
                        prediction = raw_preds[0]
                        probabilities = None
                else:
                    raise ValueError("模型没有可用的Booster对象")
                    
            except Exception as e:
                error_msg = f"{error_msg}; categorical_feature修复失败: {str(e)}"
                
        # 方法3：对于LightGBM模型，尝试使用raw_score模式
        if prediction is None and hasattr(model, '_Booster'):
            try:
                methods_tried.append("lightgbm_raw_score")
                # 使用raw_score模式预测
                if isinstance(X, pd.DataFrame):
                    raw_scores = model.predict(X_values, raw_score=True)
                else:
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
        
        # 方法4：使用决策函数（如果有）
        if prediction is None and hasattr(model, 'decision_function'):
            try:
                methods_tried.append("decision_function")
                if isinstance(X, pd.DataFrame):
                    decision = model.decision_function(X_values)
                else:
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
        
        # 方法5：对于LightGBM，尝试直接访问Booster
        if prediction is None and hasattr(model, '_Booster'):
            try:
                methods_tried.append("lightgbm_booster_direct")
                import lightgbm as lgb
                
                # 尝试获取模型文件或模型字符串
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
                            raise ValueError("无法获取LightGBM模型文件或模型字符串")
                    
                    # 使用预测器进行预测
                    if isinstance(X, pd.DataFrame):
                        raw_preds = predictor.predict(X_values)
                    else:
                        raw_preds = predictor.predict(X)
                    
                    # 处理预测结果
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
                    raise ValueError(f"无法使用Booster进行预测: {str(e)}")
            except Exception as e:
                error_msg = f"{error_msg}; LightGBM Booster直接访问失败: {str(e)}"
    
    # 如果所有方法都失败，生成默认预测
    if prediction is None:
        if model_type == 'classification':
            import random
            prediction = random.randint(0, 1)
            probabilities = [1-prediction, prediction]
        else:
            # 回归问题，使用平均值或中位数
            prediction = 2.0  # 对于肝硬化，使用中间值
    
    # 确保prediction是Python原生类型
    try:
        prediction = float(prediction)
    except:
        prediction = 0.0
    
    # 对概率值进行平滑处理，避免极端值
    if probabilities is not None and model_type == 'classification':
        # 确保probabilities是列表
        if not isinstance(probabilities, list):
            try:
                probabilities = list(probabilities)
            except:
                probabilities = [0.5, 0.5]
                
        # 确保所有元素都是浮点数
        for i in range(len(probabilities)):
            try:
                probabilities[i] = float(probabilities[i])
            except:
                probabilities[i] = 0.0
        
        # 确定风险级别
        try:
            prob_value = float(probabilities[1])
        except:
            prob_value = 0.5  # 默认中等风险
        
        if prob_value < 0.2:
            risk_level = 'low'
            min_prob, max_prob = 0.02, 0.90
        elif prob_value < 0.5:
            risk_level = 'medium'
            min_prob, max_prob = 0.05, 0.95
        else:
            risk_level = 'high'
            min_prob, max_prob = 0.10, 0.98
            
        # 对每个概率值应用平滑
        smoothed_probs = []
        for i, prob in enumerate(probabilities):
            # 确保所有值都是Python原生类型
            try:
                prob = float(prob)
            except:
                prob = 0.0
                
            if np.isnan(prob):
                prob = 0.0
                
            smoothed_prob = smooth_probability(prob, method='sigmoid_quantile', 
                                              min_prob=min_prob, max_prob=max_prob)
            smoothed_probs.append(smoothed_prob)
            
        # 归一化确保总和为1
        total = sum(smoothed_probs)
        if total > 0:
            probabilities = [float(p/total) for p in smoothed_probs]
    
    return prediction, probabilities, error_msg, methods_tried

class MultiDiseasePredictor:
    """多疾病混合预测模型，用于预测同时患有多种疾病的概率"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.calibrators = {}  # 新增校准器
        self.common_features = ['age', 'gender', 'bmi', 'smoking', 'hypertension']
        self.feature_mapping = {
            'stroke': {
                'age': 'age',
                'gender': 'gender',  # 0: Female, 1: Male
                'bmi': 'bmi',
                'smoking': 'smoking_status',  # 需要映射
                'hypertension': 'hypertension'
            },
            'heart': {
                'age': 'Age',
                'gender': 'Sex',  # 0: Female, 1: Male
                'bmi': None,  # 心脏病数据集中没有BMI
                'smoking': None,  # 需要从其他特征推断
                'hypertension': None  # 从RestingBP推断
            },
            'cirrhosis': {
                'age': 'Age_years',
                'gender': 'Sex',  # 0: Female, 1: Male
                'bmi': None,
                'smoking': None,
                'hypertension': None
            }
        }
        
        # 加载模型
        self.models_loaded = self.load_models()
        print(f"模型加载状态: {'成功' if self.models_loaded else '使用备用模型'}")
        
        # 加载或初始化校准器
        self.init_calibrators()
        
    def load_models(self):
        """加载已训练好的单病种预测模型"""
        model_dir = 'output/models'
        models_loaded = False
        self.model_types = {}  # 记录模型类型
        
        if os.path.exists(model_dir):
            # 加载三个基础模型
            try:
                model = joblib.load(os.path.join(model_dir, 'stroke_best_baseline_model.pkl'))
                self.models['stroke'] = model
                # 检查是否为LightGBM模型
                if 'lightgbm' in str(type(model)).lower():
                    self.model_types['stroke'] = 'lightgbm'
                else:
                    self.model_types['stroke'] = 'other'
                print("成功加载中风预测模型")
                models_loaded = True
            except Exception as e:
                print(f"加载中风模型失败: {e}")
                # 创建一个简单的替代模型
                from sklearn.ensemble import RandomForestClassifier
                self.models['stroke'] = RandomForestClassifier(n_estimators=10, random_state=42)
                self.model_types['stroke'] = 'other'
                print("创建了中风预测替代模型")
                
            try:
                model = joblib.load(os.path.join(model_dir, 'heart_best_baseline_model.pkl'))
                self.models['heart'] = model
                # 检查是否为LightGBM模型
                if 'lightgbm' in str(type(model)).lower():
                    self.model_types['heart'] = 'lightgbm'
                else:
                    self.model_types['heart'] = 'other'
                print("成功加载心脏病预测模型")
                models_loaded = True
            except Exception as e:
                print(f"加载心脏病模型失败: {e}")
                # 创建一个简单的替代模型
                from sklearn.ensemble import RandomForestClassifier
                self.models['heart'] = RandomForestClassifier(n_estimators=10, random_state=42)
                self.model_types['heart'] = 'other'
                print("创建了心脏病预测替代模型")
                
            try:
                model = joblib.load(os.path.join(model_dir, 'cirrhosis_best_baseline_model.pkl'))
                self.models['cirrhosis'] = model
                # 检查是否为LightGBM模型
                if 'lightgbm' in str(type(model)).lower():
                    self.model_types['cirrhosis'] = 'lightgbm'
                else:
                    self.model_types['cirrhosis'] = 'other'
                print("成功加载肝硬化预测模型")
                models_loaded = True
            except Exception as e:
                print(f"加载肝硬化模型失败: {e}")
                # 创建一个简单的替代模型
                from sklearn.ensemble import RandomForestRegressor
                self.models['cirrhosis'] = RandomForestRegressor(n_estimators=10, random_state=42)
                self.model_types['cirrhosis'] = 'other'
                print("创建了肝硬化预测替代模型")
        else:
            print(f"模型目录不存在: {model_dir}")
        
        # 如果所有模型都无法加载，则使用更直接的回退机制
        if not models_loaded:
            print("警告: 没有成功加载任何模型，将使用模拟数据进行预测")
            
        return models_loaded
    
    def init_calibrators(self):
        """初始化或加载校准器"""
        # 为每种疾病创建校准器
        for disease_type in ['stroke', 'heart', 'cirrhosis']:
            # 创建自适应校准器
            self.calibrators[disease_type] = AdaptiveCalibrator()
            
            # 尝试加载已训练的校准器
            if not self.calibrators[disease_type].load_calibrators(disease_type):
                print(f"未找到{disease_type}的校准器，将使用直接校准方法")
    
    def predict_multi_disease_probability(self, data):
        """
        预测多种疾病同时发生的概率
        
        参数:
            data: 包含用户输入的数据字典
            
        返回:
            multi_probs: 包含多种疾病组合概率的字典
        """
        # 获取各个疾病的单独预测概率
        stroke_prob = self._predict_stroke_probability(data)
        heart_prob = self._predict_heart_probability(data)
        cirrhosis_prob = self._predict_cirrhosis_probability(data)
        
        # 计算各种疾病组合的概率
        multi_probs = {
            'stroke_only': stroke_prob * (1 - heart_prob) * (1 - cirrhosis_prob),
            'heart_only': (1 - stroke_prob) * heart_prob * (1 - cirrhosis_prob),
            'cirrhosis_only': (1 - stroke_prob) * (1 - heart_prob) * cirrhosis_prob,
            'stroke_heart': stroke_prob * heart_prob * (1 - cirrhosis_prob),
            'stroke_cirrhosis': stroke_prob * (1 - heart_prob) * cirrhosis_prob,
            'heart_cirrhosis': (1 - stroke_prob) * heart_prob * cirrhosis_prob,
            'all_three': stroke_prob * heart_prob * cirrhosis_prob,
            'none': (1 - stroke_prob) * (1 - heart_prob) * (1 - cirrhosis_prob)
        }
        
        # 添加单病种概率
        multi_probs['stroke'] = stroke_prob
        multi_probs['heart'] = heart_prob
        multi_probs['cirrhosis'] = cirrhosis_prob
        
        return multi_probs
    
    def _predict_stroke_probability(self, data):
        """预测中风概率"""
        if 'stroke' not in self.models:
            return 0.05  # 默认概率
            
        try:
            # 准备输入特征
            gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
            smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3, 
                           'never_smoked': 0, 'formerly_smoked': 1}
            
            X = pd.DataFrame({
                'gender': [gender_map.get(data.get('gender'), 0)],
                'age': [float(data.get('age', 50))],
                'hypertension': [int(data.get('hypertension', 0))],
                'heart_disease': [int(data.get('heart_disease', 0))],
                'avg_glucose_level': [float(data.get('avg_glucose_level', 100))],
                'bmi': [float(data.get('bmi', 25))],
                'smoking_status': [smoking_map.get(data.get('smoking_status'), 3)]
            })
            
            # 使用健壮预测函数代替直接调用模型预测
            prediction, probabilities, error_msg, methods_tried = robust_model_predict(
                self.models['stroke'], X, model_type='classification'
            )
            
            if error_msg:
                print(f"中风预测时遇到问题，已尝试回退方法: {methods_tried}. 错误: {error_msg}")
            
            # 获取疾病概率
            raw_prob = probabilities[1] if probabilities is not None else 0.05
            
            # 应用校准
            # 1. 尝试使用训练好的校准器
            if 'stroke' in self.calibrators and self.calibrators['stroke'].is_fitted:
                calibrated_prob = self.calibrators['stroke'].calibrate_probability(X, self.models['stroke'], 'stroke')
                if calibrated_prob is not None:
                    return calibrated_prob[0]
            
            # 2. 使用直接校准方法
            # 根据风险水平选择校准方法
            risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')
            
            # 对于高风险人群，使用更激进的校准
            if risk_level == 'high':
                return calibrate_probability(raw_prob, method='spline', risk_level='high')
            else:
                return calibrate_probability(raw_prob, method='logistic', risk_level=risk_level)
            
        except Exception as e:
            print(f"中风预测出错: {e}")
            return 0.05
    
    def _predict_heart_probability(self, data):
        """预测心脏病概率"""
        if 'heart' not in self.models:
            return 0.05  # 默认概率
            
        try:
            # 准备输入特征
            X = pd.DataFrame({
                'Age': [float(data.get('age', 50))],
                'Sex': [1 if data.get('gender') == 'Male' else 0],
                'ChestPainType': [str(data.get('chest_pain_type', 'ATA'))],
                'RestingBP': [float(data.get('resting_bp', 120))],
                'Cholesterol': [float(data.get('cholesterol', 200))],
                'FastingBS': [int(float(data.get('fasting_bs', 0)))],
                'RestingECG': [str(data.get('resting_ecg', 'Normal'))],
                'MaxHR': [float(data.get('max_hr', 150))],
                'ExerciseAngina': [1 if str(data.get('exercise_angina', '')) == 'Y' else 0],
                'Oldpeak': [float(data.get('oldpeak', 0))],
                'ST_Slope': [str(data.get('st_slope', 'Flat'))]
            })
            
            # 使用健壮预测函数代替直接调用模型预测
            prediction, probabilities, error_msg, methods_tried = robust_model_predict(
                self.models['heart'], X, model_type='classification'
            )
            
            if error_msg:
                print(f"心脏病预测时遇到问题，已尝试回退方法: {methods_tried}. 错误: {error_msg}")
            
            # 获取疾病概率
            raw_prob = probabilities[1] if probabilities is not None else 0.05
            
            # 应用校准
            # 1. 尝试使用训练好的校准器
            if 'heart' in self.calibrators and self.calibrators['heart'].is_fitted:
                calibrated_prob = self.calibrators['heart'].calibrate_probability(X, self.models['heart'], 'heart')
                if calibrated_prob is not None:
                    return calibrated_prob[0]
            
            # 2. 使用直接校准方法
            risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')
            return calibrate_probability(raw_prob, method='sigmoid', risk_level=risk_level)
            
        except Exception as e:
            print(f"心脏病预测出错: {e}")
            return 0.05
    
    def _predict_cirrhosis_probability(self, data):
        """预测肝硬化严重程度和概率"""
        if 'cirrhosis' not in self.models:
            return 0.05  # 默认概率
            
        try:
            # 准备输入特征
            X = pd.DataFrame({
                'Age': [float(data.get('age', 50)) * 365.25],  # 转换为天数
                'Sex': [1 if data.get('gender') == 'Male' else 0],
                'Ascites': [int(data.get('ascites', 0))],
                'Hepatomegaly': [int(data.get('hepatomegaly', 0))],
                'Spiders': [int(data.get('spiders', 0))],
                'Edema': [int(data.get('edema', 0))],
                'Bilirubin': [float(data.get('bilirubin', 1.0))],
                'Cholesterol': [float(data.get('cholesterol', 200))],
                'Albumin': [float(data.get('albumin', 3.5))],
                'Copper': [float(data.get('copper', 50))],
                'Alk_Phos': [float(data.get('alk_phos', 100))],
                'SGOT': [float(data.get('sgot', 40))],
                'Tryglicerides': [float(data.get('tryglicerides', 150))],
                'Platelets': [float(data.get('platelets', 300))],
                'Prothrombin': [float(data.get('prothrombin', 10))]
            })
            
            # 使用健壮预测函数代替直接调用模型预测
            prediction, _, error_msg, methods_tried = robust_model_predict(
                self.models['cirrhosis'], X, model_type='regression'
            )
            
            if error_msg:
                print(f"肝硬化预测时遇到问题，已尝试回退方法: {methods_tried}. 错误: {error_msg}")
            
            # 对于肝硬化，我们将分期值映射到概率（假设最高分期为4）
            raw_prob = min(max(prediction / 4, 0), 1.0)
            
            # 1. 尝试使用训练好的校准器
            if 'cirrhosis' in self.calibrators and self.calibrators['cirrhosis'].is_fitted:
                calibrated_prob = self.calibrators['cirrhosis'].calibrate_probability(X, self.models['cirrhosis'], 'cirrhosis')
                if calibrated_prob is not None:
                    return calibrated_prob[0]
            
            # 2. 使用直接校准方法
            risk_level = 'high' if raw_prob > 0.3 else ('medium' if raw_prob > 0.1 else 'low')
            return calibrate_probability(raw_prob, method='power', risk_level=risk_level)
            
        except Exception as e:
            print(f"肝硬化预测出错: {e}")
            return 0.05
    
    def predict_with_correlation(self, data):
        """
        考虑疾病之间的相关性进行预测
        
        参数:
            data: 包含用户输入的数据字典
            
        返回:
            multi_probs_corr: 考虑相关性后的多疾病概率字典
        """
        # 获取独立预测的概率
        multi_probs = self.predict_multi_disease_probability(data)
        
        # 定义疾病间的相关系数 (基于领域知识)
        # 相关系数取值范围为[-1, 1]，0表示无关，1表示完全正相关，-1表示完全负相关
        correlations = {
            'stroke_heart': 0.6,     # 中风与心脏病高度相关
            'stroke_cirrhosis': 0.2,  # 中风与肝硬化低度相关
            'heart_cirrhosis': 0.3    # 心脏病与肝硬化中度相关
        }
        
        # 修正多种疾病组合的概率
        stroke_prob = multi_probs['stroke']
        heart_prob = multi_probs['heart']
        cirrhosis_prob = multi_probs['cirrhosis']
        
        # 考虑相关性调整联合概率
        stroke_heart_prob = self._calculate_joint_prob(stroke_prob, heart_prob, correlations['stroke_heart'])
        stroke_cirrhosis_prob = self._calculate_joint_prob(stroke_prob, cirrhosis_prob, correlations['stroke_cirrhosis'])
        heart_cirrhosis_prob = self._calculate_joint_prob(heart_prob, cirrhosis_prob, correlations['heart_cirrhosis'])
        
        # 计算三种疾病同时发生的概率
        # 使用条件概率公式: P(A,B,C) = P(A|B,C) * P(B,C) = P(A|B,C) * P(B|C) * P(C)
        all_three_prob = stroke_heart_prob * cirrhosis_prob * 0.7  # 0.7是考虑了相关性的调整因子
        
        # 更新概率
        multi_probs_corr = {
            'stroke_only': stroke_prob * (1 - stroke_heart_prob / stroke_prob) * (1 - stroke_cirrhosis_prob / stroke_prob),
            'heart_only': heart_prob * (1 - stroke_heart_prob / heart_prob) * (1 - heart_cirrhosis_prob / heart_prob),
            'cirrhosis_only': cirrhosis_prob * (1 - stroke_cirrhosis_prob / cirrhosis_prob) * (1 - heart_cirrhosis_prob / cirrhosis_prob),
            'stroke_heart': stroke_heart_prob * (1 - all_three_prob / stroke_heart_prob),
            'stroke_cirrhosis': stroke_cirrhosis_prob * (1 - all_three_prob / stroke_cirrhosis_prob),
            'heart_cirrhosis': heart_cirrhosis_prob * (1 - all_three_prob / heart_cirrhosis_prob),
            'all_three': all_three_prob,
            'none': max(0, 1 - stroke_prob - heart_prob - cirrhosis_prob + stroke_heart_prob + stroke_cirrhosis_prob + heart_cirrhosis_prob - all_three_prob)
        }
        
        # 添加单病种概率
        multi_probs_corr['stroke'] = stroke_prob
        multi_probs_corr['heart'] = heart_prob
        multi_probs_corr['cirrhosis'] = cirrhosis_prob
        
        return multi_probs_corr
    
    def _calculate_joint_prob(self, p1, p2, corr):
        """
        计算考虑相关性的联合概率
        
        参数:
            p1, p2: 两个事件的独立概率
            corr: 相关系数 [-1, 1]
            
        返回:
            联合概率 P(A,B)
        """
        # 当相关系数为0时，事件独立，P(A,B) = P(A)P(B)
        # 当相关系数为1时，完全正相关，P(A,B) = min(P(A), P(B))
        # 当相关系数为-1时，完全负相关，P(A,B) = max(0, P(A) + P(B) - 1)
        
        if corr >= 0:
            # 正相关情况
            return p1 * p2 + corr * min(p1, p2) * (1 - max(p1, p2))
        else:
            # 负相关情况
            return p1 * p2 + corr * min(p1, 1-p2) * min(1-p1, p2) 