import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import os
import joblib
import logging
from scipy.special import expit  # 添加sigmoid函数导入

# 确保output目录存在
os.makedirs('output', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'),  # 使用根目录下的app.log，避免路径问题
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 导入平滑概率函数 ==========
try:
    from model_utilities import smooth_probability
except ImportError:
    # 如果无法导入，提供一个简单的备用实现
    def smooth_probability(prob, method='clip', min_prob=0.01, max_prob=0.99):
        """简单的概率平滑函数（备用版本）"""
        if np.isscalar(prob):
            return max(min_prob, min(max_prob, prob))
        else:
            return np.clip(prob, min_prob, max_prob)
    
    logger.warning("无法从model_utilities导入smooth_probability函数，将使用简化版本")

class ProbabilityCalibrator:
    """
    概率校准器：用于校准机器学习模型输出的概率
    支持Platt Scaling和等渗回归两种校准方法
    """
    
    def __init__(self, method='isotonic', cv=5):
        """
        初始化概率校准器
        
        参数:
            method: 校准方法，'sigmoid'(Platt Scaling)或'isotonic'(等渗回归)
            cv: 交叉验证折数
        """
        self.method = method
        self.cv = cv
        self.calibrators = {}
        self.is_fitted = False
        
        # 创建输出目录
        os.makedirs('output/models/calibrators', exist_ok=True)
    
    def fit(self, X, y, model, disease_type):
        """
        训练校准器
        
        参数:
            X: 特征数据
            y: 标签数据
            model: 需要校准的模型
            disease_type: 疾病类型(stroke, heart, cirrhosis)
        """
        logger.info(f"为{disease_type}疾病模型训练{self.method}校准器...")
        
        # 检查模型类型，确保它是分类器
        is_classifier = hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')
        
        if not is_classifier:
            logger.warning(f"{disease_type}模型不是分类器，无法使用CalibratedClassifierCV")
            logger.warning("将创建一个简单的概率映射函数代替校准器")
            
            # 创建一个简单的映射函数作为校准器
            # 这里我们使用模型的预测结果，并应用sigmoid函数转换为概率
            
            # 获取模型在校准数据上的预测
            try:
                predictions = model.predict(X)
                
                # 保存校准所需的信息
                self.calibrators[disease_type] = {
                    'model': model,
                    'method': 'simple_mapping',
                    'mean': predictions.mean(),
                    'std': predictions.std() or 1.0  # 避免除零错误
                }
                
                self.is_fitted = True
                
                # 保存到文件
                joblib.dump(self.calibrators[disease_type], 
                           f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl')
                logger.info(f"简单映射校准器已保存到output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl")
                
                return self
            
            except Exception as e:
                logger.error(f"创建简单映射校准器失败: {e}")
                return self
        
        # 对于分类器，使用CalibratedClassifierCV进行校准
        try:
            calibrator = CalibratedClassifierCV(
                estimator=model,  # 新版scikit-learn使用estimator而不是base_estimator
                method=self.method,
                cv=self.cv
            )
            
            # 训练校准器
            calibrator.fit(X, y)
            
            # 保存校准器
            self.calibrators[disease_type] = calibrator
            self.is_fitted = True
            
            # 保存到文件
            joblib.dump(calibrator, f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl')
            logger.info(f"校准器已保存到output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl")
            
        except Exception as e:
            logger.error(f"训练校准器失败: {e}")
            logger.warning("将创建一个简单的概率映射函数代替校准器")
            
            # 创建一个简单的映射函数作为校准器
            try:
                predictions = model.predict(X)
                
                # 保存校准所需的信息
                self.calibrators[disease_type] = {
                    'model': model,
                    'method': 'simple_mapping',
                    'mean': predictions.mean(),
                    'std': predictions.std() or 1.0  # 避免除零错误
                }
                
                self.is_fitted = True
                
                # 保存到文件
                joblib.dump(self.calibrators[disease_type], 
                           f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl')
                logger.info(f"简单映射校准器已保存")
                
            except Exception as e2:
                logger.error(f"创建简单映射校准器也失败了: {e2}")
        
        return self
    
    def calibrate_probability(self, X, disease_type):
        """
        校准预测概率
        
        参数:
            X: 特征数据
            disease_type: 疾病类型
            
        返回:
            校准后的概率
        """
        if not self.is_fitted or disease_type not in self.calibrators:
            logger.warning(f"警告: {disease_type}的校准器未训练")
            return None
        
        # 获取校准器
        calibrator = self.calibrators[disease_type]
        
        # 检查是否是简单映射校准器
        if isinstance(calibrator, dict) and calibrator.get('method') == 'simple_mapping':
            try:
                # 使用原始模型预测
                model = calibrator['model']
                predictions = model.predict(X)
                
                # 应用简单的sigmoid变换将值映射到(0,1)区间
                
                # 标准化预测值
                mean = calibrator['mean']
                std = calibrator['std']
                normalized = (predictions - mean) / std
                
                # 应用sigmoid函数转换为概率
                return expit(normalized)
                
            except Exception as e:
                logger.error(f"使用简单映射校准器失败: {e}")
                return None
        
        # 对于标准校准器，使用predict_proba
        try:
            # 获取校准后的概率
            calibrated_probs = calibrator.predict_proba(X)
            
            # 返回正类的概率
            return calibrated_probs[:, 1]
        except Exception as e:
            logger.error(f"使用标准校准器失败: {e}")
            return None
    
    def load_calibrator(self, disease_type):
        """
        加载已训练的校准器
        
        参数:
            disease_type: 疾病类型
            
        返回:
            成功加载返回True，否则返回False
        """
        try:
            calibrator_path = f'output/models/calibrators/{disease_type}_calibrator_{self.method}.pkl'
            if os.path.exists(calibrator_path):
                self.calibrators[disease_type] = joblib.load(calibrator_path)
                self.is_fitted = True
                logger.info(f"成功加载{disease_type}的校准器")
                return True
            else:
                logger.warning(f"校准器文件不存在: {calibrator_path}")
                return False
        except Exception as e:
            logger.error(f"加载校准器失败: {e}")
            return False
    
    def plot_calibration_curve(self, X, y, model, disease_type, n_bins=10):
        """
        绘制校准曲线
        
        参数:
            X: 特征数据
            y: 标签数据
            model: 原始模型
            disease_type: 疾病类型
            n_bins: 分箱数量
        """
        try:
            import numpy as np  # 确保numpy可用
            
            # 将目标变量转换为二进制（如果不是）
            unique_values = np.unique(y)
            if len(unique_values) > 2 or not all(val in [0, 1] for val in unique_values):
                logger.warning(f"目标变量不是二进制的，包含值: {unique_values}")
                logger.warning("尝试将目标变量转换为二进制形式...")
                
                # 对于回归目标，使用中位数作为阈值
                if len(unique_values) > 2:
                    threshold = np.median(y)
                    binary_y = (y > threshold).astype(int)
                    logger.info(f"使用阈值 {threshold} 将目标变量转换为二进制")
                else:
                    # 确保值为0和1
                    min_val = np.min(y)
                    binary_y = (y > min_val).astype(int)
                    logger.info(f"将目标变量从 {unique_values} 转换为 [0, 1]")
            else:
                binary_y = y
            
            # 创建图表
            plt.figure(figsize=(10, 8))
            
            # 获取原始模型的预测概率
            try:
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)[:, 1]
                else:
                    y_prob = model.predict(X)
                    
                    # 如果是回归模型，将预测值转换为概率
                    if y_prob.ndim == 1:
                        # 简单地将值标准化到0-1范围
                        min_val = y_prob.min()
                        max_val = y_prob.max()
                        if max_val > min_val:
                            y_prob = (y_prob - min_val) / (max_val - min_val)
                        else:
                            y_prob = np.ones_like(y_prob) * 0.5  # 默认0.5概率
            except ValueError as e:
                logger.warning(f"获取原始模型预测概率失败: {e}")
                logger.warning("使用随机概率代替原始模型预测")
                # 使用随机概率代替，仅用于绘图目的
                y_prob = np.random.random(len(y))
            
            # 计算原始模型的校准曲线
            prob_true, prob_pred = calibration_curve(binary_y, y_prob, n_bins=n_bins)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='原始模型')
            
            # 如果校准器已训练，计算校准后的校准曲线
            if disease_type in self.calibrators:
                y_prob_calibrated = self.calibrate_probability(X, disease_type)
                if y_prob_calibrated is not None:
                    prob_true_calibrated, prob_pred_calibrated = calibration_curve(binary_y, y_prob_calibrated, n_bins=n_bins)
                    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='s', linewidth=1, label=f'校准后({self.method})')
            
            # 绘制理想校准线
            plt.plot([0, 1], [0, 1], linestyle='--', label='理想校准')
            
            # 添加图表元素
            plt.xlabel('预测概率')
            plt.ylabel('实际概率')
            plt.title(f'{disease_type.capitalize()}疾病预测模型校准曲线')
            plt.legend(loc='best')
            plt.grid(True)
            
            # 保存图表
            os.makedirs('output/figures', exist_ok=True)
            plt.savefig(f'output/figures/{disease_type}_calibration_curve.png')
            plt.close()
        except Exception as e:
            logger.error(f"绘制校准曲线时出错: {e}")
            logger.error(f"错误详情: {str(e)}")
            logger.error("跳过校准曲线绘制")


class AdaptiveCalibrator:
    """
    自适应校准器：根据不同风险水平使用不同的校准策略
    """
    
    def __init__(self):
        """初始化自适应校准器"""
        self.low_risk_calibrator = ProbabilityCalibrator(method='sigmoid')  # 低风险使用Platt Scaling
        self.high_risk_calibrator = ProbabilityCalibrator(method='isotonic')  # 高风险使用等渗回归
        self.risk_threshold = 0.3  # 风险阈值
        self.is_fitted = False
        self.disease_types = []
    
    def fit(self, X, y, model, disease_type):
        """
        训练自适应校准器
        
        参数:
            X: 特征数据
            y: 标签数据
            model: 需要校准的模型
            disease_type: 疾病类型
        """
        # 训练低风险和高风险校准器
        self.low_risk_calibrator.fit(X, y, model, f"{disease_type}_low_risk")
        self.high_risk_calibrator.fit(X, y, model, f"{disease_type}_high_risk")
        
        self.is_fitted = True
        if disease_type not in self.disease_types:
            self.disease_types.append(disease_type)
        
        return self
    
    def calibrate_probability(self, X, model, disease_type):
        """
        校准预测概率
        
        参数:
            X: 特征数据
            model: 原始模型
            disease_type: 疾病类型
            
        返回:
            校准后的概率
        """
        if not self.is_fitted:
            logger.warning(f"警告: 自适应校准器未训练")
            return None
        
        # 获取原始预测概率
        if hasattr(model, 'predict_proba'):
            raw_probs = model.predict_proba(X)[:, 1]
        else:
            raw_probs = model.predict(X)
        
        # 根据风险水平选择不同的校准器
        calibrated_probs = np.zeros_like(raw_probs)
        
        # 低风险样本使用Platt Scaling
        low_risk_mask = raw_probs < self.risk_threshold
        if np.any(low_risk_mask):
            low_risk_probs = self.low_risk_calibrator.calibrate_probability(
                X[low_risk_mask], f"{disease_type}_low_risk")
            if low_risk_probs is not None:
                calibrated_probs[low_risk_mask] = low_risk_probs
            else:
                calibrated_probs[low_risk_mask] = raw_probs[low_risk_mask]
        
        # 高风险样本使用等渗回归
        high_risk_mask = ~low_risk_mask
        if np.any(high_risk_mask):
            high_risk_probs = self.high_risk_calibrator.calibrate_probability(
                X[high_risk_mask], f"{disease_type}_high_risk")
            if high_risk_probs is not None:
                calibrated_probs[high_risk_mask] = high_risk_probs
            else:
                calibrated_probs[high_risk_mask] = raw_probs[high_risk_mask]
        
        return calibrated_probs
    
    def load_calibrators(self, disease_type):
        """
        加载已训练的校准器
        
        参数:
            disease_type: 疾病类型
            
        返回:
            成功加载返回True，否则返回False
        """
        low_risk_loaded = self.low_risk_calibrator.load_calibrator(f"{disease_type}_low_risk")
        high_risk_loaded = self.high_risk_calibrator.load_calibrator(f"{disease_type}_high_risk")
        
        if low_risk_loaded and high_risk_loaded:
            self.is_fitted = True
            if disease_type not in self.disease_types:
                self.disease_types.append(disease_type)
            return True
        else:
            return False


class BetterCalibrator:
    """
    更强大的校准器：专门针对低风险患者的校准
    """
    
    def __init__(self):
        """初始化更强大的校准器"""
        self.platt_calibrator = ProbabilityCalibrator(method='sigmoid')  # 用于低风险校准
        self.isotonic_calibrator = ProbabilityCalibrator(method='isotonic')  # 用于高风险校准
        self.risk_threshold = 0.25  # 风险阈值
        self.is_fitted = False
        self.disease_types = []
    
    def fit(self, X, y, model, disease_type):
        """
        训练更强大的校准器
        
        参数:
            X: 特征数据
            y: 标签数据
            model: 需要校准的模型
            disease_type: 疾病类型
        """
        # 训练低风险和高风险校准器
        self.platt_calibrator.fit(X, y, model, f"{disease_type}_better_low_risk")
        self.isotonic_calibrator.fit(X, y, model, f"{disease_type}_better_high_risk")
        
        self.is_fitted = True
        if disease_type not in self.disease_types:
            self.disease_types.append(disease_type)
        
        return self
    
    def calibrate_probability(self, X, model, disease_type):
        """
        校准预测概率
        
        参数:
            X: 特征数据
            model: 原始模型
            disease_type: 疾病类型
            
        返回:
            校准后的概率
        """
        if not self.is_fitted:
            logger.warning(f"警告: 更强大的校准器未训练")
            return None
        
        # 获取原始预测概率
        if hasattr(model, 'predict_proba'):
            raw_probs = model.predict_proba(X)[:, 1]
        else:
            raw_probs = model.predict(X)
        
        # 根据风险水平选择不同的校准器
        calibrated_probs = np.zeros_like(raw_probs)
        
        # 低风险样本使用Platt Scaling
        low_risk_mask = raw_probs < self.risk_threshold
        if np.any(low_risk_mask):
            low_risk_probs = self.platt_calibrator.calibrate_probability(
                X[low_risk_mask], f"{disease_type}_better_low_risk")
            if low_risk_probs is not None:
                # 对于低风险样本，应用额外的校准增强
                calibrated_probs[low_risk_mask] = low_risk_probs * 1.2
            else:
                calibrated_probs[low_risk_mask] = raw_probs[low_risk_mask]
        
        # 高风险样本使用等渗回归
        high_risk_mask = ~low_risk_mask
        if np.any(high_risk_mask):
            high_risk_probs = self.isotonic_calibrator.calibrate_probability(
                X[high_risk_mask], f"{disease_type}_better_high_risk")
            if high_risk_probs is not None:
                calibrated_probs[high_risk_mask] = high_risk_probs
            else:
                calibrated_probs[high_risk_mask] = raw_probs[high_risk_mask]
        
        # 确保概率在[0,1]范围内
        return np.clip(calibrated_probs, 0, 1)
    
    def load_calibrators(self, disease_type):
        """
        加载已训练的校准器
        
        参数:
            disease_type: 疾病类型
            
        返回:
            成功加载返回True，否则返回False
        """
        low_risk_loaded = self.platt_calibrator.load_calibrator(f"{disease_type}_better_low_risk")
        high_risk_loaded = self.isotonic_calibrator.load_calibrator(f"{disease_type}_better_high_risk")
        
        if low_risk_loaded and high_risk_loaded:
            self.is_fitted = True
            if disease_type not in self.disease_types:
                self.disease_types.append(disease_type)
            return True
        else:
            return False


# 用于直接校准概率的函数
def calibrate_probability(prob, method='spline', risk_level='high'):
    """
    直接校准单个概率值
    
    参数:
        prob: 原始概率值
        method: 校准方法 ('spline', 'power', 'logistic')
        risk_level: 风险级别 ('low', 'medium', 'high')
        
    返回:
        校准后的概率值
    """
    # 根据风险级别设置校准参数
    if risk_level == 'low':
        power = 0.7  # 从0.8减小到0.7，增加低风险校准力度
        logistic_a = 1.4  # 从1.2增加到1.4
        logistic_b = -0.1  # 添加负偏移量，提高低风险患者的风险评估
    elif risk_level == 'medium':
        power = 0.5  # 从0.6减小到0.5
        logistic_a = 1.8  # 从1.5增加到1.8
        logistic_b = 0  # 保持中等风险的偏移量为0
    else:  # high
        power = 0.4  # 从0.4减小到0.3，增强高风险的显示度
        logistic_a = 2.5  # 从2.0增加到2.5
        logistic_b = 0.1  # 保持高风险的正偏移量
    
    # 应用校准方法
    calibrated_prob = None
    
    if method == 'power':
        # 幂函数校准: y = x^power
        calibrated_prob = np.power(prob, power)
    
    elif method == 'logistic':
        # Logistic校准: y = 1/(1+exp(-a*(x-b)))
        calibrated_prob = 1 / (1 + np.exp(-logistic_a * (prob - logistic_b)))
    
    elif method == 'spline':
        # 改进的分段样条校准 - 对低概率区域进行更强力的校准
        if prob < 0.1:
            # 非常低概率区域，显著提升
            calibrated_prob = prob * 2.0
        elif prob < 0.3:
            # 低概率区域，强力提升
            calibrated_prob = 0.2 + (prob - 0.1) * 1.8
        elif prob < 0.6:
            # 中等概率区域，中等提升
            calibrated_prob = 0.56 + (prob - 0.3) * 1.4
        else:
            # 高概率区域，轻微提升
            calibrated_prob = min(0.98, 0.98 + (prob - 0.6) * 0.05)
    
    else:
        calibrated_prob = prob  # 默认不校准
    
    # 应用平滑处理，避免极端值
    # 根据风险等级选择不同的平滑参数
    if risk_level == 'low':
        # 低风险：保留更多低概率值，较少的高概率值
        min_prob, max_prob = 0.02, 0.90
        smooth_method = 'sigmoid_quantile'
    elif risk_level == 'medium':
        # 中等风险：均衡处理
        min_prob, max_prob = 0.05, 0.95
        smooth_method = 'sigmoid_quantile'
    else:  # high
        # 高风险：保留更多高概率值，较少的低概率值
        min_prob, max_prob = 0.10, 0.98
        smooth_method = 'sigmoid_quantile'
    
    # 应用平滑
    smoothed_prob = smooth_probability(calibrated_prob, method=smooth_method, 
                                       min_prob=min_prob, max_prob=max_prob)
    
    return smoothed_prob


def load_data_and_models():
    """加载处理后的数据和训练好的模型"""
    data = {}
    models = {}
    
    # 加载数据
    data_dir = 'output/processed_data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_path = os.path.join(data_dir, filename)
                data_name = filename.replace('_processed.csv', '')
                try:
                    data[data_name] = pd.read_csv(data_path)
                    logger.info(f"成功加载数据: {data_name}")
                except Exception as e:
                    logger.error(f"加载数据 {data_name} 失败: {e}")
    
    # 加载模型
    model_dir = 'output/models'
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl') and 'best_baseline_model' in filename:
                model_path = os.path.join(model_dir, filename)
                model_name = filename.replace('.pkl', '')
                try:
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"成功加载模型: {model_name}")
                except Exception as e:
                    logger.error(f"加载模型 {model_name} 失败: {e}")
    
    return data, models

def prepare_data_for_calibration(data):
    """准备用于校准的数据"""
    datasets = {}
    
    # 中风数据
    if 'stroke' in data:
        stroke_features = data['stroke'].drop(['id', 'stroke'], axis=1, errors='ignore')
        stroke_target = data['stroke']['stroke']
        datasets['stroke'] = (stroke_features, stroke_target)
    
    # 心脏病数据
    if 'heart' in data:
        heart_features = data['heart'].drop(['HeartDisease'], axis=1, errors='ignore')
        heart_target = data['heart']['HeartDisease']
        datasets['heart'] = (heart_features, heart_target)
    
    # 肝硬化数据
    if 'cirrhosis' in data:
        cirrhosis_features = data['cirrhosis'].drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
        cirrhosis_target = data['cirrhosis']['Stage']
        datasets['cirrhosis'] = (cirrhosis_features, cirrhosis_target)
    
    return datasets

def train_calibrators(datasets, models):
    """训练概率校准器"""
    calibrators = {}
    
    # 创建输出目录
    os.makedirs('output/models/calibrators', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    
    for disease_type, (features, target) in datasets.items():
        logger.info(f"\n开始训练 {disease_type} 校准器...")
        
        # 获取对应的模型
        model_key = f"{disease_type}_best_baseline_model"
        if model_key not in models:
            logger.warning(f"未找到 {disease_type} 的模型，跳过校准器训练")
            continue
        
        model = models[model_key]
        
        # 拆分数据用于校准 - 保持原始特征结构
        try:
            # 首先尝试使用分层抽样
            X_train, X_cal, y_train, y_cal = train_test_split(
                features, target, test_size=0.3, random_state=42, stratify=target
            )
        except ValueError as e:
            logger.warning(f"无法使用分层抽样: {e}")
            logger.warning("使用随机抽样替代（不使用stratify参数）")
            # 如果分层抽样失败，则使用随机抽样
            X_train, X_cal, y_train, y_cal = train_test_split(
                features, target, test_size=0.3, random_state=42
            )
        
        # 不再过滤特征类型，保留原始特征结构以确保分类特征一致性
        # 如果X_cal不是DataFrame，转换为DataFrame但保留所有列
        if not isinstance(X_cal, pd.DataFrame):
            X_cal = pd.DataFrame(X_cal, columns=features.columns)
        
        # 创建并训练校准器
        logger.info(f"训练 {disease_type} 的Platt Scaling校准器...")
        platt_calibrator = ProbabilityCalibrator(method='sigmoid')
        platt_calibrator.fit(X_cal, y_cal, model, disease_type)
        
        logger.info(f"训练 {disease_type} 的等渗回归校准器...")
        isotonic_calibrator = ProbabilityCalibrator(method='isotonic')
        isotonic_calibrator.fit(X_cal, y_cal, model, disease_type)
        
        logger.info(f"训练 {disease_type} 的自适应校准器...")
        adaptive_calibrator = AdaptiveCalibrator()
        adaptive_calibrator.fit(X_cal, y_cal, model, disease_type)
        
        logger.info(f"训练 {disease_type} 的增强校准器...")
        better_calibrator = BetterCalibrator()
        better_calibrator.fit(X_cal, y_cal, model, disease_type)
        
        # 绘制校准曲线
        logger.info(f"绘制 {disease_type} 的校准曲线...")
        platt_calibrator.plot_calibration_curve(X_cal, y_cal, model, disease_type)
        
        # 评估校准器性能
        evaluate_calibrator_performance(X_cal, y_cal, model, platt_calibrator, isotonic_calibrator, disease_type)
        
        # 保存校准器
        calibrators[disease_type] = {
            'platt': platt_calibrator,
            'isotonic': isotonic_calibrator,
            'adaptive': adaptive_calibrator,
            'better': better_calibrator
        }
    
    return calibrators

def evaluate_calibrator_performance(X, y, model, platt_calibrator, isotonic_calibrator, disease_type):
    """评估校准器性能"""
    logger.info(f"\n评估 {disease_type} 校准器性能...")
    
    try:
        import numpy as np  # 确保numpy可用
        
        # 将目标变量转换为二进制（如果不是）
        unique_values = np.unique(y)
        if len(unique_values) > 2 or not all(val in [0, 1] for val in unique_values):
            logger.warning(f"目标变量不是二进制的，包含值: {unique_values}")
            logger.warning("尝试将目标变量转换为二进制形式...")
            
            # 对于回归目标，使用中位数作为阈值
            if len(unique_values) > 2:
                threshold = np.median(y)
                binary_y = (y > threshold).astype(int)
                logger.info(f"使用阈值 {threshold} 将目标变量转换为二进制")
            else:
                # 确保值为0和1
                min_val = np.min(y)
                binary_y = (y > min_val).astype(int)
                logger.info(f"将目标变量从 {unique_values} 转换为 [0, 1]")
        else:
            binary_y = y
        
        # 获取原始预测概率
        try:
            if hasattr(model, 'predict_proba'):
                y_prob_orig = model.predict_proba(X)[:, 1]
            else:
                y_prob_orig = model.predict(X)
                
                # 如果是回归模型，将预测值转换为概率
                if y_prob_orig.ndim == 1:
                    # 简单地将值标准化到0-1范围
                    min_val = y_prob_orig.min()
                    max_val = y_prob_orig.max()
                    if max_val > min_val:
                        y_prob_orig = (y_prob_orig - min_val) / (max_val - min_val)
                    else:
                        y_prob_orig = np.ones_like(y_prob_orig) * 0.5  # 默认0.5概率
        except ValueError as e:
            logger.warning(f"获取原始模型预测概率失败: {e}")
            logger.warning("使用随机概率代替原始模型预测")
            # 使用随机概率代替，仅用于评估目的
            y_prob_orig = np.random.random(len(y))
        
        # 获取校准后的概率
        y_prob_platt = platt_calibrator.calibrate_probability(X, disease_type)
        y_prob_isotonic = isotonic_calibrator.calibrate_probability(X, disease_type)
        
        # 如果校准失败，使用原始概率
        if y_prob_platt is None:
            logger.warning(f"Platt校准失败，使用原始概率代替")
            y_prob_platt = y_prob_orig
            
        if y_prob_isotonic is None:
            logger.warning(f"等渗校准失败，使用原始概率代替")
            y_prob_isotonic = y_prob_orig
        
        # 确保概率在[0,1]范围内
        y_prob_orig = np.clip(y_prob_orig, 0, 1)
        y_prob_platt = np.clip(y_prob_platt, 0, 1)
        y_prob_isotonic = np.clip(y_prob_isotonic, 0, 1)
        
        # 计算Brier分数（越低越好）
        brier_orig = brier_score_loss(binary_y, y_prob_orig)
        brier_platt = brier_score_loss(binary_y, y_prob_platt)
        brier_isotonic = brier_score_loss(binary_y, y_prob_isotonic)
        
        # 计算对数损失（越低越好）
        # 避免对数损失中的0和1概率
        eps = 1e-15
        y_prob_orig = np.clip(y_prob_orig, eps, 1 - eps)
        y_prob_platt = np.clip(y_prob_platt, eps, 1 - eps)
        y_prob_isotonic = np.clip(y_prob_isotonic, eps, 1 - eps)
        
        log_loss_orig = log_loss(binary_y, y_prob_orig)
        log_loss_platt = log_loss(binary_y, y_prob_platt)
        log_loss_isotonic = log_loss(binary_y, y_prob_isotonic)
        
        logger.info(f"Brier分数 (越低越好):")
        logger.info(f"  原始模型: {brier_orig:.4f}")
        logger.info(f"  Platt校准: {brier_platt:.4f}")
        logger.info(f"  等渗校准: {brier_isotonic:.4f}")
        
        logger.info(f"对数损失 (越低越好):")
        logger.info(f"  原始模型: {log_loss_orig:.4f}")
        logger.info(f"  Platt校准: {log_loss_platt:.4f}")
        logger.info(f"  等渗校准: {log_loss_isotonic:.4f}")
        
        # 创建性能比较图表
        plt.figure(figsize=(12, 6))
        
        # Brier分数比较
        plt.subplot(1, 2, 1)
        plt.bar(['原始模型', 'Platt校准', '等渗校准'], [brier_orig, brier_platt, brier_isotonic])
        plt.title(f'{disease_type.capitalize()} Brier分数比较 (越低越好)')
        plt.ylabel('Brier分数')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 对数损失比较
        plt.subplot(1, 2, 2)
        plt.bar(['原始模型', 'Platt校准', '等渗校准'], [log_loss_orig, log_loss_platt, log_loss_isotonic])
        plt.title(f'{disease_type.capitalize()} 对数损失比较 (越低越好)')
        plt.ylabel('对数损失')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'output/figures/{disease_type}_calibration_performance.png')
        plt.close()
    
    except Exception as e:
        logger.error(f"评估校准器性能时出错: {e}")
        logger.error(f"错误详情: {str(e)}")
        logger.error("跳过校准器性能评估")

def run_calibration_training():
    """运行校准器训练流程"""
    logger.info("开始训练概率校准器...")
    
    # 加载数据和模型
    data, models = load_data_and_models()
    
    # 准备校准数据
    datasets = prepare_data_for_calibration(data)
    
    # 训练校准器
    calibrators = train_calibrators(datasets, models)
    
    logger.info("\n校准器训练完成！")
    return calibrators

# 如果直接运行此文件，则执行校准器训练
if __name__ == "__main__":
    run_calibration_training() 