import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

class DNN(nn.Module):
    """简单的深度神经网络模型"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, dropout_rate=0.3):
        super(DNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.is_regression = False  # 默认为分类任务
        
        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 如果output_dim不是1，则为多分类问题，不使用sigmoid
        if output_dim != 1:
            self.is_regression = True
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        # 只对二分类问题使用sigmoid激活函数
        if self.output_layer.out_features == 1 and not self.is_regression:
            x = self.sigmoid(x)
            
        return x

class AttentionDNN(nn.Module):
    """带注意力机制的深度神经网络模型"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, dropout_rate=0.3):
        super(AttentionDNN, self).__init__()
        
        self.feature_layers = nn.ModuleList()
        self.is_regression = False  # 默认为分类任务
        
        # 特征提取层
        self.feature_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.feature_layers.append(nn.ReLU())
        self.feature_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.feature_layers.append(nn.Dropout(dropout_rate))
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 2, 1),
            nn.Sigmoid()
        )
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.hidden_layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 如果output_dim不是1，则为多分类问题，不使用sigmoid
        if output_dim != 1:
            self.is_regression = True
        
    def forward(self, x):
        # 特征提取
        for layer in self.feature_layers:
            x = layer(x)
        
        # 注意力权重
        attention_weights = self.attention(x)
        
        # 应用注意力权重
        x = x * attention_weights
        
        # 通过隐藏层
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        # 只对二分类问题使用sigmoid激活函数
        if self.output_layer.out_features == 1 and not self.is_regression:
            x = self.sigmoid(x)
            
        return x

class ModelTrainer:
    def __init__(self, data_splits=None):
        """初始化模型训练类"""
        self.models = {}
        self.best_models = {}
        self.model_results = {}
        self.feature_importances = {}
        self.shap_values = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 存储深度学习模型
        self.dl_models = {}
        self.teacher_models = {}
        self.student_models = {}
        
        # 存储集成模型
        self.ensemble_models = {}
        
        # 存储混合模型
        self.blended_models = {}
        
        # 存储类权重
        self.class_weights = {}
        
        # 存储数据分割
        self.data_splits = data_splits if data_splits else {}
        
        # 创建输出目录
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/figures', exist_ok=True)
        
    def load_processed_data(self, stroke_path='output/processed_data/stroke_processed.csv',
                           heart_path='output/processed_data/heart_processed.csv',
                           cirrhosis_path='output/processed_data/cirrhosis_processed.csv'):
        """加载处理后的数据"""
        self.stroke_data = pd.read_csv(stroke_path)
        self.heart_data = pd.read_csv(heart_path)
        self.cirrhosis_data = pd.read_csv(cirrhosis_path)
        
        # 准备训练数据
        # 中风数据集
        stroke_features = self.stroke_data.drop(['id', 'stroke'], axis=1, errors='ignore')
        stroke_target = self.stroke_data['stroke']
        
        # 心脏病数据集
        heart_features = self.heart_data.drop(['HeartDisease'], axis=1, errors='ignore')
        heart_target = self.heart_data['HeartDisease']
        
        # 肝硬化数据集
        cirrhosis_features = self.cirrhosis_data.drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
        cirrhosis_target = self.cirrhosis_data['Stage']
        
        self.datasets = {
            'stroke': (stroke_features, stroke_target),
            'heart': (heart_features, heart_target),
            'cirrhosis': (cirrhosis_features, cirrhosis_target)
        }
        
        print("处理后的数据加载完成！")
        return self.datasets
    
    def split_data(self, test_size=0.2, random_state=42):
        """将数据拆分为训练集和测试集"""
        self.train_test_data = {}
        
        for name, (features, target) in self.datasets.items():
            # 移除非数值列
            features = features.select_dtypes(include=['float64', 'int64'])
            
            # 检查是否可以使用分层采样
            use_stratify = True
            if name == 'cirrhosis':  # 对于肝硬化数据集特殊处理
                # 检查每个类别的样本数量
                value_counts = pd.Series(target).value_counts()
                if value_counts.min() < 2:  # 如果最小类别数量小于2，不能使用分层采样
                    use_stratify = False
                    print(f"警告: {name} 数据集中有类别样本数量过少，不使用分层采样")
            
            # 拆分数据
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=test_size, random_state=random_state, stratify=target
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=test_size, random_state=random_state
                )
            
            self.train_test_data[name] = (X_train, X_test, y_train, y_test)
            print(f"{name} 数据集拆分完成: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
        
        return self.train_test_data
    
    def handle_imbalanced_data(self, method='smote'):
        """处理不平衡数据"""
        self.class_weights = {}  # 初始化类别权重字典
        
        for name, (X_train, X_test, y_train, y_test) in self.train_test_data.items():
            # 判断是否为回归任务
            is_regression = False
            if name == 'cirrhosis':  # 肝硬化数据集的Stage列是连续值，应该视为回归任务
                is_regression = True
                
            # 对于分类任务，检查是否存在类别不平衡
            unique_counts = np.unique(y_train, return_counts=True)
            class_ratio = min(unique_counts[1]) / max(unique_counts[1])
            
            if class_ratio < 0.5:  # 如果小类别样本数少于大类别的一半，认为存在不平衡
                print(f"{name} 数据集存在类别不平衡, 应用 {method} 方法处理...")
                
                if method == 'smote':
                    # 应用SMOTE过采样
                    smote = SMOTE(random_state=42)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    
                    # 更新训练数据
                    self.train_test_data[name] = (X_train_resampled, X_test, y_train_resampled, y_test)
                    
                    # 输出重采样后的类别分布
                    unique_counts_after = np.unique(y_train_resampled, return_counts=True)
                    print(f"  重采样前: {dict(zip(unique_counts[0], unique_counts[1]))}")
                    print(f"  重采样后: {dict(zip(unique_counts_after[0], unique_counts_after[1]))}")
                    
                elif method == 'class_weight':
                    # 计算类别权重（后面训练模型时使用）
                    counts = np.bincount(y_train)
                    class_weight = {i: max(counts) / counts[i] for i in range(len(counts))}
                    self.class_weights[name] = class_weight
                    print(f"  应用类别权重: {class_weight}")
            else:
                print(f"{name} 数据集类别分布较为平衡，无需特殊处理")
        
        return self.train_test_data
    
    def train_baseline_models(self):
        """训练并评估所有模型，并找出表现最好的模型"""
        # 基线模型列表 - 分类模型
        self.baseline_models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
        }
        
        # 回归模型列表 - 用于肝硬化数据集
        self.regression_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'SVR': SVR(kernel='rbf'),
            'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=5),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=10),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, max_depth=5),
            'XGBRegressor': XGBRegressor(n_estimators=100, max_depth=5),
            'LGBMRegressor': LGBMRegressor(n_estimators=100, max_depth=5)
        }
        
        results = {}
        best_models = {}
        
        # 对每个数据集
        for dataset_name, (X_train, X_test, y_train, y_test) in self.data_splits.items():
            print(f"\n训练 {dataset_name} 数据集的基线模型...")
            dataset_scores = {}
            
            # 判断是回归任务还是分类任务
            is_regression = False
            if dataset_name == 'cirrhosis':  # 肝硬化数据集的Stage是连续值
                is_regression = True
            
            if not is_regression:
                # 分类任务
                for model_name, model in self.baseline_models.items():
                    try:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        print(f"  训练 {model_name}...")
                        print(f"    CV 准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                        print(f"    测试集准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
                        
                        dataset_scores[model_name] = {
                            'model': model,
                            'cv_accuracy': cv_scores.mean(),
                            'test_accuracy': accuracy,
                            'test_f1': f1
                        }
                    except Exception as e:
                        print(f"  训练 {model_name} 时出错: {e}")
            else:
                # 回归任务
                for model_name, model in self.regression_models.items():
                    try:
                        # 使用KFold进行交叉验证
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        cv_scores = []
                        
                        # 手动进行交叉验证计算MSE
                        for train_idx, val_idx in kf.split(X_train):
                            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                            
                            model.fit(X_train_cv, y_train_cv)
                            y_pred_cv = model.predict(X_val_cv)
                            mse_cv = mean_squared_error(y_val_cv, y_pred_cv)
                            cv_scores.append(mse_cv)
                        
                        cv_mse = np.mean(cv_scores)
                        cv_std = np.std(cv_scores)
                        
                        # 在完整训练集上训练模型
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        print(f"  训练 {model_name}...")
                        print(f"    CV MSE: {cv_mse:.4f} ± {cv_std:.4f}")
                        print(f"    测试集 MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                        
                        dataset_scores[model_name] = {
                            'model': model,
                            'cv_mse': cv_mse,
                            'test_mse': mse,
                            'test_mae': mae,
                            'test_r2': r2
                        }
                    except Exception as e:
                        print(f"  训练 {model_name} 时出错: {e}")
            
            # 选择最佳模型
            if dataset_scores:
                if not is_regression:
                    # 分类任务使用F1分数
                    best_model_name = max(dataset_scores, key=lambda k: dataset_scores[k]['test_f1'])
                    print(f"  {dataset_name} 数据集的最佳基线模型: {best_model_name}")
                    best_models[dataset_name] = dataset_scores[best_model_name]['model']
                else:
                    # 回归任务使用R²分数
                    best_model_name = max(dataset_scores, key=lambda k: dataset_scores[k]['test_r2'])
                    print(f"  {dataset_name} 数据集的最佳基线模型: {best_model_name}")
                    best_models[dataset_name] = dataset_scores[best_model_name]['model']
            else:
                print(f"  没有成功训练任何模型用于 {dataset_name} 数据集")
            
            results[dataset_name] = dataset_scores
        
        self.baseline_results = results
        self.best_baseline_models = best_models
        
        # 保存最佳基线模型
        for dataset_name, model in best_models.items():
            joblib.dump(model, f'output/models/{dataset_name}_best_baseline_model.pkl')
        
        return best_models
    
    def optimize_best_models(self, cv=5):
        """优化最佳模型的超参数"""
        for name, best_model_info in self.best_models.items():
            print(f"\n优化 {name} 数据集的 {best_model_info['name']} 模型...")
            
            X_train, X_test, y_train, y_test = self.train_test_data[name]
            model_name = best_model_info['name']
            
            # 根据模型类型定义参数网格
            if model_name == 'LogisticRegression':
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['liblinear', 'saga']
                }
            elif model_name == 'DecisionTree':
                param_grid = {
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            elif model_name == 'RandomForest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            elif model_name == 'GradientBoosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif model_name == 'XGBoost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            elif model_name == 'LightGBM':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50, 70],
                    'subsample': [0.8, 0.9, 1.0]
                }
            elif model_name == 'CatBoost':
                param_grid = {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3, 5, 7]
                }
            else:
                print(f"  没有为 {model_name} 定义参数网格，跳过优化")
                continue
            
            # 创建网格搜索对象
            grid_search = GridSearchCV(
                estimator=best_model_info['model'],
                param_grid=param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # 执行网格搜索
            try:
                grid_search.fit(X_train, y_train)
                
                # 获取最佳模型
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                print(f"  最佳参数: {best_params}")
                print(f"  CV 得分: {grid_search.best_score_:.4f}")
                
                # 在测试集上评估最佳模型
                y_pred = best_model.predict(X_test)
                
                # 计算评估指标
                accuracy = accuracy_score(y_test, y_pred)
                
                if len(np.unique(y_test)) == 2:  # 二分类问题
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    # 计算ROC AUC
                    try:
                        y_prob = best_model.predict_proba(X_test)[:, 1]
                        auc_score = roc_auc_score(y_test, y_prob)
                    except:
                        auc_score = None
                else:  # 多分类问题
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    auc_score = None
                
                print(f"  测试集准确率: {accuracy:.4f}")
                print(f"  测试集精确率: {precision:.4f}")
                print(f"  测试集召回率: {recall:.4f}")
                print(f"  测试集F1分数: {f1:.4f}")
                if auc_score:
                    print(f"  测试集AUC: {auc_score:.4f}")
                
                # 更新最佳模型
                self.best_models[name]['model'] = best_model
                self.best_models[name]['params'] = best_params
                self.best_models[name]['performance'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc_score
                }
                
                # 保存模型
                joblib.dump(best_model, f'output/models/{name}_{model_name}_optimized.pkl')
                print(f"  优化后的模型已保存为 output/models/{name}_{model_name}_optimized.pkl")
                
            except Exception as e:
                print(f"  优化 {model_name} 时出错: {e}")
        
        return self.best_models
    
    def evaluate_and_visualize_models(self):
        """评估最佳模型并可视化结果"""
        for name, best_model_info in self.best_models.items():
            print(f"\n评估 {name} 数据集的 {best_model_info['name']} 模型...")
            
            X_train, X_test, y_train, y_test = self.train_test_data[name]
            model = best_model_info['model']
            
            # 预测测试集
            y_pred = model.predict(X_test)
            
            # 混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name.capitalize()} {best_model_info["name"]} 混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.tight_layout()
            plt.savefig(f'output/figures/{name}_confusion_matrix.png')
            plt.close()
            
            # 分类报告
            report = classification_report(y_test, y_pred)
            print(f"  分类报告:\n{report}")
            
            # ROC曲线（仅适用于二分类）
            if len(np.unique(y_test)) == 2:
                try:
                    # 获取预测概率
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # 计算ROC曲线
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    # 绘制ROC曲线
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, 
                             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('假正例率')
                    plt.ylabel('真正例率')
                    plt.title(f'{name.capitalize()} {best_model_info["name"]} ROC曲线')
                    plt.legend(loc='lower right')
                    plt.savefig(f'output/figures/{name}_roc_curve.png')
                    plt.close()
                    
                except Exception as e:
                    print(f"  绘制ROC曲线时出错: {e}")
            
            # 特征重要性（如果模型支持）
            try:
                # 获取特征重要性
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = X_train.columns
                    
                    # 排序
                    indices = np.argsort(importances)[::-1]
                    
                    # 绘制特征重要性
                    plt.figure(figsize=(10, 6))
                    plt.title(f'{name.capitalize()} {best_model_info["name"]} 特征重要性')
                    plt.bar(range(len(indices)), importances[indices], align='center')
                    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{name}_feature_importance.png')
                    plt.close()
                    
                    # 保存特征重要性
                    self.feature_importances[name] = {
                        'importance': importances,
                        'names': feature_names
                    }
                    
                    # SHAP值分析
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)
                        
                        # 保存SHAP值
                        self.shap_values[name] = {
                            'values': shap_values,
                            'data': X_test
                        }
                        
                        # 绘制SHAP摘要图
                        plt.figure(figsize=(10, 8))
                        if isinstance(shap_values, list):  # 多分类的情况
                            shap.summary_plot(shap_values[1], X_test, show=False)
                        else:  # 二分类的情况
                            shap.summary_plot(shap_values, X_test, show=False)
                        plt.title(f'{name.capitalize()} {best_model_info["name"]} SHAP值')
                        plt.tight_layout()
                        plt.savefig(f'output/figures/{name}_shap_summary.png')
                        plt.close()
                        
                    except Exception as e:
                        print(f"  SHAP分析时出错: {e}")
                
            except Exception as e:
                print(f"  分析特征重要性时出错: {e}")
        
        return self.feature_importances, self.shap_values
    
    def train_deep_learning_models(self, epochs=100, batch_size=32, patience=10):
        """训练深度学习模型"""
        print("\n开始训练深度学习模型...")
        
        # 创建模型结构
        for dataset_name, (X_train, X_test, y_train, y_test) in self.data_splits.items():
            print(f"\n训练 {dataset_name} 数据集的深度学习模型...")
            
            # 确定任务类型
            is_regression = False
            if dataset_name == 'cirrhosis':  # 肝硬化数据集是回归任务
                is_regression = True
            
            # 准备数据 - 确保所有数据都是数值类型
            try:
                # 尝试直接转换
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            except (TypeError, ValueError):
                # 如果失败，尝试先转换为数值类型
                X_train = X_train.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_test = X_test.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            
            y_train_tensor = torch.FloatTensor(y_train.values)
            y_test_tensor = torch.FloatTensor(y_test.values)
            
            # 重塑标签为列向量
            y_train_tensor = y_train_tensor.reshape(-1, 1)
            y_test_tensor = y_test_tensor.reshape(-1, 1)
            
            # 创建DataLoader
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # 创建模型
            input_dim = X_train.shape[1]
            
            if not is_regression:
                # 分类模型
                dnn_model = DNN(input_dim=input_dim).to(self.device)
                attention_model = AttentionDNN(input_dim=input_dim).to(self.device)
                criterion = nn.BCELoss()
            else:
                # 回归模型 - 设置is_regression=True
                dnn_model = DNN(input_dim=input_dim)
                dnn_model.is_regression = True
                dnn_model = dnn_model.to(self.device)
                
                attention_model = AttentionDNN(input_dim=input_dim)
                attention_model.is_regression = True
                attention_model = attention_model.to(self.device)
                
                criterion = nn.MSELoss()
            
            # 优化器
            dnn_optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)
            attention_optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
            
            # 训练模型
            best_dnn_loss = float('inf')
            best_att_loss = float('inf')
            dnn_patience_counter = 0
            att_patience_counter = 0
            
            for epoch in range(1, epochs + 1):
                dnn_model.train()
                attention_model.train()
                
                dnn_epoch_loss = 0
                att_epoch_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # 训练DNN
                    dnn_optimizer.zero_grad()
                    dnn_outputs = dnn_model(batch_X)
                    dnn_loss = criterion(dnn_outputs, batch_y)
                    dnn_loss.backward()
                    dnn_optimizer.step()
                    dnn_epoch_loss += dnn_loss.item()
                    
                    # 训练Attention DNN
                    attention_optimizer.zero_grad()
                    att_outputs = attention_model(batch_X)
                    att_loss = criterion(att_outputs, batch_y)
                    att_loss.backward()
                    attention_optimizer.step()
                    att_epoch_loss += att_loss.item()
                
                dnn_epoch_loss /= len(train_loader)
                att_epoch_loss /= len(train_loader)
                
                # 打印训练进度
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}/{epochs}, DNN Loss: {dnn_epoch_loss:.4f}, Attention Loss: {att_epoch_loss:.4f}")
                
                # 早停
                if dnn_epoch_loss < best_dnn_loss:
                    best_dnn_loss = dnn_epoch_loss
                    dnn_patience_counter = 0
                else:
                    dnn_patience_counter += 1
                
                if att_epoch_loss < best_att_loss:
                    best_att_loss = att_epoch_loss
                    att_patience_counter = 0
                else:
                    att_patience_counter += 1
                
                if dnn_patience_counter >= patience and att_patience_counter >= patience:
                    print(f"  提前停止训练，没有改进: {patience} epochs")
                    break
            
            # 评估模型
            dnn_model.eval()
            attention_model.eval()
            
            with torch.no_grad():
                dnn_outputs = dnn_model(X_test_tensor.to(self.device))
                att_outputs = attention_model(X_test_tensor.to(self.device))
                
                dnn_preds = dnn_outputs.cpu().numpy()
                att_preds = att_outputs.cpu().numpy()
                y_test_np = y_test.values
                
                if not is_regression:
                    # 分类任务评估
                    dnn_predictions = (dnn_preds > 0.5).astype(int).flatten()
                    att_predictions = (att_preds > 0.5).astype(int).flatten()
                    
                    dnn_accuracy = accuracy_score(y_test_np, dnn_predictions)
                    att_accuracy = accuracy_score(y_test_np, att_predictions)
                    
                    dnn_f1 = f1_score(y_test_np, dnn_predictions, average='weighted')
                    att_f1 = f1_score(y_test_np, att_predictions, average='weighted')
                    
                    print(f"  DNN 准确率: {dnn_accuracy:.4f}, F1: {dnn_f1:.4f}")
                    print(f"  Attention DNN 准确率: {att_accuracy:.4f}, F1: {att_f1:.4f}")
                    
                    # 尝试计算AUC（如果是二分类）
                    try:
                        dnn_auc = roc_auc_score(y_test_np, dnn_preds)
                        att_auc = roc_auc_score(y_test_np, att_preds)
                        print(f"  DNN AUC: {dnn_auc:.4f}")
                        print(f"  Attention DNN AUC: {att_auc:.4f}")
                    except:
                        pass
                    
                    # 选择表现更好的模型作为教师
                    if att_f1 > dnn_f1:
                        self.teacher_models[dataset_name] = attention_model
                        print(f"  选择 Attention DNN 作为 {dataset_name} 数据集的教师模型")
                    else:
                        self.teacher_models[dataset_name] = dnn_model
                        print(f"  选择 DNN 作为 {dataset_name} 数据集的教师模型")
                else:
                    # 回归任务评估
                    dnn_preds = dnn_preds.flatten()
                    att_preds = att_preds.flatten()
                    
                    dnn_mse = mean_squared_error(y_test_np, dnn_preds)
                    att_mse = mean_squared_error(y_test_np, att_preds)
                    
                    dnn_mae = mean_absolute_error(y_test_np, dnn_preds)
                    att_mae = mean_absolute_error(y_test_np, att_preds)
                    
                    dnn_r2 = r2_score(y_test_np, dnn_preds)
                    att_r2 = r2_score(y_test_np, att_preds)
                    
                    print(f"  DNN MSE: {dnn_mse:.4f}, MAE: {dnn_mae:.4f}, R²: {dnn_r2:.4f}")
                    print(f"  Attention DNN MSE: {att_mse:.4f}, MAE: {att_mae:.4f}, R²: {att_r2:.4f}")
                    
                    # 选择表现更好的模型作为教师 (使用R²)
                    if att_r2 > dnn_r2:
                        self.teacher_models[dataset_name] = attention_model
                        print(f"  选择 Attention DNN 作为 {dataset_name} 数据集的教师模型")
                    else:
                        self.teacher_models[dataset_name] = dnn_model
                        print(f"  选择 DNN 作为 {dataset_name} 数据集的教师模型")
            
            # 保存两个模型
            self.dl_models[dataset_name] = {
                'dnn': dnn_model,
                'attention_dnn': attention_model
            }
        
        print("\n深度学习模型训练完成!")
        return self.teacher_models
    
    def train_student_models_with_distillation(self, epochs=50, batch_size=32, temperature=3.0, alpha=0.5):
        """使用知识蒸馏训练更小的学生模型"""
        for dataset_name, teacher_model in self.teacher_models.items():
            print(f"\n为 {dataset_name} 数据集进行知识蒸馏...")
            
            X_train, X_test, y_train, y_test = self.data_splits[dataset_name]
            
            # 确定任务类型
            is_regression = False
            if dataset_name == 'cirrhosis':  # 肝硬化数据集是回归任务
                is_regression = True
            
            # 准备数据 - 确保所有数据都是数值类型
            try:
                # 尝试直接转换
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            except (TypeError, ValueError):
                # 如果失败，尝试先转换为数值类型
                X_train = X_train.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_test = X_test.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).copy()
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_test_tensor = torch.FloatTensor(X_test.values)
            
            y_train_tensor = torch.FloatTensor(y_train.values)
            y_test_tensor = torch.FloatTensor(y_test.values)
            
            # 重塑标签为列向量
            y_train_tensor = y_train_tensor.reshape(-1, 1)
            y_test_tensor = y_test_tensor.reshape(-1, 1)
            
            # 创建学生模型 - 较小的网络
            input_dim = X_train.shape[1]
            
            if not is_regression:
                # 分类任务
                y_train_tensor = y_train_tensor.reshape(-1, 1)
                student_model = DNN(input_dim=input_dim, hidden_dims=[64, 32], output_dim=1).to(self.device)
                
                # 获取教师模型的软标签
                teacher_model.eval()
                with torch.no_grad():
                    soft_targets = teacher_model(X_train_tensor.to(self.device))
                
                # 创建DataLoader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, soft_targets.cpu())
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # 优化器和损失函数
                optimizer = optim.Adam(student_model.parameters(), lr=0.001)
                hard_loss_fn = nn.BCELoss()
                soft_loss_fn = nn.MSELoss()  # 用于软标签
                
                # 训练学生模型
                for epoch in range(1, epochs + 1):
                    student_model.train()
                    epoch_loss = 0.0
                    
                    for batch_X, batch_y_hard, batch_y_soft in train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y_hard = batch_y_hard.to(self.device)
                        batch_y_soft = batch_y_soft.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = student_model(batch_X)
                        
                        # 计算硬标签损失
                        hard_loss = hard_loss_fn(outputs, batch_y_hard)
                        
                        # 计算软标签损失
                        soft_loss = soft_loss_fn(outputs, batch_y_soft)
                        
                        # 总损失
                        loss = alpha * hard_loss + (1 - alpha) * soft_loss
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    epoch_loss /= len(train_loader)
                    if (epoch % 10 == 0) or (epoch == 1):
                        print(f"  Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
                
                # 评估学生模型
                student_model.eval()
                with torch.no_grad():
                    outputs = student_model(X_test_tensor.to(self.device))
                    probs = outputs.cpu().numpy()
                    predictions = (probs > 0.5).astype(int).flatten()
                    
                    y_test_np = y_test.values
                    
                    accuracy = accuracy_score(y_test_np, predictions)
                    f1 = f1_score(y_test_np, predictions, average='weighted')
                    
                    print(f"  学生模型性能 - 准确率: {accuracy:.4f}, F1: {f1:.4f}")
                    
                    try:
                        auc = roc_auc_score(y_test_np, probs)
                        print(f"  AUC: {auc:.4f}")
                    except:
                        pass
            else:
                # 回归任务
                y_train_tensor = y_train_tensor.reshape(-1, 1)
                student_model = DNN(input_dim=input_dim, hidden_dims=[64, 32], output_dim=1)
                student_model.is_regression = True  # 设置为回归模型
                student_model = student_model.to(self.device)
                
                # 获取教师模型的软标签
                teacher_model.eval()
                with torch.no_grad():
                    soft_targets = teacher_model(X_train_tensor.to(self.device))
                
                # 创建DataLoader
                train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor, soft_targets.cpu())
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # 优化器和损失函数
                optimizer = optim.Adam(student_model.parameters(), lr=0.001)
                hard_loss_fn = nn.MSELoss()  # 回归任务使用MSE
                soft_loss_fn = nn.MSELoss()  # 对软标签也使用MSE
                
                # 训练学生模型
                for epoch in range(1, epochs + 1):
                    student_model.train()
                    epoch_loss = 0.0
                    
                    for batch_X, batch_y_hard, batch_y_soft in train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y_hard = batch_y_hard.to(self.device)
                        batch_y_soft = batch_y_soft.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = student_model(batch_X)
                        
                        # 计算硬标签损失
                        hard_loss = hard_loss_fn(outputs, batch_y_hard)
                        
                        # 计算软标签损失
                        soft_loss = soft_loss_fn(outputs, batch_y_soft)
                        
                        # 总损失
                        loss = alpha * hard_loss + (1 - alpha) * soft_loss
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    epoch_loss /= len(train_loader)
                    if (epoch % 10 == 0) or (epoch == 1):
                        print(f"  Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
                
                # 评估学生模型 - 使用回归指标
                student_model.eval()
                with torch.no_grad():
                    outputs = student_model(X_test_tensor.to(self.device))
                    predictions = outputs.cpu().numpy().flatten()
                    
                    y_test_np = y_test.values
                    
                    mse = mean_squared_error(y_test_np, predictions)
                    mae = mean_absolute_error(y_test_np, predictions)
                    r2 = r2_score(y_test_np, predictions)
                    
                    print(f"  学生模型性能 - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # 保存学生模型
            self.student_models[dataset_name] = student_model
        
        return self.student_models
    
    def train_ensemble_models(self):
        """训练集成模型，整合各个单一模型的预测能力"""
        for dataset_name in self.data_splits.keys():
            print(f"\n为 {dataset_name} 数据集训练集成模型...")
            
            X_train, X_test, y_train, y_test = self.data_splits[dataset_name]
            
            # 确定任务类型
            is_regression = False
            if dataset_name == 'cirrhosis':  # 肝硬化数据集是回归任务
                is_regression = True
            
            # 获取该数据集的基线模型
            if dataset_name not in self.baseline_results:
                print(f"  没有可用于 {dataset_name} 数据集的基线模型，跳过集成模型训练")
                continue
            
            if not is_regression:
                # 分类任务 - 投票与堆叠分类器
                model_results = self.baseline_results[dataset_name]
                
                # 选择表现较好的前3个模型
                top_models = sorted(model_results.keys(), 
                                key=lambda k: model_results[k]['test_f1'] if 'test_f1' in model_results[k] else 0, 
                                reverse=True)[:3]
                
                estimators = [(name, model_results[name]['model']) for name in top_models]
                
                print("  训练投票集成模型...")
                voting_clf = VotingClassifier(estimators=estimators, voting='soft')
                voting_clf.fit(X_train, y_train)
                
                print("  训练堆叠集成模型...")
                meta_clf = LGBMClassifier()
                stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_clf)
                stacking_clf.fit(X_train, y_train)
                
                # 评估投票分类器
                voting_preds = voting_clf.predict(X_test)
                voting_acc = accuracy_score(y_test, voting_preds)
                voting_f1 = f1_score(y_test, voting_preds, average='weighted')
                print(f"  投票集成模型 - 准确率: {voting_acc:.4f}, F1: {voting_f1:.4f}")
                
                # 评估堆叠分类器
                stacking_preds = stacking_clf.predict(X_test)
                stacking_acc = accuracy_score(y_test, stacking_preds)
                stacking_f1 = f1_score(y_test, stacking_preds, average='weighted')
                print(f"  堆叠集成模型 - 准确率: {stacking_acc:.4f}, F1: {stacking_f1:.4f}")
                
                # 尝试计算AUC（如果是二分类）
                try:
                    voting_probs = voting_clf.predict_proba(X_test)[:, 1]
                    stacking_probs = stacking_clf.predict_proba(X_test)[:, 1]
                    
                    voting_auc = roc_auc_score(y_test, voting_probs)
                    stacking_auc = roc_auc_score(y_test, stacking_probs)
                    
                    print(f"  投票集成模型 - AUC: {voting_auc:.4f}")
                    print(f"  堆叠集成模型 - AUC: {stacking_auc:.4f}")
                except:
                    pass
                
                # 选择最佳集成模型
                if stacking_f1 > voting_f1:
                    best_ensemble = stacking_clf
                    best_type = "stacking"
                else:
                    best_ensemble = voting_clf
                    best_type = "voting"
                    
                self.ensemble_models[dataset_name] = best_ensemble
                print(f"  {dataset_name} 数据集的最佳模型: {best_type} 集成")
            else:
                # 回归任务 - 使用投票回归器和堆叠回归器
                model_results = self.baseline_results[dataset_name]
                
                # 选择表现较好的前3个回归模型
                top_models = sorted(model_results.keys(), 
                                key=lambda k: model_results[k]['test_r2'] if 'test_r2' in model_results[k] else 0, 
                                reverse=True)[:3]
                
                estimators = [(name, model_results[name]['model']) for name in top_models]
                
                print("  训练投票回归集成模型...")
                voting_reg = VotingRegressor(estimators=estimators)
                voting_reg.fit(X_train, y_train)
                
                print("  训练堆叠回归集成模型...")
                meta_reg = LGBMRegressor()
                stacking_reg = StackingRegressor(estimators=estimators, final_estimator=meta_reg)
                stacking_reg.fit(X_train, y_train)
                
                # 评估投票回归器
                voting_preds = voting_reg.predict(X_test)
                voting_mse = mean_squared_error(y_test, voting_preds)
                voting_mae = mean_absolute_error(y_test, voting_preds)
                voting_r2 = r2_score(y_test, voting_preds)
                print(f"  投票回归集成模型 - MSE: {voting_mse:.4f}, MAE: {voting_mae:.4f}, R²: {voting_r2:.4f}")
                
                # 评估堆叠回归器
                stacking_preds = stacking_reg.predict(X_test)
                stacking_mse = mean_squared_error(y_test, stacking_preds)
                stacking_mae = mean_absolute_error(y_test, stacking_preds)
                stacking_r2 = r2_score(y_test, stacking_preds)
                print(f"  堆叠回归集成模型 - MSE: {stacking_mse:.4f}, MAE: {stacking_mae:.4f}, R²: {stacking_r2:.4f}")
                
                # 选择最佳集成模型 (使用R²)
                if stacking_r2 > voting_r2:
                    best_ensemble = stacking_reg
                    best_type = "stacking"
                else:
                    best_ensemble = voting_reg
                    best_type = "voting"
                    
                self.ensemble_models[dataset_name] = best_ensemble
                print(f"  {dataset_name} 数据集的最佳模型: {best_type} 集成")
        
        return self.ensemble_models
    
    def build_multi_disease_model(self):
        """构建多疾病关联模型，预测同时患有多种疾病的概率"""
        print("\n构建多疾病关联模型...")
        
        # 合并数据集
        # 为了简化示例，我们假设这里已经有了多疾病预测的特征数据
        # 实际应用中，这部分需要根据具体数据进行特征工程
        
        # 这里我们使用一个虚拟的方法来说明多疾病预测的思路
        print("多疾病预测需要根据实际患者数据进行建模")
        print("由于数据集中没有同一患者的多疾病信息，我们可以采用以下方法：")
        print("1. 构建疾病对之间的关系模型（如中风-心脏病，中风-肝硬化，心脏病-肝硬化）")
        print("2. 利用已有疾病作为特征，预测其他疾病的可能性")
        print("3. 计算条件概率，估计多疾病共病的概率")
        
        # 示例：中风和心脏病的关联（stroke数据集中包含heart_disease特征）
        if 'heart_disease' in self.stroke_data.columns and 'stroke' in self.stroke_data.columns:
            # 分析中风和心脏病的关系
            cross_table = pd.crosstab(
                self.stroke_data['heart_disease'], 
                self.stroke_data['stroke'],
                normalize='index'
            )
            
            print("\n心脏病与中风关系的列联表（行归一化）:")
            print(cross_table)
            
            # 计算条件概率：P(中风|心脏病)
            prob_stroke_given_heart = cross_table.loc[1, 1]
            print(f"患有心脏病的情况下中风的概率: {prob_stroke_given_heart:.4f}")
            
            # 可视化
            plt.figure(figsize=(8, 6))
            cross_table.plot(kind='bar', stacked=True)
            plt.title('心脏病与中风的关系')
            plt.xlabel('是否患有心脏病')
            plt.ylabel('比例')
            plt.xticks([0, 1], ['无心脏病', '有心脏病'])
            plt.legend(['无中风', '有中风'])
            plt.tight_layout()
            plt.savefig('output/figures/heart_stroke_relationship.png')
            plt.close()
            
            # 构建多疾病风险评估模型的方法说明
            print("\n构建多疾病风险评估模型的方法：")
            print("1. 贝叶斯网络：建立疾病之间的概率依赖关系")
            print("2. 多标签分类：同时预测多个疾病标签")
            print("3. 级联预测：先预测一种疾病，然后将该预测结果作为特征预测其他疾病")
            print("4. 共享表示学习：为多个疾病预测任务学习共享特征表示")
            
            print("\n由于现有数据限制，我们可以通过以下方式估计多疾病共病的概率：")
            print("- P(疾病A和疾病B) = P(疾病A) * P(疾病B|疾病A)")
            print("- P(疾病A、B和C) = P(疾病A) * P(疾病B|疾病A) * P(疾病C|疾病A,疾病B)")
            
        else:
            print("数据集中没有足够的信息来分析多疾病关系")
        
        return None
    
    def run_full_model_pipeline(self):
        """运行完整的模型训练流程"""
        # 1. 训练基线模型
        self.train_baseline_models()
        
        # 2. 训练深度学习模型
        self.train_deep_learning_models()
        
        # 3. 训练学生模型（知识蒸馏）
        self.train_student_models_with_distillation()
        
        # 4. 训练集成模型
        self.train_ensemble_models()
        
        # 5. 创建混合模型
        # self.create_blended_models()
        
        return self.ensemble_models
        
def create_blended_models(self):
    """创建混合模型，结合机器学习和深度学习模型的预测"""
    print("\n创建混合模型...")
    
    for dataset_name, (X_train, X_test, y_train, y_test) in self.data_splits.items():
        print(f"\n为 {dataset_name} 数据集创建混合模型...")
        
        # 确定任务类型
        is_regression = False
        if dataset_name == 'cirrhosis':  # 肝硬化数据集是回归任务
            is_regression = True
        
        # 检查是否有可用的集成模型
        if dataset_name not in self.ensemble_models:
            print(f"  没有可用于 {dataset_name} 数据集的集成模型，跳过混合模型创建")
            continue
            
        # 检查是否有可用的深度学习模型
        if dataset_name not in self.teacher_models:
            print(f"  没有可用于 {dataset_name} 数据集的深度学习模型，跳过混合模型创建")
            continue
        
        ensemble_model = self.ensemble_models[dataset_name]
        dl_model = self.teacher_models[dataset_name]
        
        if not is_regression:
            # 分类任务 - 混合预测
            # 机器学习模型预测
            ml_probs = ensemble_model.predict_proba(X_test)[:, 1]
            
            # 深度学习模型预测
            dl_model.eval()
            with torch.no_grad():
                dl_probs = dl_model(torch.FloatTensor(X_test.values).to(self.device)).cpu().numpy().flatten()
            
            # 混合预测 (简单平均)
            blend_probs = (ml_probs + dl_probs) / 2
            blend_preds = (blend_probs > 0.5).astype(int)
            
            # 评估混合模型
            blend_accuracy = accuracy_score(y_test, blend_preds)
            blend_f1 = f1_score(y_test, blend_preds, average='weighted')
            blend_auc = roc_auc_score(y_test, blend_probs)
            
            print(f"  混合模型性能 - 准确率: {blend_accuracy:.4f}, F1: {blend_f1:.4f}, AUC: {blend_auc:.4f}")
            
            # 与单独模型比较
            ml_preds = (ml_probs > 0.5).astype(int)
            dl_preds = (dl_probs > 0.5).astype(int)
            
            ml_accuracy = accuracy_score(y_test, ml_preds)
            ml_f1 = f1_score(y_test, ml_preds, average='weighted')
            ml_auc = roc_auc_score(y_test, ml_probs)
            
            dl_accuracy = accuracy_score(y_test, dl_preds)
            dl_f1 = f1_score(y_test, dl_preds, average='weighted')
            dl_auc = roc_auc_score(y_test, dl_probs)
            
            print(f"  机器学习模型 - 准确率: {ml_accuracy:.4f}, F1: {ml_f1:.4f}, AUC: {ml_auc:.4f}")
            print(f"  深度学习模型 - 准确率: {dl_accuracy:.4f}, F1: {dl_f1:.4f}, AUC: {dl_auc:.4f}")
            
            # 选择最佳模型（基于F1分数）
            if blend_f1 > ml_f1 and blend_f1 > dl_f1:
                best_name = "混合模型"
            elif ml_f1 > dl_f1:
                best_name = "机器学习模型"
            else:
                best_name = "深度学习模型"
            print(f"  {dataset_name} 数据集的最佳模型: {best_name}")
            
            # 保存混合模型信息
            self.blended_models[dataset_name] = {
                'ml_model': ensemble_model,
                'dl_model': dl_model,
                'performance': {
                    'accuracy': blend_accuracy,
                    'f1': blend_f1,
                    'auc': blend_auc
                }
            }
        else:
            # 回归任务 - 混合预测
            # 机器学习模型预测
            ml_preds = ensemble_model.predict(X_test)
            
            # 深度学习模型预测
            dl_model.eval()
            with torch.no_grad():
                dl_preds = dl_model(torch.FloatTensor(X_test.values).to(self.device)).cpu().numpy().flatten()
            
            # 混合预测 (简单平均)
            blend_preds = (ml_preds + dl_preds) / 2
            
            # 评估混合模型
            blend_mse = mean_squared_error(y_test, blend_preds)
            blend_mae = mean_absolute_error(y_test, blend_preds)
            blend_r2 = r2_score(y_test, blend_preds)
            
            print(f"  混合模型性能 - MSE: {blend_mse:.4f}, MAE: {blend_mae:.4f}, R²: {blend_r2:.4f}")
            
            # 与单独模型比较
            ml_mse = mean_squared_error(y_test, ml_preds)
            ml_mae = mean_absolute_error(y_test, ml_preds)
            ml_r2 = r2_score(y_test, ml_preds)
            
            dl_mse = mean_squared_error(y_test, dl_preds)
            dl_mae = mean_absolute_error(y_test, dl_preds)
            dl_r2 = r2_score(y_test, dl_preds)
            
            print(f"  机器学习模型 - MSE: {ml_mse:.4f}, MAE: {ml_mae:.4f}, R²: {ml_r2:.4f}")
            print(f"  深度学习模型 - MSE: {dl_mse:.4f}, MAE: {dl_mae:.4f}, R²: {dl_r2:.4f}")
            
            # 选择最佳模型（基于R²分数）
            if blend_r2 > ml_r2 and blend_r2 > dl_r2:
                best_name = "混合模型"
            elif ml_r2 > dl_r2:
                best_name = "机器学习模型"
            else:
                best_name = "深度学习模型"
            print(f"  {dataset_name} 数据集的最佳模型: {best_name}")
            
            # 保存混合模型信息
            self.blended_models[dataset_name] = {
                'ml_model': ensemble_model,
                'dl_model': dl_model,
                'performance': {
                    'mse': blend_mse,
                    'mae': blend_mae,
                    'r2': blend_r2
                }
            }
    
    return self.blended_models

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_full_model_pipeline() 