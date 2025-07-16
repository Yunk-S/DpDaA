import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
import json
import traceback
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, precision_recall_curve
)

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保输出目录存在
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

def load_models_and_data(dataset_name=None):
    """加载模型和预处理数据"""
    models = {}
    data = {}
    
    # 加载模型
    model_dir = 'output/models'
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                model_path = os.path.join(model_dir, filename)
                model_name = filename.replace('.pkl', '')
                
                # 如果指定了数据集，只加载该数据集的模型
                if dataset_name and not model_name.startswith(dataset_name):
                    continue
                    
                try:
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"加载模型: {model_name}")
                except Exception as e:
                    logger.error(f"加载模型 {model_name} 失败: {e}")
    
    # 加载数据
    data_dir = 'output/processed_data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_path = os.path.join(data_dir, filename)
                data_name = filename.replace('_processed.csv', '')
                
                # 如果指定了数据集，只加载该数据集的数据
                if dataset_name and data_name != dataset_name:
                    continue
                    
                try:
                    data[data_name] = pd.read_csv(data_path)
                    logger.info(f"加载数据: {data_name}")
                except Exception as e:
                    logger.error(f"加载数据 {data_name} 失败: {e}")
    
    return models, data

def prepare_test_data(data):
    """准备测试数据"""
    test_datasets = {}
    
    for dataset_name, df in data.items():
        if dataset_name == 'stroke':
            X = df.drop(['stroke', 'ID', 'N_Days'], axis=1, errors='ignore')
            y = df['stroke']
            test_datasets[dataset_name] = (X, y)
        elif dataset_name == 'heart':
            X = df.drop(['HeartDisease', 'ID', 'N_Days'], axis=1, errors='ignore') 
            y = df['HeartDisease']
            test_datasets[dataset_name] = (X, y)
        elif dataset_name == 'cirrhosis':
            X = df.drop(['Stage', 'ID', 'N_Days'], axis=1, errors='ignore')
            y = df['Stage']
            test_datasets[dataset_name] = (X, y)
    
    return test_datasets

def generate_roc_curves(models, test_datasets):
    """生成ROC曲线"""
    logger.info("生成ROC曲线...")
    
    for dataset_name, (X_test, y_test) in test_datasets.items():
        # 对于回归任务（肝硬化），我们不生成ROC曲线
        if dataset_name == 'cirrhosis':
            logger.info(f"{dataset_name}是回归任务，跳过ROC曲线生成")
            continue
            
        # 查找最佳模型
        best_model_name = f"{dataset_name}_best_baseline_model"
        
        if best_model_name not in models:
            logger.warning(f"没有找到{dataset_name}的最佳模型，跳过ROC曲线生成")
            continue
            
        model = models[best_model_name]
        
        try:
            # 检查模型是否支持概率预测
            if hasattr(model, 'predict_proba'):
                # 预测概率
                y_prob = model.predict_proba(X_test)
                
                # 对于二分类问题
                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                    
                    # 应用平滑处理，避免极端值
                    y_prob_smoothed = np.zeros_like(y_prob)
                    for i, prob in enumerate(y_prob):
                        # 确定风险级别和平滑参数
                        if prob < 0.2:
                            risk_level = 'low'
                            min_prob, max_prob = 0.02, 0.90
                        elif prob < 0.5:
                            risk_level = 'medium'
                            min_prob, max_prob = 0.05, 0.95
                        else:
                            risk_level = 'high'
                            min_prob, max_prob = 0.10, 0.98
                            
                        # 应用平滑
                        y_prob_smoothed[i] = smooth_probability(prob, method='sigmoid_quantile', 
                                                              min_prob=min_prob, max_prob=max_prob)
                    
                    # 计算ROC曲线（同时使用原始和平滑后的概率）
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fpr_smooth, tpr_smooth, thresholds_smooth = roc_curve(y_test, y_prob_smoothed)
                    roc_auc_smooth = auc(fpr_smooth, tpr_smooth)
                    
                    # 绘制ROC曲线
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'原始 ROC曲线 (AUC = {roc_auc:.2f})')
                    plt.plot(fpr_smooth, tpr_smooth, color='green', lw=2, label=f'平滑后 ROC曲线 (AUC = {roc_auc_smooth:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机预测 (AUC = 0.5)')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('假阳性率')
                    plt.ylabel('真阳性率')
                    plt.title(f'{dataset_name.capitalize()} 模型的ROC曲线')
                    plt.legend(loc="lower right")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_roc_curve.png')
                    plt.close()
                    
                    logger.info(f"{dataset_name} 的ROC曲线已生成")
                else:
                    logger.warning(f"{dataset_name} 不是二分类问题，跳过ROC曲线生成")
            else:
                logger.warning(f"{dataset_name} 模型不支持概率预测，跳过ROC曲线生成")
                
        except Exception as e:
            logger.error(f"为{dataset_name}生成ROC曲线时出错: {e}")
            logger.error(traceback.format_exc())

def generate_feature_importance(models, test_datasets):
    """生成特征重要性图"""
    logger.info("生成特征重要性图...")
    
    for dataset_name, (X_test, y_test) in test_datasets.items():
        # 查找最佳模型
        best_model_name = f"{dataset_name}_best_baseline_model"
        
        if best_model_name not in models:
            logger.warning(f"没有找到{dataset_name}的最佳模型，跳过特征重要性图生成")
            continue
            
        model = models[best_model_name]
        
        try:
            # 检查模型是否有feature_importances_属性（树模型通常有）
            if hasattr(model, 'feature_importances_'):
                # 获取特征重要性
                importances = model.feature_importances_
                
                # 获取特征名称
                feature_names = X_test.columns
                
                # 创建一个DataFrame用于排序
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
                # 按重要性降序排序
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
                
                # 取前15个特征
                top_n = min(15, len(feature_importance_df))
                feature_importance_df = feature_importance_df.head(top_n)
                
                # 绘制特征重要性条形图
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance_df)
                plt.title(f'{dataset_name.capitalize()} 模型的特征重要性')
                plt.xlabel('重要性')
                plt.ylabel('特征')
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_feature_importance.png')
                plt.close()
                
                logger.info(f"{dataset_name} 的特征重要性图已生成")
                
            # 对于其他模型，尝试获取coef_属性（线性模型通常有）
            elif hasattr(model, 'coef_'):
                # 获取系数
                coefs = model.coef_
                
                # 对于多分类问题，可能有多组系数
                if coefs.ndim > 1:
                    # 使用系数的绝对值平均值
                    coefs = np.abs(coefs).mean(axis=0)
                    
                # 获取特征名称
                feature_names = X_test.columns
                
                # 创建一个DataFrame用于排序
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(coefs)  # 使用系数的绝对值
                })
                
                # 按重要性降序排序
                feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
                
                # 取前15个特征
                top_n = min(15, len(feature_importance_df))
                feature_importance_df = feature_importance_df.head(top_n)
                
                # 绘制特征重要性条形图
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', data=feature_importance_df)
                plt.title(f'{dataset_name.capitalize()} 模型的特征重要性（系数绝对值）')
                plt.xlabel('系数绝对值')
                plt.ylabel('特征')
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_feature_importance.png')
                plt.close()
                
                logger.info(f"{dataset_name} 的特征重要性图（基于系数）已生成")
                
            else:
                logger.warning(f"{dataset_name} 模型不支持特征重要性可视化，跳过特征重要性图生成")
                
        except Exception as e:
            logger.error(f"为{dataset_name}生成特征重要性图时出错: {e}")

def generate_confusion_matrices(models, test_datasets):
    """生成混淆矩阵"""
    logger.info("生成混淆矩阵...")
    
    for dataset_name, (X_test, y_test) in test_datasets.items():
        # 对于回归任务（肝硬化），我们不生成混淆矩阵
        if dataset_name == 'cirrhosis':
            logger.info(f"{dataset_name}是回归任务，跳过混淆矩阵生成")
            continue
            
        # 查找最佳模型
        best_model_name = f"{dataset_name}_best_baseline_model"
        
        if best_model_name not in models:
            logger.warning(f"没有找到{dataset_name}的最佳模型，跳过混淆矩阵生成")
            continue
            
        model = models[best_model_name]
        
        try:
            # 预测值
            y_pred = model.predict(X_test)
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{dataset_name.capitalize()} 模型的混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.tight_layout()
            plt.savefig(f'output/figures/{dataset_name}_confusion_matrix.png')
            plt.close()
            
            logger.info(f"{dataset_name} 的混淆矩阵已生成")
            
        except Exception as e:
            logger.error(f"为{dataset_name}生成混淆矩阵时出错: {e}\n{traceback.format_exc()}")

def generate_residual_plots(models, test_datasets):
    """生成残差图"""
    logger.info("生成残差图...")
    
    for dataset_name, (X_test, y_test) in test_datasets.items():
        # 查找最佳模型
        best_model_name = f"{dataset_name}_best_baseline_model"
        
        if best_model_name not in models:
            logger.warning(f"没有找到{dataset_name}的最佳模型，跳过残差图生成")
            continue
            
        model = models[best_model_name]
        
        try:
            # 预测值
            y_pred = model.predict(X_test)
            
            # 对于分类任务，绘制预测概率vs真实值
            if dataset_name in ['stroke', 'heart']:
                # 计算准确率
                accuracy = accuracy_score(y_test, y_pred)
                
                # 创建条形图展示预测性能
                plt.figure(figsize=(10, 6))
                plt.bar(['准确率'], [accuracy], color='blue')
                plt.ylim(0, 1)
                plt.axhline(y=0.5, color='r', linestyle='--', label='随机猜测')
                plt.title(f'{dataset_name.capitalize()} 模型的预测准确率')
                plt.ylabel('准确率')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                plt.close()
                
                logger.info(f"{dataset_name} 的预测准确率图已生成")
                
                # 如果模型支持概率预测，生成预测概率分布图
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的概率
                        
                        # 绘制不同类别的预测概率分布
                        plt.figure(figsize=(10, 6))
                        for target_class in [0, 1]:
                            sns.kdeplot(y_prob[y_test == target_class], 
                                     label=f'真实类别 = {target_class}', 
                                     fill=True, alpha=0.5)
                        plt.axvline(x=0.5, color='r', linestyle='--', label='决策边界')
                        plt.title(f'{dataset_name.capitalize()} 模型的预测概率分布')
                        plt.xlabel('预测为正类的概率')
                        plt.ylabel('密度')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.tight_layout()
                        plt.savefig(f'output/figures/{dataset_name}_prob_distribution.png')
                        plt.close()
                        
                        logger.info(f"{dataset_name} 的预测概率分布图已生成")
                    except Exception as e:
                        logger.error(f"为{dataset_name}生成预测概率分布图时出错: {e}")
                
            else:  # 回归任务
                # 计算残差
                residuals = y_test.values - y_pred
                
                # 绘制残差图
                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('预测值')
                plt.ylabel('残差')
                plt.title(f'{dataset_name.capitalize()} 模型的残差图')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_residual_plot.png')
                plt.close()
                
                # 绘制预测值vs实际值
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('实际值')
                plt.ylabel('预测值')
                plt.title(f'{dataset_name.capitalize()} 模型的预测值vs实际值')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(f'output/figures/{dataset_name}_pred_vs_actual.png')
                plt.close()
                
                logger.info(f"{dataset_name} 的残差图已生成")
            
        except Exception as e:
            logger.error(f"为{dataset_name}生成残差图时出错: {e}\n{traceback.format_exc()}")

def generate_metrics_json(models, test_datasets):
    """生成模型评估指标JSON文件"""
    logger.info("生成模型评估指标...")
    
    for dataset_name, (X_test, y_test) in test_datasets.items():
        # 查找适用于该数据集的所有模型
        dataset_models = [m for m in models.keys() if m.startswith(dataset_name)]
        
        for model_name in dataset_models:
            model = models[model_name]
            metrics = {}
            
            try:
                # 预测值
                y_pred = model.predict(X_test)
                
                # 分类任务
                if dataset_name in ['stroke', 'heart']:
                    metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
                    
                    try:
                        metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted'))
                        metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted'))
                        metrics['f1'] = float(f1_score(y_test, y_pred, average='weighted'))
                    except:
                        metrics['precision'] = "N/A"
                        metrics['recall'] = "N/A"
                        metrics['f1'] = "N/A"
                    
                    # AUC（只适用于二分类）
                    if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                        try:
                            y_prob = model.predict_proba(X_test)[:, 1]
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            metrics['auc'] = float(auc(fpr, tpr))
                        except:
                            metrics['auc'] = "N/A"
                    else:
                        metrics['auc'] = "N/A"
                    
                # 回归任务
                else:
                    metrics['mse'] = float(mean_squared_error(y_test, y_pred))
                    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    metrics['r2'] = float(r2_score(y_test, y_pred))
                    
                # 保存指标到JSON文件
                with open(f'output/models/{model_name}_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=4)
                
                logger.info(f"{model_name} 的评估指标已生成")
                
            except Exception as e:
                logger.error(f"为{model_name}生成评估指标时出错: {e}\n{traceback.format_exc()}")

def generate_shap_visualizations(models, test_datasets):
    """生成SHAP值可视化"""
    logger.info("尝试生成SHAP值可视化...")
    
    try:
        import shap
        
        for dataset_name, (X_test, y_test) in test_datasets.items():
            # 使用较小的样本以加快计算速度
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42)
            
            # 查找最佳模型
            best_model_name = f"{dataset_name}_best_baseline_model"
            
            if best_model_name not in models:
                logger.warning(f"没有找到{dataset_name}的最佳模型，跳过SHAP值可视化")
                continue
                
            model = models[best_model_name]
            
            try:
                # 尝试使用TreeExplainer（适用于基于树的模型）
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # 绘制SHAP摘要图
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
                    plt.title(f'{dataset_name.capitalize()} 模型的SHAP特征重要性')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_importance.png')
                    plt.close()
                    
                    # 绘制详细的SHAP摘要图
                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(shap_values, X_sample, show=False)
                    plt.title(f'{dataset_name.capitalize()} 模型的SHAP值摘要')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_summary.png')
                    plt.close()
                    
                    logger.info(f"{dataset_name} 的SHAP值可视化已生成")
                    
                except Exception as e:
                    logger.warning(f"TreeExplainer失败，尝试使用KernelExplainer: {e}")
                    
                    # 如果TreeExplainer失败，尝试使用KernelExplainer
                    # 使用更小的样本以加快计算速度
                    smaller_sample = min(50, sample_size)
                    X_smaller = X_sample.iloc[:smaller_sample]
                    
                    # 定义预测函数
                    def model_predict(X):
                        return model.predict(X)
                    
                    explainer = shap.KernelExplainer(model_predict, X_smaller)
                    shap_values = explainer.shap_values(X_smaller)
                    
                    # 绘制SHAP摘要图
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_smaller, plot_type='bar', show=False)
                    plt.title(f'{dataset_name.capitalize()} 模型的SHAP特征重要性')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_importance.png')
                    plt.close()
                    
                    # 绘制详细的SHAP摘要图
                    plt.figure(figsize=(12, 10))
                    shap.summary_plot(shap_values, X_smaller, show=False)
                    plt.title(f'{dataset_name.capitalize()} 模型的SHAP值摘要')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_shap_summary.png')
                    plt.close()
                    
                    logger.info(f"{dataset_name} 的SHAP值可视化(KernelExplainer)已生成")
                    
            except Exception as e:
                logger.error(f"为{dataset_name}生成SHAP值可视化时出错: {e}\n{traceback.format_exc()}")
                
    except ImportError:
        logger.warning("未安装SHAP库，跳过SHAP值可视化")

def run_chart_generation(dataset_name=None, chart_types=None):
    """运行所有图表生成功能"""
    logger.info(f"开始为{'所有' if dataset_name is None else dataset_name}数据集生成{'所有' if chart_types is None or 'all' in chart_types else '、'.join(chart_types)}图表...")
    
    # 加载模型和数据
    models, data = load_models_and_data(dataset_name)
    
    if not models:
        logger.error(f"未找到{'任何' if dataset_name is None else dataset_name}模型!")
        return
        
    if not data:
        logger.error(f"未找到{'任何' if dataset_name is None else dataset_name}数据!")
        return
    
    # 准备测试数据
    test_datasets = prepare_test_data(data)
    
    # 根据指定的图表类型生成图表
    if chart_types is None or 'all' in chart_types or 'roc' in chart_types:
        generate_roc_curves(models, test_datasets)
    
    if chart_types is None or 'all' in chart_types or 'feature' in chart_types:
        generate_feature_importance(models, test_datasets)
    
    if chart_types is None or 'all' in chart_types or 'confusion' in chart_types:
        generate_confusion_matrices(models, test_datasets)
    
    if chart_types is None or 'all' in chart_types or 'residual' in chart_types:
        generate_residual_plots(models, test_datasets)
    
    if chart_types is None or 'all' in chart_types or 'metrics' in chart_types:
        generate_metrics_json(models, test_datasets)
    
    if chart_types is None or 'all' in chart_types or 'shap' in chart_types:
        generate_shap_visualizations(models, test_datasets)
    
    logger.info(f"{'所有' if dataset_name is None else dataset_name}数据集的{'所有' if chart_types is None or 'all' in chart_types else '、'.join(chart_types)}图表生成完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成模型评估图表和指标')
    parser.add_argument('--dataset', type=str, choices=['stroke', 'heart', 'cirrhosis'], 
                        help='指定要处理的数据集，不指定则处理所有数据集')
    parser.add_argument('--chart-type', type=str, nargs='+',
                        choices=['roc', 'feature', 'confusion', 'residual', 'shap', 'metrics', 'all'],
                        default=['all'], help='指定要生成的图表类型，默认生成所有类型')
    
    args = parser.parse_args()
    
    run_chart_generation(args.dataset, args.chart_type) 