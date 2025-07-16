import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_models_and_data(dataset_name=None):
    """加载模型和测试数据
    
    Args:
        dataset_name (str, optional): 指定要加载的数据集名称，例如'stroke', 'heart', 'cirrhosis'。
                                     如果为None，则加载所有数据集。
    """
    models = {}
    model_dir = 'output/models'
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                # 如果指定了数据集，则只加载该数据集的模型
                if dataset_name and not filename.startswith(dataset_name):
                    continue
                    
                try:
                    model_path = os.path.join(model_dir, filename)
                    model_name = filename.replace('.pkl', '')
                    models[model_name] = joblib.load(model_path)
                    print(f"加载模型: {model_name}")
                except Exception as e:
                    print(f"加载模型 {model_name} 失败: {e}")
    
    # 加载处理后的数据
    data = {}
    data_dir = 'output/processed_data'
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                data_name = filename.replace('_processed.csv', '')
                
                # 如果指定了数据集，则只加载该数据集
                if dataset_name and data_name != dataset_name:
                    continue
                    
                data_path = os.path.join(data_dir, filename)
                try:
                    data[data_name] = pd.read_csv(data_path)
                    print(f"加载数据: {data_name}")
                except Exception as e:
                    print(f"加载数据 {data_name} 失败: {e}")
    
    return models, data

def prepare_test_data(data):
    """准备测试数据"""
    test_datasets = {}
    
    for name, df in data.items():
        if name == 'stroke':
            features = df.drop(['id', 'stroke'], axis=1, errors='ignore')
            target = df['stroke']
        elif name == 'heart':
            features = df.drop(['HeartDisease'], axis=1, errors='ignore')
            target = df['HeartDisease']
        elif name == 'cirrhosis':
            features = df.drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
            target = df['Stage']
        
        # 只保留数值型特征
        features = features.select_dtypes(include=['float64', 'int64']).copy()
        
        # 分割测试集（为了简单，这里我们使用整个数据集的20%作为测试集）
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        test_datasets[name] = (X_test, y_test)
    
    return test_datasets

def generate_metrics_json(models, test_datasets):
    """生成模型评估指标JSON文件"""
    print("\n生成模型评估指标...")
    
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
                
                print(f"{model_name} 的评估指标已生成")
                
            except Exception as e:
                print(f"为{model_name}生成评估指标时出错: {e}")

def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='生成模型评估指标')
    parser.add_argument('--dataset', type=str, choices=['stroke', 'heart', 'cirrhosis'], 
                        help='指定要处理的数据集，不指定则处理所有数据集')
    args = parser.parse_args()
    
    print(f"开始生成{'所有' if args.dataset is None else args.dataset}模型评估指标...")
    
    # 加载模型和数据
    models, data = load_models_and_data(dataset_name=args.dataset)
    
    if not models:
        print(f"未找到{'任何' if args.dataset is None else args.dataset}模型!")
        return
        
    if not data:
        print(f"未找到{'任何' if args.dataset is None else args.dataset}数据!")
        return
    
    # 准备测试数据
    test_datasets = prepare_test_data(data)
    
    # 生成模型评估指标JSON
    generate_metrics_json(models, test_datasets)
    
    print(f"\n{'所有' if args.dataset is None else args.dataset}模型评估指标生成完成！")

if __name__ == "__main__":
    main() 