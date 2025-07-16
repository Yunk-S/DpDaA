import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
from visualization import DataVisualizer
from model_training import ModelTrainer
from model_calibration import run_calibration_training
import time
import logging
import argparse
import subprocess

# 确保output目录存在
os.makedirs('output', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # 使用根目录下的app.log，避免路径问题
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """创建必要的目录结构"""
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output/models/calibrators', exist_ok=True)
    os.makedirs('output/processed_data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    logger.info("目录结构已创建")

def run_data_preprocessing():
    """运行数据预处理流程"""
    logger.info("开始数据预处理...")
    start_time = time.time()
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_full_preprocessing()
    
    end_time = time.time()
    logger.info(f"数据预处理完成，耗时: {end_time - start_time:.2f} 秒")
    return processed_data

def run_data_visualization():
    """运行数据可视化流程"""
    logger.info("开始数据可视化...")
    start_time = time.time()
    
    visualizer = DataVisualizer()
    visualizer.load_processed_data()
    visualizer.run_all_visualizations()
    
    end_time = time.time()
    logger.info(f"数据可视化完成，耗时: {end_time - start_time:.2f} 秒")

def run_generate_charts_and_metrics():
    """运行图表和指标生成工具"""
    logger.info("开始生成各数据集的图表和评估指标...")
    start_time = time.time()
    
    # 确保必要的目录存在
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    
    # 使用统一的图表生成模块
    try:
        from chart_generator import run_chart_generation
        
        # 生成心脏病数据集的图表和指标
        logger.info("生成heart数据集的图表和评估指标...")
        run_chart_generation('heart', ['all'])
        
        # 生成肝硬化数据集的图表和指标
        logger.info("生成cirrhosis数据集的图表和评估指标...")
        run_chart_generation('cirrhosis', ['all'])
        
        # 生成中风数据集的图表和指标
        logger.info("生成stroke数据集的图表和评估指标...")
        run_chart_generation('stroke', ['all'])
        
    except Exception as e:
        logger.error(f"生成图表和评估指标时出错: {e}")
    
    end_time = time.time()
    logger.info(f"图表和评估指标生成完成，耗时: {end_time - start_time:.2f} 秒")

def run_model_training():
    """运行模型训练流程"""
    logger.info("开始模型训练...")
    start_time = time.time()
    
    # 首先获取处理好的数据
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_full_preprocessing()  # 返回一个字典，键为数据集名称，值为(features, target)
    
    # 分割数据集，准备训练
    data_splits = {}
    
    # 分别处理每个数据集
    for dataset_name, (features, target) in processed_data.items():
        # 训练集和测试集分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        data_splits[dataset_name] = (X_train, X_test, y_train, y_test)
    
    # 初始化模型训练器并传入数据
    trainer = ModelTrainer(data_splits)
    best_models = trainer.run_full_model_pipeline()
    
    end_time = time.time()
    logger.info(f"模型训练完成，耗时: {end_time - start_time:.2f} 秒")
    return best_models

def run_model_calibration():
    """运行模型校准流程"""
    logger.info("开始模型校准训练...")
    start_time = time.time()
    
    # 调用model_calibration模块中的校准训练函数
    calibrators = run_calibration_training()
    
    end_time = time.time()
    logger.info(f"模型校准完成，耗时: {end_time - start_time:.2f} 秒")
    return calibrators

def run_web_app():
    """运行Web应用"""
    logger.info("启动Web应用...")
    try:
        from app import app
        app.run(debug=False, port=5000)
    except Exception as e:
        logger.error(f"启动Web应用时出错: {e}")

def run_full_pipeline():
    """运行完整的分析和预测流程"""
    logger.info("开始运行完整的分析和预测流程...")
    setup_directories()
    
    # 步骤1: 数据预处理
    processed_data = run_data_preprocessing()
    
    # 步骤2: 数据可视化
    run_data_visualization()
    
    # 步骤3: 模型训练
    best_models = run_model_training()
    
    # 步骤4: 模型校准
    calibrators = run_model_calibration()
    
    # 步骤5: 生成图表和评估指标
    run_generate_charts_and_metrics()
    
    # 步骤6: 启动Web应用
    run_web_app()
    
    # 打印总结
    logger.info("完整的分析和预测流程已完成！")
    logger.info(f"最佳模型: {list(best_models.keys())}")
    
    return processed_data, best_models, calibrators

def print_system_info():
    """打印系统和依赖库信息"""
    import sys
    import platform
    import sklearn
    import pandas as pd
    import numpy as np
    import matplotlib
    
    logger.info("系统信息:")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    
    logger.info("依赖库版本:")
    logger.info(f"scikit-learn: {sklearn.__version__}")
    logger.info(f"pandas: {pd.__version__}")
    logger.info(f"numpy: {np.__version__}")
    logger.info(f"matplotlib: {matplotlib.__version__}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='疾病预测与大数据分析系统')
    parser.add_argument('--preprocess', action='store_true', help='只运行数据预处理')
    parser.add_argument('--visualize', action='store_true', help='只运行数据可视化')
    parser.add_argument('--train', action='store_true', help='只运行模型训练')
    parser.add_argument('--calibrate', action='store_true', help='只运行模型校准')
    parser.add_argument('--generate', action='store_true', help='只运行图表和指标生成')
    parser.add_argument('--webapp', action='store_true', help='只运行Web应用')
    parser.add_argument('--all', action='store_true', help='运行完整的分析和预测流程')
    
    args = parser.parse_args()
    
    print_system_info()
    
    if args.preprocess:
        run_data_preprocessing()
    elif args.visualize:
        run_data_visualization()
    elif args.train:
        run_model_training()
    elif args.calibrate:
        run_model_calibration()
    elif args.generate:
        run_generate_charts_and_metrics()
    elif args.webapp:
        run_web_app()
    elif args.all:
        run_full_pipeline()
    else:
        # 直接运行完整流程，无需参数
        run_full_pipeline() 