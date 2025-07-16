import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import missingno as msno
import os

class DataPreprocessor:
    def __init__(self):
        """初始化数据预处理类"""
        self.stroke_data = None
        self.heart_data = None
        self.cirrhosis_data = None
        
        # 存储编码器，用于逆转换
        self.encoders = {
            'stroke': {},
            'heart': {},
            'cirrhosis': {}
        }
        
        # 存储缩放器
        self.scalers = {
            'stroke': None,
            'heart': None,
            'cirrhosis': None
        }
    
    def load_data(self, stroke_path='stroke.csv', heart_path='heart.csv', cirrhosis_path='cirrhosis.csv'):
        """加载三个数据集"""
        self.stroke_data = pd.read_csv(stroke_path)
        self.heart_data = pd.read_csv(heart_path)
        self.cirrhosis_data = pd.read_csv(cirrhosis_path)
        print("数据加载完成！")
        
        # 创建输出目录
        os.makedirs('output', exist_ok=True)
        os.makedirs('output/figures', exist_ok=True)
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/processed_data', exist_ok=True)
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def explore_data(self, save_figures=True):
        """基本的数据探索"""
        print("开始探索数据...")
        
        # 首先加载数据（如果尚未加载）
        if self.stroke_data is None or self.heart_data is None or self.cirrhosis_data is None:
            self.load_data()
        
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        for dataset_name, df in datasets.items():
            print(f"\n探索 {dataset_name} 数据集:")
            print(f"数据维度: {df.shape}")
            print(f"数据类型:\n{df.dtypes}")
            
            # 基本统计信息
            print(f"统计摘要:\n{df.describe().T}")
            
            # 缺失值分析
            missing_count = df.isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_data = pd.DataFrame({'缺失值数量': missing_count, '缺失率': missing_percent})
            missing_data = missing_data[missing_data['缺失值数量'] > 0]
            
            if not missing_data.empty:
                print(f"缺失值分析:\n{missing_data}")
                
                if save_figures:
                    # 使用matplotlib直接可视化缺失值
                    plt.figure(figsize=(10, 6))
                    
                    # 替换非数值数据为NaN
                    data_for_viz = df.copy()
                    data_for_viz = data_for_viz.replace(['--', 'N/A', 'NA'], np.nan)
                    
                    # 计算每列的缺失值百分比
                    missing_percent = data_for_viz.isnull().mean() * 100
                    
                    # 绘制缺失值条形图
                    missing_percent.sort_values(ascending=False).plot(kind='bar')
                    plt.title(f'{dataset_name} - 缺失值百分比')
                    plt.ylabel('缺失值百分比 (%)')
                    plt.xlabel('特征')
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{dataset_name}_missing_percent.png')
                    plt.close()
                    
                    # 尝试使用missingno库的matrix图（不使用heatmap）
                    try:
                        plt.figure(figsize=(12, 6))
                        msno.matrix(data_for_viz)
                        plt.title(f'{dataset_name} - 缺失值矩阵图', fontsize=16)
                        plt.tight_layout()
                        plt.savefig(f"output/figures/{dataset_name}_missing_matrix.png")
                        plt.close()
                    except Exception as e:
                        print(f"无法生成缺失值矩阵图: {e}")
    
        print("数据探索完成!")
        return datasets
    
    def handle_missing_values(self, method='iterative'):
        """处理缺失值
        
        参数:
            method: 填充方法，可选 'mean', 'median', 'knn', 'iterative'
        """
        # 中风数据集缺失值处理
        print("处理中风数据集缺失值...")
        # 检查 bmi 列的缺失值
        self.stroke_data['bmi'] = self.stroke_data['bmi'].replace('N/A', np.nan).astype(float)
        
        # 根据不同方法处理缺失值
        if method == 'mean':
            self.stroke_data['bmi'] = self.stroke_data['bmi'].fillna(self.stroke_data['bmi'].mean())
        elif method == 'median':
            self.stroke_data['bmi'] = self.stroke_data['bmi'].fillna(self.stroke_data['bmi'].median())
        elif method == 'knn':
            # 使用KNN填充连续变量缺失值
            numeric_cols = self.stroke_data.select_dtypes(include=['float64', 'int64']).columns
            knn_imputer = KNNImputer(n_neighbors=5)
            self.stroke_data[numeric_cols] = pd.DataFrame(
                knn_imputer.fit_transform(self.stroke_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.stroke_data.index
            )
        elif method == 'iterative':
            # 使用迭代填充连续变量缺失值
            numeric_cols = self.stroke_data.select_dtypes(include=['float64', 'int64']).columns
            iter_imputer = IterativeImputer(max_iter=10, random_state=42)
            self.stroke_data[numeric_cols] = pd.DataFrame(
                iter_imputer.fit_transform(self.stroke_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.stroke_data.index
            )
            
        # 使用众数填充类别变量的缺失值
        self.stroke_data['smoking_status'] = self.stroke_data['smoking_status'].fillna(
            self.stroke_data['smoking_status'].mode()[0]
        )
        
        # 心脏病数据集缺失值处理
        print("处理心脏病数据集缺失值...")
        # 检查缺失值并填充
        if self.heart_data.isnull().sum().sum() > 0:
            if method in ['mean', 'median']:
                for col in self.heart_data.select_dtypes(include=['float64', 'int64']).columns:
                    if method == 'mean':
                        self.heart_data[col] = self.heart_data[col].fillna(self.heart_data[col].mean())
                    else:
                        self.heart_data[col] = self.heart_data[col].fillna(self.heart_data[col].median())
            elif method == 'knn':
                numeric_cols = self.heart_data.select_dtypes(include=['float64', 'int64']).columns
                knn_imputer = KNNImputer(n_neighbors=5)
                self.heart_data[numeric_cols] = pd.DataFrame(
                    knn_imputer.fit_transform(self.heart_data[numeric_cols]), 
                    columns=numeric_cols,
                    index=self.heart_data.index
                )
            elif method == 'iterative':
                numeric_cols = self.heart_data.select_dtypes(include=['float64', 'int64']).columns
                iter_imputer = IterativeImputer(max_iter=10, random_state=42)
                self.heart_data[numeric_cols] = pd.DataFrame(
                    iter_imputer.fit_transform(self.heart_data[numeric_cols]), 
                    columns=numeric_cols,
                    index=self.heart_data.index
                )
                
            # 使用众数填充分类变量
            for col in self.heart_data.select_dtypes(include=['object']).columns:
                self.heart_data[col] = self.heart_data[col].fillna(self.heart_data[col].mode()[0])
        
        # 肝硬化数据集缺失值处理
        print("处理肝硬化数据集缺失值...")
        # 将 'NA' 转换为 np.nan
        self.cirrhosis_data = self.cirrhosis_data.replace('NA', np.nan)
        
        # 处理数值型特征的缺失值
        if method in ['mean', 'median']:
            for col in self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns:
                if self.cirrhosis_data[col].isnull().sum() > 0:
                    if method == 'mean':
                        self.cirrhosis_data[col] = self.cirrhosis_data[col].fillna(self.cirrhosis_data[col].mean())
                    else:
                        self.cirrhosis_data[col] = self.cirrhosis_data[col].fillna(self.cirrhosis_data[col].median())
        elif method == 'knn':
            numeric_cols = self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns
            knn_imputer = KNNImputer(n_neighbors=5)
            self.cirrhosis_data[numeric_cols] = pd.DataFrame(
                knn_imputer.fit_transform(self.cirrhosis_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.cirrhosis_data.index
            )
        elif method == 'iterative':
            numeric_cols = self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns
            iter_imputer = IterativeImputer(max_iter=10, random_state=42)
            self.cirrhosis_data[numeric_cols] = pd.DataFrame(
                iter_imputer.fit_transform(self.cirrhosis_data[numeric_cols]), 
                columns=numeric_cols,
                index=self.cirrhosis_data.index
            )
            
        # 使用众数填充分类变量
        for col in self.cirrhosis_data.select_dtypes(include=['object']).columns:
            if self.cirrhosis_data[col].isnull().sum() > 0:
                self.cirrhosis_data[col] = self.cirrhosis_data[col].fillna(self.cirrhosis_data[col].mode()[0])
                
        print("所有数据集的缺失值处理完成！")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def detect_outliers(self, method='iqr', visualize=True):
        """检测和处理异常值
        
        参数:
            method: 异常值检测方法，可选 'iqr' 或 'zscore'
            visualize: 是否绘制异常值可视化
        """
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        outlier_summary = {}
        
        for name, data in datasets.items():
            print(f"\n{name.upper()} 数据集异常值检测:")
            
            # 只对数值型特征检测异常值
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            
            # 过滤掉ID列、标签列和日期列
            numeric_cols = [col for col in numeric_cols if col.lower() not in ['id', 'stroke', 
                                                                               'heartdisease', 'n_days', 
                                                                               'status', 'stage']]
            
            outliers_dict = {}
            
            for col in numeric_cols:
                outliers_indices = []
                
                if method == 'iqr':
                    # IQR方法
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # 找出异常值索引
                    outliers_indices = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
                    
                elif method == 'zscore':
                    # Z-score方法
                    mean = data[col].mean()
                    std = data[col].std()
                    z_scores = abs((data[col] - mean) / std)
                    outliers_indices = data[z_scores > 3].index
                
                # 保存异常值信息
                outliers_dict[col] = {
                    'count': len(outliers_indices),
                    'percentage': (len(outliers_indices) / len(data)) * 100,
                    'indices': outliers_indices.tolist()
                }
                
                print(f"特征 '{col}' 有 {len(outliers_indices)} 个异常值 ({(len(outliers_indices)/len(data)*100):.2f}%)")
                
                # 可视化
                if visualize and len(outliers_indices) > 0:
                    plt.figure(figsize=(10, 6))
                    
                    # 绘制箱线图
                    plt.subplot(1, 2, 1)
                    sns.boxplot(y=data[col])
                    plt.title(f'{name.capitalize()} 数据集 - {col} 箱线图')
                    
                    # 绘制直方图
                    plt.subplot(1, 2, 2)
                    sns.histplot(data[col], kde=True)
                    plt.title(f'{name.capitalize()} 数据集 - {col} 直方图')
                    
                    plt.tight_layout()
                    plt.savefig(f'output/figures/{name}_{col}_outliers.png')
                    plt.close()
            
            outlier_summary[name] = outliers_dict
        
        return outlier_summary
    
    def handle_outliers(self, method='cap'):
        """处理异常值
        
        参数:
            method: 处理方法，可选 'cap' (截断), 'remove' (删除), 'mean' (均值替换)
        """
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        for name, data in datasets.items():
            print(f"\n处理 {name.upper()} 数据集异常值:")
            
            # 只处理数值型特征
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            
            # 过滤掉ID列、标签列和日期列
            numeric_cols = [col for col in numeric_cols if col.lower() not in ['id', 'stroke', 
                                                                               'heartdisease', 'n_days', 
                                                                               'status', 'stage']]
            
            for col in numeric_cols:
                # 计算上下界
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 找出异常值
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                
                if len(outliers) > 0:
                    print(f"处理特征 '{col}' 的 {len(outliers)} 个异常值")
                    
                    if method == 'cap':
                        # 截断法
                        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    elif method == 'remove':
                        # 删除异常值（仅当异常值比例不高时）
                        if len(outliers) / len(data) < 0.05:  # 少于5%的异常值
                            data.drop(outliers.index, inplace=True)
                            print(f"删除了 {len(outliers)} 行异常值")
                        else:
                            print(f"异常值比例过高 ({len(outliers)/len(data)*100:.2f}%)，改用截断法")
                            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                    elif method == 'mean':
                        # 均值替换
                        mean_value = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)][col].mean()
                        data.loc[(data[col] < lower_bound) | (data[col] > upper_bound), col] = mean_value
            
            # 更新数据集
            if name == 'stroke':
                self.stroke_data = data
            elif name == 'heart':
                self.heart_data = data
            else:
                self.cirrhosis_data = data
        
        print("异常值处理完成！")
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def encode_categorical_features(self):
        """编码分类特征"""
        # 处理中风数据集
        print("\n编码中风数据集分类特征...")
        
        # 二元编码
        binary_cols = ['gender', 'ever_married', 'Residence_type']
        for col in binary_cols:
            le = LabelEncoder()
            self.stroke_data[col] = le.fit_transform(self.stroke_data[col])
            self.encoders['stroke'][col] = le
        
        # One-hot编码
        onehot_cols = ['work_type', 'smoking_status']
        for col in onehot_cols:
            dummies = pd.get_dummies(self.stroke_data[col], prefix=col, drop_first=False)
            self.stroke_data = pd.concat([self.stroke_data, dummies], axis=1)
            self.stroke_data.drop(col, axis=1, inplace=True)
        
        # 处理心脏病数据集
        print("编码心脏病数据集分类特征...")
        # 二元编码
        binary_cols = ['Sex', 'ExerciseAngina']
        for col in binary_cols:
            le = LabelEncoder()
            self.heart_data[col] = le.fit_transform(self.heart_data[col])
            self.encoders['heart'][col] = le
            
        # One-hot编码
        onehot_cols = ['ChestPainType', 'RestingECG', 'ST_Slope']
        for col in onehot_cols:
            dummies = pd.get_dummies(self.heart_data[col], prefix=col, drop_first=False)
            self.heart_data = pd.concat([self.heart_data, dummies], axis=1)
            self.heart_data.drop(col, axis=1, inplace=True)
        
        # 处理肝硬化数据集
        print("编码肝硬化数据集分类特征...")
        # 二元编码
        binary_cols = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders']
        for col in binary_cols:
            le = LabelEncoder()
            self.cirrhosis_data[col] = le.fit_transform(self.cirrhosis_data[col])
            self.encoders['cirrhosis'][col] = le
            
        # One-hot编码
        onehot_cols = ['Drug', 'Status', 'Edema']
        for col in onehot_cols:
            dummies = pd.get_dummies(self.cirrhosis_data[col], prefix=col, drop_first=False)
            self.cirrhosis_data = pd.concat([self.cirrhosis_data, dummies], axis=1)
            self.cirrhosis_data.drop(col, axis=1, inplace=True)
            
        print("分类特征编码完成！")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def scale_features(self):
        """标准化特征"""
        # 中风数据集
        print("\n标准化中风数据集特征...")
        stroke_numeric = self.stroke_data.select_dtypes(include=['float64', 'int64']).columns
        stroke_numeric = [col for col in stroke_numeric if col not in ['id', 'stroke']]
        
        scaler = StandardScaler()
        self.stroke_data[stroke_numeric] = scaler.fit_transform(self.stroke_data[stroke_numeric])
        self.scalers['stroke'] = scaler
        
        # 心脏病数据集
        print("标准化心脏病数据集特征...")
        heart_numeric = self.heart_data.select_dtypes(include=['float64', 'int64']).columns
        heart_numeric = [col for col in heart_numeric if col != 'HeartDisease']
        
        scaler = StandardScaler()
        self.heart_data[heart_numeric] = scaler.fit_transform(self.heart_data[heart_numeric])
        self.scalers['heart'] = scaler
        
        # 肝硬化数据集
        print("标准化肝硬化数据集特征...")
        cirrhosis_numeric = self.cirrhosis_data.select_dtypes(include=['float64', 'int64']).columns
        cirrhosis_numeric = [col for col in cirrhosis_numeric if col not in ['ID', 'N_Days', 'Stage']]
        
        scaler = StandardScaler()
        self.cirrhosis_data[cirrhosis_numeric] = scaler.fit_transform(self.cirrhosis_data[cirrhosis_numeric])
        self.scalers['cirrhosis'] = scaler
        
        print("特征标准化完成！")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def feature_engineering(self):
        """特征工程，创建新的特征"""
        # 中风数据集特征工程
        print("\n对中风数据集进行特征工程...")
        
        # 年龄分组
        self.stroke_data['age_group'] = pd.cut(
            self.stroke_data['age'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # BMI分类
        def bmi_category(bmi):
            if bmi < 18.5:
                return 0  # 偏瘦
            elif bmi < 24:
                return 1  # 正常
            elif bmi < 28:
                return 2  # 超重
            else:
                return 3  # 肥胖
                
        self.stroke_data['bmi_category'] = self.stroke_data['bmi'].apply(bmi_category)
        
        # 高血糖风险
        self.stroke_data['glucose_risk'] = (self.stroke_data['avg_glucose_level'] > 140).astype(int)
        
        # 多重风险因素（高血压、心脏病、高血糖）
        self.stroke_data['multiple_risks'] = (
            self.stroke_data['hypertension'] + 
            self.stroke_data['heart_disease'] + 
            self.stroke_data['glucose_risk']
        )
        
        # 心脏病数据集特征工程
        print("对心脏病数据集进行特征工程...")
        
        # 年龄分组
        self.heart_data['age_group'] = pd.cut(
            self.heart_data['Age'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # 胆固醇分类
        def chol_category(chol):
            if chol < 200:
                return 0  # 正常
            elif chol < 240:
                return 1  # 边缘高
            else:
                return 2  # 高
                
        self.heart_data['chol_category'] = self.heart_data['Cholesterol'].apply(chol_category)
        
        # 高血压风险
        self.heart_data['bp_risk'] = (self.heart_data['RestingBP'] > 140).astype(int)
        
        # 心率预警（最大心率偏低）
        self.heart_data['hr_warning'] = (self.heart_data['MaxHR'] < 100).astype(int)
        
        # 肝硬化数据集特征工程
        print("对肝硬化数据集进行特征工程...")
        
        # 年龄转换为年（原始数据是天数）
        self.cirrhosis_data['Age_years'] = self.cirrhosis_data['Age'] / 365.25
        
        # 年龄分组
        self.cirrhosis_data['age_group'] = pd.cut(
            self.cirrhosis_data['Age_years'], 
            bins=[0, 18, 35, 50, 65, 100], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # 胆红素风险
        self.cirrhosis_data['bilirubin_risk'] = (self.cirrhosis_data['Bilirubin'] > 1.2).astype(int)
        
        # 胆固醇风险
        self.cirrhosis_data['cholesterol_risk'] = (self.cirrhosis_data['Cholesterol'] > 240).astype(int)
        
        # 肝功能综合评分（基于关键指标）
        self.cirrhosis_data['liver_score'] = (
            (self.cirrhosis_data['Bilirubin'] > 1.2).astype(int) +
            (self.cirrhosis_data['Albumin'] < 3.5).astype(int) +
            (self.cirrhosis_data['Prothrombin'] > 12).astype(int)
        )
        
        print("特征工程完成！")
        
        return self.stroke_data, self.heart_data, self.cirrhosis_data
    
    def prepare_train_test_data(self):
        """准备训练和测试数据"""
        # 中风数据集
        stroke_features = self.stroke_data.drop(['id', 'stroke'], axis=1, errors='ignore')
        stroke_target = self.stroke_data['stroke']
        
        # 心脏病数据集
        heart_features = self.heart_data.drop(['HeartDisease'], axis=1, errors='ignore')
        heart_target = self.heart_data['HeartDisease']
        
        # 肝硬化数据集 (使用Stage作为目标变量)
        cirrhosis_features = self.cirrhosis_data.drop(['ID', 'N_Days', 'Stage'], axis=1, errors='ignore')
        cirrhosis_target = self.cirrhosis_data['Stage']
        
        # 保存处理后的数据
        self.stroke_data.to_csv('output/processed_data/stroke_processed.csv', index=False)
        self.heart_data.to_csv('output/processed_data/heart_processed.csv', index=False)
        self.cirrhosis_data.to_csv('output/processed_data/cirrhosis_processed.csv', index=False)
        
        print("训练和测试数据准备完成！")
        
        return {
            'stroke': (stroke_features, stroke_target),
            'heart': (heart_features, heart_target),
            'cirrhosis': (cirrhosis_features, cirrhosis_target)
        }
        
    def run_full_preprocessing(self):
        """运行完整的预处理流程"""
        self.load_data()
        self.explore_data()
        self.handle_missing_values(method='iterative')
        self.detect_outliers()
        self.handle_outliers(method='cap')
        self.encode_categorical_features()
        self.feature_engineering()
        self.scale_features()
        return self.prepare_train_test_data()

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.run_full_preprocessing()
    print("预处理完成，数据已保存到 output/processed_data/ 目录") 