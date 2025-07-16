import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 使用try-except块导入可选依赖
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("警告: Plotly包不可用，将跳过交互式可视化功能")
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("警告: SHAP包不可用，将跳过SHAP值可视化")
    SHAP_AVAILABLE = False

class DataVisualizer:
    def __init__(self, stroke_data=None, heart_data=None, cirrhosis_data=None):
        """初始化数据可视化类
        
        参数:
            stroke_data: 中风数据集
            heart_data: 心脏病数据集
            cirrhosis_data: 肝硬化数据集
        """
        self.stroke_data = stroke_data
        self.heart_data = heart_data
        self.cirrhosis_data = cirrhosis_data
        
        # 创建输出目录
        os.makedirs('output/figures', exist_ok=True)
        
        # 设置可视化样式
        sns.set(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    def load_processed_data(self, stroke_path='output/processed_data/stroke_processed.csv',
                           heart_path='output/processed_data/heart_processed.csv',
                           cirrhosis_path='output/processed_data/cirrhosis_processed.csv'):
        """加载处理后的数据"""
        self.stroke_data = pd.read_csv(stroke_path)
        self.heart_data = pd.read_csv(heart_path)
        self.cirrhosis_data = pd.read_csv(cirrhosis_path)
        print("处理后的数据加载完成！")
        
    def plot_feature_distributions(self, save_plots=True):
        """绘制特征分布"""
        datasets = {
            'stroke': (self.stroke_data, 'stroke'),
            'heart': (self.heart_data, 'HeartDisease'),
            'cirrhosis': (self.cirrhosis_data, 'Stage')
        }
        
        for name, (data, target) in datasets.items():
            print(f"绘制{name}数据集特征分布...")
            
            # 分类特征可视化
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            categorical_cols = [col for col in categorical_cols if col.lower() not in ['id']]
            
            if len(categorical_cols) > 0:
                # 计算每个分类特征需要的图表数量
                n_cat_cols = len(categorical_cols)
                n_rows = (n_cat_cols + 2) // 3  # 每行最多3个图
                
                # 创建画布
                plt.figure(figsize=(18, n_rows * 5))
                
                # 绘制每个分类特征的计数图
                for i, col in enumerate(categorical_cols):
                    plt.subplot(n_rows, 3, i+1)
                    
                    # 检查目标变量是否在数据中
                    if target in data.columns:
                        # 按目标变量分组的计数图
                        sns.countplot(x=col, hue=target, data=data)
                        plt.title(f'{name.capitalize()} - {col} 分布 (按 {target} 分组)')
                    else:
                        # 普通计数图
                        sns.countplot(x=col, data=data)
                        plt.title(f'{name.capitalize()} - {col} 分布')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_categorical_distributions.png')
                plt.close()
            
            # 数值特征可视化
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col.lower() not in ['id'] and col != target]
            
            if len(numeric_cols) > 0:
                # 创建画布
                n_num_cols = len(numeric_cols)
                n_rows = (n_num_cols + 2) // 3  # 每行最多3个图
                
                plt.figure(figsize=(18, n_rows * 5))
                
                # 绘制每个数值特征的分布
                for i, col in enumerate(numeric_cols):
                    plt.subplot(n_rows, 3, i+1)
                    
                    # 绘制直方图和密度图
                    if target in data.columns and len(data[target].unique()) <= 5:
                        # 按目标变量分组的分布图
                        for cls in sorted(data[target].unique()):
                            sns.histplot(data[data[target]==cls][col], kde=True, 
                                         label=f'{target}={cls}', alpha=0.5)
                        plt.legend()
                        plt.title(f'{name.capitalize()} - {col} 分布 (按 {target} 分组)')
                    else:
                        # 普通分布图
                        sns.histplot(data[col], kde=True)
                        plt.title(f'{name.capitalize()} - {col} 分布')
                    
                    plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_numeric_distributions.png')
                plt.close()
            
            # 绘制目标变量分布
            if target in data.columns:
                plt.figure(figsize=(8, 6))
                target_counts = data[target].value_counts()
                
                # 计算百分比
                target_percents = target_counts / target_counts.sum() * 100
                
                # 绘制饼图
                plt.pie(target_counts, labels=[f'{i} ({p:.1f}%)' for i, p in zip(target_counts.index, target_percents)],
                        autopct='%1.1f%%', startangle=90, shadow=True)
                plt.title(f'{name.capitalize()} - {target} 分布')
                plt.axis('equal')
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_target_distribution.png')
                plt.close()
                
        print("特征分布可视化完成！")
    
    def plot_correlation_matrices(self, save_plots=True):
        """绘制特征相关性矩阵"""
        datasets = {
            'stroke': self.stroke_data,
            'heart': self.heart_data,
            'cirrhosis': self.cirrhosis_data
        }
        
        for name, data in datasets.items():
            print(f"绘制{name}数据集相关性矩阵...")
            
            # 选择数值特征
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            
            # 如果特征太多，只保留部分重要特征
            if numeric_data.shape[1] > 15:
                # 可以根据领域知识选择重要特征
                if name == 'stroke':
                    important_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 
                                     'bmi', 'stroke', 'multiple_risks', 'glucose_risk']
                    numeric_data = numeric_data[important_cols]
                elif name == 'heart':
                    important_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 
                                     'MaxHR', 'Oldpeak', 'HeartDisease', 'bp_risk']
                    numeric_data = numeric_data[important_cols]
                elif name == 'cirrhosis':
                    important_cols = ['Age_years', 'Bilirubin', 'Albumin', 'Copper', 
                                     'Prothrombin', 'Stage', 'liver_score', 'bilirubin_risk']
                    numeric_data = numeric_data[important_cols]
            
            # 计算相关性矩阵
            corr_matrix = numeric_data.corr()
            
            # 绘制热图
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                        linewidths=0.5, vmin=-1, vmax=1)
            plt.title(f'{name.capitalize()} 数据集 - 特征相关性矩阵')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(f'output/figures/{name}_correlation_matrix.png')
            plt.close()
            
            # 如果有Plotly可用，则创建交互式热图
            if PLOTLY_AVAILABLE and save_plots:
                try:
                    fig = px.imshow(corr_matrix,
                                 x=corr_matrix.columns,
                                 y=corr_matrix.columns,
                                 color_continuous_scale='RdBu_r',
                                 title=f'{name.capitalize()} 数据集 - 特征相关性矩阵（交互式）')
                    
                    fig.write_html(f'output/figures/{name}_correlation_matrix_interactive.html')
                except Exception as e:
                    print(f"创建交互式热图时出错: {e}")
        
        print("相关性矩阵可视化完成！")
    
    def plot_feature_importance(self, save_plots=True):
        """绘制特征重要性"""
        datasets = {
            'stroke': (self.stroke_data, 'stroke'),
            'heart': (self.heart_data, 'HeartDisease'),
            'cirrhosis': (self.cirrhosis_data, 'Stage')
        }
        
        for name, (data, target) in datasets.items():
            print(f"计算{name}数据集特征重要性...")
            
            if target in data.columns:
                # 准备数据
                X = data.drop([target], axis=1)
                X = X.select_dtypes(include=['float64', 'int64'])  # 只保留数值特征
                
                # 删除ID列
                id_cols = [col for col in X.columns if col.lower() in ['id', 'n_days']]
                X = X.drop(id_cols, axis=1, errors='ignore')
                
                if len(X.columns) == 0:
                    print(f"{name}数据集没有合适的特征用于计算重要性")
                    continue
                    
                y = data[target]
                
                # 根据数据集名称和目标列确定任务类型
                is_classification = True
                if name == 'cirrhosis' and target == 'Stage':
                    is_classification = False
                elif len(y.unique()) > 10:  # 如果目标变量有很多不同值，可能是回归任务
                    is_classification = False
                
                # 处理分类目标变量
                if is_classification and (y.dtype == 'object' or y.dtype == 'category'):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # 训练模型
                if is_classification:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # 拟合模型
                model.fit(X, y)
                
                # 获取特征重要性
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # 绘制特征重要性条形图
                plt.figure(figsize=(12, 8))
                plt.title(f'{name.capitalize()} 数据集 - 特征重要性')
                plt.bar(range(X.shape[1]), importances[indices], align='center')
                plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_feature_importance.png')
                plt.close()
                
                # 使用SHAP值解释模型
                if SHAP_AVAILABLE:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X)
                        
                        # 绘制摘要图
                        plt.figure(figsize=(10, 8))
                        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                        plt.title(f'{name.capitalize()} 数据集 - SHAP值特征重要性')
                        plt.tight_layout()
                        
                        if save_plots:
                            plt.savefig(f'output/figures/{name}_shap_importance.png')
                        plt.close()
                        
                        # 绘制详细的SHAP值图
                        plt.figure(figsize=(12, 10))
                        shap.summary_plot(shap_values, X, show=False)
                        plt.title(f'{name.capitalize()} 数据集 - SHAP值特征影响')
                        plt.tight_layout()
                        
                        if save_plots:
                            plt.savefig(f'output/figures/{name}_shap_summary.png')
                        plt.close()
                        
                    except Exception as e:
                        print(f"SHAP值计算失败: {e}")
                else:
                    print("SHAP不可用，跳过SHAP值可视化")
            
            else:
                print(f"{name}数据集中没有找到目标变量 {target}")
        
        print("特征重要性可视化完成！")
    
    def plot_pair_plots(self, save_plots=True):
        """绘制特征对图，分析特征间的关系"""
        datasets = {
            'stroke': (self.stroke_data, 'stroke'),
            'heart': (self.heart_data, 'HeartDisease'),
            'cirrhosis': (self.cirrhosis_data, 'Stage')
        }
        
        for name, (data, target) in datasets.items():
            print(f"绘制{name}数据集特征对图...")
            
            if target in data.columns:
                # 选择最重要的特征
                if name == 'stroke':
                    selected_features = ['age', 'avg_glucose_level', 'bmi', 'stroke']
                elif name == 'heart':
                    selected_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'HeartDisease']
                elif name == 'cirrhosis':
                    selected_features = ['Age_years', 'Bilirubin', 'Albumin', 'Prothrombin', 'Stage']
                
                # 过滤数据
                plot_data = data[selected_features].copy()
                
                # 绘制对图
                plt.figure(figsize=(12, 10))
                sns.pairplot(plot_data, hue=target, diag_kind='kde')
                plt.suptitle(f'{name.capitalize()} 数据集 - 特征对图', y=1.02)
                
                if save_plots:
                    plt.savefig(f'output/figures/{name}_pair_plot.png')
                plt.close()
            
            else:
                print(f"{name}数据集中没有找到目标变量 {target}")
        
        print("特征对图可视化完成！")
    
    def plot_disease_comparison(self, save_plots=True):
        """比较三种疾病的共同特征"""
        print("比较三种疾病的共同特征...")
        
        if self.stroke_data is None or self.heart_data is None or self.cirrhosis_data is None:
            print("警告：数据未加载，请先调用load_processed_data方法")
            return
        
        # 提取共同特征
        # 年龄是三个数据集的共同特征
        stroke_age = self.stroke_data[['age', 'stroke']].copy()
        stroke_age['disease'] = 'Stroke'
        stroke_age.rename(columns={'stroke': 'has_disease', 'age': 'Age'}, inplace=True)
        
        heart_age = self.heart_data[['Age', 'HeartDisease']].copy()
        heart_age['disease'] = 'Heart Disease'
        heart_age.rename(columns={'HeartDisease': 'has_disease'}, inplace=True)
        
        cirrhosis_age = self.cirrhosis_data[['Age_years', 'Stage']].copy()
        cirrhosis_age['disease'] = 'Cirrhosis'
        cirrhosis_age['has_disease'] = (cirrhosis_age['Stage'] > 2).astype(int)  # 将3,4期视为重度疾病
        cirrhosis_age.rename(columns={'Age_years': 'Age'}, inplace=True)
        cirrhosis_age.drop('Stage', axis=1, inplace=True)
        
        # 合并数据
        combined_age = pd.concat([stroke_age, heart_age, cirrhosis_age], ignore_index=True)
        
        # 绘制年龄分布对比
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='disease', y='Age', hue='has_disease', 
                      data=combined_age, split=True, inner="quart")
        plt.title('三种疾病的年龄分布对比')
        plt.xlabel('疾病类型')
        plt.ylabel('年龄')
        plt.legend(title='是否患病', loc='best')
        
        if save_plots:
            plt.savefig('output/figures/disease_age_comparison.png')
        plt.close()
        
        # 绘制年龄段患病率
        plt.figure(figsize=(14, 8))
        
        # 加载原始数据获取年龄的均值和标准差
        try:
            # 加载原始数据
            stroke_orig = pd.read_csv('stroke.csv')
            heart_orig = pd.read_csv('heart.csv')
            cirrhosis_orig = pd.read_csv('cirrhosis.csv')
            
            # 计算原始数据的统计值
            stroke_mean = stroke_orig['age'].mean()
            stroke_std = stroke_orig['age'].std()
            
            heart_mean = heart_orig['Age'].mean()
            heart_std = heart_orig['Age'].std()
            
            cirrhosis_age_years = cirrhosis_orig['Age'] / 365.25  # 转换为年
            cirrhosis_mean = cirrhosis_age_years.mean()
            cirrhosis_std = cirrhosis_age_years.std()
            
            print(f"原始数据年龄统计:")
            print(f"中风数据: 均值 = {stroke_mean:.2f}, 标准差 = {stroke_std:.2f}")
            print(f"心脏病数据: 均值 = {heart_mean:.2f}, 标准差 = {heart_std:.2f}")
            print(f"肝硬化数据: 均值 = {cirrhosis_mean:.2f}, 标准差 = {cirrhosis_std:.2f}")
            
            # 反转标准化 - 根据疾病类型应用不同的均值和标准差
            combined_age['Age_Original'] = combined_age.apply(
                lambda row: row['Age'] * stroke_std + stroke_mean if row['disease'] == 'Stroke' else
                           (row['Age'] * heart_std + heart_mean if row['disease'] == 'Heart Disease' else
                            row['Age'] * cirrhosis_std + cirrhosis_mean),
                axis=1
            )
            
            # 打印处理前后的年龄分布
            print("\n年龄分布对比:")
            print("标准化年龄范围:", combined_age['Age'].min(), "到", combined_age['Age'].max())
            print("反标准化年龄范围:", combined_age['Age_Original'].min(), "到", combined_age['Age_Original'].max())
            
        except Exception as e:
            print(f"无法加载原始数据进行反标准化: {e}")
            # 如果无法加载原始数据，使用估计的均值和标准差
            combined_age['Age_Original'] = combined_age['Age'] * 20 + 50  # 粗略估计
        
        # 定义年龄段 - 使用反标准化的年龄
        age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']
        
        # 使用反标准化后的年龄值进行分组
        combined_age['age_group'] = pd.cut(combined_age['Age_Original'], bins=age_bins, labels=age_labels)
        
        # 检查年龄组分布
        print("\n反标准化后年龄组分布:")
        print(combined_age['age_group'].value_counts().sort_index())
        
        # 计算各年龄段各疾病的患病率
        disease_rates = combined_age.groupby(['disease', 'age_group'])['has_disease'].mean().reset_index()
        disease_rates['percentage'] = disease_rates['has_disease'] * 100
        
        # 打印结果
        print("\n各年龄段疾病患病率:")
        print(disease_rates)
        
        # 绘制分组柱状图
        plt.figure(figsize=(16, 8))
        sns.barplot(x='age_group', y='percentage', hue='disease', data=disease_rates)
        plt.title('各年龄段三种疾病患病率对比')
        plt.xlabel('年龄段')
        plt.ylabel('患病率(%)')
        plt.legend(title='疾病类型')
        plt.xticks(rotation=45)
        
        if save_plots:
            plt.savefig('output/figures/disease_rate_by_age.png')
        plt.close()
        
        # 使用Plotly创建交互式图表 (如果可用)
        if PLOTLY_AVAILABLE and save_plots:
            try:
                fig = px.line(disease_rates, x='age_group', y='percentage', color='disease',
                            title='各年龄段三种疾病患病率对比（交互式）',
                            labels={'age_group': '年龄段', 'percentage': '患病率(%)', 'disease': '疾病类型'},
                            markers=True)
                
                fig.write_html('output/figures/disease_rate_by_age_interactive.html')
            except Exception as e:
                print(f"创建交互式年龄疾病图表时出错: {e}")
            
        # 性别对比（中风和心脏病数据集都有性别特征）
        # 首先检查数据集中是否存在性别特征
        if 'gender' in self.stroke_data.columns and 'Sex' in self.heart_data.columns:
            print("处理性别数据中...")
            
            try:
                # 提取中风数据的性别和患病情况
                stroke_gender = self.stroke_data[['gender', 'stroke']].copy()
                stroke_gender['disease'] = 'Stroke'
                stroke_gender.rename(columns={'stroke': 'has_disease', 'gender': 'Sex'}, inplace=True)
                
                # 提取心脏病数据的性别和患病情况
                heart_gender = self.heart_data[['Sex', 'HeartDisease']].copy()
                heart_gender['disease'] = 'Heart Disease'
                heart_gender.rename(columns={'HeartDisease': 'has_disease'}, inplace=True)
                
                # 打印调试信息
                print("中风数据性别唯一值:", stroke_gender['Sex'].unique())
                print("心脏病数据性别唯一值:", heart_gender['Sex'].unique())
                print("中风数据类型:", stroke_gender['Sex'].dtype)
                print("心脏病数据类型:", heart_gender['Sex'].dtype)
                
                # 统一性别表示 - 使用简单方法：将所有小于0的值归为0(女性)，大于0的值归为1(男性)
                # 这是因为标准化后的值可能不再是0和1，但仍保留正负关系
                
                # 处理中风数据 - 保留原始的数值关系
                stroke_gender['Sex_Numeric'] = np.where(stroke_gender['Sex'] < 0, 0, 1)
                
                # 处理心脏病数据 - 保留原始的数值关系
                heart_gender['Sex_Numeric'] = np.where(heart_gender['Sex'] < 0, 0, 1)
                
                # 将两个数据集合并
                combined_gender = pd.concat([stroke_gender, heart_gender], ignore_index=True)
                
                # 映射为可显示的文本
                combined_gender['Sex_Display'] = combined_gender['Sex_Numeric'].map({0: 'Female', 1: 'Male'})
                
                # 打印调试信息
                print("合并后性别唯一值:", combined_gender['Sex_Numeric'].unique())
                print("映射后显示值:", combined_gender['Sex_Display'].unique())
                print("数据记录总数:", len(combined_gender))
                
                # 计算各性别各疾病的患病率
                gender_rates = combined_gender.groupby(['disease', 'Sex_Display'])['has_disease'].mean().reset_index()
                gender_rates['percentage'] = gender_rates['has_disease'] * 100
                
                # 打印患病率
                print("性别疾病患病率:")
                print(gender_rates)
                
                # 绘制分组柱状图
                plt.figure(figsize=(10, 6))
                
                # 确保数据存在再绘图
                if not gender_rates.empty and len(gender_rates) >= 2:
                    sns.barplot(x='Sex_Display', y='percentage', hue='disease', data=gender_rates)
                    plt.title('性别对中风和心脏病患病率的影响')
                    plt.xlabel('性别')
                    plt.ylabel('患病率(%)')
                    plt.legend(title='疾病类型')
                    
                    if save_plots:
                        plt.savefig('output/figures/gender_disease_comparison.png')
                    plt.close()
                else:
                    print("警告：性别患病率数据不足，无法绘制图表")
                
            except Exception as e:
                print(f"处理性别数据时出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("数据集中缺少性别特征，无法进行性别比较")
        
        print("疾病比较可视化完成！")
    
    def run_all_visualizations(self):
        """运行所有可视化函数"""
        self.plot_feature_distributions()
        self.plot_correlation_matrices()
        self.plot_feature_importance()
        self.plot_pair_plots()
        self.plot_disease_comparison()
        print("所有可视化任务完成！")

if __name__ == "__main__":
    # 创建可视化器并加载数据
    visualizer = DataVisualizer()
    visualizer.load_processed_data()
    
    # 运行所有可视化
    visualizer.run_all_visualizations() 