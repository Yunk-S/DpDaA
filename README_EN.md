<div align="center">

**Language Version / 语言版本**

**English** | [中文](./README.md)

</div>

---

# Disease Prediction and Big Data Analysis System

A machine learning and deep learning-based system for predicting stroke, heart disease, and cirrhosis, providing data analysis, risk prediction, and health management recommendations.

## Project Structure

```
├── app.py                        # Web application main program
├── main.py                       # System entry program
├── data_preprocessing.py         # Data preprocessing module
├── model_training.py             # Model training module
├── model_calibration.py          # Model calibration module
├── model_utilities.py            # Model utilities (metrics generation, chart generation)
├── train_deep_learning_models.py # Deep learning model training module
├── visualization.py              # Data visualization module
├── chart_generator.py            # Chart generation tool
├── generate_metrics.py           # Model metrics generation tool
├── multi_disease_model.py        # Multi-disease hybrid prediction model
├── predict.py                    # Command-line model prediction module
├── predict_cmd.bat               # Interactive prediction batch file
├── run.bat                       # System startup batch file
├── static/                       # Static resources directory
│   ├── style.css                 # Stylesheet
│   ├── script.js                 # JavaScript scripts
│   └── images/                   # Image resources
├── templates/                    # HTML templates
│   ├── index.html                # Homepage
│   ├── data_analysis.html        # Data analysis page
│   ├── predict.html              # Prediction page
│   ├── multi_disease.html        # Multi-disease correlation analysis page
│   └── ...                       # Other pages
├── output/                       # Output directory
│   ├── models/                   # Saved trained models
│   │   ├── calibrators/          # Saved calibrator models
│   ├── figures/                  # Generated charts
│   └── processed_data/           # Processed data
├── requirements.txt              # Project dependency packages list
└── *.csv                         # Raw datasets (stroke.csv, heart.csv, cirrhosis.csv)
```

## Installation and Running

### Environment Requirements

- Python 3.8+
- Required dependency packages (see requirements.txt)

### Installation Steps

1. Create and activate virtual environment
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependency packages
   ```
   pip install -r requirements.txt
   ```

   If installation is slow in mainland China, you can use Tsinghua mirror source:
   ```
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
   ```

### Running the System

#### Method 1: Using Batch File (Windows)

Run the `run.bat` batch file directly and follow the prompts:
```
run.bat
```

The system will prompt in the following order:
1. Whether to create/use virtual environment (recommended to choose Y)
   - If you choose Y and no virtual environment exists, it will create a new virtual environment and activate it automatically
   - If you choose Y and a virtual environment already exists, it will directly activate the existing virtual environment
   - If you choose N, it will use the system Python environment
2. Whether to install dependency packages
   - First-time users need to install dependencies
   - Can choose to use default installation source or Tsinghua mirror source (recommended for mainland China)
3. Select function mode:
   1. Run complete analysis and prediction workflow
   2. Run data preprocessing only
   3. Run data visualization only
   4. Run model training only
   5. Run chart and metrics generation only
   6. Run web application only
   7. Execute main.py directly (all steps)
   8. Exit

**Note:** 
- Options 1 and 7 have the same functionality, both execute the complete workflow. Choose either one.
- Using a virtual environment can avoid dependency conflicts, especially recommended for new users.
- After the program ends, if a virtual environment was used, you need to manually input the `deactivate` command to exit the virtual environment.

#### Method 2: Using Python Commands

Run the complete workflow (data preprocessing, visualization, model training, chart metrics generation, and web application):
```
python main.py --all
```

Or use different parameters to run specific functions separately:
- `--preprocess`: Run data preprocessing only
- `--visualize`: Run data visualization only
- `--train`: Run model training only
- `--generate`: Run chart and metrics generation only
- `--webapp`: Run web application only

Example: Start web application only
```
python main.py --webapp
```

#### Method 3: Using Command-line Prediction Tool Note!! Use this for prediction first!!!

1. Use command-line tool for direct prediction:
```
python predict.py [disease_type] [param1=value1] [param2=value2] ...
```

For example, predict stroke risk:
```
python predict.py stroke age=60 gender=Male hypertension=1 heart_disease=0 avg_glucose_level=120 bmi=28 smoking_status="formerly smoked"
```

2. Or use interactive batch file for prediction:
```
predict_cmd.bat
```
Follow prompts to select disease type and input relevant health indicators.

**Parameter Description:**
- `disease_type`: Disease type, available values are stroke, heart, cirrhosis, and multi
- Subsequent parameters are health indicators, format is parameter_name=parameter_value

#### Interactive Prediction Tool Usage

`predict_cmd.bat` is an interactive command-line prediction tool that provides an English interface for disease risk prediction. Usage steps:

1. Run `predict_cmd.bat` file
2. Select the disease type to predict according to prompts (number between 1-4)
   - 1: Stroke prediction
   - 2: Heart disease prediction
   - 3: Cirrhosis prediction
   - 4: Multi-disease prediction
3. Input health indicator data step by step according to prompts
4. The system will display prediction results, including risk probability and level

This tool is particularly suitable when the web application cannot run normally, or when you want to quickly get prediction results without starting the complete web interface.

## Web Application Usage

1. After running the web application, the system will automatically open the homepage in the browser (http://localhost:5000)
2. The homepage provides system introduction and basic information about three diseases
3. The navigation bar provides the following function entries:
   - **Data Analysis**: View distribution, correlation, and feature importance of three disease datasets
   - **Model Performance**: View performance metrics and evaluation results of prediction models
   - **Disease Prediction**: Input personal health indicators to get disease risk predictions and health recommendations (frontend-backend connection bug, please use predict_cmd.bat for prediction)
   - **Multi-disease Correlation**: View correlation analysis and comprehensive risk assessment between three diseases

### Single Disease Prediction Function
1. Click the "Disease Prediction" dropdown menu in the top navigation bar and select "Single Disease Prediction"
2. On the prediction page, you can choose one of three diseases (stroke, heart disease, or cirrhosis) for risk prediction
3. Fill in the corresponding health indicator form and click the "Predict Risk" button
4. The system will display prediction results, including risk probability, risk level, and health recommendations

### Multi-disease Joint Prediction Function
1. Click the "Disease Prediction" dropdown menu in the top navigation bar and select "Multi-disease Joint Prediction"
2. Fill in the comprehensive health indicator form, including basic information, lifestyle, and clinical indicators
3. Click the "Comprehensive Prediction" button
4. The system will display individual risks for three diseases and joint risk probabilities for multiple disease combinations

## Project Overview

This project aims to build a comprehensive disease prediction and analysis system. Through in-depth analysis of data for three common major diseases (stroke, heart disease, and cirrhosis), it establishes prediction models and provides visualization interfaces to display results and health management recommendations. The system uses various advanced machine learning and deep learning algorithms, combined with medical domain knowledge, to achieve high-precision disease risk prediction.

### Initial Concept

The project design is based on the following points:
1. **Multi-disease Integrated Analysis**: Unlike traditional single-disease prediction systems, this project simultaneously handles three highly correlated major diseases, exploring correlations and common risk factors between them
2. **Data-driven Decision Making**: Through analysis of large amounts of real medical data, extract valuable features and patterns to assist medical decision-making
3. **Model Integration and Optimization**: Combine advantages of multiple machine learning algorithms, using ensemble learning and model fusion techniques to improve prediction accuracy
4. **Interpretability Analysis**: Use SHAP values and feature importance analysis to provide interpretability of model prediction results
5. **User-friendly Interface**: Design intuitive web interface so that both medical professionals and ordinary users can use the system conveniently

## Features

- **Multi-disease Prediction**: Risk prediction for three major diseases: stroke, heart disease, and cirrhosis
- **Data Visualization**: Intuitive display of data distribution, feature correlation, and prediction results
- **Risk Assessment**: Provide personalized health risk scores and prevention recommendations
- **Interactive Interface**: User-friendly web interface supporting real-time prediction and result display
- **Multi-disease Correlation Analysis**: Analyze correlations and comorbidity risks between different diseases
- **Age and Disease Relationship Analysis**: Show disease incidence trends across different age groups
- **Error Recovery Mechanism**: When model loading fails, the system automatically uses backup models or simulated data for prediction, ensuring user experience is not affected

## Technical Architecture

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning Models**:
  - Traditional machine learning algorithms (Random Forest, XGBoost, LightGBM, CatBoost, SVM, etc.)
  - Deep learning models (DNN, attention mechanism networks)
  - Ensemble learning (voting, stacking, hybrid fusion)
- **Model Interpretability**: SHAP value analysis

## Multi-disease Hybrid Prediction Model Details

### Model Architecture

The multi-disease hybrid prediction model (`MultiDiseasePredictor` class) is one of the core components of the system, implementing the following functions:

1. **Model Loading and Fallback Mechanism**:
   - Attempts to load trained single-disease prediction models
   - Automatically creates backup models (RandomForest) when model loading fails
   - Provides simulated data prediction functionality, ensuring the system can run normally under any circumstances

2. **Single Disease Risk Prediction**:
   - Stroke prediction: Predicts stroke risk based on age, gender, hypertension, and other features
   - Heart disease prediction: Predicts heart disease risk based on chest pain type, blood pressure, cholesterol, and other features
   - Cirrhosis prediction: Predicts cirrhosis risk or staging based on biochemical indicators

3. **Multi-disease Joint Prediction**:
   - Independent prediction: Calculates the probability of each disease occurring independently
   - Joint probability: Calculates joint probabilities of multiple disease combinations
   - Considers correlations: Adjusts joint probabilities through correlation coefficients, more consistent with actual medical situations

### Mathematical Principles

1. **Multi-task Learning Framework**:
   The system adopts a multi-task learning framework that simultaneously learns multiple related tasks (disease prediction) by sharing underlying feature representations.

   Mathematical representation:
   
   Shared feature extraction layer:
   ```math
   \mathbf{h} = f_{\text{shared}}(\mathbf{x}; \theta_s)
   ```
   
   Task-specific output layer:
   ```math
   \hat{y}^{(k)} = f_{\text{task}_k}(\mathbf{h}; \theta_k)
   ```

2. **Joint Probability Calculation**:
   For independent probabilities of diseases A, B, C as $P(A)$, $P(B)$, $P(C)$ respectively, the system calculates the following joint probabilities:
   
   - Single disease:
   ```math
   P(A)(1-P(B))(1-P(C))
   ```
   
   - Two diseases:
   ```math
   P(A)P(B)(1-P(C))
   ```
   
   - Three diseases:
   ```math
   P(A)P(B)P(C)
   ```
   
   - No disease:
   ```math
   (1-P(A))(1-P(B))(1-P(C))
   ```

3. **Correlation Adjustment**:
   The system uses correlation coefficients to adjust joint probabilities, calculating correlation-adjusted joint probabilities:
   
   ```math
   P(A,B) = P(A)P(B) + \text{corr}_{AB} \cdot \min(P(A), P(B)) \cdot (1 - \max(P(A), P(B)))
   ```
   
   Where $\text{corr}_{AB}$ is the correlation coefficient between diseases A and B.

4. **Attention Mechanism**:
   Attention mechanisms are used in deep learning models to enhance the model's focus on important features:
   
   ```math
   \alpha_i = \text{Attention}(h_i)
   ```
   
   ```math
   h'_i = h_i \cdot \alpha_i
   ```
   
   Where $\alpha_i$ is the attention weight and $h_i$ is the feature representation.

### Model Optimization Techniques

1. **Knowledge Distillation**:
   - Teacher model: Complex deep neural networks or ensemble models
   - Student model: Lightweight models that improve performance by learning "soft labels" from teacher models
   - Distillation loss:
   ```math
   \mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \hat{y}_{\text{student}}) + (1-\alpha) \cdot \mathcal{L}_{\text{KL}}(\hat{y}_{\text{teacher}}, \hat{y}_{\text{student}})
   ```

2. **Ensemble Learning**:
   - Voting ensemble: Multiple base models obtain final predictions through voting or averaging
   - Stacking ensemble: Use meta-learners to integrate prediction results from base models
   - Hybrid fusion: Combine advantages of traditional machine learning and deep learning models

3. **Collaborative Regularization**:
   Encourage parameters of different tasks to share information, prompting the model to learn comorbidity patterns between diseases:
   
   ```math
   \mathcal{L}_{\text{reg}} = \lambda \cdot \sum_{i,j} \| \theta_i - \theta_j \|_2^2
   ```

## Mathematical Knowledge and Model Techniques Used

### Data Preprocessing Techniques
- **Missing Value Handling**: Iterative Imputer, KNN Imputer, mean/median filling
- **Outlier Detection**: IQR method, Z-score method, Isolation Forest
- **Feature Engineering**: Feature selection, feature interaction, polynomial features, risk score calculation
- **Data Standardization**: Z-score standardization, Min-Max scaling
- **Class Imbalance Handling**: SMOTE oversampling, class weight adjustment

### Machine Learning Algorithms
- **Classification Algorithms**: Logistic regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM
- **Regression Algorithms**: Linear regression, Ridge regression, LASSO regression, Elastic Net, SVR, Random Forest regression, Gradient Boosting regression
- **Ensemble Learning**:
  - Voting ensemble: Combines prediction results from multiple base models through voting or averaging
  - Stacking ensemble: Uses meta-learners to integrate prediction results from base models
  - Hybrid fusion: Combines advantages of different types of models, such as fusion of traditional machine learning and deep learning models

### Deep Learning Techniques
- **Deep Neural Networks (DNN)**: Multi-layer perceptron structure for complex feature learning
- **Attention Mechanism**: Introduces attention layers in neural networks to focus on important features
- **Batch Normalization**: Accelerates training process and improves model stability
- **Dropout Regularization**: Prevents overfitting
- **Knowledge Distillation**: Transfers knowledge from complex models to simple models

### Model Evaluation and Optimization
- **Cross Validation**: K-fold cross validation, stratified K-fold cross validation
- **Hyperparameter Optimization**: Grid search, random search
- **Performance Metrics**:
  - Classification tasks: Accuracy, precision, recall, F1-score, AUC-ROC
  - Regression tasks: MSE, RMSE, MAE, R²

### Interpretability Analysis
- **SHAP Value Analysis**: Explains the contribution of each feature in model predictions
- **Feature Importance Analysis**: Identifies features with the greatest impact on prediction results
- **Partial Dependence Plots**: Shows the relationship between specific features and prediction results

## Dataset Description

The project uses the following three datasets:

1. **Stroke Dataset (stroke.csv)**
   - Features: Age, gender, hypertension, heart disease history, marital status, work type, residence type, average glucose level, BMI, smoking status
   - Target variable: Whether stroke occurred (binary classification)
   - Sample size: Approximately 5,110 records
   - Characteristics: Class imbalance, missing values mainly concentrated in BMI feature

2. **Heart Disease Dataset (heart.csv)**
   - Features: Age, gender, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, maximum heart rate, angina, ST depression, ST slope
   - Target variable: Whether has heart disease (binary classification)
   - Sample size: Approximately 918 records
   - Characteristics: Complex feature combinations, multiple types of chest pain and ECG results require specialized encoding

3. **Cirrhosis Dataset (cirrhosis.csv)**
   - Features: Age, gender, bilirubin, cholesterol, albumin, copper, alkaline phosphatase, SGOT, triglycerides, platelets, prothrombin time, etc.
   - Target variable: Cirrhosis stage (stages 1-4, regression task)
   - Sample size: Approximately 418 records
   - Characteristics: Target variable is continuous, requires regression models; contains multiple biochemical indicators

### Multi-disease Correlation Analysis Function

The multi-disease correlation analysis page provides several key sections:
1. **Inter-disease Correlation Overview**: Introduces correlations and common pathological mechanisms between three diseases
2. **Gender and Disease Relationship Comparison**: Shows risk differences between genders for three diseases
3. **Age Distribution Comparison**: Shows age distribution characteristics of patients with three diseases through violin plots
4. **Age and Disease Incidence Relationship**: Shows disease incidence trends across different age groups for three diseases
5. **Comorbidity Risk Factor Analysis**: Shows common risk factors affecting multiple diseases through tables and radar charts
6. **Multi-disease Prediction Model and Comprehensive Risk Assessment**: Introduces multi-objective learning methods and comprehensive risk scoring systems
7. **Prevention Recommendations and Health Management**: Provides comprehensive prevention and health management recommendations for multiple diseases

## System Performance and Limitations

### Performance Metrics
- Stroke prediction: Accuracy approximately 92.3%, AUC approximately 0.89
- Heart disease prediction: Accuracy approximately 88.5%, AUC approximately 0.91
- Cirrhosis staging: R² approximately 0.28, RMSE approximately 0.71

### Limitations
1. Limited dataset size may affect model generalization ability
2. Lack of temporal data prevents disease progression prediction
3. No integration of genomic data limits personalized medicine potential
4. Model interpretability still has room for improvement

## Error Handling and Fallback Mechanisms

The system implements multi-level error handling and fallback mechanisms to ensure normal operation under various circumstances:

1. **Model Loading Failure Handling**:
   - When pre-trained models cannot be loaded, the system automatically creates simple replacement models
   - Uses RandomForestClassifier for classification tasks (stroke, heart disease)
   - Uses RandomForestRegressor for regression tasks (cirrhosis)

2. **Prediction Error Handling**:
   - Catches exceptions during prediction process, returns default probability values
   - Displays simulated data usage tips in frontend, ensuring users understand result reliability

3. **Web Application Error Handling**:
   - Implements global exception handling, catches and logs errors
   - Provides user-friendly error pages instead of directly displaying technical error information

### Model Calibration Function

To address the problem of low prediction probabilities, especially for high-risk populations, the system has added model calibration functionality:

1. **Probability Calibration Techniques**:
   - Platt Scaling (logistic regression calibration): Uses logistic regression to calibrate original prediction probabilities
   - Isotonic Regression: Uses non-parametric methods for probability calibration, more flexible
   - Piecewise spline calibration: Uses different calibration strategies for different probability intervals

2. **Adaptive Calibration Strategy**:
   - Automatically selects the best calibration method based on risk level
   - Uses gentle calibration for low-risk populations
   - Uses more aggressive calibration strategies for high-risk populations

3. **Calibration Evaluation Metrics**:
   - Brier score evaluation: Measures accuracy of probability predictions
   - Log loss evaluation: Measures uncertainty of probability predictions
   - Calibration curve visualization: Intuitively shows effects before and after calibration

The calibrated models significantly improve prediction accuracy, especially for high-risk populations, making prediction results more consistent with clinical reality.

### Running Calibration Function

Model calibration can be run in the following ways:

1. Using command line parameters:
   ```
   python main.py --calibrate
   ```

2. Automatically run in complete workflow:
   ```
   python main.py --all
   ```

3. Using batch file, select corresponding function:
   ```
   run.bat
   ```

### Calibration Principles

Model calibration is based on the following mathematical principles:

1. **Platt Scaling**: Uses logistic regression to map original prediction scores to calibrated probabilities
   
   Mathematical expression:
   ```math
   P(y=1|s) = \sigma(As + B)
   ```
   
   Where $s$ is the original score, $A$ and $B$ are parameters, $\sigma$ is the sigmoid function

2. **Isotonic Regression**: Uses piecewise constant functions for non-parametric calibration
   - Maintains probability monotonicity while minimizing mean squared error

3. **Adaptive Calibration**:
   - For low-risk samples $(p < 0.3)$: Use Platt Scaling
   - For high-risk samples $(p \geq 0.3)$: Use isotonic regression or spline calibration

4. **Piecewise Spline Calibration**:
   - Low probability region $(p < 0.2)$:
   ```math
   p' = p \times 1.2
   ```
   
   - Medium probability region $(0.2 \leq p < 0.5)$:
   ```math
   p' = 0.24 + (p - 0.2) \times 1.5
   ```
   
   - High probability region $(p \geq 0.5)$:
   ```math
   p' = 0.69 + (p - 0.5) \times 1.8
   ```

## Project Packaging Instructions

### Necessary Files and Directories

When packaging the project, please include the following content:
1. All Python source code files (*.py)
2. Dataset files (stroke.csv, heart.csv, cirrhosis.csv)
3. static directory (contains static resources)
4. templates directory (contains HTML templates)
5. requirements.txt (dependency packages list)
6. README.md and README_EN.md (documentation files)
7. run.bat batch script

### Excludable Directories

The following content can be excluded when packaging:
1. __pycache__ directory (Python cache files)
2. .idea directory (IDE configuration files)
3. .venv or venv directory (Python virtual environment)
4. catboost_info directory (CatBoost temporary files)
5. app.log (application log files)
6. output directory (can be automatically generated at runtime)

## Contributors

- Shi Yunkun
- Zeng Zihang
