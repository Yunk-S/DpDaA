# Disease Prediction and Big Data Analysis System

Machine learning and deep learning-based stroke, heart disease, and cirrhosis prediction system, providing data analysis, risk prediction, and health management recommendations.
This project is still being continuously updated and is based on the research of question B in the 15th APMCM Asia Pacific Mathematical Model Competition (Chinese competition) in 2025. Due to limited training data for the competition, the current model is slightly insufficient. We apologize for any inconvenience caused and will only provide reference.

> [中文版 README](README.md)

## Project Structure

```
├── app.py                        # Web application main program
├── main.py                       # System entry program
├── data_preprocessing.py         # Data preprocessing module
├── model_training.py             # Model training module
├── model_calibration.py          # Model calibration module
├── model_utilities.py            # Model utility library (metrics generation, chart generation)
├── train_deep_learning_models.py # Deep learning model training module
├── visualization.py              # Data visualization module
├── chart_generator.py            # Chart generation tool
├── generate_metrics.py           # Model metrics generation tool
├── multi_disease_model.py        # Multi-disease hybrid prediction model
├── run.bat                       # System startup batch file
├── static/                       # Static resource directory
│   ├── style.css                 # Style sheet
│   ├── script.js                 # JavaScript script
│   └── images/                   # Image resources
├── templates/                    # HTML templates
│   ├── index.html                # Homepage
│   ├── data_analysis.html        # Data analysis page
│   ├── predict.html              # Prediction page
│   ├── multi_disease.html        # Multi-disease association analysis page
│   └── ...                       # Other pages
├── output/                       # Output directory
│   ├── models/                   # Saved trained models
│   │   ├── calibrators/          # Saved calibrator models
│   ├── figures/                  # Generated charts
│   └── processed_data/           # Processed data
├── requirements.txt              # Project dependency list
└── *.csv                         # Original datasets (stroke.csv, heart.csv, cirrhosis.csv)
```

## Installation and Running

### Requirements

- Python 3.8+
- Related dependencies (see requirements.txt)

### Installation Steps

1. Create and activate a virtual environment
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

   If installation is slow in mainland China, you can use the Tsinghua mirror:
   ```
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
   ```

### Running the System

#### Method 1: Using batch file (Windows)

Run the `run.bat` batch file and follow the prompts:
```
run.bat
```

Running will prompt in the following order:
1. Whether to create/use a virtual environment (recommended to select Y)
   - If Y is selected and a virtual environment doesn't exist, a new virtual environment will be created and automatically activated
   - If Y is selected and a virtual environment already exists, the existing virtual environment will be activated directly
   - If N is selected, the system Python environment will be used
2. Whether to install dependencies
   - Dependencies need to be installed when using for the first time
   - You can choose to use the default installation source or Tsinghua mirror (recommended for mainland China)
3. Select function mode:
   1. Run the complete analysis and prediction process
   2. Only run data preprocessing
   3. Only run data visualization
   4. Only run model training
   5. Only run chart and metric generation
   6. Only run the Web application
   7. Execute main.py directly (all steps)
   8. Exit

**Note:** 
- Options 1 and 7 have the same functionality and both execute the complete process. Choose either one.
- Using a virtual environment can avoid dependency conflict issues, especially recommended for new users.
- After the program ends, if you used a virtual environment, you need to manually enter the `deactivate` command to exit the virtual environment.

#### Method 2: Using Python commands

Run the complete process (data preprocessing, visualization, model training, chart metrics generation, and Web application):
```
python main.py --all
```

Or use different parameters to run specific functions separately:
- `--preprocess`: Only run data preprocessing
- `--visualize`: Only run data visualization
- `--train`: Only run model training
- `--generate`: Only run chart and metric generation
- `--webapp`: Only run the Web application

Example: Only start the Web application
```
python main.py --webapp
```

## Web Application Instructions 

1. After running the Web application, the system will automatically open the homepage in the browser (http://localhost:5000)
2. On the homepage, you can view the system introduction and basic information about the three diseases
3. The navigation bar provides the following function entries:
   - **Data Analysis**: View the distribution, correlation, and feature importance of the three disease data
   - **Model Performance**: View the performance metrics and evaluation results of each prediction model
   - **Disease Prediction**: Input personal health indicators to get disease risk predictions and health advice
   - **Multi-disease Association**: View the association analysis and comprehensive risk assessment between the three diseases

### Single Disease Prediction Function
1. Click on the "Disease Prediction" dropdown menu in the top navigation bar and select "Single Disease Prediction"
2. On the prediction page, you can select one of the three diseases (stroke, heart disease, or cirrhosis) for risk prediction
3. Fill in the corresponding health indicator form and click the "Predict Risk" button
4. The system will display prediction results, including risk probability, risk level, and health advice

### Multi-disease Joint Prediction Function
1. Click on the "Disease Prediction" dropdown menu in the top navigation bar and select "Multi-disease Joint Prediction"
2. Fill in the comprehensive health indicator form, including basic information, lifestyle, and clinical indicators
3. Click the "Comprehensive Prediction" button
4. The system will display the individual risk of three diseases and the joint risk probability of multiple disease combinations

## Project Overview

This project aims to build a comprehensive disease prediction and analysis system by conducting in-depth analysis of data on three common major diseases—stroke, heart disease, and cirrhosis—to establish prediction models and provide a visualization interface to display results and health management recommendations. The system employs various advanced machine learning and deep learning algorithms, combined with knowledge from the medical field, to achieve high-precision disease risk prediction.

### Initial Approach

The design approach of the project is based on the following points:
1. **Multi-disease Integrated Analysis**: Unlike traditional single-disease prediction systems, this project simultaneously processes three highly related major diseases, exploring their associations and common risk factors
2. **Data-Driven Decision Making**: Through the analysis of large amounts of real medical data, extracting valuable features and patterns to aid medical decision-making
3. **Model Integration and Optimization**: Combining the advantages of multiple machine learning algorithms, using ensemble learning and model fusion techniques to improve prediction accuracy
4. **Explainable Analysis**: Using SHAP values and feature importance analysis to provide explainability for model prediction results
5. **User-Friendly Interface**: Designing an intuitive Web interface for convenient use by both medical professionals and ordinary users

## Features

- **Multi-disease Prediction**: Risk prediction for three major diseases: stroke, heart disease, and cirrhosis
- **Data Visualization**: Intuitively display data distribution, feature correlation, and prediction results
- **Risk Assessment**: Provide personalized health risk scores and prevention recommendations
- **Interactive Interface**: User-friendly Web interface, supporting real-time prediction and result display
- **Multi-disease Association Analysis**: Analyze the associations and comorbidity risks between different diseases
- **Age and Disease Relationship Analysis**: Show trends in disease incidence rates across different age groups
- **Error Recovery Mechanism**: When model loading fails, the system automatically uses backup models or simulated data for prediction, ensuring user experience is not affected

## Technical Architecture

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning Models**:
  - Traditional machine learning algorithms (Random Forest, XGBoost, LightGBM, CatBoost, SVM, etc.)
  - Deep learning models (DNN, Attention Mechanism Networks)
  - Ensemble learning (Voting, Stacking, Hybrid Fusion)
- **Model Explainability**: SHAP value analysis

## Multi-disease Hybrid Prediction Model Explanation

### Model Architecture

The multi-disease hybrid prediction model (`MultiDiseasePredictor` class) is one of the core components of the system, implementing the following functions:

1. **Model Loading and Fallback Mechanism**:
   - Attempts to load trained single-disease prediction models
   - Automatically creates backup models (RandomForest) when model loading fails
   - Provides simulated data prediction functionality, ensuring the system can run normally under any circumstances

2. **Single Disease Risk Prediction**:
   - Stroke prediction: Based on features such as age, gender, hypertension, etc.
   - Heart disease prediction: Based on features such as chest pain type, blood pressure, cholesterol, etc.
   - Cirrhosis prediction: Based on biochemical indicators to predict cirrhosis risk or staging

3. **Multi-disease Joint Prediction**:
   - Independent prediction: Calculate the probability of each disease occurring independently
   - Joint probability: Calculate the joint probability of multiple disease combinations
   - Consider correlation: Adjust joint probability through correlation coefficients, more consistent with actual medical situations

### Mathematical Principles

1. **Multi-task Learning Framework**:
   The system adopts a multi-task learning framework, learning multiple related tasks (disease prediction) simultaneously by sharing underlying feature representations.

   Mathematical representation:
   - Shared feature extraction layer: $\mathbf{h} = f_{\text{shared}}(\mathbf{x}; \theta_s)$
   - Task-specific output layer: $\hat{y}^{(k)} = f_{\text{task}_k}(\mathbf{h}; \theta_k)$

2. **Joint Probability Calculation**:
   For independent probabilities of diseases A, B, C as $P(A)$, $P(B)$, $P(C)$, the system calculates the following joint probabilities:
   - Single disease: $P(A)(1-P(B))(1-P(C))$
   - Two diseases: $P(A)P(B)(1-P(C))$
   - Three diseases: $P(A)P(B)P(C)$
   - No disease: $(1-P(A))(1-P(B))(1-P(C))$

3. **Correlation Adjustment**:
   The system uses correlation coefficients to adjust joint probabilities, calculating correlation-considered joint probabilities:
   
   $P(A,B) = P(A)P(B) + \text{corr}_{AB} \cdot \min(P(A), P(B)) \cdot (1 - \max(P(A), P(B)))$
   
   where $\text{corr}_{AB}$ is the correlation coefficient between diseases A and B.

4. **Attention Mechanism**:
   Attention mechanism is used in deep learning models to enhance the model's focus on important features:
   
   $\alpha_i = \text{Attention}(h_i)$
   $h'_i = h_i \cdot \alpha_i$
   
   where $\alpha_i$ is the attention weight and $h_i$ is the feature representation.

### Model Optimization Techniques

1. **Knowledge Distillation**:
   - Teacher model: Complex deep neural networks or ensemble models
   - Student model: Lightweight model, improving performance by learning the "soft labels" of the teacher model
   - Distillation loss: $\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \hat{y}_{\text{student}}) + (1-\alpha) \cdot \mathcal{L}_{\text{KL}}(\hat{y}_{\text{teacher}}, \hat{y}_{\text{student}})$

2. **Ensemble Learning**:
   - Voting ensemble: Multiple base models obtain final predictions through voting or averaging
   - Stacking ensemble: Using meta-learners to integrate prediction results from base models
   - Hybrid fusion: Combining the advantages of traditional machine learning and deep learning models

3. **Collaborative Regularization**:
   Encourages information sharing between parameters of different tasks, promoting the model to learn comorbidity patterns between diseases:
   
   $\mathcal{L}_{\text{reg}} = \lambda \cdot \sum_{i,j} \| \theta_i - \theta_j \|_2^2$

## Mathematical Knowledge and Model Techniques Used

### Data Preprocessing Techniques
- **Missing Value Handling**: Iterative imputation (IterativeImputer), KNN imputation, mean/median filling
- **Outlier Detection**: IQR method, Z-score method, isolation forest
- **Feature Engineering**: Feature selection, feature interaction, polynomial features, risk score calculation
- **Data Standardization**: Z-score standardization, Min-Max scaling
- **Class Imbalance Handling**: SMOTE oversampling, class weight adjustment

### Machine Learning Algorithms
- **Classification Algorithms**: Logistic Regression, Random Forest, Gradient Boosting Trees, XGBoost, LightGBM, CatBoost, SVM
- **Regression Algorithms**: Linear Regression, Ridge Regression, LASSO Regression, Elastic Net, SVR, Random Forest Regression, Gradient Boosting Regression
- **Ensemble Learning**:
  - Voting ensemble: Combining prediction results from multiple base models through voting or averaging to obtain final predictions
  - Stacking ensemble: Using meta-learners to integrate prediction results from base models
  - Hybrid fusion: Combining advantages of different types of models, such as fusion of traditional machine learning and deep learning models

### Deep Learning Techniques
- **Deep Neural Networks (DNN)**: Multilayer perceptron structure for complex feature learning
- **Attention Mechanism**: Introducing attention layers in neural networks to focus on important features
- **Batch Normalization**: Accelerating the training process and improving model stability
- **Dropout Regularization**: Preventing overfitting
- **Knowledge Distillation**: Transferring knowledge from complex models to simple models

### Model Evaluation and Optimization
- **Cross-validation**: K-fold cross-validation, stratified K-fold cross-validation
- **Hyperparameter Optimization**: Grid search, random search
- **Performance Metrics**:
  - Classification tasks: Accuracy, precision, recall, F1 score, AUC-ROC
  - Regression tasks: MSE, RMSE, MAE, R²

### Explainability Analysis
- **SHAP Value Analysis**: Explaining the contribution of each feature in model predictions
- **Feature Importance Analysis**: Identifying features with the greatest impact on prediction results
- **Partial Dependence Plots**: Showing the relationship between specific features and prediction results

## Dataset Description

The project uses the following three datasets:

1. **Stroke Dataset (stroke.csv)**
   - Features: Age, gender, hypertension, heart disease history, marital status, work type, residence type, average glucose level, BMI, smoking status
   - Target variable: Whether a stroke occurred (binary classification)
   - Sample size: Approximately 5,110 records
   - Characteristics: Class imbalance, missing values mainly concentrated in the BMI feature

2. **Heart Disease Dataset (heart.csv)**
   - Features: Age, gender, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, electrocardiogram results, maximum heart rate, angina, ST depression, ST slope
   - Target variable: Whether heart disease is present (binary classification)
   - Sample size: Approximately 918 records
   - Characteristics: Complex feature combinations, multiple types of chest pain and electrocardiogram results requiring specialized encoding

3. **Cirrhosis Dataset (cirrhosis.csv)**
   - Features: Age, gender, bilirubin, cholesterol, albumin, copper, alkaline phosphatase, SGOT, triglycerides, platelets, prothrombin time, etc.
   - Target variable: Cirrhosis staging (stages 1-4, regression task)
   - Sample size: Approximately 418 records
   - Characteristics: Target variable is continuous, requiring regression models; multiple biochemical indicators

### Multi-disease Association Analysis Function

The multi-disease association analysis page provides the following key sections:
1. **Disease Association Overview**: Introduction to the associations and common pathological mechanisms between the three diseases
2. **Gender and Disease Relationship Comparison**: Showing risk differences between genders for the three diseases
3. **Age Distribution Comparison**: Using violin plots to show age distribution characteristics of patients with the three diseases
4. **Age and Disease Incidence Relationship**: Showing trends in incidence rates of the three diseases across different age groups
5. **Comorbidity Risk Factor Analysis**: Using tables and radar charts to show common risk factors affecting multiple diseases
6. **Multi-disease Prediction Model and Comprehensive Risk Assessment**: Introduction to multi-objective learning methods and comprehensive risk scoring system
7. **Prevention Recommendations and Health Management**: Providing comprehensive prevention and health management recommendations for multiple diseases

## System Performance and Limitations

### Performance Metrics
- Stroke prediction: Accuracy approximately 92.3%, AUC approximately 0.89
- Heart disease prediction: Accuracy approximately 88.5%, AUC approximately 0.91
- Cirrhosis staging: R² approximately 0.28, RMSE approximately 0.71

### Limitations
1. Limited dataset scale, potentially affecting model generalization ability
2. Lack of time-series data, unable to predict disease progression
3. No integration of genomic data, limiting the potential for personalized medicine
4. Room for improvement in model explainability

## Error Handling and Fallback Mechanism

The system implements multi-level error handling and fallback mechanisms to ensure normal operation under various conditions:

1. **Model Loading Failure Handling**:
   - When pre-trained models cannot be loaded, the system automatically creates simple substitute models
   - Using RandomForestClassifier for classification tasks (stroke, heart disease)
   - Using RandomForestRegressor for regression tasks (cirrhosis)

2. **Prediction Error Handling**:
   - Capturing exceptions during the prediction process, returning default probability values
   - Displaying simulated data use prompts in the frontend, ensuring users understand result reliability

3. **Web Application Error Handling**:
   - Implementing global exception handling, capturing and recording errors
   - Providing user-friendly error pages instead of directly displaying technical error messages

### Model Calibration Function

To address the issue of low prediction probabilities, especially for high-risk populations, the system has added model calibration functionality:

1. **Probability Calibration Techniques**:
   - Platt Scaling (logistic regression calibration): Using logistic regression to calibrate original prediction probabilities
   - Isotonic Regression: Using non-parametric methods for probability calibration, more flexible
   - Piecewise spline calibration: Using different calibration strategies for different probability intervals

2. **Adaptive Calibration Strategy**:
   - Automatically selecting the best calibration method based on risk level
   - Using mild calibration for low-risk populations
   - Using more aggressive calibration strategies for high-risk populations

3. **Calibration Evaluation Metrics**:
   - Brier score evaluation: Measuring the accuracy of probability predictions
   - Log loss evaluation: Measuring the uncertainty of probability predictions
   - Calibration curve visualization: Intuitively showing the effects before and after calibration

Calibrated models significantly improve prediction accuracy, especially for high-risk populations, making prediction results more consistent with clinical reality.

### Running the Calibration Function

Model calibration can be run through the following methods:

1. Using command line parameters:
   ```
   python main.py --calibrate
   ```

2. Automatically run in the complete process:
   ```
   python main.py --all
   ```

3. Using batch file, select the corresponding function:
   ```
   run.bat
   ```

### Calibration Principles

Model calibration is based on the following mathematical principles:

1. **Platt Scaling**: Using logistic regression to map original prediction scores to calibrated probabilities
   - Mathematical expression: P(y=1|s) = σ(As + B), where s is the original score, A and B are parameters, σ is the sigmoid function

2. **Isotonic Regression**: Using piecewise constant functions for non-parametric calibration
   - Maintaining probability monotonicity while minimizing mean squared error

3. **Adaptive Calibration**:
   - For low-risk samples (p < 0.3): Using Platt Scaling
   - For high-risk samples (p ≥ 0.3): Using isotonic regression or spline calibration

4. **Piecewise Spline Calibration**:
   - Low probability region (p < 0.2): Slight enhancement p' = p * 1.2
   - Medium probability region (0.2 ≤ p < 0.5): Moderate enhancement p' = 0.24 + (p - 0.2) * 1.5
   - High probability region (p ≥ 0.5): Significant enhancement p' = 0.69 + (p - 0.5) * 1.8

## Project Packaging Instructions

### Required Files and Directories

Please include the following when packaging the project:
1. All Python source code files (*.py)
2. Dataset files (stroke.csv, heart.csv, cirrhosis.csv)
3. static directory (containing static resources)
4. templates directory (containing HTML templates)
5. requirements.txt (dependency list)
6. README.md (this documentation file)
7. README_EN.md (English version documentation file)
8. run.bat batch script

### Excludable Directories

The following can be excluded when packaging:
1. __pycache__ directory (Python cache files)
2. .idea directory (IDE configuration files)
3. .venv or venv directory (Python virtual environment)
4. catboost_info directory (CatBoost temporary files)
5. app.log (application log file)
6. output directory (can be automatically generated at runtime)


## Probability Smoothing Processing Function

To address the issue of extreme prediction results (0% or 100%), we have implemented various probability smoothing processing strategies:

### 1. Improved Model Calibration Methods

- Implemented multiple calibration methods (spline functions, Platt scaling, Beta distribution)
- Using weighted fusion strategies to combine the advantages of various calibration methods
- Adopting different calibration parameters for different risk levels

### 2. Prediction Result Smoothing Processing

- Added the `smooth_probability` function, providing multiple smoothing methods:
  - `clip`: Simple truncation method, ensuring probabilities are between set minimum and maximum values
  - `beta`: Beta distribution smoothing, using probability distribution for smoothing
  - `sigmoid_quantile`: Sigmoid function and quantile combined smoothing method (recommended)
- Adjusting smoothing parameters based on risk level (low, medium, high)
- For regression prediction results, using value domain limitations and smooth quintile distribution

### 3. System Robustness Enhancement

- Added multiple fallback strategies, ensuring models can provide reasonable predictions under various conditions:
  - Standard prediction method
  - LightGBM raw_score mode
  - Decision function method
  - Direct Booster access
  - Model parameter reconstruction
- Strengthened error handling mechanisms, providing detailed error information and lists of attempted methods

### Usage Method

```python
from model_utilities import smooth_probability

# Smooth a single probability value
smoothed_prob = smooth_probability(
    prob=0.99,               # Original probability
    method='sigmoid_quantile', # Smoothing method
    min_prob=0.05,           # Minimum probability value
    max_prob=0.95            # Maximum probability value
)

# Smooth probability array
import numpy as np
probs = np.array([0.01, 0.99, 0.5, 0])
smoothed_probs = smooth_probability(
    prob=probs,
    method='beta',
    min_prob=0.02,
    max_prob=0.98
)
```

These improvements have solved the previously faced issue of prediction models outputting extreme values (0% or 100%); now the system outputs smoother, more reasonable probability distributions, improving the reliability and practicality of prediction results, better aligning with the inherent uncertainty in medical predictions. 

## Contributors

- Shi Yunkun
- Zeng Zihang