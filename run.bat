@echo off
chcp 65001 >nul

echo ============================================
echo           Disease Prediction System         
echo ============================================
echo.

echo Do you want to create/use a virtual environment? (Y/N)
echo [Recommended for avoiding dependency conflicts]
set /p venv_choice=

if /i "%venv_choice%"=="Y" (
    echo.
    if exist venv\Scripts\activate.bat (
        echo Virtual environment already exists. Activating...
        call venv\Scripts\activate.bat
    ) else (
        echo Creating new virtual environment...
        python -m venv venv
        if %ERRORLEVEL% neq 0 (
            echo Failed to create virtual environment.
            goto DEP_CHECK
        )
        echo Virtual environment created successfully!
        call venv\Scripts\activate.bat
        echo Virtual environment activated!
    )
    echo.
    echo You are now using the Python virtual environment.
) else (
    echo Not using virtual environment.
)

:DEP_CHECK
echo.
echo Do you want to install dependencies? (Y/N)
echo [First time users should install dependencies]
set /p dep_choice=

if /i "%dep_choice%"=="Y" (
    echo.
    echo Choose installation method:
    echo 1. Default installation
    echo 2. Install using Tsinghua mirror (recommended for China)
    echo.
    set /p mirror=

    if "%mirror%"=="1" (
        set "mirror_url="
        set "mirror_arg="
    ) else if "%mirror%"=="2" (
        set "mirror_url=https://pypi.tuna.tsinghua.edu.cn/simple"
        set "mirror_arg=--trusted-host pypi.tuna.tsinghua.edu.cn"
    ) else (
        echo Invalid option, using default PyPI...
        set "mirror_url="
        set "mirror_arg="
    )

    echo.
    echo Installing dependencies (this may take several minutes)...
    echo.
    
    echo Step 1/3: Installing basic and machine learning packages...
    if not "%mirror_url%"=="" (
        python -m pip install --only-binary :all: numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn missingno catboost -i %mirror_url% %mirror_arg%
    ) else (
        python -m pip install --only-binary :all: numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn missingno catboost
    )
    
    echo.
    echo Step 2/3: Installing PyTorch and other deep learning packages...
    if not "%mirror_url%"=="" (
        python -m pip install torch torchvision -i %mirror_url% %mirror_arg%
        python -m pip install --only-binary :all: keras -i %mirror_url% %mirror_arg%
        
        echo Installing TensorFlow 2.15.0...
        python -m pip install tensorflow==2.15.0 -i %mirror_url% %mirror_arg%
        
        if %ERRORLEVEL% neq 0 (
            echo TensorFlow 2.15.0 installation failed. Trying alternative method...
            python -m pip install tensorflow==2.15.0 --no-deps -i %mirror_url% %mirror_arg%
            echo Note: TensorFlow dependencies may be incomplete.
        )
    ) else (
        python -m pip install torch torchvision
        python -m pip install --only-binary :all: keras
        
        echo Installing TensorFlow 2.15.0...
        python -m pip install tensorflow==2.15.0
        
        if %ERRORLEVEL% neq 0 (
            echo TensorFlow 2.15.0 installation failed. Trying alternative method...
            python -m pip install tensorflow==2.15.0 --no-deps
            echo Note: TensorFlow dependencies may be incomplete.
        )
    )
    
    echo.
    echo Step 3/3: Installing visualization, web application and model calibration packages...
    if not "%mirror_url%"=="" (
        python -m pip install --only-binary :all: shap plotly flask werkzeug joblib tqdm -i %mirror_url% %mirror_arg%
        echo Installing model calibration packages...
        python -m pip install scikit-learn-intelex>=2021.5.0 netcal>=1.3.0 calibration>=0.1.0 -i %mirror_url% %mirror_arg%
    ) else (
        python -m pip install --only-binary :all: shap plotly flask werkzeug joblib tqdm
        echo Installing model calibration packages...
        python -m pip install scikit-learn-intelex>=2021.5.0 netcal>=1.3.0 calibration>=0.1.0
    )
    
    echo.
    echo Installation complete!
    
    echo.
    echo Verifying critical packages...
    python -c "import numpy; print('NumPy:', numpy.__version__)" 2>nul
    python -c "import pandas; print('Pandas:', pandas.__version__)" 2>nul
    python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)" 2>nul
    python -c "import torch; print('PyTorch:', torch.__version__)" 2>nul
    python -c "try: import tensorflow as tf; print('TensorFlow:', tf.__version__); except: print('TensorFlow: Not installed')" 2>nul
    echo.
    echo If any packages failed to verify, you may need to install them manually.
    echo TensorFlow may be missing but the system should still work for most functions.
) else (
    echo Skipping dependency installation...
)

:MENU
echo.
echo Please select an option:
echo ============================================
echo 1. Run complete analysis and prediction pipeline
echo 2. Run data preprocessing only
echo 3. Run data visualization only
echo 4. Run model training only
echo 5. Run model calibration only
echo 6. Run charts and metrics generation only
echo 7. Run web application only
echo 8. Direct execution (python main.py)
echo 9. Exit
echo.

set /p user_option=
echo You selected option: %user_option%

if "%user_option%"=="1" goto OPTION_1
if "%user_option%"=="2" goto OPTION_2
if "%user_option%"=="3" goto OPTION_3
if "%user_option%"=="4" goto OPTION_4
if "%user_option%"=="5" goto OPTION_5
if "%user_option%"=="6" goto OPTION_6
if "%user_option%"=="7" goto OPTION_7
if "%user_option%"=="8" goto OPTION_8
if "%user_option%"=="9" goto OPTION_9

echo Invalid option! Please try again.
goto MENU

:OPTION_1
echo.
echo Running complete analysis and prediction...
echo Command: python main.py --all
python main.py --all
echo.
echo Execution completed (exit code: %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 (
    echo WARNING: An error occurred during execution.
    echo If you see TensorFlow errors, you can try options 2-6 which may work without TensorFlow.
)
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_2
echo.
echo Running data preprocessing...
echo Command: python main.py --preprocess
python main.py --preprocess
echo.
echo Execution completed (exit code: %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 (
    echo WARNING: An error occurred during execution.
)
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_3
echo.
echo Running data visualization...
echo Command: python main.py --visualize
python main.py --visualize
echo.
echo Execution completed (exit code: %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 (
    echo WARNING: An error occurred during execution.
)
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_4
echo.
echo Running model training...
echo Command: python main.py --train
python main.py --train
echo.
echo Execution completed (exit code: %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 (
    echo WARNING: An error occurred during execution.
)
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_5
echo.
echo Running model calibration...
echo Command: python main.py --calibrate
python main.py --calibrate
echo.
echo Execution completed (exit code: %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 (
    echo WARNING: An error occurred during execution.
)
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_6
echo.
echo Running charts and metrics generation...
echo Command: python main.py --generate
python main.py --generate
echo.
echo Execution completed (exit code: %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 (
    echo WARNING: An error occurred during execution.
)
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_7
echo.
echo Running web application...
echo Command: python main.py --webapp
start python main.py --webapp

echo.
echo Web application started in a separate window.
echo.
echo Now launching prediction tool...
timeout /t 5 > nul
start predict_cmd.bat

echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_8
echo.
echo Direct execution of main.py (all steps)...
echo Command: python main.py
python main.py
echo.
echo Execution completed (exit code: %ERRORLEVEL%).
if %ERRORLEVEL% neq 0 (
    echo WARNING: An error occurred during execution.
)
echo.
echo Press any key to return to the menu...
pause >nul
goto MENU

:OPTION_9
echo.
echo Exiting...
if /i "%venv_choice%"=="Y" (
    echo Deactivating virtual environment...
    call deactivate
)
exit 