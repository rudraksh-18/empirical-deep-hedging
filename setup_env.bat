@echo off
echo ============================================================
echo   Empirical Deep Hedging — Environment Setup
echo ============================================================

echo [1/4] Creating Python virtual environment...
py -3.11 -m venv venv
if errorlevel 1 (
    echo ERROR: python not found. Please install Python 3.9+ and add to PATH.
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo [4/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================================
echo   Setup complete!
echo   To activate the environment: venv\Scripts\activate.bat
echo   To train the model:          python train.py
echo   To evaluate results:         python evaluate.py
echo ============================================================
