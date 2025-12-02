@echo off
echo ===================================================
echo Setting up Python 3.12 Environment for GPU Support
echo ===================================================

REM Check if Python 3.12 is available
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.12 is not detected!
    echo Please install Python 3.12 first.
    echo You can run: winget install -e --id Python.Python.3.12
    pause
    exit /b 1
)

echo [1/4] Removing old environment...
if exist .venv (
    rmdir /s /q .venv
)

echo [2/4] Creating new virtual environment with Python 3.12...
py -3.12 -m venv .venv

echo [3/4] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo [4/4] Installing PyTorch with CUDA 12.1 support...
.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip.exe install bitsandbytes transformers flask flask-cors accelerate

echo.
echo ===================================================
echo Setup Complete!
echo You can now run: launch_exe.bat
echo ===================================================
pause
