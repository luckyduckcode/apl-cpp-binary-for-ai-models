@echo off
REM APL Chat Interface - Build Executable Script
REM This script builds the APL Chat application as a standalone .exe file

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║        APL Chat Interface - Build Executable              ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

REM Check if Python virtual environment is active
if not defined VIRTUAL_ENV (
    echo [!] Virtual environment not activated
    echo [*] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

echo [*] Installing PyInstaller (if not already installed)...
python -m pip install pyinstaller -q

echo [*] Building executable... This may take 2-5 minutes...
echo.
python build_exe.py

if errorlevel 1 (
    echo.
    echo [!] Build failed. Check the output above for errors.
    pause
    exit /b 1
)

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║            Build Complete!                                ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo [✓] Executable created: dist\APL-Chat.exe
echo [✓] File size: ~300-400 MB (includes all dependencies)
echo.
echo Next steps:
echo   1. Test the executable:  dist\APL-Chat.exe
echo   2. Share with others:    Copy dist\APL-Chat.exe anywhere
echo   3. Create shortcut:      Right-click exe - Create shortcut
echo.
pause
