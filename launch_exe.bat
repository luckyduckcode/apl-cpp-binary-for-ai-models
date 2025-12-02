@echo off
REM APL Chat Interface Launcher
REM Simple script to run the APL Chat .exe

title APL Chat Interface
color 0A

REM Check if executable exists
if not exist "dist\APL-Chat.exe" (
    echo.
    echo Error: APL-Chat.exe not found in dist\ folder
    echo.
    echo To build the executable:
    echo   1. Open PowerShell
    echo   2. Run: python build_exe_simple.py
    echo.
    pause
    exit /b 1
)

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║          APL Chat Interface - Starting                     ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Starting server and opening chat interface...
echo.

REM Run the executable
start "" "dist\APL-Chat.exe"

REM Give it a moment to start
timeout /t 2 /nobreak

REM Open browser to localhost
echo Opening browser...
start http://localhost:5000

echo.
echo Chat interface is running at: http://localhost:5000
echo.
echo To stop the server:
echo   - Close the application window
echo   - OR press Ctrl+C in the console
echo.
