@echo off
setlocal

echo ============================================================
echo  ASL CNN branch - one-click environment setup (Windows)
echo ============================================================

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PYTHON_LAUNCHER=py -3.11"
) else (
    set "PYTHON_LAUNCHER=python"
)

if not exist ".venv" (
    echo Creating virtual environment with: %PYTHON_LAUNCHER%
    %PYTHON_LAUNCHER% -m venv .venv
    if errorlevel 1 (
        echo.
        echo Failed to create the virtual environment. Make sure Python 3.10/3.11
        echo is installed and available on PATH, then re-run this script.
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing packages from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo One or more packages failed to install. Scroll up for the error.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo  This window's virtual environment is now active.
echo  Next time, activate it manually with:
echo      .venv\Scripts\activate.bat
echo ============================================================
pause
endlocal
