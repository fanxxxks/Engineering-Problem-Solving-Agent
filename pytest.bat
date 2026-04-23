@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

call :try_python "%SCRIPT_DIR%.venv\Scripts\python.exe" %*
if not errorlevel 9009 exit /b %errorlevel%

call :try_python "D:\pythonc++\Thonny\python.exe" %*
if not errorlevel 9009 exit /b %errorlevel%

where python >nul 2>nul
if not errorlevel 1 (
    python -m pytest -q %*
    exit /b %errorlevel%
)

py -3 -m pytest -q %*
exit /b %errorlevel%

:try_python
set "CANDIDATE=%~1"
shift
if not exist "%CANDIDATE%" exit /b 9009
"%CANDIDATE%" -c "import sys" >nul 2>nul
if errorlevel 1 exit /b 9009
"%CANDIDATE%" -m pytest -q %*
exit /b %errorlevel%
