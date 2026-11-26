@echo off
setlocal
set "INTEGRATION_DIR=%~dp0"
set "PROJECT_PYTHON=%INTEGRATION_DIR%..\.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo Nie znaleziono Pythona w projekcie: "%PROJECT_PYTHON%"
    pause
    exit /b 1
)
"%PROJECT_PYTHON%" -u -B "%INTEGRATION_DIR%scripts\mark_contacts.py" --start-library --port 0
set "SERVER_EXIT=%ERRORLEVEL%"
if not "%SERVER_EXIT%"=="0" pause
exit /b %SERVER_EXIT%
