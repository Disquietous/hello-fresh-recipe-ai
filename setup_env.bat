@echo off
echo Setting up virtual environment for HelloFresh Recipe AI...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
echo To deactivate, run: deactivate
pause