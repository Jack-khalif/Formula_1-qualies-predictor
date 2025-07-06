@echo off
REM This script activates the virtual environment and runs the Streamlit app.

echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo Starting Streamlit app...
streamlit run app.py

pause
