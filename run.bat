@echo off
cd /d C:\Users\PJ\python\test\elgo_app
echo add venv...
call .\Scripts\activate.bat
echo Flask start...
python app.py
pause