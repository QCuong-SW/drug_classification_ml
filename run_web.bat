@echo off
chcp 65001 > nul
echo ===================================================
echo   KHỞI ĐỘNG DRUG CLASSIFICATION ML WEB STUDIO
echo ===================================================
echo Đang mở server tại http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
python server.py
pause
