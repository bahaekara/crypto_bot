@echo off
chcp 65001 > nul
echo Execution du systeme de prediction...
python prediction/prediction_system.py
echo Prediction terminee.
pause
