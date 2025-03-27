@echo off
chcp 65001 > nul
echo Execution du bot d'analyse technique...
python analysis/technical_indicators.py
echo Analyse technique terminee.
pause
