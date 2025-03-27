@echo off
chcp 65001 > nul
echo Execution de l'analyse de sentiment...
python analysis/sentiment_analysis.py
echo Analyse de sentiment terminee.
pause
