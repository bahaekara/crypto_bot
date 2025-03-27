@echo off
chcp 65001 > nul
echo Generation directe des donnees simulees...

REM Vérifier si Python est accessible
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR: Python n'est pas accessible. Verifiez votre installation Python.
    pause
    exit /b 1
)

REM Exécuter le script de génération de données
python data/generate_data.py

echo Generation des donnees terminee.
pause
