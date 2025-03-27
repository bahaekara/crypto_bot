@echo off
chcp 65001 > nul
echo Installation des dependances pour le bot d'analyse de cryptomonnaies...

REM Verifier si le fichier requirements.txt existe
if not exist "%~dp0requirements.txt" (
    echo ERREUR: Le fichier requirements.txt n'a pas ete trouve dans le repertoire actuel.
    echo Chemin actuel: %~dp0
    echo Veuillez vous assurer que le fichier requirements.txt est present.
    goto :end
)

REM Verifier si Python est installe
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERREUR: Python n'est pas installe ou n'est pas dans le PATH.
    echo Veuillez installer Python depuis https://www.python.org/downloads/
    goto :end
)

REM Installer les dependances
echo Installation des packages depuis requirements.txt...
python -m pip install -r "%~dp0requirements.txt"
if %ERRORLEVEL% neq 0 (
    echo AVERTISSEMENT: Certains packages n'ont pas pu etre installes.
    echo Le bot pourrait ne pas fonctionner correctement.
) else (
    echo Toutes les dependances ont ete installees avec succes.
)

:end
echo Installation terminee.
pause
