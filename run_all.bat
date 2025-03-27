@echo off
chcp 65001 > nul
echo Execution du bot d'analyse de cryptomonnaies...

REM Execution des differentes etapes du bot
echo 1. Collecte des donnees...
python data/collect_data.py

echo 2. Analyse technique...
python analysis/technical_indicators.py

echo 3. Analyse de sentiment...
python analysis/sentiment_analysis.py

echo 4. Analyse on-chain...
python analysis/on_chain_analysis.py

echo 5. Prediction...
python prediction/prediction_system.py

echo 6. Generation des newsletters...
python newsletters/newsletter_generator.py

echo Traitement termine.
pause
