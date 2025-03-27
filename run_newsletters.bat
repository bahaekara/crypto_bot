@echo off
chcp 65001 > nul
echo Generation des newsletters...
python newsletters/newsletter_generator.py
echo Generation des newsletters terminee.
pause
