@echo off
cd /d "%~dp0"
title Flood Segmentation Training Live Monitor
echo Starting Flood Segmentation Training Monitor...
.\venv\Scripts\python.exe ml\monitor_training.py
pause
