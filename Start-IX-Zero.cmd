@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python -m ix_zero
) else (
  py -3 -m ix_zero
)
pause
