@echo off
setlocal
cd /d "%~dp0"
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
  "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" edge_review.py
) else (
  py -3 edge_review.py
)
echo.
echo The key is read from your local .env or environment. If successful, open the report path printed above.
pause
