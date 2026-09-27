@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Launch from the repository even when invoked through a shortcut or another directory.
pushd "%~dp0"
if errorlevel 1 goto directory_failed
if not exist ".venv\Scripts\python.exe" goto missing_runtime

".venv\Scripts\python.exe" -m brilliance.launcher %*
set "brilliance_exit_code=%errorlevel%"
popd
goto finish

:missing_runtime
echo BRILLIANCE could not find its configured project Python:
echo   "%~dp0.venv\Scripts\python.exe"
echo Complete the Install steps in README.md, then run Brilliance.cmd again.
set "brilliance_exit_code=1"
popd
goto finish

:directory_failed
echo BRILLIANCE could not open its project directory:
echo   "%~dp0"
set "brilliance_exit_code=1"

:finish
if "%brilliance_exit_code%"=="0" exit /b 0
echo.
echo BRILLIANCE exited with code %brilliance_exit_code%.
rem Keep double-click failures visible. Automation can set BRILLIANCE_NO_PAUSE=1.
if not "%BRILLIANCE_NO_PAUSE%"=="1" pause
exit /b %brilliance_exit_code%
