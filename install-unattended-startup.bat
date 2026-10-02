@echo off
setlocal
set "BAS_SCRIPT=%~dp0run-unattended.bat"
set "BAS_SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\BAS AI Monitor.lnk"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut($env:BAS_SHORTCUT); $s.TargetPath=$env:BAS_SCRIPT; $s.WorkingDirectory='%~dp0'; $s.Save()"

if exist "%BAS_SHORTCUT%" (
  echo BAS AI Monitor will now start automatically with Windows.
) else (
  echo ERROR: The startup shortcut could not be created.
)
pause
