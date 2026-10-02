@echo off
setlocal

set "BAS_URL=%~1"
if not defined BAS_URL set "BAS_URL=http://localhost:8000"
set "BAS_PROFILE=%LOCALAPPDATA%\BASVoiceKioskProfile"

set "CHROME_64=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
set "CHROME_32=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
set "EDGE=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"

if exist "%CHROME_64%" (
  start "" "%CHROME_64%" --user-data-dir="%BAS_PROFILE%" --no-first-run --kiosk --autoplay-policy=no-user-gesture-required --disable-session-crashed-bubble "%BAS_URL%"
  exit /b 0
)

if exist "%CHROME_32%" (
  start "" "%CHROME_32%" --user-data-dir="%BAS_PROFILE%" --no-first-run --kiosk --autoplay-policy=no-user-gesture-required --disable-session-crashed-bubble "%BAS_URL%"
  exit /b 0
)

if exist "%EDGE%" (
  start "" "%EDGE%" --user-data-dir="%BAS_PROFILE%" --no-first-run --kiosk --autoplay-policy=no-user-gesture-required --disable-session-crashed-bubble "%BAS_URL%"
  exit /b 0
)

echo Chrome or Microsoft Edge was not found. Opening the default browser.
echo The default browser may require one click before it permits sound.
start "" "%BAS_URL%"
exit /b 1
