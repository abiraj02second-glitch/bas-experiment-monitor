@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo [1/3] Starting Docker Desktop...
docker desktop start >nul 2>&1

set /a BAS_TRIES=0
:WAIT_DOCKER
docker info >nul 2>&1 && goto DOCKER_READY
set /a BAS_TRIES+=1
if %BAS_TRIES% GEQ 60 goto DOCKER_FAILED
timeout /t 2 /nobreak >nul
goto WAIT_DOCKER

:DOCKER_READY
echo [2/3] Starting BAS AI Monitor...
docker compose up -d
if errorlevel 1 goto APP_FAILED

set /a BAS_TRIES=0
:WAIT_APP
curl.exe -fsS http://localhost:8000/state >nul 2>&1 && goto APP_READY
set /a BAS_TRIES+=1
if %BAS_TRIES% GEQ 60 goto APP_FAILED
timeout /t 2 /nobreak >nul
goto WAIT_APP

:APP_READY
echo [3/3] Opening unattended voice display...
call "%~dp0run-kiosk.bat" "http://localhost:8000"
exit /b 0

:DOCKER_FAILED
echo ERROR: Docker Desktop did not become ready within two minutes.
echo Open Docker Desktop and run this file again.
pause
exit /b 1

:APP_FAILED
echo ERROR: BAS AI Monitor did not start correctly.
docker compose ps
docker compose logs --tail 50
pause
exit /b 1
