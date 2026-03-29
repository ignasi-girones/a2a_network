@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%"

echo ==========================================
echo   A2A Debate Network - Launcher
echo ==========================================
echo.

if "%1"=="--split" goto :split
if "%1"=="-s" goto :split
goto :single

:split
echo Modo: terminales separadas
echo.

REM Obtener resolucion de pantalla aproximada (1920x1080 default)
set W=960
set H=540

REM Fila superior: Normalizer, AE1, AE2
start "Normalizer :9001" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Normalizer :9001 && mode con: cols=80 lines=25 && python -m agents.normalizer"
timeout /t 1 /nobreak >nul

start "AE1 :9002" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title AE1 :9002 && mode con: cols=80 lines=25 && python -m agents.specialized --port 9002 --agent-id ae1"
timeout /t 1 /nobreak >nul

start "AE2 :9003" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title AE2 :9003 && mode con: cols=80 lines=25 && python -m agents.specialized --port 9003 --agent-id ae2"
timeout /t 1 /nobreak >nul

REM Fila inferior: Feedback, Orchestrator, Frontend
start "Feedback :9004" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Feedback :9004 && mode con: cols=80 lines=25 && python -m agents.feedback"
timeout /t 1 /nobreak >nul

start "Orchestrator :9000" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Orchestrator :9000 && mode con: cols=80 lines=25 && python -m agents.orchestrator"
timeout /t 2 /nobreak >nul

start "Frontend :5173" cmd /k "cd /d %ROOT%frontend && title Frontend :5173 && mode con: cols=80 lines=25 && npm run dev"

echo.
echo 6 terminales lanzadas.
echo Organiza manualmente o usa Win+Flechas para distribuirlas.
echo.
echo   Frontend:     http://localhost:5173
echo   Orchestrator: http://localhost:9000
echo.
echo Para cerrar todo: ejecuta  start.bat --stop
goto :end

:single
echo Modo: terminal unica (todos los procesos aqui)
echo   Usa "start.bat --split" o "start.bat -s" para terminales separadas.
echo   Usa "start.bat --stop" para cerrar todos los agentes.
echo.

if "%1"=="--stop" goto :stop

echo Arrancando agentes...
start /b "" python -m agents.normalizer
start /b "" python -m agents.feedback
start /b "" python -m agents.specialized --port 9002 --agent-id ae1
start /b "" python -m agents.specialized --port 9003 --agent-id ae2
timeout /t 2 /nobreak >nul
start /b "" python -m agents.orchestrator
timeout /t 1 /nobreak >nul

echo.
echo ==========================================
echo   Agentes arrancados en background.
echo   Arrancando frontend...
echo ==========================================
echo.
echo   Frontend:     http://localhost:5173
echo   Orchestrator: http://localhost:9000
echo.

cd /d "%ROOT%frontend"
npm run dev
goto :end

:stop
echo Cerrando todos los procesos de agentes y frontend...
taskkill /f /fi "WINDOWTITLE eq Normalizer*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq AE1*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq AE2*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Feedback*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Orchestrator*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Frontend*" >nul 2>&1
REM Fallback: matar procesos python/node en los puertos
for %%p in (9000 9001 9002 9003 9004) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING') do (
        taskkill /f /pid %%a >nul 2>&1
    )
)
echo Todos los procesos cerrados.
goto :end

:end
endlocal
