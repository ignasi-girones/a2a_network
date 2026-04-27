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

REM Fila superior: MCP Tools, Normalizer, Analyst, Seeker
start "MCP Tools :8085" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title MCP Tools :8085 && mode con: cols=80 lines=25 && python -m agents.mcp_tools.server"
timeout /t 1 /nobreak >nul

start "Normalizer :8081" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Normalizer :8081 && mode con: cols=80 lines=25 && python -m agents.normalizer"
timeout /t 1 /nobreak >nul

start "Analyst :8082" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Analyst :8082 && mode con: cols=80 lines=25 && python -m agents.specialized --port 8082 --agent-id analyst --role analyst"
timeout /t 1 /nobreak >nul

start "Seeker :8083" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Seeker :8083 && mode con: cols=80 lines=25 && python -m agents.specialized --port 8083 --agent-id seeker --role seeker"
timeout /t 1 /nobreak >nul

REM Fila central: Devil's Advocate, Empiricist, Pragmatist, Synthesizer
start "DevilsAdvocate :8087" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title DevilsAdvocate :8087 && mode con: cols=80 lines=25 && python -m agents.specialized --port 8087 --agent-id devils_advocate --role devils_advocate"
timeout /t 1 /nobreak >nul

start "Empiricist :8089" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Empiricist :8089 && mode con: cols=80 lines=25 && python -m agents.specialized --port 8089 --agent-id empiricist --role empiricist"
timeout /t 1 /nobreak >nul

start "Pragmatist :8090" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Pragmatist :8090 && mode con: cols=80 lines=25 && python -m agents.specialized --port 8090 --agent-id pragmatist --role pragmatist"
timeout /t 1 /nobreak >nul

start "Synthesizer :8088" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Synthesizer :8088 && mode con: cols=80 lines=25 && python -m agents.specialized --port 8088 --agent-id synthesizer --role synthesizer"
timeout /t 1 /nobreak >nul

REM Fila inferior: Feedback, Orchestrator, Frontend
start "Feedback :8084" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Feedback :8084 && mode con: cols=80 lines=25 && python -m agents.feedback"
timeout /t 1 /nobreak >nul

start "Orchestrator :8080" cmd /k "cd /d %ROOT% && set PYTHONPATH=%ROOT% && title Orchestrator :8080 && mode con: cols=80 lines=25 && python -m agents.orchestrator"
timeout /t 2 /nobreak >nul

start "Frontend :8086" cmd /k "cd /d %ROOT%frontend && title Frontend :8086 && mode con: cols=80 lines=25 && npm run dev"

echo.
echo 8 terminales lanzadas.
echo Organiza manualmente o usa Win+Flechas para distribuirlas.
echo.
echo   Frontend:     http://localhost:8086
echo   Orchestrator: http://localhost:8080
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
start /b "" python -m agents.mcp_tools.server
start /b "" python -m agents.normalizer
start /b "" python -m agents.feedback
start /b "" python -m agents.specialized --port 8082 --agent-id analyst --role analyst
start /b "" python -m agents.specialized --port 8083 --agent-id seeker --role seeker
start /b "" python -m agents.specialized --port 8087 --agent-id devils_advocate --role devils_advocate
start /b "" python -m agents.specialized --port 8089 --agent-id empiricist --role empiricist
start /b "" python -m agents.specialized --port 8090 --agent-id pragmatist --role pragmatist
start /b "" python -m agents.specialized --port 8088 --agent-id synthesizer --role synthesizer
timeout /t 2 /nobreak >nul
start /b "" python -m agents.orchestrator
timeout /t 1 /nobreak >nul

echo.
echo ==========================================
echo   Agentes arrancados en background.
echo   Arrancando frontend...
echo ==========================================
echo.
echo   Frontend:     http://localhost:8086
echo   Orchestrator: http://localhost:8080
echo.

cd /d "%ROOT%frontend"
npm run dev
goto :end

:stop
echo Cerrando todos los procesos de agentes y frontend...
taskkill /f /fi "WINDOWTITLE eq MCP Tools*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Normalizer*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Analyst*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Seeker*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq DevilsAdvocate*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Empiricist*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Pragmatist*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Synthesizer*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Feedback*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Orchestrator*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Frontend*" >nul 2>&1
REM Fallback: matar procesos python/node en los puertos
for %%p in (8080 8081 8082 8083 8084 8085 8086 8087 8088 8089 8090) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%p ^| findstr LISTENING') do (
        taskkill /f /pid %%a >nul 2>&1
    )
)
echo Todos los procesos cerrados.
goto :end

:end
endlocal
