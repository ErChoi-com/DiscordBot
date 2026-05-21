@echo off
setlocal

cd /d "%~dp0"

set "LAST_START_FILE=%~dp0.last_start.txt"
set "START_LOCK_DIR=%~dp0.start.lock"
set "ARG_FORCERUN=0"
set "ARG_SCHEDULER=0"
set "ARG_NOVENV=0"
for %%A in (%*) do (
  if /i "%%~A"=="forcerun" set "ARG_FORCERUN=1"
  if /i "%%~A"=="scheduler" set "ARG_SCHEDULER=1"
  if /i "%%~A"=="novenv" set "ARG_NOVENV=1"
)

2>nul mkdir "%START_LOCK_DIR%"
if errorlevel 1 (
  echo [run.bat] Another launcher is already evaluating startup. Skipping.
  exit /b 0
)

if "%ARG_FORCERUN%"=="1" (
  echo [run.bat] forcerun: killing any existing bot process...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$pidPath = '.bot.pid'; $lockPath = '.bot.lock'; $candidatePids = @(); if (Test-Path $pidPath) { try { $value = (Get-Content $pidPath -ErrorAction Stop | Select-Object -First 1).Trim(); if ($value -match '^[0-9]+$') { $candidatePids += [int]$value } } catch {} }; if (Test-Path $lockPath) { try { $lockRaw = (Get-Content $lockPath -ErrorAction Stop | Select-Object -First 1).Trim(); if ($lockRaw.StartsWith('{')) { $lockJson = $lockRaw | ConvertFrom-Json -ErrorAction Stop; if ($lockJson.pid -match '^[0-9]+$') { $candidatePids += [int]$lockJson.pid } } elseif ($lockRaw -match '^[0-9]+$') { $candidatePids += [int]$lockRaw } } catch {} }; $candidatePids = $candidatePids | Select-Object -Unique; foreach ($procId in $candidatePids) { $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue; if ($proc) { Write-Output ('[run.bat] Killing PID ' + $procId + ' from runtime ownership files.'); Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue } }; Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(pythonw?|py)(\.exe)?$' -and $_.CommandLine -and ($_.CommandLine -match 'src[/\\]app\.py' -or $_.CommandLine -match 'rebuilt_app[/\\]src[/\\]app\.py') } | ForEach-Object { Write-Output ('[run.bat] Killing PID ' + $_.ProcessId + ' (detected via WMI).'); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Start-Sleep -Milliseconds 1500; $remaining = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(pythonw?|py)(\.exe)?$' -and $_.CommandLine -and ($_.CommandLine -match 'src[/\\]app\.py' -or $_.CommandLine -match 'rebuilt_app[/\\]src[/\\]app\.py') }; if ($remaining) { Write-Output '[run.bat] forcerun failed: one or more bot processes survived kill attempts.'; foreach ($proc in $remaining) { Write-Output ('[run.bat] survivor PID ' + $proc.ProcessId + ' :: ' + $proc.Name) }; Write-Output '[run.bat] Hint: process may be elevated (Task Scheduler "Run with highest privileges"). Start this shell as Administrator or disable elevation for that task.'; exit 20 }; foreach ($path in @($pidPath, $lockPath)) { if (Test-Path $path) { try { Remove-Item -Force $path -ErrorAction Stop; Write-Output ('[run.bat] Removed stale file: ' + $path) } catch { Write-Output ('[run.bat] Could not remove ' + $path + ' (it may be in use).') } } }"
  if errorlevel 20 (
    call :release_lock
    exit /b 1
  )
  goto :start_bot
)

echo [run.bat] Checking whether the bot should start...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$existing = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(pythonw?|py)(\.exe)?$' -and $_.CommandLine -and ($_.CommandLine -match 'src[/\\]app\.py' -or $_.CommandLine -match 'rebuilt_app[/\\]src[/\\]app\.py') }; if ($existing) { Write-Output ('[run.bat] Existing bot process(es) detected: ' + $existing.Count + '. Skipping start.'); foreach ($proc in $existing) { Write-Output ('[run.bat] active PID ' + $proc.ProcessId + ' :: ' + $proc.Name) }; exit 10 }; $lastStartFile = $env:LAST_START_FILE; if (Test-Path $lastStartFile) { try { $lastStart = [DateTimeOffset]::Parse((Get-Content $lastStartFile -ErrorAction Stop | Select-Object -First 1).Trim()); $elapsed = (Get-Date) - $lastStart.LocalDateTime; if ($elapsed.TotalHours -lt 24) { Write-Output ('[run.bat] Last start was at ' + $lastStart.ToString('u') + '. Skipping because it has been less than 24 hours.'); exit 11 } } catch { Write-Output '[run.bat] Last start timestamp is invalid. Ignoring file and continuing.' } }"
set "CHECK_EXIT=%ERRORLEVEL%"
if "%CHECK_EXIT%"=="10" (
  call :release_lock
  exit /b 0
)
if "%CHECK_EXIT%"=="11" (
  call :release_lock
  exit /b 0
)

:start_bot

if "%ARG_NOVENV%"=="1" echo [run.bat] novenv: skipping .venv interpreter discovery.
if not "%ARG_NOVENV%"=="1" if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not "%ARG_NOVENV%"=="1" if not defined PYTHON_EXE if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Set-Content -Path $env:LAST_START_FILE -Value (Get-Date -Format o) -Encoding ascii"

echo [run.bat] Starting bot with: "%PYTHON_EXE%" src\app.py
if "%ARG_SCHEDULER%"=="1" (
  echo [run.bat] Scheduler mode: running bot in foreground so Task Scheduler can enforce single instance.
  "%PYTHON_EXE%" src\app.py
) else (
  start "Discord Bot" "%PYTHON_EXE%" src\app.py
  timeout /t 1 /nobreak >nul
)

call :release_lock

echo [run.bat] Done.
endlocal
exit /b 0

:release_lock
if exist "%START_LOCK_DIR%" rmdir "%START_LOCK_DIR%" >nul 2>nul
exit /b 0
