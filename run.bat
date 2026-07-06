@echo off
setlocal

cd /d "%~dp0"

set "LAST_START_FILE=%~dp0.last_start.txt"
set "START_LOCK_DIR=%~dp0.start.lock"
set "ARG_FORCERUN=0"
set "ARG_SCHEDULER=0"
for %%A in (%*) do (
  if /i "%%~A"=="forcerun" set "ARG_FORCERUN=1"
  if /i "%%~A"=="scheduler" set "ARG_SCHEDULER=1"
)

if "%ARG_FORCERUN%"=="1" if exist "%START_LOCK_DIR%" (
  echo [run.bat] forcerun: removing stale start lock.
  rmdir "%START_LOCK_DIR%" >nul 2>nul
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
  "$existing = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(pythonw?|py)(\.exe)?$' -and $_.CommandLine -and ($_.CommandLine -match 'src[/\\]app\.py' -or $_.CommandLine -match 'rebuilt_app[/\\]src[/\\]app\.py') }; if ($existing) { $lastStartFile = $env:LAST_START_FILE; if (Test-Path $lastStartFile) { try { $lastStart = [DateTimeOffset]::Parse((Get-Content $lastStartFile -ErrorAction Stop | Select-Object -First 1).Trim()); $elapsed = (Get-Date) - $lastStart.LocalDateTime; if ($elapsed.TotalHours -lt 24) { Write-Output ('[run.bat] Bot is running and last start was ' + $lastStart.ToString('u') + '. Skipping restart.'); exit 10 } } catch {} }; Write-Output ('[run.bat] Bot is running but 24h+ since last start. Graceful restart...'); foreach ($proc in $existing) { Write-Output ('[run.bat] Requesting graceful stop for PID ' + $proc.ProcessId); try { Stop-Process -Id $proc.ProcessId -ErrorAction Stop } catch { Write-Output ('[run.bat] Graceful stop failed, will force kill.') } }; Start-Sleep -Seconds 5; $survivors = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(pythonw?|py)(\.exe)?$' -and $_.CommandLine -and ($_.CommandLine -match 'src[/\\]app\.py' -or $_.CommandLine -match 'rebuilt_app[/\\]src[/\\]app\.py') }; if ($survivors) { Write-Output '[run.bat] Bot still running after graceful wait. Force killing.'; foreach ($proc in $survivors) { Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue } ; Start-Sleep -Milliseconds 1500 }; exit 0 } else { Write-Output '[run.bat] No bot process found. Starting fresh.'; exit 0 }"
set "CHECK_EXIT=%ERRORLEVEL%"
if "%CHECK_EXIT%"=="10" (
  call :release_lock
  exit /b 0
)

:start_bot

echo [run.bat] Syncing ATS company lists...
if defined PYTHON_EXE (
  "%PYTHON_EXE%" sync_ats_companies.py
) else (
  py -3.11 sync_ats_companies.py
)

if not defined PYTHON_EXE (
  where py >nul 2>nul
  if errorlevel 1 (
    echo [run.bat] ERROR: PYTHON_EXE is not set and py launcher was not found.
    echo [run.bat] Set PYTHON_EXE to a non-virtual interpreter path and try again.
    call :release_lock
    exit /b 1
  )
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Set-Content -Path $env:LAST_START_FILE -Value (Get-Date -Format o) -Encoding ascii"

if defined PYTHON_EXE (
  echo [run.bat] Starting bot with: "%PYTHON_EXE%" src\app.py
  if "%ARG_SCHEDULER%"=="1" (
    echo [run.bat] Scheduler mode: running bot in foreground so Task Scheduler can enforce single instance.
    "%PYTHON_EXE%" src\app.py
  ) else (
    start "Discord Bot" "%PYTHON_EXE%" src\app.py
    timeout /t 1 /nobreak >nul
  )
) else (
  echo [run.bat] Starting bot with: py -3.11 src\app.py
  if "%ARG_SCHEDULER%"=="1" (
    echo [run.bat] Scheduler mode: running bot in foreground so Task Scheduler can enforce single instance.
    py -3.11 src\app.py
  ) else (
    start "Discord Bot" py -3 src\app.py
    timeout /t 1 /nobreak >nul
  )
)

call :release_lock

echo [run.bat] Done.
endlocal
exit /b 0

:release_lock
if exist "%START_LOCK_DIR%" rmdir "%START_LOCK_DIR%" >nul 2>nul
exit /b 0
