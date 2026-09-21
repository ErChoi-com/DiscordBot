@echo off
rem Double-click entry point for scripts\fix_watchdog_window.ps1.
rem
rem That script repoints the \DiscordBot scheduled task away from
rem "cmd.exe /c run.bat" -- which draws a terminal window every 60 seconds --
rem and at run_hidden.vbs, which runs under wscript.exe and allocates no console
rem at all. The task is owned by an administrator, so the script asks for
rem elevation itself; expect one UAC prompt.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fix_watchdog_window.ps1"
echo.
pause
