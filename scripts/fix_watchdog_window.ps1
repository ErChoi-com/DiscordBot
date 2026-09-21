<#
.SYNOPSIS
    Stop the \DiscordBot watchdog task from drawing a terminal window every minute.

.DESCRIPTION
    The \DiscordBot scheduled task repeats every 60 seconds. Its action was

        cmd.exe /c <repo>\run.bat

    and cmd.exe allocates a console, so the watchdog drew a window 1,440 times a
    day even though it almost always finds the bot already running and exits
    within seconds. On Windows 11 that console is hosted by Windows Terminal
    rather than conhost, so none of the "start it minimised / hide the console
    window" workarounds apply: GetConsoleWindow() under Windows Terminal is a
    hidden proxy window, and hiding it changes nothing on screen.

    The only durable fix is to never ask for a console. This script repoints the
    task at run_hidden.vbs, which runs under wscript.exe -- a GUI-subsystem
    binary that allocates no console at all and starts run.bat with SW_HIDE.
    Nothing is drawn, by any terminal host, now or after the next Windows
    update.

    The task also gains the "scheduler" argument, so run.bat uses its existing
    foreground branch. That is what lets Task Scheduler's IgnoreNew policy
    enforce a single instance, and it means no detached child is needed.

.NOTES
    The task was registered by an administrator, so the current user has only
    Read access to it and cannot change it. This script therefore relaunches
    itself elevated. Everything it changes is one task action; triggers,
    principal, and settings are left exactly as they are.

    To revert:
        $t = Get-ScheduledTask -TaskPath '\' -TaskName 'DiscordBot'
        $a = New-ScheduledTaskAction -Execute 'cmd.exe' `
               -Argument '/c C:\Users\ernes\.vscode\discordbot\rebuilt_app\run.bat' `
               -WorkingDirectory 'C:\Users\ernes\.vscode\discordbot\rebuilt_app'
        Set-ScheduledTask -TaskPath '\' -TaskName 'DiscordBot' -Action $a
#>
[CmdletBinding()]
param(
    # Set when the script has already relaunched itself with admin rights, so a
    # failure to elevate cannot turn into an infinite chain of UAC prompts.
    [switch]$Elevated
)

$ErrorActionPreference = 'Stop'

$TaskPath = '\'
$TaskName = 'DiscordBot'

# Resolve the repo from this script's own location rather than hard-coding it,
# so the fix still applies after the checkout is moved or renamed.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Launcher = Join-Path $RepoRoot 'run_hidden.vbs'

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal]$id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Path -LiteralPath $Launcher)) {
    Write-Error "Launcher not found: $Launcher`nRun this from the repo's scripts\ directory."
}

if (-not (Test-Admin)) {
    if ($Elevated) {
        # Already came back from a UAC round trip without gaining rights.
        # Trying again would just prompt forever.
        Write-Error 'Elevation was declined or did not take effect. Nothing was changed.'
    }
    Write-Host 'This task is owned by an administrator; requesting elevation...'
    $args = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', $MyInvocation.MyCommand.Path, '-Elevated'
    )
    $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $args `
                       -Verb RunAs -Wait -PassThru
    exit $p.ExitCode
}

# The elevated run happens in its own window, which closes the instant it is
# done, and -Verb RunAs cannot be combined with output redirection. Without a
# transcript a failure here leaves nothing but an exit code behind.
$LogPath = Join-Path $env:TEMP 'fix_watchdog_window.log'
try { Start-Transcript -Path $LogPath -Force | Out-Null } catch { }

Write-Host "Repo:     $RepoRoot"
Write-Host "Launcher: $Launcher"
Write-Host "Log:      $LogPath"

$task = Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName
$before = ($task.Actions | ForEach-Object { "$($_.Execute) $($_.Arguments)" }) -join '; '
Write-Host "Before:   $before"

$action = New-ScheduledTaskAction `
    -Execute 'wscript.exe' `
    -Argument "//B //Nologo `"$Launcher`" scheduler" `
    -WorkingDirectory $RepoRoot

# The task shipped with a single BootTrigger carrying the one-minute
# repetition. That works until something edits the task: editing re-arms the
# triggers, a BootTrigger cannot fire again without a boot, and the repetition
# hangs off that trigger -- so the watchdog goes silent (NextRunTime empty)
# until the machine is next restarted. Changing the action alone is enough to
# trigger it, which is exactly what this script does.
#
# Keep the BootTrigger, and add a time-based trigger that repeats forever from
# a moment already in the past. That one arms immediately and re-arms on its
# own, so the watchdog survives future edits, a missed boot event, and this
# script being run twice.
$repeat = New-TimeSpan -Minutes 1

# The repetition rides on the clock trigger, not the boot trigger. A trigger
# built by New-ScheduledTaskTrigger -AtStartup has a null Repetition property,
# so it cannot carry one without dropping to raw CIM -- and it does not need to:
# the clock trigger already repeats every minute, boot included. The boot
# trigger stays only so the watchdog still has something to fire on if the clock
# trigger is ever removed.
$bootTrigger = New-ScheduledTaskTrigger -AtStartup

# Start boundary is midnight today, i.e. already in the past, so the repetition
# is armed the moment the task is written rather than at some future wall-clock
# time.
#
# No -RepetitionDuration: leaving the Duration element out of the task XML is
# what "repeat forever" looks like, and it is what this task already did. Do not
# be tempted to pass [TimeSpan]::Zero to say the same thing -- that serialises as
# Duration:PT0S, which the task XML schema rejects outright.
$clockTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
                                         -RepetitionInterval $repeat

Set-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName `
                  -Action $action -Trigger @($bootTrigger, $clockTrigger) | Out-Null

$after = (Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName).Actions |
    ForEach-Object { "$($_.Execute) $($_.Arguments)" }
Write-Host "After:    $($after -join '; ')"

if ($after -notmatch 'wscript\.exe') {
    Write-Error 'The action did not change. The task may be locked by policy.'
}

# An armed repeating trigger reports a NextRunTime. An empty one means the task
# is dormant until reboot, which is the failure this script exists to avoid, so
# check rather than assume.
$info = Get-ScheduledTaskInfo -TaskPath $TaskPath -TaskName $TaskName
Write-Host "NextRun:  $($info.NextRunTime)"
if (-not $info.NextRunTime) {
    Write-Error ('The task has no armed next run, so the watchdog would stay ' +
                 'dormant until the next boot. Check the triggers in taskschd.msc.')
}

Write-Host ''
Write-Host 'Done. The watchdog now runs with no console, so no window is drawn,'
Write-Host 'and it re-arms on a clock trigger instead of only at boot.'
