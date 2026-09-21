' Launch run.bat -- or one command it asks for -- with no visible console.
'
' Why this exists
' ---------------
' The \DiscordBot scheduled task repeats every minute. Its action used to be
' "cmd.exe /c run.bat", and cmd.exe allocates a console, so the watchdog drew a
' terminal on the desktop 1,440 times a day even though it almost always finds
' the bot already alive and exits within seconds.
'
' On Windows 11 that console is hosted by Windows Terminal (the default terminal
' application), not conhost, so "start it minimised" tricks do not help: the
' host decides how the window is shown. The only durable fix is to never ask for
' a visible console at all.
'
' wscript.exe is a GUI-subsystem binary -- it has no console of its own -- and
' WshShell.Run(..., 0, ...) starts the child with SW_HIDE. Nothing is drawn at
' any point, by any terminal host, now or after the next Windows update.
'
' Usage
' -----
'   wscript //B //Nologo run_hidden.vbs [args...]
'       Run "<script dir>\run.bat args..." hidden, wait for it, exit with its
'       code. This is the scheduled-task entry point: waiting is what lets the
'       task's "do not start a new instance" policy see a run still in progress.
'
'   wscript //B //Nologo run_hidden.vbs --nowait-env VARNAME
'       Run the command line held in the environment variable VARNAME, hidden,
'       and return immediately. The command travels in the environment rather
'       than in argv on purpose: it contains nested quotes and >> redirection,
'       and argv would have to survive both cmd.exe's parser and VBScript's.
'       The environment is inherited verbatim, so nothing can re-split it.
'
' Reverting is one command: point the task action back at cmd.exe /c run.bat.
Option Explicit

Dim fso, shell, here, args, i, cmd, bat, name

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

here = fso.GetParentFolderName(WScript.ScriptFullName)
Set args = WScript.Arguments

If args.Count >= 1 And LCase(args(0)) = "--nowait-env" Then
    If args.Count < 2 Then
        WScript.Quit 2   ' told to read a variable but not which one
    End If
    name = args(1)
    cmd = shell.Environment("PROCESS")(name)
    If Len(Trim(cmd)) = 0 Then
        WScript.Quit 3   ' the variable was empty; running "" would be silent
    End If
    shell.Run cmd, 0, False
    WScript.Quit 0
End If

' Default mode: run.bat, with any arguments passed straight through.
bat = here & "\run.bat"
If Not fso.FileExists(bat) Then
    WScript.Quit 2
End If

cmd = """" & bat & """"
For i = 0 To args.Count - 1
    ' run.bat's own arguments are bare words (scheduler, forcerun, noelevated).
    ' Quote anyway so a future argument containing a space still arrives whole.
    cmd = cmd & " """ & args(i) & """"
Next

WScript.Quit shell.Run(cmd, 0, True)
