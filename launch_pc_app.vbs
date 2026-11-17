Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "c:\Users\DELL\Downloads\sar_narrative_generator"
WshShell.Run "python app_gui.py", 1, False
