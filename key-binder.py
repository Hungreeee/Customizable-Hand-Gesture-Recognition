from ahk import AHK, Window
import time

ahk = AHK(executable_path="F:\Program Files\AutoHotkey_2.0.11\AutoHotkey64.exe")
# win = Window.from_pid(ahk, pid="32928")        
win = ahk.win_get(title="Lethal Company")
win.activate()

ahk.key_down("w")
time.sleep(2)
ahk.key_up("w")

ahk.key_down("d")
time.sleep(2)
ahk.key_up("d")