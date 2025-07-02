# main.py
import tkinter as tk
from gui_app import GUIApplication

if __name__ == "__main__":
    root = tk.Tk()    
    app = GUIApplication(root)
    root.mainloop()