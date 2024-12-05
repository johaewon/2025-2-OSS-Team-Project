import tkinter as tk
import cv2
from PIL import Image, ImageTk

class App:
   def __init__(self, window):
       self.window = window
       self.vid = cv2.VideoCapture(0)
       
       self.canvas = tk.Canvas(window)
       self.canvas.pack()
       
       self.btn_capture = tk.Button(window, text="Capture", command=self.capture)
       self.btn_capture.pack()
       
       self.update()
       self.window.mainloop()
       
   def capture(self):
       _, frame = self.vid.read()
       cv2.imwrite("frame.jpg", frame)
       
   def update(self):
       _, frame = self.vid.read()
       self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
       self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
       self.window.after(10, self.update)

if __name__ == "__main__":
   App(tk.Tk())