import tkinter as tk
import cv2
from PIL import Image, ImageTk

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.lbl_instruction = tk.Label(window, text="Press 'Capture' to start processing.", font=('Helvetica', 14))
        self.lbl_instruction.grid(row=1, column=0, columnspan=2, pady=5)

        self.lbl_result = tk.Label(window, text="", font=('Helvetica', 12))
        self.lbl_result.grid(row=2, column=0, columnspan=2, pady=5)

        self.lbl_answer = tk.Label(window, text="", wraplength=500, font=('Helvetica', 12))
        self.lbl_answer.grid(row=3, column=0, columnspan=2, pady=5)

        self.btn_snapshot = ttk.Button(window, text="Capture", command=self.capture, width=20)
        self.btn_snapshot.grid(row=4, column=0, padx=5, pady=10)

        self.btn_reset = ttk.Button(window, text="Reset", command=self.reset, width=20)
        self.btn_reset.grid(row=4, column=1, padx=5, pady=10)

        self.delay = 15
        self.running = False
        self.update_thread = threading.Thread(target=self.update)
        self.update_thread.start()

        self.window.mainloop()

    def capture(self):
        if self.running:
            return
        ret, frame = self.vid.read()
        if ret:
            image_path = "frame-captured.jpg"
            cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.lbl_instruction.config(text="Speak now:")
            self.running = True
            threading.Thread(target=self.process_frame, args=(image_path,)).start()

    def reset(self):
        self.running = False
        self.lbl_instruction.config(text="Press 'Capture' to start processing.")
        self.lbl_result.config(text="")
        self.lbl_answer.config(text="")
        self.canvas.delete("all")
        if self.vid.isOpened():
            self.vid.release()
        self.vid = cv2.VideoCapture(self.video_source)

    def update(self):
        while True:
            if not self.running:
                ret, frame = self.vid.read()
                if ret:
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            time.sleep(self.delay / 1000.0)

    def process_frame(self, image_path):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language='en-US')
            self.lbl_result.config(text=f"Recognized text: {text}")
            answer = generate_answer(model, image_path, text)
            self.lbl_answer.config(text=f"Answer: {answer}")
            audio_path = text_to_speech(answer)
            play_audio(audio_path)
        except sr.UnknownValueError:
            self.lbl_result.config(text="Could not understand audio.")
        except sr.RequestError as e:
            self.lbl_result.config(text=f"Could not request results; {e}")
        except Exception as e:
            self.lbl_result.config(text=f"Error: {e}")
        self.running = False

if __name__ == "__main__":
    root = tk.Tk()
    App(root, "Tkinter and OpenCV")
