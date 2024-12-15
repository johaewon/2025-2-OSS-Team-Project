import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import speech_recognition as sr
import time
from torchvision import transforms
from transformers import GPT2Tokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, GPT2Model
import soundfile as sf
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from datasets import load_dataset
import soundfile as sf
import pygame

# VQA 및 음성-텍스트 모델
class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size)
        self.efficientnet_out_features = self.efficientnet._fc.in_features
        self.gpt2_hidden_size = self.gpt2.config.hidden_size
        combined_features_size = self.efficientnet_out_features + self.gpt2_hidden_size
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
        image_features = self.efficientnet.extract_features(images)
        image_features = image_features.mean([2, 3])
        image_features = image_features.view(image_features.size(0), -1)
        outputs = self.gpt2(question)
        output_features = outputs.last_hidden_state
        seq_length = output_features.size(1)
        image_features = image_features.unsqueeze(1).expand(-1, seq_length, -1)
        combined = torch.cat([image_features, output_features], dim=-1)
        combined = combined.view(-1, combined.size(-1))
        output = self.classifier(combined)
        output = output.view(-1, seq_length, self.vocab_size)
        return output
    
# 모델 로드
model_path = '../model/horse_EfficientNet2.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)  # 모델을 바로 로드
model.to(device)
model.eval()

# 음성 파일 재생 함수 (pygame 사용)
def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()

# 이미지 전처리 함수
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)
    
# 질문 전처리 및 답변 생성 함수
def generate_answer(model, image_path, question):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    question_encoding = tokenizer.encode_plus(
        question, return_tensors='pt', truncation=True, 
        add_special_tokens=True, max_length=32, padding='max_length'
    )
    question_tensor = question_encoding['input_ids'].to(device)

    with torch.no_grad():
        outputs = model(image_tensor, question_tensor)
        pred_ids = torch.argmax(outputs, dim=2)
        answer_tokens = pred_ids[0, :].tolist()
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # 문장으로 답변 포맷팅
    if question.lower().startswith("what color"):
        answer = f"This color is {answer}."
    elif question.lower().startswith("who"):
        answer = f"This is {answer}."
    else:
        answer = f"It's {answer}."

    return answer

# 텍스트를 음성으로 변환하는 함수
def text_to_speech(text):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Speaker embeddings 로드
    speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(speaker_embeddings_dataset[0]["xvector"]).unsqueeze(0)

    # 텍스트 처리
    inputs = processor(text=text, return_tensors="pt")

    # 음성 생성
    speech = tts_model.generate_speech(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speaker_embeddings=speaker_embeddings,  # 스피커 임베딩 추가
        vocoder=vocoder
    )

    audio_file = 'output_audio.wav'
    sf.write(audio_file, speech.numpy(), samplerate=16000)
    return audio_file

# GUI 애플리케이션 클래스
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
