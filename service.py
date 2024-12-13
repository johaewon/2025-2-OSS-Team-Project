import pygame
import soundfile as sf
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch

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