import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

def text_to_speech(text):
    """
    텍스트를 음성으로 변환하는 함수
    
    Args:
        text (str): 음성으로 변환할 텍스트
        
    Returns:
        str: 생성된 음성 파일의 경로
    """
    # 필요한 모델과 프로세서 로드
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
        speaker_embeddings=speaker_embeddings,
        vocoder=vocoder
    )

    # 음성 파일 저장
    audio_file = 'output_audio.wav'
    sf.write(audio_file, speech.numpy(), samplerate=16000)
    
    return audio_file