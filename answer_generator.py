import torch
from transformers import GPT2Tokenizer
from torchvision import transforms
from PIL import Image

def generate_answer(model, image_path, question):
    """
    이미지와 질문을 받아서 답변을 생성하는 함수
    
    Args:
        model: 학습된 VQA 모델
        image_path (str): 이미지 파일 경로
        question (str): 사용자의 질문
        
    Returns:
        str: 생성된 답변 문장
    """
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 질문 전처리
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    question_encoding = tokenizer.encode_plus(
        question, return_tensors='pt', truncation=True, 
        add_special_tokens=True, max_length=32, padding='max_length'
    )
    question_tensor = question_encoding['input_ids'].to(device)

    # 모델 추론
    with torch.no_grad():
        outputs = model(image_tensor, question_tensor)
        pred_ids = torch.argmax(outputs, dim=2)
        answer_tokens = pred_ids[0, :].tolist()
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # 답변 포맷팅
    if question.lower().startswith("what color"):
        answer = f"This color is {answer}."
    elif question.lower().startswith("who"):
        answer = f"This is {answer}."
    else:
        answer = f"It's {answer}."

    return answer