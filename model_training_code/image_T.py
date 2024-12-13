########## 이미지 전처리 함수 ##########

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 차원 추가: [C, H, W] -> [1, C, H, W]

########## 질문 전처리 후 답변 생성 ##########

def generate_answer(model, image_path, question):
    # 이미지 전처리
    image_tensor = transform_image(image_path).to(device)

    # 질문 전처리
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    question_encoding = tokenizer.encode_plus(
        question, return_tensors='pt', truncation=True,
        add_special_tokens=True, max_length=32, padding='max_length'
    )
    question_tensor = question_encoding['input_ids'].to(device)

    # 모델을 사용하여 답변 생성
    with torch.no_grad():
        outputs = model(image_tensor, question_tensor)
        pred_ids = torch.argmax(outputs, dim=2)  # 예측된 토큰 ID
        answer_tokens = pred_ids[0, :].tolist()  # 첫 번째 출력(배치 크기 1)에서 토큰 추출
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer
