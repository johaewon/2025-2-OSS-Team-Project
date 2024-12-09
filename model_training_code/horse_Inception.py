import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models # 이미지
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer, GPT2Model # 텍스트
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


########## Dataset ##########

class VQADataset(Dataset):
    def __init__(self, df, tokenizer, transform, img_path, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.img_path = img_path
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(self.img_path, row['image_id'] + '.jpg') # 이미지
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question'] # 질문
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if not self.is_test:
            answer = row['answer'] # 답변
            answer = self.tokenizer.encode_plus(
                answer,
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            return {
                'image': image.squeeze(),
                'question': question['input_ids'].squeeze(),
                'answer': answer['input_ids'].squeeze()
            }
        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
            }


import torch
from torch import nn
from torchvision import models

class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size

        # Inception v3 모델 로드, 최종 레이어를 Identity로 설정
        self.inception = models.inception_v3(pretrained=True)
        self.inception.fc = nn.Identity()  # 분류기 부분 제거

        # GPT-2 모델 로드
        from transformers import GPT2Model
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size)

        # Inception v3 출력 차원 (2048)과 GPT-2 출력 차원 합산
        combined_features_size = 2048 + self.gpt2.config.hidden_size
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
    # Inception 모델 호출
        outputs = self.inception(images)
        
        # logits 속성을 사용하여 이미지 특성을 가져옵니다.
        image_features = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
    
        # 이미지 특성의 차원 조정
        image_features = image_features.view(image_features.size(0), -1)
    
        # GPT-2 모델을 통한 텍스트 특성 처리
        text_outputs = self.gpt2(question)
        output_features = text_outputs.last_hidden_state
    
        # 이미지와 텍스트 특성 결합
        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1), -1)
        combined = torch.cat([image_features, output_features], dim=-1)
    
        # 최종 출력 계산
        output = self.classifier(combined)
    
        return output

