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


########## Model ##########

class VQAModel(nn.Module):
    def __init__(self, vocab_size):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size

        ## 수정 ##
        #self.resnet = models.resnet50(pretrained=True)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size) # 추가한 [PAD] 토큰 반영

        ## 추가 ##
        # EfficientNet 출력 차원
        self.efficientnet_out_features = self.efficientnet._fc.in_features
        # GPT2 출력 차원
        self.gpt2_hidden_size = self.gpt2.config.hidden_size
        
        ## 수정 ##
        #combined_features_size = 1000 + self.gpt2.config.hidden_size # resnet 출력 차원 + gpt2 출력 차원
        combined_features_size = self.efficientnet_out_features + self.gpt2_hidden_size # efficientnet 출력 차원 + gpt2 출력 차원
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question):
        ## 수정 ##
        image_features = self.efficientnet.extract_features(images)
        image_features = image_features.mean([2, 3])  # Global Average Pooling 사용해서 차원 맞추기
        image_features = image_features.view(image_features.size(0), -1)

        # GPT2 Feature Extraction
        outputs = self.gpt2(question)
        output_features = outputs.last_hidden_state  # [batch, sequence, hidden]

        ## 수정 & 추가 ##
        seq_length = output_features.size(1)
        image_features = image_features.unsqueeze(1).expand(-1, seq_length, -1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]

        ## Flatten the combined features ##
        combined = combined.view(-1, combined.size(-1))
        output = self.classifier(combined)
        output = output.view(-1, seq_length, self.vocab_size)

        return output
        
