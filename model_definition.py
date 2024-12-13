import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from transformers import GPT2Model

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
model_path = './model_training_code/EfficientNet.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)  
model.to(device)
model.eval()
