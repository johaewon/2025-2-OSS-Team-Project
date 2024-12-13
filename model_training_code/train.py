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


### Accuracy ###
def calculate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images = data['image'].to(device)
            question = data['question'].to(device)
            labels = data['answer'].to(device)
            outputs = model(images, question)
            _, predicted = torch.max(outputs, dim=2)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
