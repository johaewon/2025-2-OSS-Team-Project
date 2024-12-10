def calculate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images = data['image'].to(device)
            question = data['question'].to(device)
            labels = data['answer'].to(device)
            
            outputs = model(images, question)  # [batch, sequence, vocab]
            _, predicted = torch.max(outputs, dim=2)  # 가장 높은 확률을 가진 토큰 선택
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
