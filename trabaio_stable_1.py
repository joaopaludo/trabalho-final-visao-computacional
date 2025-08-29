import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# --- CONFIGURAÇÕES ---
data_dir = "./moedas"  # caminho para o dataset baixado do Kaggle
model_path = "modelo.pth"
batch_size = 16
epochs = 5
lr = 0.001

# Mapeamento das classes para valores monetários
coin_values = {
    "1 real": 1.00,
    "50 centavos": 0.50,
    "25 centavos": 0.25,
    "10 centavos": 0.10,
    "5 centavos": 0.05,
    "1 centavo": 0.01,
}

# --- TRANSFORMAÇÕES ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- DATASET & DATALOADER ---
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- MODELO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))
model = model.to(device)

# --- TREINAMENTO OU CARREGAMENTO ---
if os.path.exists(model_path):
    print("Carregando modelo salvo...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Treinando modelo...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo em {model_path}")

# --- AVALIAÇÃO ---
"""
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Acurácia no teste: {100 * correct / total:.2f}%")
"""

# --- FUNÇÃO PARA CALCULAR O VALOR TOTAL ---
def calcular_valor(imagens):
    model.eval()
    total_value = 0.0
    with torch.no_grad():
        for img in imagens:
            img = data_transforms(img).unsqueeze(0).to(device)
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            label = dataset.classes[pred.item()]
            total_value += coin_values[label]
            print(f"Moeda reconhecida: {label} -> R$ {coin_values[label]:.2f}")
    print(f"\nValor total: R$ {total_value:.2f}")
    return total_value


# --- EXEMPLO 1: Uma imagem só ---
img_path = "./moedas/1 real/100-centavos (1).JPG"  # exemplo de uma imagem do dataset
img = Image.open(img_path).convert("RGB")

# Calcula valor de uma única moeda
calcular_valor([img])
