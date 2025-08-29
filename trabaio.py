import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIGURAÇÕES ---
data_dir = "./moedas/moedas_treino"  # caminho para o dataset baixado do Kaggle
model_path = "modelos/modelo_50_epochs.pth"
batch_size = 16
epochs = 50
lr = 0.001

# Mapeamento das classes para valores monetários
coin_values = {
    "100": 1.00,
    "50": 0.50,
    "25": 0.25,
    "10": 0.10,
    "5": 0.05,
    "1": 0.01
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

def detectar_moedas_e_classificar(image_path):
    # Carregar imagem
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Pré-processamento (blur + detecção de bordas)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_value = 0.0

    for i in range(len(contours)):

        c = contours[i]

        # Ignorar contornos muito pequenos (ruído)
        if cv2.contourArea(c) < 250:
            continue

        # Obter bounding box
        x, y, w, h = cv2.boundingRect(c)

        # Recortar moeda
        coin_crop = img_rgb[y:y+h, x:x+w]
        coin_pil = Image.fromarray(coin_crop).convert("RGB")

        # Classificar moeda com o modelo treinado
        img_tensor = data_transforms(coin_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            label = dataset.classes[pred.item()]
            value = coin_values[label]
            total_value += value
            print(f"Moeda reconhecida: {label} -> R$ {coin_values[label]:.2f}")

        # Desenhar bounding box e valor na imagem
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_rgb, f"R$ {value:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Mostrar imagem final com matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Valor total: R$ {total_value:.2f}")
    plt.show()

    return total_value

detectar_moedas_e_classificar("./moedas/25 centavos/Nova/25-centavos-nova (1).JPG")
