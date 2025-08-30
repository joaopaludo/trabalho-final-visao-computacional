import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---
# 1. Defina o caminho para a imagem que você quer analisar
IMAGE_PATH = "moedas/colecao/IMG_9844.JPG"

# 2. Defina os caminhos para o modelo e os nomes das classes
MODEL_PATH = 'modelos/keras_model.h5'
CLASSES = ["1_Real", "50_centavos", "25_centavos", "10_centavos", "5_centavos"]

# 3. Defina o limiar de confiança para a classificação
CONFIDENCE_THRESHOLD = 0.7  # 70%

# 4. Defina a área mínima para um objeto ser considerado uma moeda
MIN_COIN_AREA = 2000
# --------------------

# Carrega o modelo de IA
model = load_model(MODEL_PATH, compile=False)
# Prepara a estrutura de dados para a predição
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Função para pré-processar a imagem para detecção de contornos
def preProcess(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    imgPre = cv2.Canny(imgPre, 90, 140)
    kernel = np.ones((4, 4), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=2)
    imgPre = cv2.erode(imgPre, kernel, iterations=1)
    return imgPre

# Função para classificar a imagem de uma moeda
def DetectarMoeda(img):
    imgMoeda = cv2.resize(img, (224, 224))
    imgMoedaNormalize = (np.asarray(imgMoeda).astype(np.float32) / 127.0) - 1
    data[0] = imgMoedaNormalize
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = CLASSES[index]
    return classe, percent

# --- LÓGICA PRINCIPAL ---
# Carrega a imagem do arquivo
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"Erro: não foi possível carregar a imagem em '{IMAGE_PATH}'")
else:
    # Cria uma cópia da imagem para desenhar os resultados
    output_img = img.copy()
    output_img = cv2.resize(output_img, (640, 480))
    
    # Pré-processa a imagem para encontrar os contornos
    imgPre = preProcess(output_img)
    contours, hi = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    total_value = 0.0
    # Itera sobre cada contorno detectado
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_COIN_AREA:
            # Obtém o retângulo que envolve a moeda
            x, y, w, h = cv2.boundingRect(cnt)
            # Desenha o retângulo na imagem de saída
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Recorta a moeda da imagem original
            recorte = output_img[y:y + h, x:x + w]
            
            # Classifica a moeda recortada
            classe, conf = DetectarMoeda(recorte)
            
            # Se a confiança for alta, soma o valor e exibe o texto
            if conf > CONFIDENCE_THRESHOLD:
                cv2.putText(output_img, str(classe), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if classe == '1_Real': total_value += 1.0
                if classe == '50_centavos': total_value += 0.50
                if classe == '25_centavos': total_value += 0.25
                if classe == '10_centavos': total_value += 0.10
                if classe == '5_centavos': total_value += 0.05

    # Desenha o valor total na imagem
    cv2.rectangle(output_img, (400, 20), (620, 80), (0, 0, 255), -1)
    cv2.putText(output_img, f'R$ {total_value:.2f}', (410, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Exibe as imagens
    cv2.imshow('Resultado Final', output_img)
    cv2.imshow('Pre-processamento', imgPre)
    
    # Espera o usuário pressionar uma tecla para fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()
