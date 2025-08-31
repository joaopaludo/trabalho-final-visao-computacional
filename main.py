import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---

# IMAGE_PATH = "moedas/25 centavos/Nova/25-centavos-nova (2).JPG"
# IMAGE_PATH = "moedas/1 real/100-centavos (3).JPG"
# IMAGE_PATH = "moedas/5 centavos/nova/5-centavos-nova (8).JPG"
# IMAGE_PATH = "moedas/10 centavos/nova/10-centavos-nova (2).JPG"
# IMAGE_PATH = "moedas/50 centavos/50-centavos (3).JPG"
IMAGE_PATH = "moedas/teste1.jpg"
# IMAGE_PATH = "moedas/teste2.jpg"
# IMAGE_PATH = "moedas/teste3.jpg"

MODEL_PATH = 'modelos/keras_model.h5'
CLASSES = ["1_Real", "50_centavos", "25_centavos", "10_centavos", "5_centavos"]
CONFIDENCE_THRESHOLD = 0.3  # 30%
MINIMUM_COIN_AREA = 2000
# --------------------

model = load_model(MODEL_PATH, compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Função para pré-processar a imagem para detecção de contornos
def preProcessar(img):
    imgPreprocessed = cv2.GaussianBlur(img, (5, 5), 3)
    imgPreprocessed = cv2.Canny(imgPreprocessed, 90, 140)
    kernel = np.ones((2, 2), np.uint8)
    imgPreprocessed = cv2.dilate(imgPreprocessed, kernel, iterations=2)
    imgPreprocessed = cv2.erode(imgPreprocessed, kernel, iterations=1)
    return imgPreprocessed

# Função para classificar a imagem de uma moeda
def detectarMoeda(img):
    imgResized = cv2.resize(img, (224, 224))
    imgNormalized = (np.asarray(imgResized).astype(np.float32) / 127.0) - 1
    data[0] = imgNormalized
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = CLASSES[index]
    return classe, percent

# Lê a imagem do arquivo
img = cv2.imread(IMAGE_PATH)

if img is None:
    print(f"Erro: não foi possível carregar a imagem em '{IMAGE_PATH}'")
else:
    imagemSaida = img.copy()
    imagemSaida = cv2.resize(imagemSaida, (640, 480))

    imgPreprocessada = preProcessar(imagemSaida)
    contours, _ = cv2.findContours(imgPreprocessada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valorTotal = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MINIMUM_COIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imagemSaida, (x, y), (x + w, y + h), (0, 255, 0), 2)

            recorte = imagemSaida[y:y + h, x:x + w]

            classe, confiabilidade = detectarMoeda(recorte)

            if confiabilidade > CONFIDENCE_THRESHOLD:
                cv2.putText(imagemSaida, str(classe) + " " + f'{confiabilidade:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                if classe == '1_Real': valorTotal += 1.0
                if classe == '50_centavos': valorTotal += 0.50
                if classe == '25_centavos': valorTotal += 0.25
                if classe == '10_centavos': valorTotal += 0.10
                if classe == '5_centavos': valorTotal += 0.05

    cv2.rectangle(imagemSaida, (200, 20), (20, 80), (0, 0, 255), -1)
    cv2.putText(imagemSaida, f'R$ {valorTotal:.2f}', (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow('Resultado final', imagemSaida)
    cv2.imshow('Pre-processamento', imgPreprocessada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
