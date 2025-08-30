import cv2
import os
import numpy as np

# --- CONFIGURAÇÕES ---
# 1. Defina a pasta raiz de entrada que contém as subpastas (5, 10, 25...).
PASTA_ENTRADA = "moedas/moedas_treino"

# 2. Defina a pasta raiz de saída onde os recortes serão salvos.
PASTA_SAIDA = "moedas/recortes"

# 3. Limiar mínimo da área do contorno para ser considerado uma moeda.
#    Ajuste este valor se necessário.
AREA_MINIMA = 1540
# --------------------

# Sua função de pré-processamento (não precisa mudar)
def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    imgPre = cv2.Canny(imgPre,90,140)
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre

# --- LÓGICA PRINCIPAL ---

# Verifica se a pasta de saída principal existe; se não, cria-a.
if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)
    print(f"Diretório de saída criado em: '{PASTA_SAIDA}'")

# Loop para percorrer cada subpasta de classe (ex: '5', '10', '25'...)
for nome_classe in os.listdir(PASTA_ENTRADA):
    pasta_classe_entrada = os.path.join(PASTA_ENTRADA, nome_classe)

    # Garante que estamos processando apenas diretórios
    if not os.path.isdir(pasta_classe_entrada):
        continue

    # Cria a pasta da classe correspondente no diretório de saída (ex: 'recortes/5')
    pasta_classe_saida = os.path.join(PASTA_SAIDA, nome_classe)
    if not os.path.exists(pasta_classe_saida):
        os.makedirs(pasta_classe_saida)

    print(f"\n--- Processando a pasta: {pasta_classe_entrada} ---")

    # Loop para percorrer cada arquivo de imagem na pasta da classe
    for nome_arquivo in os.listdir(pasta_classe_entrada):
        caminho_arquivo_entrada = os.path.join(pasta_classe_entrada, nome_arquivo)

        # Carrega a imagem
        img = cv2.imread(caminho_arquivo_entrada)
        
        # Pula se o arquivo não puder ser lido como uma imagem
        if img is None:
            print(f"  [Aviso] Não foi possível ler o arquivo: {nome_arquivo}")
            continue
        
        # Redimensiona a imagem para um tamanho padrão para o pré-processamento
        img = cv2.resize(img, (640, 480))

        # Aplica a sua lógica de recorte
        imgPre = preProcess(img)
        contours, _ = cv2.findContours(imgPre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contador_recortes = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > AREA_MINIMA:
                x, y, w, h = cv2.boundingRect(cnt)
                imgCrop = img[y:y+h, x:x+w]

                # Cria um nome de arquivo único para o recorte
                nome_base = os.path.splitext(nome_arquivo)[0]
                nome_arquivo_saida = f"{nome_base}_crop_{contador_recortes}.jpg"
                caminho_arquivo_saida = os.path.join(pasta_classe_saida, nome_arquivo_saida)
                
                # Salva a imagem recortada
                cv2.imwrite(caminho_arquivo_saida, imgCrop)
                contador_recortes += 1
        
        if contador_recortes > 0:
            print(f"  Processado: {nome_arquivo} -> {contador_recortes} recorte(s) salvo(s).")

print("\nProcesso concluído!")
