import tensorflow as tf
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model_path = "modelos/savedmodel" 
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
class_names = open("modelos/labels.txt", "r").readlines()

# Caminho da imagem a verificar
image_path = "moedas/moedas/25/back/25-back (24).JPG"
image = cv2.imread(image_path)

if image is None:
    print(f"Erro: não foi possível carregar a imagem em '{image_path}'")
else:
    image_resized = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

    image_normalized = (image_resized.astype(np.float32) / 127.5) - 1

    image_batch = np.expand_dims(image_normalized, axis=0)

    prediction_dict = model(image_batch)

    predictions = prediction_dict['sequential_11'] 

    index = np.argmax(predictions)
    class_name = class_names[index]
    confidence_score = predictions[0][index]

    print(f"Arquivo: {image_path}")
    print("Classe:", class_name[2:].strip())
    print("Confiança:", f"{confidence_score*100:.2f}%")

    # --- VISUALIZAÇÃO ---
    label_text = f"Classe: {class_name[2:].strip()} ({confidence_score*100:.2f}%)"
    cv2.putText(image_resized, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Resultado da Imagem", image_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
