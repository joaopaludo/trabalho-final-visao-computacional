import tensorflow as tf
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model_path = "modelos/savedmodel" 
model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

class_names = open("modelos/labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)


while True:
    ret, image = camera.read()

    image_resized = cv2.resize(image, (450, 450), interpolation=cv2.INTER_AREA)

    cv2.imshow("Webcam Image", image)

    image_normalized = (image_resized.astype(np.float32) / 127.5) - 1

    image_batch = np.expand_dims(image_normalized, axis=0)

    prediction_dict = model(image_batch)

    predictions = prediction_dict['sequential_7'] 

    index = np.argmax(predictions)
    class_name = class_names[index]
    confidence_score = predictions[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
