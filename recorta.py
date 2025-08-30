import cv2
import os
import numpy as np

def preProcess(img):
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    imgPre = cv2.Canny(imgPre,90,140)
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=2)
    imgPre = cv2.erode(imgPre,kernel,iterations=1)
    return imgPre

img = cv2.imread("moedas/moedas_treino/5/5-back (1).JPG")
img = cv2.resize(img, (640, 480))
imgPre = preProcess(img)
contours, _ = cv2.findContours(imgPre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 1540:
        x, y, w, h = cv2.boundingRect(cnt)
        imgCrop = img[y:y+h, x:x+w]
        cv2.imwrite(f"recortes_de_moedas/back_{x}_{y}.JPG", imgCrop)
