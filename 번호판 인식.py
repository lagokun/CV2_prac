# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')

#%%
# 경로에 한글이 있으면 잘 안되는 듯?
car = cv2.imread("C:/AIFactory/ETC/car/19.jpg")

height, width, channel = car.shape

plt.figure(figsize = (12,10))
plt.imshow(car, cmap = 'gray')

#%%
#이미지 처리를 쉽게하기 위해 Gray scale로 변환
gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)

plt.figure(figsize = (12,10))
plt.imshow(gray, cmap = 'gray')

#%%
#이미지 처리를 쉽게하기 위해 Hsv의 V채널만 사용하는 경우도 존재

hsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
gray = hsv[:,:,2]

plt.figure(figsize = (12,10))
plt.imshow(gray, cmap = 'gray')

#%%
#Adaptive Thresholding

#Threshold만 사용하기, 사진을 흑백 처리 시킴
img_thresh = cv2.adaptiveThreshold(gray,
                                   maxValue = 255.0, 
                                   adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresholdType = cv2.THRESH_BINARY_INV, 
                                   blockSize = 19, C = 9	)

#GaussianBlur 처리 후 Threshold 사용
img_blurred = cv2.GaussianBlur(gray, 
                               ksize = (5,5), 
                               sigmaX = 0)

img_blur_thresh = cv2.adaptiveThreshold(img_blurred,
                                        maxValue = 255.0, 
                                        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresholdType = cv2.THRESH_BINARY_INV, 
                                   blockSize = 19, C = 9	)

plt.Figure(figsize = (20,20))
plt.subplot(1,2,1)
plt.title('Threshold only')
plt.imshow(img_thresh, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title('Blur and Threshold')
plt.imshow(img_blur_thresh, cmap='gray')
plt.axis('off')

#%%
#Find Contours 윤곽선 찾고 그리기

contours, _ = cv2.findContours(img_thresh, 
                               mode = cv2.RETR_LIST, 
                               method = cv2.CHAIN_APPROX_SIMPLE)

temp_result = np.zeros((height, width, channel), dtype = np.uint8)

cv2.drawContours(temp_result, contours = contours, contourIdx = -1, color = (255,255,255))

plt.Figure(figsize = (20,20))
plt.imshow(temp_result)
plt.axis('off')

#%%

temp_result = np.zeros((height, width, channel), dtype = np.uint8)

contours_dict = []

for contour in contours:
    x,y,w,h = cv2.boundingRect(contour) # 윤곽선을 감싸는 사각형을 구한다
    cv2.rectangle(temp_result, pt1 = (x,y), pt2 = (x+w, y+h), color = (255,255,255), thickness = 2)
    
    contours_dict.append({'contour': contour,
                          'x':x,
                          'y':y,
                          'w':w,
                          'h':h,
                          'cx':x+(w/2),
                          'cy':y+(h/2)})

plt.Figure(figsize = (20,20))
plt.imshow(temp_result)
plt.axis('off')

#%%
#번호만 건지기

min_area = 80
min_width, min_height = 2, 8
min_ratio, max_ratio = 0.25, 1.0

poss_contours = []

cnt = 0

for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    if area > min_area and d['w'] > min_width and d['h'] > min_height and min_ratio < ratio < max_ratio:
        d['idx'] = cnt # 높은 확률에 인덱스 부여
        cnt += 1
        poss_contours.append(d) 
        
temp_result = np.zeros((height, width, channel), dtype = np.uint8)


    
    