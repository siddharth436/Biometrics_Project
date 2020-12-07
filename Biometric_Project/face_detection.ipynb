# run as python3 face_detection.py 'path_of_original_image' 'path_of_aligned_image'

import cv2
import numpy as np
import os
import sys

haar_model = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

dir_name = sys.argv[1]
aligned_directory = sys.argv[2]

file_names = os.listdir(dir_name)

for i in range(0,len(file_names)):
	path = dir_name+file_names[i]
#	print (path)
	img = cv2.imread(path)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face_coordinates = haar_model.detectMultiScale(img_gray, 1.3, 5)

	for (x,y,w,h) in face_coordinates:
		cv2.rectangle(img,(x-10,y-10),(x+int(np.round(1.3*w)),int(np.round(y+1.3*h))),(255,0,0),2)
		roi_color = img[y:y+h, x:x+w]
		resized_img = cv2.resize(roi_color, (96, 96), interpolation = cv2.INTER_CUBIC)
#	print (resized_img.shape[0], resized_img.shape[1])
		save_path = aligned_directory+'img_'+str(i)+'.png'
		print (save_path)
		cv2.imwrite(save_path,resized_img)


