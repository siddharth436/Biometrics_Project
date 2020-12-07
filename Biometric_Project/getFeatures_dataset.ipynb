# To obtain features using neural network
# Input : Path to image folder
# Writes features to a csv file
# first argument Path to image folder second argument csv_file_path

import os
import sys
import subprocess
import numpy as np
import csv
import cv2

dirPath = sys.argv[1]
outFile = sys.argv[2]
folderList = os.listdir(dirPath)
mainData = []
count = 0
for folderName in folderList:
	folderPath = os.path.join(dirPath,folderName)
	fileList = os.listdir(folderPath)

	for fileName in fileList:
		fullPath = os.path.join(folderPath,fileName)
		save_path = './temp.png'
		count = count + 1
		print (folderName, ",", count)
		img = cv2.imread(fullPath)
		resized_img = cv2.resize(img, (96, 96), interpolation = cv2.INTER_CUBIC)
		cv2.imwrite(save_path,resized_img)
		data = subprocess.check_output(["th","load_model_embedding.lua", save_path])
		dataArr = data.split()
		dataArr = dataArr[:-4]
		mainData.append(dataArr)
dataNparr = np.array(mainData,dtype=float)
np.savetxt(outFile,dataNparr,delimiter=",")
