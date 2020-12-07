# Run with python 2 not python3
# To obtain features using neural network
# Input : Path to image folder
# Writes features to a csv file
# first argument Path to image folder second argument csv_file_path

import os
import sys
import subprocess
import numpy as np
import csv

dirPath = sys.argv[1]
outFile = sys.argv[2]
fileList = os.listdir(dirPath)
mainData = []
for fileName in fileList:
	fullPath = os.path.join(dirPath,fileName)
	data = subprocess.check_output(["th","load_model_embedding.lua", fullPath])
	dataArr = data.split()
	dataArr = dataArr[:-4]
	mainData.append(dataArr)
dataNparr = np.array(mainData,dtype=float)
np.savetxt(outFile,dataNparr,delimiter=",")
