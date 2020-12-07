"""
knn 5 fold
"""

from __future__ import division
import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import xlsxwriter

#-----------------------------------------------------------------------------------------------
def allscores(predictions,Y_test):
    count=0
    for i in range(len(predictions)):
        if predictions[i]!=Y_test[i]:
            count+=1
    accuracy=((len(Y_test)-count)/len(Y_test))*100 
    [precision,recall,f1score,support]=precision_recall_fscore_support(Y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(Y_test,predictions).ravel()
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    precision1=precision.mean()
    recall1=recall.mean()
    f1score1=f1score.mean()
    print("accuracy:%0.2f"%accuracy)
    print("precision:%0.2f"%precision1)
    print("recall:%0.2f"%recall1)
    print("f1score:%0.2f"%f1score1)    
    print("sensitivity:%0.2f"%sensitivity)
    print("specificity:%0.2f"%specificity)
    return accuracy,precision1,recall1,sensitivity,specificity,f1score1
    
#----------------------------------------------------------------------------------------------- 

"""
Distance calculation
"""
def distancecalc(row1):
    distance=np.zeros(len(X_train))
    row1v=np.multiply(row1,np.ones([len(X_train),len(X_train[0])]))
    disc=np.zeros(len(distance))
    distance=abs(np.square(row1v)-np.square(X_train))
    disc=[]
    for disrow in distance:
        dis=np.sum(disrow)
        disc.append(dis)
    disc=np.sqrt(disc)    
    disc=np.column_stack((disc,Y_train))
    dicdisc=dict(disc)
    sdisc=sorted(dicdisc)
    findist=[]
    for i in range(9):
        f=(dicdisc[sdisc[i]])
        findist.append(f)
    labelfin=np.array(findist)    
    labelfin=labelfin.astype(int)
    result=(np.bincount(labelfin).argmax())
    return result
#----------------------------------------------------------------------------------------------

dataset1=np.genfromtxt('all_csv_files_Subhasis/james_phelps_non_affine.csv',delimiter=',')
dataset2=np.genfromtxt('all_csv_files_Subhasis/oliver_phelps_non_affine.csv',delimiter=',')
Y_label1=np.ones((len(dataset1),1),dtype=int)
dataset1=np.column_stack((dataset1,Y_label1))
Y_label2=np.zeros((len(dataset2),1),dtype=int)
dataset2=np.column_stack((dataset2,Y_label2))
dataset=np.concatenate((dataset1,dataset2),axis=0)
np.random.shuffle(dataset)
feat=np.array(dataset)
lab=np.array(dataset[:,[128]],dtype='int')
feat=np.delete(feat,128,1)
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(feat,lab,test_size=0.2)
X_train=feat
Y_train=lab


#----------------------------------------------------------------------------------------
kf = KFold(n_splits=10)
accuracynfold=[]
recallnfold=[]
sensitivitynfold=[]
specificitynfold=[]
f1scorenfold=[]
for train_index, test_index in kf.split(X_train):
    X_train1, X_test1 = X_train[train_index], X_train[test_index]
    y_train1, y_test1 = Y_train[train_index], Y_train[test_index]
    
    predictions=[]
    for row in X_test1:
        distance=distancecalc(row)
        predictions.append(distance)
    

    [accuracy,precision,recall,sensitivity,specificity,f1score]=allscores(predictions,y_test1)
    accuracynfold.append(accuracy)
    recallnfold.append(recall)
    sensitivitynfold.append(sensitivity)
    specificitynfold.append(specificity)
    f1scorenfold.append(f1score)
#----------------------------------------------------------------------------------------

workbook = xlsxwriter.Workbook('Output.xlsx')
worksheet = workbook.add_worksheet()

titles = ['accuracy-nfold', 'recall-nfold', 'sensitivity-nfold', 'f1score-nfold']
for i,item in enumerate(titles):
	worksheet.write(0,i,item)
	
for i,item in enumerate(accuracynfold):
	worksheet.write(i+1,0,item)

for i,item in enumerate(recallnfold):
	worksheet.write(i+1,1,item)
	
for i,item in enumerate(sensitivitynfold):
	worksheet.write(i+1,2,item)
	
for i,item in enumerate(f1scorenfold):
	worksheet.write(i+1,3,item)

worksheet.write(0,5,'Mean Accuracy')
worksheet.write(1,5,(sum(accuracynfold)/len(accuracynfold)))

workbook.close()

