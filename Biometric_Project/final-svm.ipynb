from __future__ import division
import numpy as np
from sklearn import cross_validation,svm,preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import xlsxwriter

#-----------------------------------------------------------------------------------------------
def getresults(clf,X_test):
    resultsprob = clf.predict_proba(X_test)
    """saves the probabilities"""
    np.savetxt("output_probabilities.csv", resultsprob, delimiter=",")
    predictions=[]
    predictions=clf.predict(X_test)
    return resultsprob,predictions 

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
    print ("accuracy:%0.2f"%accuracy)
    print("precision:%0.2f"%precision1)
    print("recall:%0.2f"%recall1)
    print("f1score:%0.2f"%f1score1)    
    print("sensitivity:%0.2f"%sensitivity)
    print("specificity:%0.2f"%specificity)
    return accuracy,precision1,recall1,sensitivity,specificity,f1score1
    
#-----------------------------------------------------------------------------------------------    
dataset1=np.genfromtxt('all_csv_files_Subhasis/aaron_ashmore.csv',delimiter=',')
dataset2=np.genfromtxt('all_csv_files_Subhasis/shawn_ashmore.csv',delimiter=',')
Y_label1=np.ones((len(dataset1),1),dtype=int)
dataset1=np.column_stack((dataset1,Y_label1))
Y_label2=np.zeros((len(dataset2),1),dtype=int)
dataset2=np.column_stack((dataset2,Y_label2))
dataset=np.concatenate((dataset1,dataset2),axis=0)
np.random.shuffle(dataset)
#-----------------------------------------------------------------------------------------------
feat=np.array(dataset)
lab=np.array(dataset[:,[128]],dtype='int')
feat=np.delete(feat,128,1)


scaler=preprocessing.MinMaxScaler()
scaler.fit(feat)
feat=scaler.transform(feat)

results=[]
X_train=feat
Y_train=lab
#-----------------------------------------------------------------------------------------------

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',probability=True)
clf.fit(X_train,Y_train.ravel())


#-----------------------------------------------------------------------------------------------
kf = KFold(n_splits=10)
accuracynfold=[]
recallnfold=[]
sensitivitynfold=[]
specificitynfold=[]
f1scorenfold=[]
for train_index, test_index in kf.split(X_train):
    X_train1, X_test1 = X_train[train_index], X_train[test_index]
    y_train1, y_test1 = Y_train[train_index], Y_train[test_index]
    clf.fit(X_train1,y_train1.ravel())  
    [probs,predictions]=getresults(clf,X_test1)
    [accuracy,precision,recall,sensitivity,specificity,f1score]=allscores(predictions,y_test1)
    accuracynfold.append(accuracy)
    recallnfold.append(recall)
    sensitivitynfold.append(sensitivity)
    specificitynfold.append(specificity)
    f1scorenfold.append(f1score)
#-----------------------------------------------------------------------------------------------    
finalacc=np.mean(accuracynfold)
print("mean accuracy:%0.2f"%finalacc)  

#-----------------------------------------------------------------------------------------------
#output data to worksheet

#Workbook to store the data
workbook = xlsxwriter.Workbook('Output.xlsx')
worksheet = workbook.add_worksheet()

titles = ['accuracynfold', 'recallnfold', 'sensitivitynfold', 'f1scorenfold']
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
worksheet.write(1,5,finalacc)

workbook.close()
  
