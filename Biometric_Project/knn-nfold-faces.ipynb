"""
knn 5 fold
"""

import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

#-----------------------------------------------------------------------------------------------
def allscores(predictions,Y_test):
    count=0
    for i in range(len(predictions)):
        if predictions[i]!=Y_test[i]:
            count+=1
    accuracy=((len(Y_test)-count)/len(Y_test))*100 
    [precision,recall,f1score,support]=precision_recall_fscore_support(Y_test, predictions)
    precision1=precision.mean()
    recall1=recall.mean()
    f1score1=f1score.mean()
    print("accuracy:%0.2f"%accuracy)
    print("precision:%0.2f"%precision1)
    print("recall:%0.2f"%recall1)
    print("f1score:%0.2f"%f1score1)    
    return accuracy,precision1,recall1,f1score1
    
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

dataset=np.genfromtxt('D:\Study Files\My Projects\SMAI\\faces95.csv',delimiter=',')
labels=np.genfromtxt('D:\Study Files\My Projects\SMAI\\faces95_labes.csv',delimiter=',',dtype=str)
labels=np.delete(labels,1,axis=1)
le=preprocessing.LabelEncoder()
le.fit(labels)
labels=le.transform(labels)
feat=np.array(dataset)
X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(feat,labels,test_size=0.2)
X_train=feat
Y_train=labels


#----------------------------------------------------------------------------------------
kf = KFold(n_splits=10)
accuracynfold=[]
recallnfold=[]
f1scorenfold=[]
for train_index, test_index in kf.split(X_train):
    X_train1, X_test1 = X_train[train_index], X_train[test_index]
    y_train1, y_test1 = Y_train[train_index], Y_train[test_index]
    
    predictions=[]
    for row in X_test1:
        distance=distancecalc(row)
        predictions.append(distance)
    

    [accuracy,precision,recall,f1score]=allscores(predictions,y_test1)
    accuracynfold.append(accuracy)
    recallnfold.append(recall)
    f1scorenfold.append(f1score)



#----------------------------------------------------------------------------------------

