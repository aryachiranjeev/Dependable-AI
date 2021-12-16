from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
import re
from glob import glob
from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
import sys
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle


UTK_Face_train = pd.read_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_train.csv")
UTK_Face_valid = pd.read_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_valid.csv",)
UTK_Face_test = pd.read_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_test.csv")

UTK_Face_train_numpy = UTK_Face_train.iloc[:,:].values
UTK_Face_test_numpy = UTK_Face_test.iloc[:,:].values


x_train = []
x_test = []
y_train = []
y_test = []


for i in range(len(UTK_Face_train)):
    img_path = "/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/"+str(UTK_Face_train_numpy[i][0])
    image = Image.open(img_path)
    gray_image = ImageOps.grayscale(image)
    gray_image = gray_image.resize((64,64))
    gray_image = (np.asarray(gray_image) - 0.5)/0.5
    gray_image = gray_image.flatten()
    x_train.append(gray_image)
    y_train.append(UTK_Face_train_numpy[i][2])


for j in range(len(UTK_Face_test)):
    img_path = "/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/"+str(UTK_Face_test_numpy[j][0])
    image = Image.open(img_path)
    gray_image = ImageOps.grayscale(image)
    gray_image = gray_image.resize((64,64))
    gray_image = (np.asarray(gray_image) - 0.5)/0.5
    gray_image = gray_image.flatten()
    x_test.append(gray_image)
    y_test.append(UTK_Face_test_numpy[j][2])


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

np.save("x_train_svm.npy",x_train)
np.save("y_train_svm.npy",y_train)
np.save("x_test_svm.npy",x_test)
np.save("y_test_svm.npy",y_test)


# from here rest of the part was run on Googl Colab
 
x_train = np.load("/home/euclid/Desktop/Chiranjeev/DAI/svm/x_train_svm.npy")
x_test = np.load("/home/euclid/Desktop/Chiranjeev/DAI/svm/x_test_svm.npy")
y_train = np.load("/home/euclid/Desktop/Chiranjeev/DAI/svm/y_train_svm.npy")
y_test = np.load("/home/euclid/Desktop/Chiranjeev/DAI/svm/y_test_svm.npy")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


def calibrated(x_train, x_test, y_train,c=1.0):
  model = svm.LinearSVC(C=1.0)
  calibrated = CalibratedClassifierCV(model)
  calibrated.fit(x_train, y_train)
  file1 = "/home/euclid/Desktop/Chiranjeev/DAI/svm/svm_drive.pkl"
  pickle.dump(calibrated,open(file1,'wb'))
  return calibrated.predict_proba(x_test),calibrated.predict_proba(x_train)
 

ypred_test_calibrated_scores,ypred_train_calibrated_scores = calibrated(x_train, x_test, y_train)
y_pred_test = np.argmax(ypred_test_calibrated_scores,axis=1)
y_pred_train = np.argmax(ypred_train_calibrated_scores,axis=1)

accuracy_train = accuracy_score(y_train,y_pred_train) * 100
print('\nTrain accuracy_score:',accuracy_train)

accuracy = accuracy_score(y_test,y_pred_test) * 100
print('\nTest accuracy_score:',accuracy)
print("\n Confusion Matrix")
print(confusion_matrix(y_test,y_pred_test))

def tp_fp_tn_fn(y_pred,y_test,th=None,condition = None):
    
    TP =0
    FP =0
    FN =0
    TN =0

    for i in range(len(y_pred)):
        
        if y_pred[i] == 1 and y_test[i] == 1:
            TP += 1

        elif y_pred[i] == 1 and y_test[i] == 0:
            FP += 1
        
        elif y_pred[i] == 0 and y_test[i] == 1:
            FN += 1
        
        elif y_pred[i] == 0 and y_test[i] == 0:
            TN += 1

        try: 
            TPR = TP / (TP+FN) 
            FPR = FP / (FP+TN)
            Precision =  TP / (TP+FP)
            Recall =  TP / (TP+FN)

        except:
    
            TPR = 0.0
            FPR = 0.0
            Precision = 1.0
            Recall = 0.0

    if condition == "confusion":
        print("|",TP, " | " ,FP,"|")
        print("|",FN, " | " ,TN,"|" )
        print("\n")
        
    return FPR,TPR,Recall,Precision    
         

def ROC_Precision_Recall_Curve(test_labels,scores,num_epochs,optimizer_num,ctr):

    thresold_FPR_TPR_Recall_Precision  = []
    fprs = []
    tprs = []
    recall = []
    precision = []

   
    thresholds = np.linspace(0,1,1000)
    for th in thresholds:
        list1 = []
        for i in range(len(test_labels)):
            if scores[i][1] >= th:
                list1.append(1)
            else:
                list1.append(0)

            
        FPR,TPR,Recall,Precision  = tp_fp_tn_fn(np.array(list1),test_labels,th)
        thresold_FPR_TPR_Recall_Precision.append([th,FPR,TPR,Recall,Precision])
        fprs.append(FPR)
        tprs.append(TPR)
        recall.append(Recall)
        precision.append(Precision)
    
    
    df_thresold_FPR_TPR_Recall_Precision =  pd.DataFrame(np.array(thresold_FPR_TPR_Recall_Precision),index = None,columns = ['thresold','FPR','TPR','Recall','Precision'])
    df_thresold_FPR_TPR_Recall_Precision_str = "/home/euclid/Desktop/Chiranjeev/DAI/svm/"+"df_thresold_FPR_TPR_Recall_Precision_svm_"+ optimizer_num +"_"+ str(num_epochs) + ".csv"
    df_thresold_FPR_TPR_Recall_Precision.to_csv(df_thresold_FPR_TPR_Recall_Precision_str,index = None)

    #### sklearn roc curve starts ####

    y_pred_test_sk = []

    for j in range(len(test_labels)):
        y_pred_test_sk.append(scores[j][1])
    

    fpr_sk, tpr_sk, thresholds_roc_sk = metrics.roc_curve(test_labels, np.array(y_pred_test_sk))
    ctr+=1
    plot1 = plt.figure(ctr)
    plt.plot(fpr_sk,tpr_sk)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    ROC_saving_str_sk = "/home/euclid/Desktop/Chiranjeev/DAI/svm/"+"Sklearn_ROC_Curve_svm_" + optimizer_num +"_"+ str(num_epochs) + ".png"
    plt.savefig(ROC_saving_str_sk)
    plt.close()


    precision_sk, recall_sk, thresholds_roc_pr = metrics.precision_recall_curve(test_labels, np.array(y_pred_test_sk))
    ctr+=1
    plot2 = plt.figure(ctr)
    plt.plot(recall_sk,precision_sk)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    Precision_Recall_str_sk = "/home/euclid/Desktop/Chiranjeev/DAI/svm/"+"Sklearn_Precision_Recall_svm_" + optimizer_num +"_"+ str(num_epochs) + ".png"
    plt.savefig(Precision_Recall_str_sk)
    plt.close()

    #### sklearn ends ####


    ctr+=1
    plot1 = plt.figure(ctr)
    plt.plot(fprs,tprs)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    ROC_saving_str = "/home/euclid/Desktop/Chiranjeev/DAI/svm/"+"ROC_Curve_svm_" + optimizer_num +"_"+ str(num_epochs) + ".png"
    plt.savefig(ROC_saving_str)
    plt.close()

    ctr+=1
    plot2 = plt.figure(ctr)
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    Precision_Recall_str = "/home/euclid/Desktop/Chiranjeev/DAI/svm/"+"Precision_Recall_svm_" + optimizer_num +"_"+ str(num_epochs) + ".png"
    plt.savefig(Precision_Recall_str)
    plt.close()

ctr =0
ROC_Precision_Recall_Curve(y_test,ypred_test_calibrated_scores,0,"svm",ctr)

test_csv_read2 = pd.read_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_test.csv").iloc[:,:].values
file_gender_across_race_age_y_test_pred2 = np.hstack((test_csv_read2,np.array(y_pred_test).reshape(len(y_pred_test),1)))
test_gender_across_race_age_y_test_pred2 = pd.DataFrame(file_gender_across_race_age_y_test_pred2,index=None,columns=['filename','age','gender','race','test_predictions'])
str_opt2 = "/home/euclid/Desktop/Chiranjeev/DAI/svm/"+"test_gender_across_race_age_y_test_pred2_optimizer2_svm.csv"
test_gender_across_race_age_y_test_pred2.to_csv(str_opt2,index=None)