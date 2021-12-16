import numpy as np
import pandas as pd
import os
import re
from glob import glob
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image,ImageOps
import torch.nn.functional as F
from light_cnn import LightCNN_9Layers
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
from kornia.utils.one_hot import one_hot
import sys
from sklearn import metrics



## focal loss starts    
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss




class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)

## focal loss ends        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("Cuda is available")

age = []
gender = []
race = []
wrong_files = []
filenames = []

folder = glob("UTKFace/*")

for f in folder:
    numbers = re.findall(r'\d+\.*',f)

    if len(numbers) == 4:
        gender.append(int(numbers[1]))
        race.append(int(numbers[2]))
        filenames.append(f)

        if int(numbers[0]) >= 0 and int(numbers[0]) <= 28:
            age.append(0)
        elif int(numbers[0]) >= 29 and int(numbers[0]) <= 56:
            age.append(1)
        elif int(numbers[0]) >= 57 and int(numbers[0]) <= 84:
            age.append(2)
        elif int(numbers[0]) >= 85 and int(numbers[0]) <= 116:
            age.append(3)

    else:
        wrong_files.append(f)

print("wrong_files:",np.array(wrong_files).shape)



age = np.array(age).reshape(len(age),1)
gender = np.array(gender).reshape(len(gender),1)
race = np.array(race).reshape(len(race),1)
filenames = np.array(filenames).reshape(len(filenames),1)

data = np.concatenate((filenames,age,gender,race),axis = 1)
df = pd.DataFrame(data,index=None,columns = ['filename','age','gender','race'])

df_train,df_test = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
df_train,df_test = df_train.reset_index(),df_test.reset_index()
df_train,df_test = df_train.drop(['index'],axis=1),df_test.drop(['index'],axis=1)

#train,validation sets splitting
df_train,df_valid = train_test_split(df_train,test_size=0.1,random_state=22,shuffle=True)
df_train,df_valid = df_train.reset_index(),df_valid.reset_index()
df_train,df_valid = df_train.drop(['index'],axis=1),df_valid.drop(['index'],axis=1)

print("df_train shape:",df_train.shape)
print("df_valid shape:",df_valid.shape)
print("df_test shape:", df_test.shape)

df.to_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face.csv",index=None)
df_train.to_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_train.csv",index=None)
df_valid.to_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_valid.csv",index=None)
df_test.to_csv("/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_test.csv",index=None)



class UTKDataset(Dataset):

    def __init__(self,csv_file,img_dir,transform = None,target_transform = None):

        self.info_csv = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.info_csv)

    def __getitem__(self,idx):
        
        img_path = os.path.join(self.img_dir,self.info_csv.iloc[idx,0])
        image = Image.open(img_path)
        gray_image = ImageOps.grayscale(image)
        labels = self.info_csv.iloc[idx,2]

        if self.transform:
            image = self.transform(gray_image)

        if self.target_transform:
            labels = self.target_transform(labels)

        return image,labels


class ToTransform:

    def __call__(self,image):


        trans = T.Compose([
            T.Resize((64,64)),
            T.ToTensor(),
            T.Normalize(
                mean = [0.5],
                std = [0.5]
            ) 
        ]) 
        image = trans(image)

        return image


class target_transform:

    def __call__(self,y):

        y = torch.tensor(y)
        label_transform = F.one_hot(y, num_classes=2)

        return label_transform



train_csv_file = "/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_train.csv"
valid_csv_file = "/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_valid.csv"
test_csv_file = "/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/UTK_face_test.csv"

img_dir = "/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k"


#train loop
def train(num_epochs,train_dataloader,valid_dataloader,model,criterion,optimizer):
    

    store_train_losses =[]
    store_train_accuracies = []
    num_total_steps = len(train_dataloader)
    

    for epoch in range(0,num_epochs):
        running_loss = 0.0
        running_correct = 0
        train_num_samples = 0

        for i,(imgs,lbls) in enumerate(train_dataloader):
            

            imgs = imgs.to(device)
            lbls = lbls.to(device)

            
            #forward pass
            outputs = model(imgs)
            loss = criterion(outputs,lbls)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_predictions = torch.max(outputs,1)[1]
            train_num_samples += lbls.shape[0]

            running_correct += (train_predictions == lbls).sum().item()
            running_loss += loss.item()


        validation_loss,validation_accuracy = test(valid_dataloader,model,criterion,purpose="valid")
        print(f'epoch: {epoch+1} / {num_epochs} train loss: {running_loss/num_total_steps:.4f} train accuracy: {(running_correct/train_num_samples)*100:.4f}% validation loss: {validation_loss.item():.4f} validation accuracy: {validation_accuracy:.4f}%')
        store_train_losses.append(running_loss/num_total_steps)
        store_train_accuracies.append((running_correct/train_num_samples)*100)
        #save model
        torch.save(model.state_dict(),f"/home/euclid/Desktop/Chiranjeev/DAI/UTK_crop_20k/LCNN_UTK_crop_20k/LightCNN_UTK_crop_20k_model{epoch+1}_focal_loss.pth")

    return model,np.array(store_train_losses),np.array(store_train_accuracies)
        

def test(dataloader,model,criterion,purpose="test"):
    
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        test_loss = 0
        scores = []
        test_labels = []
        store_predictions = []
        
        num_total_steps_test = len(dataloader)
        for imgs,lbls in dataloader:

            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)


            predictions = torch.max(outputs,1)[1]
            num_samples += lbls.shape[0]
            num_correct += (predictions == lbls).sum().item()
            
            if purpose == "test":
                pred_scores = outputs.cpu().numpy()
                for s in pred_scores:
                    scores.append(s)

                test_lbls = lbls.cpu().numpy()
                for tl in test_lbls:
                    test_labels.append(tl)
                
                for p in predictions.cpu().numpy():
                    store_predictions.append(p)

            test_loss += loss

                
                
        accuracy = (num_correct/num_samples)*100
        test_loss = test_loss/num_total_steps_test 

        if purpose == "test": 

            print(f'test loss:{test_loss:.4f} test accuracy: {accuracy}%')
            return test_loss,accuracy,scores,test_labels,store_predictions
    
    return test_loss,accuracy



def tp_fp_tn_fn(y_pred,y_test,condition = None):
    
    TP =0
    FP =0
    FN =0
    TN =0

    for i in range(len(y_pred)):
        
        if y_pred[i] == 1 and y_test[i][0] == 1:
            TP += 1

        elif y_pred[i] == 1 and y_test[i][0] == 0:
            FP += 1
        
        elif y_pred[i] == 0 and y_test[i][0] == 1:
            FN += 1
        
        elif y_pred[i] == 0 and y_test[i][0] == 0:
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
            
        FPR,TPR,Recall,Precision  = tp_fp_tn_fn(np.array(list1),test_labels)
        thresold_FPR_TPR_Recall_Precision.append([th,FPR,TPR,Recall,Precision])
        fprs.append(FPR)
        tprs.append(TPR)
        recall.append(Recall)
        precision.append(Precision)
    
    
    df_thresold_FPR_TPR_Recall_Precision =  pd.DataFrame(np.array(thresold_FPR_TPR_Recall_Precision),index = None,columns = ['thresold','FPR','TPR','Recall','Precision'])
    df_thresold_FPR_TPR_Recall_Precision_str = "df_thresold_FPR_TPR_Recall_Precision_"+ optimizer_num +"_"+ str(num_epochs) + "_focal_loss.csv"
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
    ROC_saving_str_sk = "Sklearn_ROC_Curve_" + optimizer_num +"_"+ str(num_epochs) + "_focal_loss.png"
    plt.savefig(ROC_saving_str_sk)
    plt.close()


    precision_sk, recall_sk, thresholds_roc_pr = metrics.precision_recall_curve(test_labels, np.array(y_pred_test_sk))
    ctr+=1
    plot2 = plt.figure(ctr)
    plt.plot(recall_sk,precision_sk)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    Precision_Recall_str_sk = "Sklearn_Precision_Recall_" + optimizer_num +"_"+ str(num_epochs) + "_focal_loss.png"
    plt.savefig(Precision_Recall_str_sk)
    plt.close()

    #### sklearn ends ####


    ctr+=1
    plot1 = plt.figure(ctr)
    plt.plot(fprs,tprs)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    ROC_saving_str = "ROC_Curve_" + optimizer_num +"_"+ str(num_epochs) + "_focal_loss.png"
    plt.savefig(ROC_saving_str)
    plt.close()

    ctr+=1
    plot2 = plt.figure(ctr)
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    Precision_Recall_str = "Precision_Recall_" + optimizer_num +"_"+ str(num_epochs) + "_focal_loss.png"
    plt.savefig(Precision_Recall_str)
    plt.close()

def loss_acc_plot(num_epochs_array,store_train_losses,store_train_accuracies,num_epochs,optimizer_num,ctr):
    
    ctr+=1
    plot3 = plt.figure(ctr)
    plt.plot(num_epochs_array,store_train_losses)
    plt.xlabel("epochs")
    plt.ylabel("Train loss")
    loss_svaing_str = "EpochvsLoss_"+ optimizer_num +"_"+ str(num_epochs)+"_focal_loss.png"
    plt.savefig(loss_svaing_str)
    plt.close()

    ctr+=1
    plot4 = plt.figure(ctr)
    plt.plot(np.arange(0,num_epochs),store_train_accuracies) 
    plt.xlabel("epochs")
    plt.ylabel("train accuracy")
    acc_svaing_str = "EpochvsAccuracy_"+ optimizer_num +"_"+ str(num_epochs)+"_focal_loss.png"
    plt.savefig(acc_svaing_str)
    plt.close()




if __name__ == "__main__":

    number_of_epochs = [45]
    ctr = 0
    import random
    def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    for num_epochs in number_of_epochs:


        model2 = LightCNN_9Layers(num_classes=2).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(),lr=0.00001,weight_decay=1e-7)
        kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        criterion2 =  FocalLoss(**kwargs)

        set_seed(42)

        train_dataset = UTKDataset(train_csv_file,img_dir,transform = ToTransform())
        valid_dataset = UTKDataset(valid_csv_file,img_dir,transform = ToTransform())
        test_dataset = UTKDataset(test_csv_file,img_dir,transform = ToTransform())

        train_dataloader = DataLoader(train_dataset,batch_size=64,num_workers = 0, shuffle=True,worker_init_fn = np.random.seed(42))
        valid_dataloader = DataLoader(valid_dataset,batch_size=64,num_workers = 0, shuffle=False)
        test_dataloader = DataLoader(test_dataset,batch_size=64,num_workers = 0, shuffle=False)



        print("\nnum_epochs_optimizer2:",num_epochs)

        model2,store_train_losses,store_train_accuracies = train(num_epochs,train_dataloader,valid_dataloader,model2,criterion2,optimizer2)
        print("Training Done!!")
        
        loss,accuracy,scores,test_labels,store_predictions = test(test_dataloader,model2,criterion2)
        print("Testing Done!!")

        print("\nConfusion Matrix\n")
        tp_fp_tn_fn(np.array(store_predictions).reshape(len(store_predictions),1),np.array(test_labels).reshape(len(test_labels),1),"confusion")


        scores = np.array(scores)
        test_labels = np.array(test_labels).reshape(len(test_labels),1)

        dataframe_with_scores = pd.DataFrame(np.hstack((test_labels,scores)),index = None,columns = ['test_labels','male(0)_scores','female(1)_scores'])
        saving_str2 = "dataframe_with_scores_optimizer2_"+str(num_epochs)+"_focal_loss.csv"
        dataframe_with_scores.to_csv(saving_str2,index = None)

        
        ROC_Precision_Recall_Curve(test_labels,scores,num_epochs,"optimizer2",ctr)   
        print("ROC Done!!")
        print("Precision-Recall Done!!")     

        loss_acc_plot(np.arange(0,num_epochs),store_train_losses,store_train_accuracies,num_epochs,"optimizer2",ctr) 
        

        print("Loss graph Done!!")
        print("Accuracy graph Done!!")


        test_csv_read2 = pd.read_csv(test_csv_file).iloc[:,:].values
        file_gender_across_race_age_y_test_pred2 = np.hstack((test_csv_read2,np.array(store_predictions).reshape(len(store_predictions),1)))
        test_gender_across_race_age_y_test_pred2 = pd.DataFrame(file_gender_across_race_age_y_test_pred2,index=None,columns=['filename','age','gender','race','test_predictions'])
        str_opt2 = "test_gender_across_race_age_y_test_pred2_optimizer2_" + str(num_epochs)+"_focal_loss.csv"
        test_gender_across_race_age_y_test_pred2.to_csv(str_opt2,index=None)




        
            


            


