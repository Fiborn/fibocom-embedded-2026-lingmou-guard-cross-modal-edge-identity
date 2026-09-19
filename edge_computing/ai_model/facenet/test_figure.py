from matplotlib import pyplot as plt
import numpy as np

with open('fpr_tpr/1e-2/fpr_vsdn_x2_k7andk7_withoutedloss.txt','r') as f:
    lines=f.readlines()
    fpr1=[]
    for line in lines:
        line = line.strip()
        fpr1.append(line)
with open('fpr_tpr/1e-2/tpr_vsdn_x2_k7andk7_withoutedloss.txt','r') as f1:
    lines=f1.readlines()
    tpr1=[]
    for line in lines:
        line = line.strip()
        tpr1.append(line)
for i in range(len(fpr1)):
    fpr1[i] = float(fpr1[i])
    tpr1[i]=float(tpr1[i])

with open('fpr_tpr/1e-2/fpr_vsdn_x2_k7andk3_withoutedloss.txt','r') as f:
    lines=f.readlines()
    fpr2=[]
    for line in lines:
        line = line.strip()
        fpr2.append(line)
with open('fpr_tpr/1e-2/tpr_vsdn_x2_k7andk3_withoutedloss.txt','r') as f1:
    lines=f1.readlines()
    tpr2=[]
    for line in lines:
        line = line.strip()
        tpr2.append(line)
for i in range(len(fpr2)):
    fpr2[i] = float(fpr2[i])
    tpr2[i]=float(tpr2[i])

with open('fpr_tpr/1e-2/fpr_vsdn_x2_k7andk3_withedloss.txt','r') as f:
    lines=f.readlines()
    fpr3=[]
    for line in lines:
        line = line.strip()
        fpr3.append(line)
with open('fpr_tpr/1e-2/tpr_vsdn_x2_k7andk3_withedloss.txt','r') as f1:
    lines=f1.readlines()
    tpr3=[]
    for line in lines:
        line = line.strip()
        tpr3.append(line)
for i in range(len(fpr3)):
    fpr3[i] = float(fpr3[i])
    tpr3[i]=float(tpr3[i])

# plt.plot([0, 1], [1e-4, 1e-1], color='navy',label='ROC curve (area = %0.2f)', linestyle='--')
plt.plot(fpr1, tpr1, label='1')
plt.plot(fpr2, tpr2, label='2')
plt.plot(fpr3, tpr3, label='3')
plt.xlim([1e-4, 1e-1])
plt.ylim([0.0, 1.00])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
# plt.savefig('./figure/vsdn_x2_1e-2.png')