import torch
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,LabelBinarizer,label_binarize


batch_size=32

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

train_dataset = mnist.MNIST('../data', train=True, transform=transform,download=False)
test_dataset = mnist.MNIST('../data', train=False,transform=transform,download=False)

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)


class BP:
    def __init__(self,layers):
        self.w1=np.random.random([layers[0],layers[1]])*2-1
        self.w2=np.random.random([layers[1],layers[2]])*2-1
        self.w3=np.random.random([layers[2],layers[3]])*2-1

        self.b1=np.zeros([layers[1]])
        self.b2=np.zeros([layers[2]])
        self.b3=np.zeros([layers[3]])


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def d_sigmoid(self, x):
        return x*(1-x)

    def train(self,x_data,y_data,lr):
        x=x_data
        y=y_data

        #前向传播
        l1=self.sigmoid(np.dot(x,self.w1)+self.b1)
        l2=self.sigmoid(np.dot(l1,self.w2)+self.b2)
        l3 = self.sigmoid(np.dot(l2, self.w3) + self.b3)

        #反向传播
        delta_l3 = (y - l3) * self.d_sigmoid(l3)
        delta_l2 = delta_l3.dot(self.w3.T) * self.d_sigmoid(l2)
        delta_l1=delta_l2.dot(self.w2.T)*self.d_sigmoid(l1)

        #权值变化
        self.w3+=(lr*l2.T.dot(delta_l3))/x.shape[0]
        self.w2+=(lr*l1.T.dot(delta_l2))/x.shape[0]
        self.w1+=(lr*x.T.dot(delta_l1))/x.shape[0]

        #偏置改变
        self.b3+=lr*np.mean(delta_l3,axis=0)
        self.b2+=lr*np.mean(delta_l2,axis=0)
        self.b1+=lr*np.mean(delta_l1,axis=0)

    def predict(self,x):

        l1=self.sigmoid(np.dot(x,self.w1)+self.b1)
        l2=self.sigmoid(np.dot(l1,self.w2)+self.b2)
        l3 = self.sigmoid(np.dot(l2, self.w3) + self.b3)
        return l3

bp=BP([28*28,300,100,10])

loss=[]
accuracy=[]

lr=0.1      #学习率
num_epoches=121           #训练轮数

for epoch in range(num_epoches):

    if epoch%30==0:
        lr=lr*0.5
    for img,label in train_loader:
        img=img.view(img.size(0),-1)
        img=np.array(img)
        label=np.array(label)
        label=label_binarize(label,classes=[0,1,2,3,4,5,6,7,8,9])
        bp.train(img,label,lr)

    test_loss=0
    test_acc=0
    for img, label in test_loader:
        img = img.view(img.size(0), -1)
        img=np.array(img)
        label=np.array(label)
        predictions = bp.predict(img)
        y2 = np.argmax(predictions, axis=1)
        acc = np.equal(y2, label).mean()  # 预测准确率
        cost = (np.square(label - y2) / 2).mean()
        test_loss+=cost
        test_acc+=acc
    accuracy.append(test_acc/len(test_loader))
    loss.append(test_loss/len(test_loader))
    print('epoch:', epoch, 'accuracy:', acc, 'loss:', cost)

plt.figure(figsize=(15,20),dpi=80)
plt.subplot(2,1,1)
plt.plot(range(0,len(loss)),loss)
plt.ylabel('loss',fontsize=15)

plt.subplot(2,1,2)
plt.plot(range(0,len(accuracy)),accuracy)
plt.xlabel('epochs',fontsize=15)
plt.ylabel('accuracy',fontsize=15)
plt.savefig('../data/layer3_result2.png')
# plt.show()

