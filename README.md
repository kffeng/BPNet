# BPNet
### BP神经网络_手写数字识别


使用的数据集是pytoch中的手写数字集MNIST，使用的激活函数为sigmoid，batch_size为32，共训练120，学习率初始设置为0.01，以后每30轮学习率减少为原来的一半，做了两次对比实验。
一次为一层隐藏层的结果，代码为文件中的bp_hidden1，结构为[28*28,300,10]。结果如下图所示，从结果中可以看出，效果并不好，随着训练轮数的增加，在测试集上的准确率大概稳定在0.72左右。
![image](https://user-images.githubusercontent.com/69356569/195749666-d98b66da-8bdf-4bc9-93dd-ad3cf6c0ac17.png)


第二次实验是设计两层隐藏层，代码为文件中的bp_hidden2，网络结构为[28*28，300，100，10]，从训练结果中可以看到，随着训练轮数的增加，在测试集上的准确率大概稳定在0.95左右，效果要比一层隐藏层的效果好很多。
![image](https://user-images.githubusercontent.com/69356569/195749712-debb654c-0644-4cc1-bc44-c66ddf0af7c0.png)


2月10修改1
