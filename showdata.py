import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import cv2

'''
transforms.ToTensor()
能够将尺寸为H*W*C且位于(0,255)的所有PIL图片，或者np.uint8的Numpy数组，
转化为尺寸为(C*H*W)且位于(0.0,1.0)的Tensor
'''

# 下载训练集
train_dataset = datasets.MNIST(root='./mnist',
                train=True,
                transform=transforms.ToTensor(),
                download=True)
# 下载测试集
test_dataset = datasets.MNIST(root='./mnist',
               train=False,
               transform=transforms.ToTensor(),
               download=True)
 
# 装载训练集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                      batch_size=64,
                      shuffle=True)
# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                     batch_size=64,
                     shuffle=True)

# [batch_size,channels,height,weight]
images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images) 
img = img.numpy().transpose(1, 2, 0)
img = img*255
label=list(labels)
for i in range(len(label)):
    print(label[i],end="\t")
    if (i+1)%8==0:
        print()
cv2.imwrite('1.png', img)
