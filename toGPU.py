#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2


normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

new_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

train_dataset = datasets.ImageFolder(root='data1\\train',transform=data_transform)
#%%
trainloader = DataLoader(train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=0)
#%%
def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(4):
        label_ = labels_batch[i].item()
        image_ = np.transpose(images_batch[i])
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')
        plt.pause(0.01)

#%%
plt.figure()
ii = 0
for i_batch, sample_batch in enumerate(trainloader):
    show_batch_images(sample_batch)
    if ii > 10:
        break
    ii +=1
    plt.show()
#%%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #  conv1层，输入的灰度图，所以 in_channels=1, out_channels=6 说明使用了6个滤波器/卷积核，
        # kernel_size=5卷积核大小5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # conv2层， 输入通道in_channels 要等于上一层的 out_channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # an affine operarion: y = Wx + b
        # 全连接层fc1,因为32x32图像输入到fc1层时候，feature map为： 5x5x16
        # 因此，全连接层的输入特征维度为16*5*5，  因为上一层conv2的out_channels=16
        # out_features=84,输出维度为84，代表该层为84个神经元
        self.fc1 = nn.Linear(16*7*7, 120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x[:,0,:,:]
        x = x[:,None,:,:]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 特征图转换为一个１维的向量
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net = Net()
print(net)

# %%
input = torch.randn(1,3,40,40)
out = net(input)
print(out)

# %%
classes = ('no', 'yes')

def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#%%
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(16)))

# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)


# %%
for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')

# %%
#储存网络参数
PATH = './net.pth'
#%%
torch.save(net.state_dict(), PATH)

# %%
#测试集
test_dataset = datasets.ImageFolder(root='data1\\test',transform=data_transform)

testloader = DataLoader(test_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=0)
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
# %%
net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(16)))

# %%
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# %%
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print(predicted)
        c = (predicted == labels).squeeze()
        for i in range(2):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# %%
