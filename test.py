#%%
import numpy as np
import torch


#%%
x = np.random.random((100,100))


#%%
import torch
import torchvision
import torchvision.transforms as transforms

#%%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#%%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%%

for epoch in range(2):  # loop over the dataset multiple times

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
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#%%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

#%%
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#%%
net = Net()
net.load_state_dict(torch.load(PATH))

#%%
outputs = net(images)

#%%
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#%%
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

#%%
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

#%%
net.to(device)

#%%
inputs, labels = data[0].to(device), data[1].to(device)

#%%
import ffmpeg

#%%
import cv2

#%%

# Importing all necessary libraries 
import cv2 
import os 
  
# Read the video from specified path 
cam = cv2.VideoCapture(r"C:\Users\Jinming\Desktop\360mM-300\data2.avi") 
  
try: 
      
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        # if video is still left continue creating images 
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 

#%%
#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
def get_largest(im, n):
	# Find contours of the shape
	major = cv2.__version__.split('.')[0]
	if major == '3':
		_, contours, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		contours, _ = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Cycle through contours and add area to array
	areas = []
	for c in contours:
		areas.append(cv2.contourArea(c))

	# Sort array of areas by size
	sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

	if sorted_areas and len(sorted_areas) >= n:
		# Find nth largest using data[n-1][1]
		return sorted_areas[n - 1][1]
	else:
		return None 

#%%
get_largest(img0t, 10)

#%%
import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
im = cv.imread('./data/frame0.jpg', True)
im0 = im[:,:,0]
#im0 = cv.threshold(im0,127,255,cv.THRESH_BINARY)[1]

#%%
dx = cv.Sobel(im0,cv.CV_64F,1,0,ksize=5)
dy = cv.Sobel(im0,cv.CV_64F,0,1,ksize=5)
#%%
mag = np.hypot(dx,dy)
mag *= 255.0/np.max(mag)

#%%
plt.imshow(mag, cmap = 'gray')

#%%
el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
image = cv.dilate(mag, el, iterations=6)

#%%
import cv2
import numpy as np

original = cv2.imread("test.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
retval, image = cv2.threshold(original, 50, 255, cv2.cv.CV_THRESH_BINARY)

el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
image = cv2.dilate(image, el, iterations=6)

cv2.imwrite("dilated.png", image)

contours, hierarchy = cv2.findContours(
    image,
    cv2.cv.CV_RETR_LIST,
    cv2.cv.CV_CHAIN_APPROX_SIMPLE
)

drawing = cv2.imread("test.jpg")

centers = []
radii = []
for contour in contours:
    area = cv2.contourArea(contour)

    # there is one contour that contains all others, filter it out
    if area > 500:
        continue

    br = cv2.boundingRect(contour)
    radii.append(br[2])

    m = cv2.moments(contour)
    center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
    centers.append(center)

print("There are {} circles".format(len(centers)))

radius = int(np.average(radii)) + 5

for center in centers:
    cv2.circle(drawing, center, 3, (255, 0, 0), -1)
    cv2.circle(drawing, center, radius, (0, 255, 0), 1)

cv2.imwrite("drawing.png", drawing)