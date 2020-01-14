import net as nn
import torch
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

net = nn.Net()
net.load_state_dict(torch.load(nn.PATH))

normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
new_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

def isSingle(img):
    b = new_transform(img)
    b = b[None,:,:,:]
    output = net(b)
    _, predicted = torch.max(output, 1)
    return predicted.numpy()[0]>0.5


def candidate_filter(im):
    im_copy = im.copy()
    im0 = im[:,:,0]
    im0 = cv2.threshold(im0,127,255,cv2.THRESH_BINARY)[1]

    def f(s):
        size = im.shape
        if s[0]-20>=0 and s[0]+20<=size[1] and s[1]-20>=0 and s[1]+20<=size[0]:
            return [s[0]-20,s[0]+20,s[1]-20,s[1]+20]
        else:
            return None

    contours, _ = cv2.findContours(im0.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    radii = []
    centers_and_roi = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # there is one contour that contains all others, filter it out
        if area > 500:
            continue

        br = cv2.boundingRect(contour)
        radii.append(br[2])

        m = cv2.moments(contour)
        if np.abs(m['m00'])<1e-5:
            continue

        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)

    radius = int(np.average(radii)) + 5

    for center in centers:
        frame = f(center)
        if frame:
            img = im_copy[frame[2]:frame[3],frame[0]:frame[1]]
            if isSingle(img):
                centers_and_roi.append((center,f(center)))

    #for center, _ in centers_and_roi:
    #    cv2.circle(im, center, 3, (255, 0, 0), -1)
    #    cv2.circle(im, center, radius, (0, 255, 0), 1)

    #cv2.imwrite("test.png", im)
    return centers_and_roi


def cen(im):
    gray_image=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    r, t = cv2.threshold(gray_image, 127, 255, 0)
    contours, _ = cv2.findContours(t.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = (0,0)
    area0 = 0
    for contour in contours:
        area = cv2.contourArea(contour)

        # there is one contour that contains all others, filter it out
        if area > 500:
            continue
        
        if area>area0:
            area0 = area

            m = cv2.moments(contour)
            if np.abs(m['m00'])<1e-5:
                continue

            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            centers = center

    return centers


def orbit(cam):
    currentframe = 0
    rois = []
    #centers
    while(True):
        ret, frame = cam.read()
        if rois != []:
            for _, roi in rois:
                im0 = frame[roi[2]:roi[3],roi[0]:roi[1]]
                cen(im0)
                

        if currentframe == 0:
            rois = candidate_filter(frame)
        

        
img = cv2.imread("data/frame1.jpg", True)
candidate_filter(img)