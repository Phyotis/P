#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt



def preprocess(filename, index):
    #这里修改文件名
    im = cv2.imread(filename, True)
    im_copy = im.copy()
    im0 = im[:,:,0]
    im0 = cv2.threshold(im0,127,255,cv2.THRESH_BINARY)[1]
    #%%
    """Sobel 滤波器
    dx = cv.Sobel(im0,cv.CV_64F,1,0,ksize=5)
    dy = cv.Sobel(im0,cv.CV_64F,0,1,ksize=5)

    mag = np.hypot(dx,dy)
    mag *= 255.0/np.max(mag)

    plt.imshow(mag, cmap = 'gray')
    """

    contours, _ = cv2.findContours(im0.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if np.abs(m['m00'])<1e-5:
            continue

        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)


    radius = int(np.average(radii)) + 5
    for center in centers:
        cv2.circle(im, center, 3, (255, 0, 0), -1)
        cv2.circle(im, center, radius, (0, 255, 0), 1)

    cv2.imwrite("drawing"+index+".png", im)

    def f(s):
        size = im.shape
        if s[0]-20>=0 and s[0]+20<=size[1] and s[1]-20>=0 and s[1]+20<=size[0]:
            return [s[0]-20,s[0]+20,s[1]-20,s[1]+20]
        else:
            return None

    for i,center in enumerate(centers):
        frame = f(center)
        if frame:
            cv2.imwrite("split/"+index+str(i)+".png",(im_copy[frame[2]:frame[3],frame[0]:frame[1]]))


#preprocess('./data/frame32.jpg', 'a')
#preprocess('./data/frame1.jpg', 'b')

preprocess("data1/val/1/b94.png","aa")

# %%
