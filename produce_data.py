import v2im
import preprocess as pre
import os

PATH = r"C:\\Users\\Jinming\\Desktop\\USB-backup-09-16-2019\\9-9-2019_ML\\"
for i in range(1,8):
    v2im.process(PATH+str(i)+".avi")

pre.preprocess('./data/frame0.jpg','a')
pre.preprocess('./data/frame1.jpg','b')
pre.preprocess('./data/frame2.jpg','c')
pre.preprocess('./data/frame3.jpg','d')
pre.preprocess('./data/frame4.jpg','e')
pre.preprocess('./data/frame5.jpg','f')
pre.preprocess('./data/frame6.jpg','g')

path0 = "./data1/0/"
path1 = "./data1/1/"

for i, filename in enumerate(os.listdir(path0)):
    os.rename(path0+filename, path0+str(i)+".png")

for i, filename in enumerate(os.listdir(path1)):
    os.rename(path1+filename, path1+str(i)+".png")