# Importing all necessary libraries 
import cv2 
import os 
import sys


count = 0

def process(path, oneframe=True):
    global count
    # Read the video from specified path 
    cam = cv2.VideoCapture(path) 
    
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
        global count
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret and currentframe<100:
            if currentframe%10==0: 
                # if video is still left continue creating images 
                name = './data/frame' + str(count) + '.jpg'
                print ('Creating...' + name) 
                count += 1
        
                # writing the extracted images 
                cv2.imwrite(name, frame) 
                if oneframe == True:
                    break
                # increasing counter so that it will 
                # show how many frames are created 
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

#path_list = ["C:/Users/Jinming/Desktop/5.avi","C:/Users/Jinming/Desktop/4.avi"]
#for path in path_list:
    #process(path)

#process()