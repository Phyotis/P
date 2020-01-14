import sys
import pycurl
import os
import time
from StringIO import StringIO
import re
from PyQt4 import QtGui,QtCore
from PyQt4.QtGui import *
from PyQt4.QtCore import *



# class definition
class Pic_Label(QtGui.QLabel):
    def __init__(self):
        super(Pic_Label,self).__init__()
        self.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.cache_map={}

    def paintEvent(self, event):

        if self.extention !="gif":
            size = self.size()
            painter = QtGui.QPainter(self)
            point = QtCore.QPoint(0,0)
            scaledPix = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation)
            # start painting the label from left upper corner
            point.setX((size.width() - scaledPix.width())/2)
            point.setY((size.height() - scaledPix.height())/2)
            #print point.x(), ' ', point.y()
            painter.drawPixmap(point, scaledPix)
        else:
            QLabel.paintEvent(self, event)



    def mouseReleaseEvent(self,ev):
        #self.emit(SIGNAL('clicked()'))
        if ev.button() == Qt.RightButton:
            self.emit(SIGNAL("RightClick"))
        else:
            self.emit(SIGNAL("LeftClick"))



    def set_image(self,pic_url,index):

        if (index in self.cache_map) == False:
            self.cache_map[index]=False
        self.pixmap = QtGui.QPixmap()
        self.retrieve_from_url_cache(pic_url,index)
        


    def retrieve_from_url_cache(self,pic_url,index):

        try:
            self.extention=re.search(r"\.(\w+)$", pic_url).group(1)
        except:
            self.extention="jpg"
        cache_pic_name="Pic_"+str(index)+"."+self.extention
        cache_pic_path=os.getcwd()+"\Cache_Pic\\"+cache_pic_name



        if self.cache_map[index]==True:
            if self.extention =="gif":
                movie = QtGui.QMovie(cache_pic_path)
                self.setMovie(movie)
                movie.start()

            else:
                #print "Cached!" + cache_pic_path
                if self.pixmap.load(cache_pic_path) == False:
                #print "use jpg to try again"
                    if self.pixmap.load(cache_pic_path)== False:
                    #last resort, try again
                        self.retrieve_from_url(pic_url,index,cache_pic_path)
                self.setPixmap(self.pixmap)  # udpate immediately


        else:
            if self.extention =="gif":
                data=self.retrieve_from_url(pic_url,index,cache_pic_path)
                self.pixmap.loadFromData(data)
                f = open(cache_pic_path, 'wb')
                f.write(data)
                f.close()
                movie = QtGui.QMovie(cache_pic_path)
                self.setMovie(movie)
                movie.start()

            else:
                data=self.retrieve_from_url(pic_url,index,cache_pic_path)
                self.pixmap.loadFromData(data)
                self.pixmap.save(cache_pic_path)
                self.setPixmap(self.pixmap)  # udpate immediately


    def retrieve_from_url(self,pic_url,index,file_path):
        c = pycurl.Curl()
        c.setopt(pycurl.PROXY, 'http://192.168.87.15:8080')
        c.setopt(pycurl.PROXYUSERPWD, 'LL66269:')
        c.setopt(pycurl.PROXYAUTH, pycurl.HTTPAUTH_NTLM)
        buffer = StringIO()
        c.setopt(pycurl.URL, pic_url)
        c.setopt(c.WRITEDATA, buffer)
        c.perform()
        c.close()  
        data = buffer.getvalue()
        self.cache_map[index]=True
        return data

        


    def setMovie(self,movie):
        QLabel.setMovie(self, movie)
        s=movie.currentImage().size()
        self._movieWidth = s.width()
        self._movieHeight = s.height()



class Example(QtGui.QWidget):


    def __init__(self,thread_url_list):
        super(Example, self).__init__()
        
        self.url_list=thread_url_list
        self.current_pic_index=0

        cwd = os.getcwd()
        #print cwd
        directory=cwd+"\Cache_Pic"
        #print directory
        if not os.path.exists(directory):
            os.makedirs(directory)


        self.initUI()

        # making subfolderss to cache pictures



    def initUI(self):


        layout = QtGui.QGridLayout()
        self.label = Pic_Label()
        self.label.set_image(self.url_list[0],0)


        #self.label = QLabel()
        #movie = QtGui.QMovie("Cache_Pic/Pic_0.gif")
        #self.label.setMovie(movie)
        #movie.start()


        layout.addWidget(self.label)
        layout.setRowStretch(0,1)
        layout.setColumnStretch(0,1)
      
        #self.connect(self.label,SIGNAL('clicked()'),self.fun_next)
        self.connect(self.label,SIGNAL("LeftClick"),self.fun_next)
        self.connect(self.label,SIGNAL("RightClick"),self.fun_prev)


        #b1=QtGui.QPushButton("next")
        #b2=QtGui.QPushButton("prev")
        #b1.clicked.connect(self.fun_next)
        #b2.clicked.connect(self.fun_prev)
        #layout.addWidget(b1)
        #layout.addWidget(b2)


        self.setLayout(layout)
        self.setGeometry(300, 300, 500, 500)
        self.setWindowTitle('Picture Viewer')
        self.show()



        # Connect button to image updating 






    def fun_next(self):
        if self.current_pic_index < len(self.url_list)-1:
            self.current_pic_index=self.current_pic_index+1
        else:
            self.current_pic_index=0

        self.label.set_image(self.url_list[self.current_pic_index],self.current_pic_index)
        sys.stdout.write('\r')
        sys.stdout.write("[ %d ] out of (%d)" % (self.current_pic_index+1,len(self.url_list)))
        sys.stdout.flush()




    def fun_prev(self):
        if self.current_pic_index > 0:
            self.current_pic_index=self.current_pic_index-1
        else:
            self.current_pic_index=len(self.url_list)-1

        self.label.set_image(self.url_list[self.current_pic_index],self.current_pic_index)
        sys.stdout.write('\r')
        sys.stdout.write("[ %d ] out of (%d)" % (self.current_pic_index+1,len(self.url_list)))
        sys.stdout.flush()


    
def view_image():
    url_list=['https://i.imgur.com/waprhO3.gif','http://static.cnbetacdn.com/article/2017/0831/7f11d5ec94fa123.png','http://static.cnbetacdn.com/article/2017/0831/1b6595175fb5486.jpg']
    viewer_app = QtGui.QApplication(sys.argv)
    ex = Example(url_list)
    sys.exit(viewer_app.exec_())


view_image()
