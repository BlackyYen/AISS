# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:30:41 2020

@author: UCL
"""
from __future__ import division, print_function, absolute_import
import os
import sys
import pandas as pd
import cv2
import qdarkstyle
from window_1 import Ui_Dialog as window_1
from window_2 import Ui_MainWindow as window_2
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QMessageBox, QLabel, QCheckBox
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QDialog
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QSize

from fuzzy_system import fuzzy_system, grade

from yolo import YOLO

class Controller1(QDialog, window_1):
    def  __init__ (self, parent = None):
        super(QDialog, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.handleLogin)
        self.setStyleSheet("background-color: whitesmoke")
        self.setWindowOpacity(1) # 设置窗口透明度
        
        self.gif = QMovie('background/new_background.gif')
        #self.gif.setScaledSize(QSize().scaled(300, 370, Qt.KeepAspectRatio))
        self.label_movie.setScaledContents(True)
        self.label_movie.setMovie(self.gif)
        self.label_movie.setAlignment(Qt.AlignCenter)
        self.gif.start()
        
    def handleLogin(self):
        if self.lineEdit_1.text() == '' and self.lineEdit_2.text() == '':
            #關鍵
            self.accept()
        else:
            QMessageBox.warning(self, 'Error', '使用者名稱或密碼錯誤，\n請重新嘗試！')
        
class Controller2(QMainWindow, window_2):
    def  __init__ (self, parent = None):
        super(QMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.lcdNumber_1.setDigitCount(10)
        self.lcdNumber_2.setDigitCount(10)
        self.lcdNumber_3.setDigitCount(10)
        self.radioButton_1.toggled.connect(self.onClicked1)
        self.radioButton_2.toggled.connect(self.onClicked2)
        self.pushButton_1.clicked.connect(self.read_file)
        self.pushButton_2.clicked.connect(self.write_folder)
        self.pushButton_3.clicked.connect(self.clear)
        self.pushButton_4.clicked.connect(self.process)
        self.setStyleSheet("background-color: whitesmoke")
        self.setWindowOpacity(1) # 设置窗口透明度
        
        self.label_5.setEnabled(False)
        self.label_6.setEnabled(False)
        self.lineEdit_1.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.pushButton_1.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        
        self.gif = QMovie('background/new_background2.gif')
        self.gif.setScaledSize(QSize().scaled(400, 500, Qt.KeepAspectRatio))
        self.label_movie.setMovie(self.gif)
        self.label_movie.setAlignment(Qt.AlignCenter)
        self.label_movie.setScaledContents(True)
        self.gif.start()
        
        
        # 设置边框样式 可选样式有Box Panel等
        self.label_show_image.setFrameShape(QtWidgets.QFrame.Box)
        # 设置阴影 只有加了这步才能设置边框颜色 
        # 可选样式有Raised、Sunken、Plain（这个无法设置颜色）等
        self.label_show_image.setFrameShadow(QtWidgets.QFrame.Plain)
        # 设置线条宽度
        self.label_show_image.setLineWidth(6)

    def onClicked1(self):
        self.label_5.setEnabled(False)
        self.lineEdit_1.setEnabled(False)
        self.pushButton_1.setEnabled(False)
        self.label_6.setEnabled(True)
        self.lineEdit_2.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.radioBtn = self.sender()
        
    def onClicked2(self):
        self.label_5.setEnabled(True)
        self.lineEdit_1.setEnabled(True)
        self.pushButton_1.setEnabled(True)
        self.label_6.setEnabled(True)
        self.lineEdit_2.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.radioBtn = self.sender()

    def read_file(self):
        #選取文件
        filename, filetype =QFileDialog.getOpenFileName(self, "選取文件", "C:/")
        print(filename)
        self.lineEdit_1.setText(filename)

    def write_folder(self):
        #選取文件夾
        foldername = QFileDialog.getExistingDirectory(self, "選取文件夾", "C:/")
        foldername = foldername + '/'
        print(foldername)
        self.lineEdit_2.setText(foldername)
        
    def clear(self):
        self.video_capture.release()
        self.label_show_image.clear()
        self.label_7.clear()
        
    # 進行處理
    def process(self):
        try:
            self.label_7.setText('執行中！')
            #print(self.radioBtn.text())
            #獲取文件路徑
            file_path = self.lineEdit_1.text()
            #獲取文件夾路徑
            folder_path = self.lineEdit_2.text()
            #frame = YD(YOLO(), file_path, folder_path)
            self.YD(YOLO(), file_path, folder_path)
            self.label_show_image.clear()
            self.lcdNumber_1.display(0)
            self.lcdNumber_2.display(0)
            self.lcdNumber_3.display(0)
            self.lineEdit_1.clear()
            self.lineEdit_2.clear()
            suc_result = r'執行成功！'
            self.label_7.setText(suc_result)
        except:
            fail_result = r'執行失敗，請再試一次！'
            self.label_7.setText(fail_result)
    
    def a(a,self):
        print(a)
    
    def YD(self,yolo, video_path, output_path):
        gd, grade_system = fuzzy_system()
        
if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    if Controller1().exec_() == QDialog.Accepted:
        ui = Controller2()    
        ui.show()
        sys.exit(app.exec_())

