from __future__ import division, print_function, absolute_import
import os
import sys
import qdarkstyle
from cv2 import cv2
from pyqt.window_1 import Ui_Dialog as window_1
from pyqt.window_2 import Ui_MainWindow as window_2
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QMessageBox, QLabel, QCheckBox
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QDialog
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QSize

from yolo import YOLO

from main import main


class Controller1(QDialog, window_1):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.handleLogin)
        self.setStyleSheet("background-color: whitesmoke")
        self.setWindowOpacity(1)  # 設定視窗透明度

        self.gif = QMovie('./background/version_02/1920x1080_crop.gif')
        self.gif.setScaledSize(QSize().scaled(1916 // 1.2, 494 // 1.2,
                                              Qt.KeepAspectRatio))
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
    def __init__(self, parent=None):
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
        self.setWindowOpacity(1)  # 设置窗口透明度

        self.label_5.setEnabled(False)
        self.label_6.setEnabled(False)
        self.lineEdit_1.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.pushButton_1.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)

        self.gif = QMovie('./background/version_02/1920x1080_crop.gif')
        self.gif.setScaledSize(QSize().scaled(1916 // 1.2, 494 // 1.2,
                                              Qt.KeepAspectRatio))
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

        video_name = r'visem_03_23.mp4'
        output_name = r'visem_03_23.mp4'
        video_dir = r'./video/sperm'
        output_dir = r'./video/results'
        self.video_path = os.path.join(video_dir, video_name)
        self.output_path = os.path.join(output_dir, output_name)

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
        filename, filetype = QFileDialog.getOpenFileName(self, "選取文件", "C:/")
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
        # try:
        self.label_7.setText('Executing!')
        #print(self.radioBtn.text())
        # 獲取文件路徑
        file_path = self.lineEdit_1.text()
        # 獲取文件夾路徑
        folder_path = self.lineEdit_2.text()

        self.video_capture = cv2.VideoCapture(self.video_path)

        main(
            YOLO(),
            video_capture=self.video_capture,
            output_path=self.output_path,
            label_show_image=self.label_show_image,
            lcdNumber_1=self.lcdNumber_1,
            lcdNumber_2=self.lcdNumber_2,
            lcdNumber_3=self.lcdNumber_3,
        )

        self.label_show_image.clear()
        self.lcdNumber_1.display(0)
        self.lcdNumber_2.display(0)
        self.lcdNumber_3.display(0)
        self.lineEdit_1.clear()
        self.lineEdit_2.clear()
        suc_result = r'Detection Finished!'
        self.label_7.setText(suc_result)
        # except:
        #     fail_result = r'Detection failed!'
        #     self.label_7.setText(fail_result)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    if Controller1().exec_() == QDialog.Accepted:
        ui = Controller2()
        ui.show()
        sys.exit(app.exec_())
