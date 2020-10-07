# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window_1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(311, 176)
        self.label_movie = QtWidgets.QLabel(Dialog)
        self.label_movie.setGeometry(QtCore.QRect(0, 0, 311, 71))
        self.label_movie.setText("")
        self.label_movie.setObjectName("label_movie")
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(48, 86, 213, 77))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_1 = QtWidgets.QLabel(self.layoutWidget)
        self.label_1.setTextFormat(QtCore.Qt.AutoText)
        self.label_1.setIndent(-1)
        self.label_1.setObjectName("label_1")
        self.gridLayout.addWidget(self.label_1, 0, 0, 1, 1)
        self.lineEdit_1 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.gridLayout.addWidget(self.lineEdit_1, 0, 1, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 2)
        self.pushButton_1 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_1.setObjectName("pushButton_1")
        self.gridLayout.addWidget(self.pushButton_1, 2, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "智慧型精子篩選系統"))
        self.label_1.setText(_translate("Dialog", "使用者名稱："))
        self.label_2.setText(_translate("Dialog", "密碼："))
        self.pushButton_1.setText(_translate("Dialog", "登入"))

