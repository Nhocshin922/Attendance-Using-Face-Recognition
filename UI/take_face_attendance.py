from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QVBoxLayout, QLabel, QWidget

class Ui_Main_Screen(object):
    def setupUi(self, Main_Screen):
        Main_Screen.setObjectName("Main_Screen")
        Main_Screen.resize(800, 600)
        self.centralwidget = QWidget(parent=Main_Screen)
        self.centralwidget.setObjectName("centralwidget")

        self.layout = QVBoxLayout(self.centralwidget)
        self.camera_label = QLabel(self.centralwidget)
        self.layout.addWidget(self.camera_label)

        Main_Screen.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=Main_Screen)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        Main_Screen.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=Main_Screen)
        self.statusbar.setObjectName("statusbar")
        Main_Screen.setStatusBar(self.statusbar)

        self.retranslateUi(Main_Screen)
        QtCore.QMetaObject.connectSlotsByName(Main_Screen)

    def retranslateUi(self, Main_Screen):
        _translate = QtCore.QCoreApplication.translate
        Main_Screen.setWindowTitle(_translate("Main_Screen", "MainWindow"))
