from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 579)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Create a vertical layout for central widget
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(20, 50, 20, 20)  
        self.verticalLayout.setSpacing(20)
        
        # Add register label
        self.register_label = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Algerian")
        font.setPointSize(30)
        self.register_label.setFont(font)
        self.register_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.register_label.setObjectName("register_label")
        self.verticalLayout.addWidget(self.register_label)
        
        # Add information group box
        self.information_Box = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.information_Box.setObjectName("information_Box")
        
        # Create a form layout inside the group box for better alignment
        self.formLayout = QtWidgets.QFormLayout(self.information_Box)
        self.formLayout.setContentsMargins(80, 80, 80, 80)
        self.formLayout.setSpacing(20)
        
        self.name_label = QtWidgets.QLabel(parent=self.information_Box)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.name_label.setFont(font)
        self.name_label.setObjectName("name_label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.name_label)
        
        self.name_Edit = QtWidgets.QLineEdit(parent=self.information_Box)
        self.name_Edit.setObjectName("name_Edit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.name_Edit)
        
        self.age_label = QtWidgets.QLabel(parent=self.information_Box)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.age_label.setFont(font)
        self.age_label.setObjectName("age_label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.age_label)
        
        self.date_of_birth_Edit = QtWidgets.QDateEdit(parent=self.information_Box)
        self.date_of_birth_Edit.setObjectName("date_of_birth_Edit")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.date_of_birth_Edit)
        
        self.gender_label = QtWidgets.QLabel(parent=self.information_Box)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.gender_label.setFont(font)
        self.gender_label.setObjectName("gender_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.gender_label)
        
        self.gender_box = QtWidgets.QComboBox(parent=self.information_Box)
        self.gender_box.setObjectName("gender_box")
        self.gender_box.addItem("")
        self.gender_box.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.gender_box)
        
        self.department_label = QtWidgets.QLabel(parent=self.information_Box)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.department_label.setFont(font)
        self.department_label.setObjectName("department_label")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.department_label)
        
        self.department_Box = QtWidgets.QComboBox(parent=self.information_Box)
        self.department_Box.setObjectName("department_Box")
        self.department_Box.addItem("")
        self.department_Box.addItem("")
        self.department_Box.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.department_Box)
        
        self.verticalLayout.addWidget(self.information_Box)
        
        # Add buttons with horizontal layout
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(20)
        
        self.face_capture_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.face_capture_button.setObjectName("face_capture_button")
        self.face_capture_button.setMinimumSize(QtCore.QSize(150, 40))  # Thu nhỏ độ rộng và chiều cao
        self.face_capture_button.setStyleSheet("""
            QPushButton {
                border: 2px solid black;
                border-radius: 5px;
                background-color: #f0f0f0;
                color: black;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        self.horizontalLayout.addWidget(self.face_capture_button)
        
        self.back_push_Button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.back_push_Button.setObjectName("back_push_Button")
        self.back_push_Button.setMinimumSize(QtCore.QSize(150, 40))  # Thu nhỏ độ rộng và chiều cao
        self.back_push_Button.setStyleSheet("""
            QPushButton {
                border: 2px solid black;
                border-radius: 5px;
                background-color: #f0f0f0;
                color: black;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        self.horizontalLayout.addWidget(self.back_push_Button)
        
        self.verticalLayout.addLayout(self.horizontalLayout)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.register_label.setText(_translate("MainWindow", "REGISTER"))
        self.information_Box.setTitle(_translate("MainWindow", "INFORMATION"))
        self.name_label.setText(_translate("MainWindow", "Full name"))
        self.age_label.setText(_translate("MainWindow", "Date of birth"))
        self.gender_box.setItemText(0, _translate("MainWindow", "Male"))
        self.gender_box.setItemText(1, _translate("MainWindow", "Female"))
        self.gender_label.setText(_translate("MainWindow", "Gender"))
        self.department_label.setText(_translate("MainWindow", "Department"))
        self.department_Box.setItemText(0, _translate("MainWindow", "AI"))
        self.department_Box.setItemText(1, _translate("MainWindow", "Tester"))
        self.department_Box.setItemText(2, _translate("MainWindow", "Developer"))
        self.face_capture_button.setText(_translate("MainWindow", "TAKE FACE CAPTURE"))
        self.back_push_Button.setText(_translate("MainWindow", "Back"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
