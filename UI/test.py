import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout, QWidget, QLabel, QMessageBox
from PyQt6.QtCore import Qt,QTimer
from PIL import Image
import numpy as np
from PyQt6.QtGui import QPalette, QColor, QImage, QPixmap
from take_face_attendance import Ui_Main_Screen as Ui_MainScreen
from register import Ui_MainWindow as Ui_Register
from my_utils import get_feature
from get_save_features import FaceInfoSaver
from verification_multi_face import FaceVerification
from capture_face import CaptureFaceWindow
from datetime import datetime, timedelta
from attendance_confirm import Ui_MainWindow as Ui_AttendanceConfirm

class AttendanceConfirmWindow(QMainWindow, Ui_AttendanceConfirm):
    def __init__(self, main_screen, employee_info, frame):
        super().__init__()
        self.setupUi(self)
        self.main_screen = main_screen
        self.employee_info = employee_info

        # Set up the captured image label
        self.camera_label = QLabel(self.centralwidget)
        self.camera_label.setGeometry(20, 40, 400, 300)
        self.display_image(frame)

        # Set up employee info
        self.name_edit_label.setText(employee_info['full_name'])
        self.department_edit_label.setText(employee_info['department_name'])
        self.time_edit_label.setText(datetime.now().strftime("%H:%M:%S"))
        self.date_edit_label.setText(datetime.now().strftime("%Y-%m-%d"))

        self.retake_Button.clicked.connect(self.retake)
        self.confirmation_cancel_Button.clicked.connect(self.confirm)

        # Set up countdown
        self.countdown = 10
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)

    def display_image(self, img):
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        q_image = q_image.rgbSwapped()
        self.camera_label.setPixmap(QPixmap.fromImage(q_image))
        self.camera_label.setScaledContents(True)

    def update_countdown(self):
        self.countdown -= 1
        self.countdown_label.setText(str(self.countdown))
        if self.countdown == 0:
            self.timer.stop()
            self.confirm()

    def confirm(self):
        if self.timer.isActive():
            self.timer.stop()
        employee_id = self.employee_info['employee_id']
        self.main_screen.face_verification.face_info_saver.save_attendance(employee_id)
        
        # Đóng cửa sổ xác nhận và khởi động lại quá trình xác minh sau một khoảng thời gian ngắn
        QTimer.singleShot(100, self.finish_confirmation)

    def finish_confirmation(self):
        self.close()
        self.main_screen.face_verification.reset_recognition()
        self.main_screen.start_verification()

    def retake(self):
        if self.timer.isActive():
            self.timer.stop()
        self.close()
        self.main_screen.restart_verification()

class MainScreen(QMainWindow, Ui_MainScreen):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.attendance_confirm_window = None
        
        # Create a central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add camera label to the layout
        layout.addWidget(self.camera_label, 1) 

        # Create a widget for the button and add it to the layout
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        layout.addWidget(button_widget)

        # Add Register button with black border
        self.register_button = QtWidgets.QPushButton("Register", button_widget)
        self.register_button.setFixedSize(200, 40)
        self.register_button.setStyleSheet("""
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
        button_layout.addWidget(self.register_button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.register_button.clicked.connect(self.open_register)
        self.register_button.hide()  

        self.face_info_saver = FaceInfoSaver(
            model_path="ArcFace/model/finetuned.pth",
            users_path="users"
        )

        self.face_verification = FaceVerification(
            saved_model_path="ArcFace/model/finetuned.pth",
            users_dir="users",
            camera_label=self.camera_label,
            face_info_saver=self.face_info_saver
        )
        self.face_verification.face_recognized.connect(self.show_success_message)
        self.face_verification.unknown_face_detected.connect(self.show_register_button)

    def show_success_message(self, person_name, frame):
        current_date = datetime.now().strftime("%Y-%m-%d")
            
        query = """
        SELECT e.employee_id, e.full_name, d.department_name
        FROM Employees e
        JOIN Departments d ON e.department_id = d.department_id
        WHERE e.full_name = %s
        """
        self.face_verification.face_info_saver.db_cursor.execute(query, (person_name,))
        result = self.face_verification.face_info_saver.db_cursor.fetchone()
            
        if result:
            employee_info = {
                'employee_id': result[0],
                'full_name': result[1],
                'department_name': result[2]
            }
            self.attendance_confirm_window = AttendanceConfirmWindow(self, employee_info, frame)
            self.attendance_confirm_window.show()
            self.face_verification.stop()
        else:
            QMessageBox.warning(self, "Error", f"Không tìm thấy thông tin nhân viên cho {person_name}")

    def start_verification(self):
        self.face_verification.reset_recognition()
        self.face_verification.start_verification()
            
    def restart_verification(self):
        if hasattr(self, 'face_verification'):
            self.face_verification.stop()
        self.face_verification = FaceVerification(
            saved_model_path="ArcFace/model/finetuned.pth",
            users_dir="users",
            camera_label=self.camera_label,
            face_info_saver=self.face_info_saver
        )
        self.face_verification.face_recognized.connect(self.show_success_message)
        self.face_verification.unknown_face_detected.connect(self.show_register_button)
        self.face_verification.start_verification()

    def show_register_button(self):
        self.register_button.show()
    
    def show_and_restart_verification(self):
        self.show()
        self.start_verification()

    def open_register(self):
        self.face_verification.stop()  
        self.register_window = Register(self)
        self.register_window.show()
        self.hide()

class Register(QMainWindow, Ui_Register):
    def __init__(self, previous_window):
        super().__init__()
        self.setupUi(self)
        self.previous_window = previous_window
        self.face_capture_button.clicked.connect(self.open_capture_window)
        self.back_push_Button.clicked.connect(self.go_back)

        self.face_info_saver = FaceInfoSaver(
            model_path="ArcFace/model/finetuned.pth",
            users_path="users"
        )

    def open_capture_window(self):
        full_name = self.name_Edit.text()
        date_of_birth = self.date_of_birth_Edit.date().toString(QtCore.Qt.DateFormat.ISODate)
        gender = "Male" if self.gender_box.currentText() == "Male" else "Female"
        department_id = self.department_Box.currentIndex() + 1  

        if full_name:
            self.face_info_saver.name = full_name
            self.capture_window = CaptureFaceWindow(self.face_info_saver)
            self.capture_window.capture_success.connect(lambda: self.save_info(full_name, gender, date_of_birth, department_id))
            self.capture_window.start_camera()
            self.capture_window.show()
        else:
            QMessageBox.warning(self, "Warning", "Please enter a name.")

    def save_info(self, full_name, gender, date_of_birth, department_id):
        try:
            employee_id = self.face_info_saver.save_to_database(full_name, gender, date_of_birth, department_id)
            QMessageBox.information(self, "Success", f"Information saved successfully! Employee ID: {employee_id}")
            self.go_back()  
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save information: {str(e)}")

    def go_back(self):
        self.previous_window.show_and_restart_verification()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainScreen()
    main_window.show()
    main_window.start_verification()
    sys.exit(app.exec())