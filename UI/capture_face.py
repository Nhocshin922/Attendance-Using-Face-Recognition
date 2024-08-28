from PyQt6 import QtCore, QtGui, QtWidgets
import cv2

class Ui_Capture_face(object):
    def setupUi(self, Capture_face):
        Capture_face.setObjectName("Capture_face")
        Capture_face.resize(700, 500)  # Reduced window size
        self.centralwidget = QtWidgets.QWidget(parent=Capture_face)
        self.centralwidget.setObjectName("centralwidget")

        self.video_frame = QtWidgets.QLabel(parent=self.centralwidget)
        self.video_frame.setGeometry(QtCore.QRect(50, 20, 600, 400))  # Adjusted size and position
        self.video_frame.setObjectName("video_frame")

        self.capture_button = QtWidgets.QPushButton("CAPTURE", parent=self.centralwidget)
        self.capture_button.setGeometry(QtCore.QRect(250, 430, 200, 50))  # Adjusted size and position
        self.capture_button.setObjectName("capture_button")
        self.capture_button.setStyleSheet("""
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

        Capture_face.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=Capture_face)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 700, 26))  # Adjusted width
        self.menubar.setObjectName("menubar")
        Capture_face.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=Capture_face)
        self.statusbar.setObjectName("statusbar")
        Capture_face.setStatusBar(self.statusbar)

        self.retranslateUi(Capture_face)
        QtCore.QMetaObject.connectSlotsByName(Capture_face)

    def retranslateUi(self, Capture_face):
        _translate = QtCore.QCoreApplication.translate
        Capture_face.setWindowTitle(_translate("Capture_face", "Capture Face"))

class CaptureFaceWindow(QtWidgets.QMainWindow, Ui_Capture_face):
    capture_success = QtCore.pyqtSignal()

    def __init__(self, face_info_saver):
        super().__init__()
        self.setupUi(self)
        self.face_info_saver = face_info_saver
        self.capture_button.clicked.connect(self.capture_face)
        self.cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_video_stream)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise Exception('Failed to open camera!')
        self.timer.start(30)

    def display_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to grayscale for face detection
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_info_saver.face_cascade.detectMultiScale(
                frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Draw rectangle around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Convert the frame back to RGB for displaying
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], QtGui.QImage.Format.Format_RGB888)
            self.video_frame.setPixmap(QtGui.QPixmap.fromImage(image))

    def capture_face(self):
        name = self.face_info_saver.name
        if name:
            ret, frame = self.cap.read()
            if ret:
                self.face_info_saver.save_face(frame)
                QtWidgets.QMessageBox.information(self, "Success", f"Face captured and saved for {name}.")
                self.capture_success.emit()  # Emit signal when capture is successful
                self.close()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Name not provided.")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    import sys
    from get_save_features import FaceInfoSaver

    app = QtWidgets.QApplication(sys.argv)
    face_info_saver = FaceInfoSaver(model_path="ArcFace/model/finetuned.pth", users_path="users")
    ui = CaptureFaceWindow(face_info_saver)
    ui.start_camera()
    ui.show()
    sys.exit(app.exec())