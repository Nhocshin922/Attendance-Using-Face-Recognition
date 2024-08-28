import cv2
import torch as t
from ArcFace.mobile_model import mobileFaceNet
from my_utils import cosin_metric, draw_ch_zn
import numpy as np
import os
from torchvision import transforms
from PIL import Image, ImageFont
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap

class FaceVerification(QThread):
    face_recognized = pyqtSignal(str, np.ndarray)
    unknown_face_detected = pyqtSignal()

    def __init__(self, saved_model_path, users_dir, camera_label, face_info_saver):
        super().__init__()
        self.saved_model_path = saved_model_path
        self.users_dir = users_dir
        self.camera_label = camera_label
        self.face_info_saver = face_info_saver
        self.model = self.load_model()
        self.total_features = self.load_features()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.threshold = 0.65
        self.font_path = os.path.join("simhei.ttf")
        self.font = ImageFont.truetype(self.font_path, 20, encoding='utf-8')
        self.cap = cv2.VideoCapture(0)
        self.recognized = False
        self.recognized_person = None
        self.recognized_score = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def load_model(self):
        model = mobileFaceNet()
        checkpoint = t.load(self.saved_model_path, map_location=t.device('cpu'))
        if 'backbone_net_list' in checkpoint:
            model.load_state_dict(checkpoint['backbone_net_list'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model

    def load_features(self):
        name_list = os.listdir(self.users_dir)
        path_list = [os.path.join(self.users_dir, i, f'{i}.txt') for i in name_list]
        total_features = np.empty((128,), np.float32)
        for i in path_list:
            temp = np.loadtxt(i)
            total_features = np.vstack((total_features, temp))
        return total_features[1:]

    def start_verification(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.start()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        show_img = frame.copy()
        for (x, y, w, h) in faces:
            person_img = frame[y:y + h, x:x + w].copy()
            pil_image = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            transformed_image = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])(pil_image).unsqueeze(0).to(self.device)
            feature = self.model(transformed_image).detach().cpu().numpy().squeeze()

            cos_distance = cosin_metric(self.total_features, feature)
            index = np.argmax(cos_distance)

            if cos_distance[index] > self.threshold:
                person = os.listdir(self.users_dir)[index]
                score = cos_distance[index]
                if not self.recognized:
                    self.recognized = True
                    self.recognized_person = person
                    self.recognized_score = score
                    self.face_recognized.emit(person, show_img) 
                
                display_text = f"{person} ({score:.2f})"
            else:
                display_text = "Unknown face"
                self.unknown_face_detected.emit()
            
            show_img = draw_ch_zn(show_img, display_text, self.font, (x, y - 30))
            cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        self.display_image(show_img)

    def display_image(self, img):
        qformat = QImage.Format.Format_RGB888
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.camera_label.setPixmap(QPixmap.fromImage(out_image))
        self.camera_label.setScaledContents(True)

    def stop(self):
        self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            self.update_frame()
    
    def reset_recognition(self):
        self.recognized = False
        self.recognized_person = None
        self.recognized_score = None
    
    def get_current_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None