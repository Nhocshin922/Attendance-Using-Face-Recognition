from ArcFace.mobile_model import mobileFaceNet
import torch as t
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import os
from my_utils import get_feature
import mysql.connector
from datetime import datetime, timedelta

class FaceInfoSaver:
    def __init__(self, model_path, users_path):
        self.model_path = model_path
        self.users_path = users_path
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.name = None
        self.db_connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            database="attendance_management"
        )
        self.db_cursor = self.db_connection.cursor()

    def load_model(self):
        model = mobileFaceNet()
        checkpoint = t.load(self.model_path, map_location=self.device)
        if 'backbone_net_list' in checkpoint:
            model.load_state_dict(checkpoint['backbone_net_list'])
        else:
            model.load_state_dict(checkpoint)
        model.eval().to(self.device)
        return model

    def save_face(self, frame):
        if self.name is None:
            raise Exception("Name is not set in FaceInfoSaver")
        
        info_path = os.path.join(self.users_path, self.name)
        if not os.path.exists(info_path):
            os.makedirs(info_path)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            person_img = frame[y:y + h, x:x + w]

            cv2.imwrite(os.path.join(info_path, f'{self.name}.jpg'), person_img)

            person_img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            feature = np.squeeze(get_feature(np.array(person_img_rgb), self.model, self.trans, self.device))
            np.savetxt(os.path.join(info_path, f'{self.name}.txt'), feature)
            
    def save_to_database(self, full_name, gender, date_of_birth, department_id):
        query = """
        INSERT INTO Employees (full_name, gender, date_of_birth, department_id)
        VALUES (%s, %s, %s, %s)
        """
        values = (full_name, gender, date_of_birth, department_id)
        self.db_cursor.execute(query, values)
        self.db_connection.commit()
        return self.db_cursor.lastrowid

    def __del__(self):
        if hasattr(self, 'db_connection') and self.db_connection.is_connected():
            self.db_cursor.close()
            self.db_connection.close()
            
    def save_attendance(self, employee_id):
        current_time = datetime.now()
        current_date = current_time.date()

        query = """
        SELECT attendance_id, check_in_time, check_out_time
        FROM Attendance
        WHERE employee_id = %s AND date = %s
        ORDER BY check_in_time DESC
        LIMIT 1
        """
        self.db_cursor.execute(query, (employee_id, current_date))
        result = self.db_cursor.fetchone()

        if result:
            attendance_id, check_in_time, check_out_time = result
            if check_in_time:
                # Update check-out time
                query = "UPDATE Attendance SET check_out_time = %s WHERE attendance_id = %s"
                self.db_cursor.execute(query, (current_time, attendance_id))
                self.db_connection.commit()
        else:
            # First check-in of the day
            query = "INSERT INTO Attendance (employee_id, check_in_time, date) VALUES (%s, %s, %s)"
            self.db_cursor.execute(query, (employee_id, current_time, current_date))
            self.db_connection.commit()