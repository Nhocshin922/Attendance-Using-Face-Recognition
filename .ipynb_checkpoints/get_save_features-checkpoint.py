from ArcFace.mobile_model import mobileFaceNet
from mtcnn import MTCNN
import torch as t
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import os
from utils import get_feature

def save_person_information():
    # Yêu cầu người dùng nhập tên
    name = input("Nhập tên của người dùng: ")

    saved_model = os.path.join("ArcFace", "model", "068.pth")
    info_path = os.path.join("users", name)
    if not os.path.exists(info_path):
        os.makedirs(info_path)

    use_cuda = t.cuda.is_available()
    device = t.device("cuda" if use_cuda else "cpu")

    model = mobileFaceNet()
    if use_cuda:
        model.load_state_dict(t.load(saved_model)['backbone_net_list'])
    else:
        model.load_state_dict(t.load(saved_model, map_location=t.device('cpu'))['backbone_net_list'])

    model.eval()
    model.to(device)

    trans = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    detector = MTCNN()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('failed open camera!!!')
    ret, frame = cap.read()
    while ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)
        
        for face in faces:
            bounding_box = face['box']
            keypoints = face['keypoints']
            
            cv2.rectangle(frame, 
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]),
                          (0,155,255),
                          2)

            cv2.circle(frame, (keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(frame, (keypoints['mouth_right']), 2, (0,155,255), 2)
        
        cv2.imshow('img', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            if faces:
                face = faces[0]
                bounding_box = face['box']
                person_img = frame_rgb[bounding_box[1]:bounding_box[1]+bounding_box[3], 
                                    bounding_box[0]:bounding_box[0]+bounding_box[2]]
                
                cv2.imshow('crop', cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(info_path, f'{name}.jpg'), cv2.cvtColor(person_img, cv2.COLOR_RGB2BGR))
                
                # Chuyển đổi person_img thành numpy array nếu nó chưa phải
                if isinstance(person_img, Image.Image):
                    person_img = np.array(person_img)
                
                # Đảm bảo person_img là một numpy array 3 chiều (height, width, channels)
                if len(person_img.shape) == 2:
                    person_img = np.stack((person_img,)*3, axis=-1)
                
                feature = np.squeeze(get_feature(person_img, model, trans, device))
                np.savetxt(os.path.join(info_path, f'{name}.txt'), feature)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    save_person_information()
