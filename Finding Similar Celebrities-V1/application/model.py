### import and model load

import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import glob

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')


### 얼굴 찾는 함수

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)
        
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np


### 얼굴을 인코딩 해주는 함수(랜드마크 추출)

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)


### 이미지 불러오기 and 라벨 저장

label_name = []
label_class = {}
img_paths = glob.glob("kpop_img/*")

for path in img_paths:
    name = path.split(".")[0][9:]
    label_name.append(name)
    label_class[name] = path


### descs 파일 저장

descs = []

for name, label_path in label_class.items():
    img = cv2.imread(label_path)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB
    _, img_shapes, _ = find_faces(img)
    descs.append([name, encode_faces(img, img_shapes)[0]])

np.save('descs.npy', descs)


### descs 파일 불러오기

# descs = np.load('descs.npy', allow_pickle=True)