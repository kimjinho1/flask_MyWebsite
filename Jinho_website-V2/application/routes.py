from application import app
import flask
from flask import Flask, request, render_template

import dlib, cv2
import numpy as np
# from scipy import misc
import imageio

from util import detector, sp, facerec, find_faces, encode_faces

descs = np.load('application/descs.npy', allow_pickle=True)

@app.route("/")
@app.route("/index")
@app.route("/home")
def index():
    return render_template("index.html", index=True)

@app.route("/")
@app.route("/simliar_celebrities")
def celebrities():
    return render_template("celebrities.html", celebrities=True)

# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: return render_template('index.html', label="No Files")

        # 이미지 픽셀 정보 읽기
        img = imageio.imread(file)
        img = img[:, :, :3]

        img = cv2.flip(img, 1) # 좌우 대칭
        rects, shapes, _ = find_faces(img) # 얼굴 찾기
        descriptors = encode_faces(img, shapes) # 인코딩

        if(len(descriptors) == 0):
            return render_template('index.html', label="Face is not recognized. try again")
        elif(len(descriptors) > 1):
            return render_template('index.html', label="More than one face was detected. try again")
        else:
            desc = descriptors[0]
            x = rects[0][0][0] # 얼굴 X 좌표
            y = rects[0][0][1] # 얼굴 Y 좌표
            w = rects[0][1][1]-rects[0][0][1] # 얼굴 너비 
            h = rects[0][1][0]-rects[0][0][0] # 얼굴 높이        

            descs1 = sorted(descs, key=lambda x: np.linalg.norm([desc] - x[1]))
            dist = np.linalg.norm([desc] - descs1[0][1], axis=1)
            if dist < 0.45:
                name = descs1[0][0]
                comment = "{0}을 닮으셨네요. 올~~\n\n".format(name) 
            else:
                comment = "풉ㅋ 닮은 연예인이 없네요ㅜㅜ\n 성형하고 오세요!\n\n"
#                 comment = "그 얼굴에 닮은 연예인이 있을 거라고 생각하세요? 정신 차리세요^^"
        
            return render_template('celebrities.html', label=comment)