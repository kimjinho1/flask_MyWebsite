# flask_MyWebsite

## Download Models
- [shape_predictor_68_face_landmarks.dat.bz2](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)
- [dlib_face_recognition_resnet_model_v1.dat](https://github.com/kairess/simple_face_recognition/raw/master/models/dlib_face_recognition_resnet_model_v1.dat)

## Jinho_website-V1  
- **main.py 또는 main.ipynb를 실행시킨 후에 홈페이지 접속 가능**  
- **아래에  github, instagram 아이콘 누르면 내 github, instagram 창이 열림!**  
- layout.html -> title 수정 가능  
- index.html -> 본문 내용 수정 가능  
- footer.html -> gituhb, instagram 주소 변경 가능  
- static/images/layout -> 아이콘(로고)이미지 변경 가능  


## Finding Similar Celebrities-V1
![sa](https://user-images.githubusercontent.com/29765855/81027739-c85f3c80-8eb9-11ea-8bb8-f5382838f5b0.PNG)  

- **이미지 업로드를 눌러서 사진을 올리면 닮은 연예인을 출력해준다.  EX) Chen을 닮으셨네요~**  
- **닮은 연예인이 없다면 아주 기분 좋은 말이 출력된다.**  
- **얼굴이 딱 하나만 인식되어야 한다. 하나도 인식 못하거나 여러개를 인식한 경우 "try again"이 출력된다.**   
- https://github.com/davisking/dlib-models 에 있는 shape_predictor_68_face_landmarks.dat 파일이 model 폴더에 추가해야 됨.
- https://niceman.tistory.com/192 를 참고해서 만듬.  
- routes.py -> 예측 모델과 출력 결과 수정 가능  


## Jinho_website-V2  
- **위의 2개를 대충 섞어보았다.**   

## 
