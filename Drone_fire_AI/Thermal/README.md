# 열화상 산불 영상 모델 진행 과정
1. 초창기 모델 아이디어
- 비전처럼 동일한 yolo 모델을 사용하면 어떨지에 대한 의문이 들었음
  - 절대 온도가 아닌 상대 온도였기 때문에 색에 대한 확연한 차이가 없어 불가능할 거라고 생각함(region proposal 공부 후 확실히 깨달음-> 안된다)
  - 하지만 일단 해봐야지 알기 때문에 yolo 모델 적용해보기로 결정

- yolo 모델 적용하기 위한 준비
   1. 이미지 수집
      - 아르고스 다인 쪽에서 받은 열화상 동영상에서 프레임 단위로 이미지 추출
   2. 이미지 라벨링
      - 산불처럼 labelImg을 이용하여 불이 있는 위치에 라벨링 진행
   3. anchor clusturing
      - 해당 방법을 통해 anchor box clusturing 구해서 넣기
   4. 모델 학습
      - 산불 비전과 같이 동일한 yolo v2 모델 사용 및 학습 진행
   5. 학습 결과

|모델 결과1|모델 결과2|
|----|--------|
|![image](https://user-images.githubusercontent.com/49123169/135958906-31d15be1-3a5d-49c3-aca6-36a15510e119.png)|![image](https://user-images.githubusercontent.com/49123169/135958921-a12c8b91-e955-4062-b921-c3cab0985508.png)|

  - 심각할 정도로 낮은 예측률과 더불어 불을 인식한다 하더라도 불 위치를 제대로 잡지 못하는 것을 확인
  - 다른 방안 모색하기로 결정

2. OCR 모델
- 열화상 영상에서 오른쪽 아래에 현재 시점의 MAX 온도가 표시된다는 것을 알게 됨
  - 해당 온도를 정확하게 인지할 수 있다면 충분히 가능성이 있다고 판단함
  - 그래서 글자 인식하고 해당 값을 추출하는 OCR 방식을 적용

- OCR 모델 실험 전에 이미지 자르기부터 진행

~~~
def im_trim (img): #함수로 만든다
    x = 1700; y = 845; #자르고 싶은 지점의 x좌표와 y좌표 지정
    w = 160; h = 55; #x로부터 width, y로부터 height를 지정
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    cv2.imwrite("./webcam/frame.jpg",img_trim) #org_trim.jpg 라는 이름으로 저장
    return img_trim
~~~

- (1) OCR 모델을 사용하기 위한 실험들(Tessart OCR)
  - 많은 사람들이 사용하는 OCR인 Tessart OCR을 가지고 실험
    - Tessart OCR에 필요한 환경을 설치 후, 바로 적용해본 결과 몇 개 빼고는 정확한 결과가 나오지 않았음
      - 원본 색깔에서 좋은 성능이 안 나와서 강제로 색상을 바꾸게 한 뒤에도 잘 쳐주어야 50%인 결과가 나왔음

|정확하게 인식했을 때|인식 못했을 때|
|------------------|-------------|
|![image](https://user-images.githubusercontent.com/49123169/135984677-2d087509-96e9-48e3-a233-55d68a0591c5.png)|![image](https://user-images.githubusercontent.com/49123169/135984712-f57fe863-03c7-4da4-85e6-39907c095cd3.png)|

~~~
import pytesseract
import cv2
import numpy as np
train_image_folder='./fire_thermal_image/IR_Color2.mp4'
image_nms = list(np.random.choice(os.listdir(train_image_folder),300,replace=False))
for img_nm in image_nms:
    org_image=cv2.imread(os.path.join(train_image_folder,img_nm))
    # size 축소 
    # 0 < B < 100 , 128 < G < 255 , 0 < R < 100 
    org_image= cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)

    dst1 = cv2.inRange(org_image, (199, 199, 199), (255, 255, 255)) 
    img_result = cv2.bitwise_and(src_hsv, src_hsv, mask = dst1) 
    dst2 = cv2.inRange(org_image, (0, 0, 99), (0, 0, 100)) 
    img_result2 = cv2.bitwise_and(src_hsv, src_hsv, mask = dst2) 
    # cv2.imshow('src', src) 
    # cv2.moveWindow('src',400,100) 
    # cv2.imshow('dst1', dst1) 
    # cv2.moveWindow('dst1',400,450) 
    # cv2.imshow('img_result', img_result) 
    # cv2.moveWindow('img_result',800,450) 
    # cv2.imshow('dst2', dst2) 
    # cv2.moveWindow('dst2',400,800) 
    trim_image = im_trim(dst1) #trim_image 변수에 결과물을 넣는다
    filename = "{}.png".format(os.getpid()) 
    cv2.imwrite(filename, trim_image) # Simple image to string 
    text = pytesseract.image_to_string(Image.open(filename), lang=None) 
    os.remove(filename) 
    print(text) 
    plt.imshow(trim_image)
    plt.show()
~~~

- 그래서 Tessart OCR을 학습시켜야 되겠다는 생각이 들었지만 학습하는 방법을 몰라 진행하지 못했음
  - 차후에 학습하는 방법을 발견했는데 온통 영어에다가 무슨 소리인지 몰라 학습하지 않았음
  - >https://tesseract-ocr.github.io/tessdoc/

- (2) OCR 모델을 사용하기 위한 실험들(Keras OCR) 
  - tensorflow 기반인 tessart ocr이 안 되니 keras쪽에서 ocr이 있나 확인해보았음
  - 확인한 결과, keras쪽에 ocr이 존재하는 것을 확인
    - 명칭 자체가 keras-ocr이며 tessart처럼 여러 가지 설치를 안 해도 바로 사용 가능한 것을 확인
    - 바로 적용해보았음
  - 적용 결과, 제대로 인식 못하는 것을 확인

|사진1|사진2|
|-----|-----|
|![image](https://user-images.githubusercontent.com/49123169/135988267-5c298365-af76-46c5-b864-1617bd65866f.png)|![image](https://user-images.githubusercontent.com/49123169/135988284-24950f08-f5e8-4e2f-93fe-70e9643f8ae3.png)|

~~~
import matplotlib.pyplot as plt

import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
images = [
    keras_ocr.tools.read('/content/IR_Color10.jpg'),keras_ocr.tools.read('/content/IR_Color16.jpg')
    ]


# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

# Plot the predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
~~~

- (3) OCR 모델을 사용하기 위한 실험들(Easy-OCR)
  - tensorflow, keras쪽의 ocr은 성능이 낮아서 pytorch쪽까지 건드리기로 결정
  - pytorch쪽을 찾아본 결과, tessart처럼 많이 사용하는 easy-ocr을 발견
  - keras-ocr과 마찬가지로 easy-ocr로 손쉽게 설치가 가능하여 바로 넣어서 실험해봄
  - 확인 결과, 성능이 좋진 않지만 기존 ocr보다 더 좋은 성능을 보임(학습을 안했는데도 불구하고)
  - 그래서 해당 모델을 기반으로 학습시킬 수 있는 방법을 찾기 시작함

~~~
!pip install easyocr
import easyocr
reader = easyocr.Reader(['ko','en']) # need to run only once to load model into memory
result = reader.readtext('/content/gdrive/MyDrive/ocr/121.9.jpg')
~~~

- OCR 모델 학습(Naver clova OCR github)
  - Easy-ocr github 내에서 custom image로 train하고 싶으면 naver clova쪽을 활용하라는 글이 있었음(저자 공식)
  - 그래서 naver clova github에서 ocr 문서를 찾음
    - https://github.com/clovaai/deep-text-recognition-benchmark
  - 하지만 어떻게 custom 시키는지 몰라서 당황하던 도중, 하는 방법을 알려주는 곳을 찾음
    - https://davelogs.tistory.com/76?category=928468
    - https://ropiens.tistory.com/35
  - 위의 사이트를 기반으로 드론 산불 열화상 기반의 custom 모델 만들기를 시작함

3. OCR 모델 학습
- (1) 데이터 전처리 
  - Naver clova 모델에 넣을 데이터 형태가 lmdb임
    - 해당 형태로 만들기 위해선 training과 validation 폴더 별로 이미지를 넣고 그 폴더 밖에 각 이미지의 이미지 이름과 그 이미지의 글자 결과를 하나의 txt로 저장해야 함
  - 처음에는 lmdb로 만들 수 있는 방법을 찾음
    - easy-ocr하고 naver clova에서 밑의 깃허브의 방법으로 lmdb데이터 형태를 만들라고 권유함
    - >https://github.com/Belval/TextRecognitionDataGenerator
  - 그래서 먼저 train하고 validation과 test로 이미지를 나누 후에 
  - 1차 문제 발생
    - TextRecognitionDataGenerator로 만드는데 온도를 파일명에 넣어야함
      - 직접 입력하며 파일명에 넣으려고 했지만 ℃ 특수문자가 파일명으로 넣지 못하는 경우가 발생
    - 해결 방법 : 직접 txt파일을 만들어서 이미지 이름을 입력하고 그 옆에(\t를 사용) 해당 이미지의 텍스트값을 입력함
    - 이후에 직접 만든 txt파일과 폴더를 가지고 lmdb 데이터 만들기 진행
  - 2차 문제 발생
    - create_lmdb_dataset.py를 통해 만들려고 했으나 실행에서 오류가 나는 문제 발생
    - 해결 방법 : txt 파일의 이미지에서 경로를 제대로 설정 안해서 나타나는 오류였음
    - 경로를 제대로 설정한 결과, lmdb 파일로 만들어져서 모델 학습을 진행하려고 했음

- (2) 데이터 학습1
  - 3차 문제 발생
    - 모델 학습을 진행하려고 했더니 lmdb파일을 인식하지 못하는 경우가 발생
    - 해결 방법 
      - 처음 lmdb 파일을 만들 때는 colab 환경 내에서 만들었는데 새로 만들어진 폴더와 파일을 colab이 인식하지 못하는 것으로 판단
      - 그래서 로컬에서 lmdb 파일을 만들고 colab에서는 학습만 진행했더니 문제없이 잘 돌아감
  - 대략적으로 5000번 학습을 하고 test하려고 시도
  - 4차 문제 발생
    - test.py를 통해 test 이미지로 학습 결과를 test하려고 했더니 여러 오류가 발생
    - 해결 방법 
      - 해당 github에서도 test.py를 사용하려다가 오류가 발생하는 사람들이 많은 것을 봄
      - 그래서 issue쪽에서 이에 대한 답변으로 demo.py를 통해 학습 결과를 테스트해도 된다는 글을 보고 실행
    - 해결하여 테스트를 진행한 결과, 90% 정도의 정확도를 보였음
  - 틀린 5%의 이미지와 형태를 분석한 결과 다음과 같은 판단을 내림
    - 생각보다 세자리 수 틀린 횟수가 존재 -> 학습 데이터 부족으로 판단
    - 사진 화질이 깨지거나 안 좋은 경우, 맞추지 못한다.

- (3) 데이터 학습2
  - 영상쪽 데이터를 넣었을 때 바로 결과가 나올 수 있는 코드를 짜던 도중, rgb를 제외하고 학습시켜서 흑백으로 학습시킨 것으로 알게 됨.
  - 그래서 rgb가 있다는 것을 코드로 인식시킨 후에 다시 학습을 진행
  - 그 전보다 빠르게 수렴했고 예측률도 95%로 5% 더 올라감

4. 실시간 모델 결과 코드 작성
- 첫 번째 코드
  - 기존 코드에서 frame으로 바로 받지 않고 이미지로 저장시킨 뒤에 image_folder 방향을 frame 이미지가 저장된 위치로 해놓아서 학습 결과 도출
~~~
def im_trim (img): #함수로 만든다
    x = 1700; y = 845; #자르고 싶은 지점의 x좌표와 y좌표 지정
    w = 160; h = 55; #x로부터 width, y로부터 height를 지정
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    cv2.imwrite("./webcam/frame.jpg",img_trim) #org_trim.jpg 라는 이름으로 저장
    return img_trim
    
        while(vs.isOpened()):
            grabbed, frame = vs.read()
            
            if not grabbed:
                continue  #break -> continue로 교체(프레임이 잘 못 되도 처리를 안하고 넘어가게 만들기 위해))

            # Resize frame
            im_trim(frame)
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                print(batch_size)
                image = image_tensors.to(device)
                print(image.shape)
                # For max length prediction
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)     
~~~

- 두 번째 코드
  - 영상 frame으로 바로 받아서 넣어서 결과를 도출하려고 시도
  - 하지만 끝내 해당 방법을 구현하지 못함
~~~
    while(vs.isOpened()):
        grabbed, frame = vs.read()

    # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            continue  #break -> continue로 교체(프레임이 잘 못 되도 처리를 안하고 넘어가게 만들기 위해))
        with torch.no_grad():
            imgs=im_trim(frame)
            plt.imshow(imgs)
            plt.show()
            #imgs=NormalizePAD2(imgs)
            image=prep_image(imgs)
~~~

- 세 번째 코드
  - 첫 번째 코드와 다르지 않지만 첫 번째 코드에서 GPU를 사용하지 않고 했을 때는 영상 대비 결과 나오는 속도가 매우 느렸음
  - GPU를 사용한 결과, 준 실시간급으로 영상 결과 도출
~~~
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
~~~

- 네 번째 코드
  - 영상 크기에 따라 해당 영상에서 보이는 높은 온도 위치가 달라지는 것을 확인
  - 또한, 결과를 이미지로 저장해야 했음
  - 그래서 영상 크기에 따라 detect할 온도 위치를 다시 찾아주고 결과에 따른 사각형 박스와 결과 text 코드 작성

~~~
from PIL import ImageFont,ImageDraw, Image 
font=ImageFont.truetype("/home/dj/work/drone_fire/ocr-train_v3/fonts/H2GTRE.TTF",30)
def im_trim (img): #함수로 만든다
    h,w,c=img.shape
    if h==1080:
        x = 1700; y = 845; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 160; h = 55; #x로부터 width, y로부터 height를 지정
        img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
        cv2.imwrite("./webcam/frame.jpg",img_trim) #org_trim.jpg 라는 이름으로 저장
    elif h==480:        
        x = 630; y = 355; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 80; h = 20; #x로부터 width, y로부터 height를 지정
        img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
        cv2.imwrite("./webcam/frame.jpg",img_trim)
    elif h==720:
        x = 1137; y =565; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 110; h = 35; #x로부터 width, y로부터 height를 지정
        img_trim = img[y:y+h, x:x+w]
        cv2.imwrite("./webcam/frame.jpg",img_trim)
    return img_trim

                    img= frame.copy()
                    h,w,c=img.shape
                    if h==1080:
                        img=Image.fromarray(img)
                        draw=ImageDraw.Draw(img)
                        draw.text((1720,810),pred,font=font)
                        img = np.array(img)
                        cv2.rectangle(img, 
                            pt1=(1700,845), 
                            pt2=(1860,900),
                            color=(171,242,0),
                            thickness=1
                            )
                        cv2.imwrite('/home/dj/work/drone_fire/ocr-train_v3/pngfolder/frame%d.jpg'%i,img)
                    elif h==720:
                        img=Image.fromarray(img)
                        draw=ImageDraw.Draw(img)
                        draw.text((1148 , 530),pred,font=font)
                        img = np.array(img)
                        cv2.rectangle(img, 
                            pt1=(1137,565), 
                            pt2=(1247,600),
                            color=(171,242,0),
                            thickness=1
                            )
                        cv2.imwrite('/home/dj/work/drone_fire/ocr-train_v3/pngfolder/frame%d.jpg'%i,img)
                    elif h==480:        
                        img=Image.fromarray(img)
                        draw=ImageDraw.Draw(img)
                        draw.text((650 , 320),pred,font=font)
                        img = np.array(img)
                        cv2.rectangle(img, 
                            pt1=(630,355), 
                            pt2=(710,375),
                            color=(171,242,0),
                            thickness=1
                            )
                        cv2.imwrite('/home/dj/work/drone_fire/ocr-train_v3/pngfolder/frame%d.jpg'%i,img)

~~~
