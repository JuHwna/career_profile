# 프로젝트 진행 상황
1. 데이터 수집 및 전처리
- 데이터 수집
  - 산불 사진
    - 깃허브, 사이트 등에서 드론 시점에서 찍은 화재 사진 수집
    - 아르고스 다인 측에서 보낸 동영상을 프레임 단위로 일정 부분 수집
~~~
import cv2
import os
import shutil

videoPath = './VideoFile/smoke'
imagePath = './images/'
file_list = os.listdir(videoPath)


for file in file_list:
    print(file)
    try:
        if not (os.path.isdir(videoPath + file)):
            os.makedirs(os.path.join(imagePath + file))
            cap = cv2.VideoCapture(videoPath + '/'+file)
            print(cap)
            count = 0
            ext = os.path.splitext(file)[0]
            while True:
                ret, image = cap.read()
                
            #    cv2.imwrite(imagePath + file + "/frame%d.jpg" % count, image)

             #   print('%d.jpg done' % count)
              #  count += 1
                image = cv2.resize(image, (512, 512))
                if(int(cap.get(1)) % 30 == 0):
                    savePath=os.path.join(imagePath+file)

                    cv2.imwrite(imagePath+"frame%d.jpg"  % count, image)
                    shutil.move(imagePath+"/frame" +str(count)+".jpg",savePath)
                    print('Saved frame%d.png' % count)
                    count += 1
            cap.release()

    except OSError as e:
        if e.errno != e.EEXIST:
            print("Failed to create directory!!!!!")
            raise
~~~


  - 산불 이외의 사진
    - 산불이라고 착각할만한 것(단풍, 빨간 꽃, 석양, 빨간 구름, 도시 불빛 등)을 구글 이미지에서 크롤링을 통해 수집

![image](https://user-images.githubusercontent.com/49123169/135943624-f6365b8e-2fa1-4bb9-8620-58cd8a2f17d3.png)

~~~
from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
import time 
import os 
import urllib.request 
from multiprocessing import Pool 
import pandas as pd

key=pd.read_csv('./keyword3.txt',encoding='UTF-8',names=['keyword']) 
keyword=[] 
[keyword.append(key['keyword'][x]) for x in range(len(key))]

def createFolder(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print ('Error: Creating directory. ' + directory)

def image_download(keyword): 
    createFolder('./'+keyword+'_high resolution') 
    
    chromedriver = 'C:/Users/admin/chromedriver.exe' 
    driver = webdriver.Chrome(chromedriver) 
    print(keyword, '검색') 
    driver.get('https://www.google.co.kr/imghp?hl=ko') 
    
    Keyword=driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input') 
    Keyword.send_keys(keyword) 
    
    driver.find_element_by_xpath('//*[@id="sbtc"]/button').click() 
    print(keyword+' 스크롤 중 .............') 
    elem = driver.find_element_by_tag_name("body") 
    for i in range(45):
        try:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.1)
        except:
            try:
                driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[4]/div[2]/input').click()
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.1)
            except:
                pass 
    
    images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") 
    print(keyword+' 찾은 이미지 개수:',len(images))
    count=1
    
    for image in images: 
        try:
            image.click() 
            time.sleep(2)
            imgUrl =driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img').get_attribute('src')
            urllib.request.urlretrieve(imgUrl, "./"+keyword+"_high resolution/"+keyword+"_"+str(count)+".jpg") 
            print(str(k+1)+'/'+str(len(links))+' '+keyword+' 다운로드 중....... Download time : '+str(time.time() - start)[:5]+' 초')
            count+=1
        except: 
            count+=1
            continue
    print(keyword+' ---다운로드 완료---') 
    driver.close()
~~~

- 전처리
  - 산불
    - labelImg를 통해 산불을 메인으로, 연기는 부수적으로 라벨링 진행(산불 : fire, 연기 : smoke)
    - 산불은 큰 불보다는 작은 불씨들을 위주로 잡게 작게 라벨링 작업을 진행함
    - 연기는 산불 주변 위주로 라벨링 진행
  - 산불 이외
    - labelImg은 빈 라벨링 파일(xml형식)을 만들 수 없었음
    - 모델 학습에서는 꼭 xml 파일이 있어야지 학습하는 파일로 인식하기 때문에 임의로 xml 파일을 만들어서 적용
~~~
ss=os.path.join('C:/Users/admin/Desktop/non-fire/train')
ss_list=os.listdir(ss)
for i in range(len(ss_list)):
    filename=ss_list[i].split('.')[0]
    file_path=os.path.realpath(ss_list[i])
    img=cv2.imread(ss+'/'+ss_list[i])
    h,w,c=img.shape
    root = Element('annotation')
    SubElement(root, 'folder').text = '250'
    SubElement(root, 'filename').text = filename + '.jpg'
    SubElement(root, 'path').text = file_path
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(w)
    SubElement(size, 'height').text = str(h)
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    tree = ElementTree(root)
    tree.write('C:/Users/admin/Desktop/non-fire/xml/' + filename +'.xml')
~~~
2. 모델 학습

- 모델 학습 환경
  - 초창기 환경
    - 회사 내 GPU 2개로 학습 진행하려고 했으나 세팅에서 문제가 발생
    - Colab 환경에서 진행(Colab 기본 환경에서 GPU로 학습 진행)
  - 중후반기 환경
    - Colab 환경에서 진행 

- 모델 학습 진행 상황
1. 모델 학습 진행 상황 - 초창기
   - 이 때에는 산불 이미지만 가지고 학습 진행
   - 주어진 YOLO V2에서 어떠한 튜닝도 없이 똑같은 하이퍼파라미터와 에포크로 진행
       - Epoch : 35, early stopping : 0.001

|초창기 모델|accuracy|
|----------|--------|
|![초창기 모델](https://user-images.githubusercontent.com/49123169/135943317-197abdcc-f9bb-44da-bd5e-82f43d07cb86.png)|![image](https://user-images.githubusercontent.com/49123169/135943527-9016f091-bdea-407e-8e3c-ff9ccca7a0c3.png)


2. 모델 학습 진행 상황 - 중반기
   - 산불 이외의 사진들이 불로 잡는 현상을 발견
   - 그래서 산불로 인식할 수 있는 요소들을 찾아서 데이터 수집 후 모델 학습에 넣어서 진행
       - 초반에 산불로 인식할 수 있는 요소들을 xml 파일을 넣어서 진행했는데 학습 데이터로 인식 안되는 경우가 발생
         - 코드 수정하여 bounding box 요소가 없어도 학습 데이터로 인식할 수 있게 변경
         - >parse_annotation쪽에 있는 if len(img['object'])>0 지움

|초창기 모델|accuracy|
|----------|--------|
|![두번째 모델](https://user-images.githubusercontent.com/49123169/135944465-dff08fd9-b102-4a46-9b3f-3cdbd5f0685d.png)|![image](https://user-images.githubusercontent.com/49123169/135944053-21b2a038-faab-44d6-b29d-1807d73f2bb9.png)|


- 오탐지는 줄었지만 산불 인식률을 똑같은 문제 발생
   - 하이퍼 파라미터를 조정하기 시작함
       - 먼저 optimizer 조정 시도(adam -> sgd 등)
       - 더 안 좋은 결과 발견
       - adam으로 돌아가기로 결정
   - epoch을 늘리기로 시도
       - 더 학습이 될 거 같은 생각과 기존에 진행한 cnn 프로젝트에서도 70은 했기 때문에 더 늘리기로 결정
       - 이 과정에서 early stopping도 조정함
       - ![image](https://user-images.githubusercontent.com/49123169/135945372-64aa1e0d-c431-4bdf-88fc-f0e8f135f71e.png)
       - epoch 100까지 학습했을 때 더 좋은 결과를 나타냄

  3. 모델 학습 진행 상황 - 마지막
     - 연기까지 탐지하는 모델 개발 필요(중요하지 않지만 일부분이라도 잡으면 좋겠다는 요청)
     - 그래서 기존 산불 라벨링된 이미지에서 연기를 다시 라벨링 작업 실시
       - 연기도 연기로 인식될 수 있는 요소들을 데이터 수집 진행
       - ![image](https://user-images.githubusercontent.com/49123169/135945547-484d31f5-a81a-47b1-b415-75917cfea8c6.png)
     - 이 과정에서 산불 이미지도 더 수집하여 라벨링 작업 진행
       - 산불로 착각할 수 있는 요소들을 더 찾아 수집 후 모델에 넣음
       - ![image](https://user-images.githubusercontent.com/49123169/135945538-189d1b6f-16f2-4962-a684-326922d2abe9.png)
     - epoch을 기존보다 더 늘리기 시도
       - epoch 150 vs 최대 epoch(early stopping으로 멈출 때까지)


|epoch 150|epoch 223(최대)|
|---------|---------------|
|![image](https://user-images.githubusercontent.com/49123169/135945613-cd97c30c-107f-4a54-a15f-67ccd2f5a99a.png)|![image](https://user-images.githubusercontent.com/49123169/135945625-4f440c09-58b5-4535-b587-dd309b99d5f9.png)|



       - epoch 150일 때가 제일 성능이 괜찮아서 해당 가중치로 진행




4. 드론 실증 실험

- 드론 실증 실험 전 test로 몇 번 진행
  - 이 때는 모의 산불을 발생시키지 않고 서버와 연동이 잘 되는지 테스트 진행
  - rtsp로 영상이 잘 들어오는 거 확인했으며 모델이 잘 돌아가는지를 확인함
  - 문제 발생
    - ![image (1)](https://user-images.githubusercontent.com/49123169/143828068-6fe04401-eafc-4db3-93bf-d3bb680988f0.png)
    - 실시간 영상 프레임을 모델에 넣고 돌리는 도중 해당 에러 발생
    - 코덱 문제인 줄 알고 영상 코덱을 변경하거나 코덱을 인식하는 코드 수정
      - 실패
      - 원인 파악 : 25프레임 실시간 영상을 모델에 넣고 돌리기에는 현재 서버 환경에서는 힘듬
      - 해결 방안 : 중개 서버를 만들어 5프레임 영상으로 송출되게 만들었음(DE팀이 해결)
- 드론 실증 실험 진행
  - 1차 실험 진행
    - 이 때는 모델을 실시간으로 돌리지 않고 DE팀이 만든 중개 서버까지 영상이 5프레임으로 잘 송출되는지를 보려고 작업함
    - Vision, IR 영상 둘 다 찍으면서 모의 산불 실험까지 진행
    - 이후, DE팀에서 저장한 5프레임 영상을 가지고 모델에 돌려 모델 정확도 테스트 진행(720p, 1080p 영상 각각 진행)
    - 결과
      - 720p로 했을 때
        - 모의 산불을 발생시켰는데 불이랑 연기가 육안으로 보이지 않을 정도로 약해서 탐지하는 게 불가능했음
        - 거기에다가 720p라고 했지만 화질이 엄청 좋지 않아 탐지가 가능할지 의문이었음
        - 오탐지 위주로 살펴보았는데 모의 실험 장 내에 빨간 지붕과 회색 지붕, 단풍이 조금 있었는데 해당 부분을 불이나 연기로 잡는 현상이 자주 보였음
      - 1080p로 했을 때
        - 720p보다 모의 산불의 크기가 커져서 연기가 육안으로 보였음
        - 1080p 화질이라고 믿기 힘들 정도였지만 720p보다는 좋다는 것을 확인함
        - 연기가 프레임에 들어왔을 때 탐지하는 모습을 보였음(위치 정확도 최소 30~50%)(탐지 정확도 65%)
        - 오탐지는 있긴 했지만 자주 오탐지 되지는 않았음(총 오탐지 횟수 11개)
  - 2차 실험 진행
    - 이 때는 모델을 실시간으로 돌리지 않고 DE팀이 만든 중개 서버까지 영상이 5프레임으로 잘 송출되는지를 보려고 작업함
    - Vision, IR 영상 둘 다 찍으면서 모의 산불 실험까지 진행
    - 이후, DE팀에서 저장한 5프레임 영상을 가지고 모델에 돌려 모델 정확도 테스트 진행(1080p 영상 진행)
    - 결과
      - 송출이 너무 안 되서 영상 저장이 제대로 되지 않았음
      - 거기에다가 화질이 더 안 좋아져서 오탐지가 꽤 많이 나옴

5. End-to-End 과정
- 송출부터 관제시스템까지 e2e 개발 진행
  - da팀은 모델 결과를 thinkboard에 잘 안착하는 것만 하면 되었음(탐지 이미지와 로그 파일) 
- 탐지 이미지는 aws s3 적재로, 로그는 api서버로 보내면 되었음
  - 탐지 이미지 aws s3적재 코드      

~~~
def handle_upload_img(j,dt):
    s3_client=boto3.client('s3',aws_access_key_id=ACCESS_KEY_ID,aws_secret_access_key=ACCESS_SECRET_KEY)
    s3_client.upload_file('/home/dj/work/drone_fire/ocr-train_v3/pngfolder/DroneID_date_mode_sequence%d.jpg'%j,BUCKET_NAME,'OCR/images/DroneID_{0}_OCR_{1}.jpg'.format(dt,j))
    handle_upload_img(cnt,now.strftime('%Y%m%d'))
~~~


  - 로그 파일은 json 파일로 보내줘야 했음

~~~
def send_log_as_json(url, h, d):
    response=requests.post(url, headers=h, data=d)

file_data=OrderedDict()
file_data["ts"]=int(time.time())*1000


order_values=OrderedDict()
order_values["droneID"]="sample_ID"
order_values["ts"]=int(time.time())*1000
order_values["mode"]="IR"
order_values["cnt"]=cnt
#order_values["accuracyfire"]=fin_score2
order_values["accuracyfire"]="null"
order_values["accuracysmoke"]="null"
order_values["predictedlabelFire"]="null"
order_values["predictedlabelSmoke"]="null"
temp_pred=pred[2:-1]
temp_jum=[i for i in range(len(temp_pred)) if temp_pred[i]=='.']

if len(temp_jum)>=2:
    for j in temp_jum[1:]:
        temp_pred[j]='0'
    order_values["temp"]=temp_pred
else:
    order_values["temp"]=temp_pred

order_values["fireDangerIR"]=200
order_values["fireConfirmIR"]=800
order_values['imageSrc']="https://firedrones3.s3.ap-northeast2.amazonaws.com/OCR/images/DroneID_{0}_OCR_{1}.jpg".format(now.strftime('%Y%m%d'),cnt)
order_values["finish"]="finish"
order_values["addinfo"]={}
file_data["values"]=order_values
payload=json.dumps(file_data,ensure_ascii=False,indent='\t').encode("utf-8") 

t=threading.Thread(target=send_log_as_json, args=(log_url, headers, payload))
t.start()
~~~

