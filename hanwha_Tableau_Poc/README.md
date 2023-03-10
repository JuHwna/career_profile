# 한화생명 Tableau 도입 PoC
## 1. 기간
- 2021.04.19 ~ 2021.05.18
## 2. 목표
- 태블로의 기능이 적절한지 보고 도입하기 위해 PoC 진행
- 태블로로 시각화했을 때 회사 내 업무에 적용하고 더 나은지 보기 위해

## 3. 인원
- 3명

## 4. 사용 툴
- 태블로, PostgreSQL

## 5. 진행 과정
(1) 태블로 서버 및 태블로 데스크탑 설치
- 태블로가 설치되지 않았기 때문에 사전에 태블로 설치를 위해 필요한 OS 환경과 포트 환경 구성 요청
- 이후, 태블로 서버 설치 진행
  - 이때 태블로 서버를 설치하는 방법을 몰라 처음에 헤맸었음
  - 결국 협력사 직원의 도움을 통해 태블로 설치하는 법을 옆에서 보면서 설치 서포트를 함
- 고객사 컴퓨터에 태블로 데스크탑 설치

(2) 데이터 마트 구축 및 수정
- 기존에 고객사가 데이터 마트를 새로 구축하고 있었고 시각화에 필요한 쿼리를 주었음
  - 확인해본 결과, 대부분 사용 가능했지만 일부분은 시각화에 필요한 열들이 들어 있지 않았음
  - 그래서 사용 테이블을 모두 조회해본 뒤에 필요한 열들이 들어간 테이블을 조인시켜서 새로운 데이터 마트를 구성했음
  
(3) 태블로 시각화 진행
- 사전에 제시한 대시보드 구성이 존재하여 해당 구성에 맞추어서 진행
- 처음에는 틀을 먼저 만든 후에 디자인까지 비슷하게 구성
- 이후에는 해당 틀 안에 각 그래프들을 넣어 대시보드 개발

(4) 값 검증 과정 진행
- 고객사와 같이 태블로로 만든 대시보드에 보이는 값들이 자신들이 가진 데이터와 일치하는지 확인
- 확인 결과, 다른 값들이 존재하는 것을 확인
  - 원인이 데이터 마트에 쿼리가 일부 잘못 된 것을 확인
  - 쿼리 수정 후 다시 확인 결과 모든 값이 맞다는 것을 확인
  
