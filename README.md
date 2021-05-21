Fast Campus Data Science School 17th <b> Regression project </b>

# Regression for factors affecting Life Expectancy🧬
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
  <img src="https://img.shields.io/apm/l/vim-mode"/>

## :pencil: 개요
### 1️⃣ 주제 선정 동기
- Covid 19로 건강에 대한 관심 ↑
- 100세 시대를 코 앞에 두고 있는 요즘, 어떤 요인이 장수에 영향을 미치는지 탐구해보자
<!-- - ※ 기대 수명이란?
    - 특정 시기에 태어난 인구의 예상되는 수명 -->

### 2️⃣ 프로젝트 목적
- 기대 수명과 연관된 요인 분석
- 다중 선형 회귀 기반 회귀 모델 공식화
- 개발도상국 등 기대 수명이 낮은 국가 대상, 기대 수명이 낮은 이유 및 수명 제고 방안 분석

### 3️⃣ Dataset
- [Kaggle "Life-Expextancy (WHO)](https://www.kaggle.com/kumarajarshi/life-expectancy-who)
  - 2000~2015년 193개국의 기대수명 및 관련 요인 데이터셋
  - 종속변수(Target): Life expactancy
  - 독립변수(Features): 경제, 사회 (예방접종, 교육 등), 사망률 등 19개 요인
- [THE WORLD BANK](https://www.worldbank.org/en/home)
  - GDP per capita, Death rates, Population 등 결측치 처리 및 추가용 자료 수집

## 📈 Modeling
- Pipeline & GridSearchCV
    <div>
   <img width="400" src="https://user-images.githubusercontent.com/71582831/119071725-4e96e400-ba25-11eb-9de2-a8ed4f6249d9.png">
  <img width="500" alt="model_result" src="https://user-images.githubusercontent.com/71582831/118845152-7e9b9580-b906-11eb-8448-e5f4bd538049.png"></div>

- Predict
  - 2019년 한국 기대수명: 83세
  - Linear Regression Predicted: 81.35세

- Results visualization 
  <div>
  <img width="400" alt="linear_results" src="https://user-images.githubusercontent.com/71582831/118846047-49437780-b907-11eb-9a6c-7ff9043aeffa.png">
  <img width="400" alt="randomforest_results" src="https://user-images.githubusercontent.com/71582831/118846172-69733680-b907-11eb-9c73-82941920e573.png"></div>
  <br>
  
## 🔎 EDA
#### Life expectancy와 높은 상관관계를 가지는 Features
  - 양의 상관관계:  교육 정도(Schooling) 0.8, 자원의 소득 구성(Income composition of resources) 0.8
  - 음의 상관관계: 영아 사망률(Infant deaths) -0.9, 성인 사망률(Adult mortality) -0.7
  <img width="400" alt="heatmap_new" src="https://user-images.githubusercontent.com/71582831/118845035-63308a80-b906-11eb-97e3-f1ae3ca98d77.png">

#### Status에 따른 국가별 Features 분포
  - Developed(Status 1 - Deep color) / Developing (Status 0 - Light color)
  - Life expectancy
      <div><img width="700" src= "https://user-images.githubusercontent.com/71582831/119067183-76357e80-ba1c-11eb-839d-3c90f263789c.png"></div>
  - Life expectancy 상위/하위 10% 국가들 비교
     - Life expectancy와 양의 상관관계를 가진 Features 일수록, Developing 국가가 상위권을 차지
      <div><img width="350" src= "https://user-images.githubusercontent.com/71582831/119073677-b4389f80-ba28-11eb-98d1-ca33de326c9d.png">
      <img width="350" src= "https://user-images.githubusercontent.com/71582831/119073681-b569cc80-ba28-11eb-9379-59c6d370a605.png">
      <img width="350" src= "https://user-images.githubusercontent.com/71582831/119073836-ecd87900-ba28-11eb-80ec-eee1a54efb41.png">
      <img width="350" src= "https://user-images.githubusercontent.com/71582831/119073841-ee09a600-ba28-11eb-9dc5-d38908014a92.png"></div>


## 👩🏻‍🤝‍🧑🏻 Contributors
- [Seyoung Ko](https://github.com/SeyoungKo) & [Hyunjin Kim](https://github.com/HyunjinKIM-Chloe)
<img width="300" src="https://user-images.githubusercontent.com/71582831/118420774-aae2c680-b6fa-11eb-87c9-9b14e002eced.gif">
