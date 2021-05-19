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

## 🔎 EDA
<img width="500" alt="heatmap_new" src="https://user-images.githubusercontent.com/71582831/118845035-63308a80-b906-11eb-97e3-f1ae3ca98d77.png">


## 📈 Modeling
- Pipeline 구축
  - Linear Regression
  - Decison Tree Regressor
  - PCA + Linear Regression
  - PLS Regression
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Random Forest Regressor
  - Support Vector Regressor
  - Lasso
  - Ridge
  - Kernel Ridge Regression

- Model Evaluation
<img width="500" alt="model_result" src="https://user-images.githubusercontent.com/71582831/118845152-7e9b9580-b906-11eb-8448-e5f4bd538049.png">

- Results visualization
<div>
<img width="400" alt="linear_results" src="https://user-images.githubusercontent.com/71582831/118846047-49437780-b907-11eb-9a6c-7ff9043aeffa.png">
<img width="400" alt="randomforest_results" src="https://user-images.githubusercontent.com/71582831/118846172-69733680-b907-11eb-9c73-82941920e573.png"></div>
<br>

## 👩🏻‍🤝‍🧑🏻 Contributors
- [Seyoung Ko](https://github.com/SeyoungKo) & [Hyunjin Kim](https://github.com/HyunjinKIM-Chloe)
<img width="300" src="https://user-images.githubusercontent.com/71582831/118420774-aae2c680-b6fa-11eb-87c9-9b14e002eced.gif">
