Fast Campus Data Science School 17th <b> Regression project </b>

# Regression for factors affecting Life Expectancy๐งฌ
  <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
  <img src="https://img.shields.io/apm/l/vim-mode"/>

## :pencil: ๊ฐ์
### 1๏ธโฃ ์ฃผ์  ์ ์  ๋๊ธฐ
- Covid 19๋ก ๊ฑด๊ฐ์ ๋ํ ๊ด์ฌ โ
- 100์ธ ์๋๋ฅผ ์ฝ ์์ ๋๊ณ  ์๋ ์์ฆ, ์ด๋ค ์์ธ์ด ์ฅ์์ ์ํฅ์ ๋ฏธ์น๋์ง ํ๊ตฌํด๋ณด์
<!-- - โป ๊ธฐ๋ ์๋ช์ด๋?
    - ํน์  ์๊ธฐ์ ํ์ด๋ ์ธ๊ตฌ์ ์์๋๋ ์๋ช -->

### 2๏ธโฃ ํ๋ก์ ํธ ๋ชฉ์ 
- ๊ธฐ๋ ์๋ช๊ณผ ์ฐ๊ด๋ ์์ธ ๋ถ์
- ๋ค์ค ์ ํ ํ๊ท ๊ธฐ๋ฐ ํ๊ท ๋ชจ๋ธ ๊ณต์ํ
- ๊ฐ๋ฐ๋์๊ตญ ๋ฑ ๊ธฐ๋ ์๋ช์ด ๋ฎ์ ๊ตญ๊ฐ ๋์, ๊ธฐ๋ ์๋ช์ด ๋ฎ์ ์ด์  ๋ฐ ์๋ช ์ ๊ณ  ๋ฐฉ์ ๋ถ์

### 3๏ธโฃ Dataset
- [Kaggle "Life-Expextancy (WHO)](https://www.kaggle.com/kumarajarshi/life-expectancy-who)
  - 2000~2015๋ 193๊ฐ๊ตญ์ ๊ธฐ๋์๋ช ๋ฐ ๊ด๋ จ ์์ธ ๋ฐ์ดํฐ์
  - ์ข์๋ณ์(Target): Life expactancy
  - ๋๋ฆฝ๋ณ์(Features): ๊ฒฝ์ , ์ฌํ (์๋ฐฉ์ ์ข, ๊ต์ก ๋ฑ), ์ฌ๋ง๋ฅ  ๋ฑ 19๊ฐ ์์ธ
- [THE WORLD BANK](https://www.worldbank.org/en/home)
  - GDP per capita, Death rates, Population ๋ฑ ๊ฒฐ์ธก์น ์ฒ๋ฆฌ ๋ฐ ์ถ๊ฐ์ฉ ์๋ฃ ์์ง

## ๐ Modeling
- Pipeline & GridSearchCV
    <div>
   <img width="345" src="https://user-images.githubusercontent.com/71582831/119071725-4e96e400-ba25-11eb-9de2-a8ed4f6249d9.png">
   <img width="490" alt="model_result" src="https://user-images.githubusercontent.com/71582831/118845152-7e9b9580-b906-11eb-8448-e5f4bd538049.png"></div>

- Predict
  - 2019๋ ํ๊ตญ ๊ธฐ๋์๋ช: 83์ธ
  - Linear Regression Predicted: 81.35์ธ

- Results visualization 
  <div>
  <img width="500" alt="linear_results" src="https://user-images.githubusercontent.com/71582831/118846047-49437780-b907-11eb-9a6c-7ff9043aeffa.png">
  <img width="500" alt="randomforest_results" src="https://user-images.githubusercontent.com/71582831/118846172-69733680-b907-11eb-9c73-82941920e573.png"></div>
  <br>
  
## ๐ EDA
#### Life expectancy์ ๋์ ์๊ด๊ด๊ณ๋ฅผ ๊ฐ์ง๋ Features
  - ์์ ์๊ด๊ด๊ณ:  ๊ต์ก ์ ๋(Schooling) 0.8, ์์์ ์๋ ๊ตฌ์ฑ(Income composition of resources) 0.8
  - ์์ ์๊ด๊ด๊ณ: ์์ ์ฌ๋ง๋ฅ (Infant deaths) -0.9, ์ฑ์ธ ์ฌ๋ง๋ฅ (Adult mortality) -0.7
    <img width="500" alt="heatmap_new" src="https://user-images.githubusercontent.com/71582831/118845035-63308a80-b906-11eb-97e3-f1ae3ca98d77.png">

#### Status์ ๋ฐ๋ฅธ ๊ตญ๊ฐ๋ณ Features ๋ถํฌ
  - Developed(Status 1 - Deep color) / Developing (Status 0 - Light color)
  - Life expectancy
      <div><img width="700" src= "https://user-images.githubusercontent.com/71582831/119067183-76357e80-ba1c-11eb-839d-3c90f263789c.png"></div>
  - Life expectancy ์์/ํ์ 10% ๊ตญ๊ฐ๋ค ๋น๊ต
     - Life expectancy์ ์์ ์๊ด๊ด๊ณ๋ฅผ ๊ฐ์ง Features: Developed ๊ตญ๊ฐ๊ฐ ์์๊ถ์ ์ฐจ์ง
     - Life expectancy์ ์์ ์๊ด๊ด๊ณ๋ฅผ ๊ฐ์ง Features: Developing ๊ตญ๊ฐ๊ฐ ์์๊ถ์ ์ฐจ์ง
      <div><img width="350" src= "https://user-images.githubusercontent.com/71582831/119073677-b4389f80-ba28-11eb-98d1-ca33de326c9d.png">
      <img width="350" src= "https://user-images.githubusercontent.com/71582831/119073681-b569cc80-ba28-11eb-9379-59c6d370a605.png">
      <img width="350" src= "https://user-images.githubusercontent.com/71582831/119073836-ecd87900-ba28-11eb-80ec-eee1a54efb41.png">
      <img width="350" src= "https://user-images.githubusercontent.com/71582831/119073841-ee09a600-ba28-11eb-9dc5-d38908014a92.png"></div>


## ๐ฉ๐ปโ๐คโ๐ง๐ป Contributors
- [Seyoung Ko](https://github.com/SeyoungKo) & [Hyunjin Kim](https://github.com/HyunjinKIM-Chloe)
<img width="300" src="https://user-images.githubusercontent.com/71582831/118420774-aae2c680-b6fa-11eb-87c9-9b14e002eced.gif">
