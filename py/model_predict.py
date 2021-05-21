from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import constant
import joblib
import model_pipeline
import plotly.express as px

original = pd.read_csv(constant.PATH + 'life_expectancy_data_fillna.csv')

class ModelPredict:
    # 2019년 한국 데이터셋
    def set_testset(self, data):
        testset = pd.DataFrame(columns=data.columns)
        new_dict = {'COUNTRYCODE': 'KOR',
                    'YEAR': 2019, 'STATUS_Developing': 0,
                    'LIFE_EXPECTANCY': 83,
                    'GDP': 1646739.22,
                    'GDPPERCAP': 31846.2,
                    'SCHOOLING': 14,
                    'INFANT_DEATHS': 3,
                    'ADULT_MORTALITY': 110,
                    'INCOME_COMPOSITION_OF_RESOURCES': 0.916,
                    'POPULATION': 51709000,
                    }
        testset = testset.append(new_dict, ignore_index=True)
        testset.fillna(data.mean(), inplace=True)

        return testset

    # 컬럼 추가
    def add_feature(self, original, filename=None):
        path = "../datas/worldbank_"
        original.columns = [cols.upper() for cols in original.columns.tolist()]

        if not filename == None:
            df = pd.read_csv(f"{path}{filename}.csv").groupby('Country Code').mean()
            df.drop(columns=['2016', '2017', '2018', '2019', '2020'], axis=1, inplace=True)
            col_name = filename.upper()
            original[col_name] = [df.loc[original['COUNTRYCODE'][i]][str(original['YEAR'][i])]
                                  for i in range(len(original))]
        return original

    def preprocessing(self, data):
        # GDP per capita 데이터 추가
        data = self.add_feature(data, "gdppercap")

        # Nan값 GDP/POP으로 대체
        data["GDPPERCAP"].fillna(data["GDP"] / data["POPULATION"], inplace=True)
        data.columns = [cols.upper() for cols in original.columns.tolist()]

        if 'STATUS' in data.columns.tolist():
            data = pd.get_dummies(original, columns=['STATUS'], drop_first=True)

        return data

    # 저장한 모델을 불러와 predict하는 함수
    def model_predict(self, prep_data, model_idx, test_df):
        saved_model = joblib.load(f"../models/{constant.GRID_DICT.get(model_idx)}.pkl")

        X = prep_data.drop(['COUNTRYCODE', 'ISO3166', 'COUNTRY', 'YEAR', 'LIFE_EXPECTANCY', 'REGION', 'INCOMEGROUP'], axis=1)
        y = prep_data['LIFE_EXPECTANCY']
        # X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=13)

        X_pr = test_df[X.columns]
        y_pr = test_df['LIFE_EXPECTANCY']
        y_pred = saved_model.predict(X_pr)

        r2_score = r2(y_pr, y_pred)

        print(f"{constant.GRID_DICT.get(model_idx)} | 예측 기대 수명:{y_pred.mean().round(2)} / RMSE:, {np.sqrt(mse(y_pr, y_pred)).round(2)}")

        if r2_score is True:
            print("저장된 모델들의 R2:", r2_score)

    # 추가 데이터 예측 결과
    def extra_predict(self, extra_df):
        for idx, model_idx in enumerate(range(len(constant.GRID_DICT))):
            self.model_predict(model_idx, extra_df)

class Visualization():
    def split_data(self, prep_data, X):
        data_df = pd.concat([prep_data[X.columns], prep_data['LIFE_EXPECTANCY']], axis=1)

        X = data_df[X.columns]
        y = data_df['LIFE_EXPECTANCY']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

        data_df['split'] = 'train'
        data_df.loc[X_test.index, 'split'] = 'test'

        return (X_train, X_test), (y_train, y_test), data_df

    # reg plot 함수
    def reg_plotly(self, prep_data, X, model_idx):
        model = joblib.load(f"../models/{constant.GRID_DICT.get(model_idx)}.pkl")

        X, y, data_df = self.split_data(prep_data, X)
        model.fit(X[0], y[0])

        y_pred = model.predict(X)
        data_df['prediction'], data_df['y_test'], data_df['y_train'] = y_pred, y[1], y_pred

        fig = px.scatter(
            data_df, x=data_df['LIFE_EXPECTANCY'], y=data_df['prediction'],
            marginal_y='histogram', color='split', title=f"{constant.GRID_DICT.get(model_idx)} Model",
            color_discrete_sequence=['red', 'cornflowerblue'])

        fig.update_traces(histnorm='probability', selector={'type': 'histogram'})

        fig.add_shape(type='line', line=dict(dash='dash'),
                      x0=y.min(), y0=y.min(),
                      x1=y.max(), y1=y.max()
                      )

        fig.update_layout(font_size=14, width=1000, height=600, template='plotly_white')

        # return fig.show()

if __name__ == '__main__':
    pp = model_pipeline.Preprocessing()
    pipe = model_pipeline.ModelPipeline()

    mp = ModelPredict()
    data = pp.preprocessing(original)

    top_features = pp.get_top_features(data, 5)
    fence = pp.get_fence(data, top_features)
    prep_data = pp.drop_outlier(data, fence, top_features)
    prep_data.reset_index(inplace=True, drop=True)
    print(prep_data)

    kor_df = mp.set_testset(original)
    mp.model_predict()
    mp.extra_predict(kor_df)

    v = Visualization()
    model_idx = 0
    # v.reg_plotly(prep_data, X, model_idx)



