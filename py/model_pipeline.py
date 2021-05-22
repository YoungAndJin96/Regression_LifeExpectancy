from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LassoCV , ElasticNetCV , RidgeCV
from sklearn.pipeline import Pipeline, FeatureUnion
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor as GBR
from xgboost import XGBRegressor as XGBR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression as  PLS
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.io as pio
init_notebook_mode(connected=True)
pio.renderers.default = "notebook_connected"

import pandas as pd
import numpy as np
import time
import pickle
import joblib
import constant

original = pd.read_csv(constant.PATH + 'life_expectancy_data_fillna.csv')

class ModelPipeline:
    def pipe_model(self):
        pipe_linear = Pipeline([
            ('scl', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('fit', LinearRegression())])
        pipe_tree = Pipeline([
            ('scl', StandardScaler()),
            ('fit', DTR())])
        pipe_lasso = Pipeline([
            ('scl', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('fit', Lasso(random_state=13))])
        pipe_ridge = Pipeline([
            ('scl', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('fit', Ridge(random_state=13))])
        pipe_pca = Pipeline([
            ('scl', StandardScaler()),
            ('pca', PCA()),
            ('fit', LinearRegression())])
        pipe_pls = Pipeline([
            ('scl', StandardScaler()),
            ('fit', PLS())])
        pipe_gbr = Pipeline([
            ('scl', StandardScaler()),
            ('fit', GBR())])
        pipe_xgbr = Pipeline([
            ('scl', StandardScaler()),
            ('fit', XGBR(random_state=13))])
        pipe_rfr = Pipeline([
            ('scl', StandardScaler()),
            ('fit', RFR(random_state=13))])
        pipe_svr = Pipeline([
            ('scl', StandardScaler()),
            ('fit', SVR())])
        pipe_KR = Pipeline([
            ('scl', StandardScaler()),
            ('fit', KernelRidge())])

        return [pipe_linear, pipe_tree, pipe_lasso, pipe_ridge, pipe_pca, pipe_pls,
                pipe_gbr, pipe_xgbr, pipe_rfr, pipe_svr, pipe_KR]

    def grid_params(self, max_depth, split_range):
        max_depth = max_depth
        min_samples_split_range = split_range

        grid_params_linear = [{
            "poly__degree": np.arange(1, 3),
            "fit__fit_intercept": [True, False]
        }]
        grid_params_tree = [{

        }]
        grid_params_lasso = [{
            "poly__degree": np.arange(1, 3),
            "fit__tol": np.logspace(-5, 0, 10),
            "fit__alpha": np.logspace(-5, 1, 10)
        }]
        grid_params_ridge = [{
            "poly__degree": np.arange(1, 3),
            "fit__alpha": np.linspace(2, 5, 10),
            "fit__solver": ["cholesky", "lsqr", "sparse_cg"],
            "fit__tol": np.logspace(-5, 0, 10)
        }]
        grid_params_pca = [{
            "pca__n_components": np.arange(2, 8)
        }]
        grid_params_pls = [{
            "fit__n_components": np.arange(2, 8)
        }]
        grid_params_gbr = [{
            "fit__max_features": ["sqrt", "log2"],
            "fit__loss": ["ls", "lad", "huber", "quantile"],
            "fit__max_depth": max_depth,
            "fit__min_samples_split": min_samples_split_range
        }]
        grid_params_xgbr = [{
            "fit__max_features": ["sqrt", "log2"],
            "fit__loss": ["ls", "lad", "huber", "quantile"],
            "fit__max_depth": max_depth,
            "fit__min_samples_split": min_samples_split_range
        }]
        grid_params_rfr = [{

        }]
        grid_params_svr = [{
            "fit__kernel": ["rbf", "linear"],
            "fit__degree": [2, 3, 5],
            "fit__gamma": np.logspace(-5, 1, 10)
        }]
        grid_params_KR = [{
            "fit__kernel": ["rbf", "linear"],
            "fit__gamma": np.logspace(-5, 1, 10)
        }]

        return [grid_params_linear, grid_params_tree, grid_params_lasso, grid_params_ridge, grid_params_pca,
                grid_params_pls, grid_params_gbr, grid_params_xgbr, grid_params_rfr, grid_params_svr, grid_params_KR]

    def grid_cv(self, pipe, params):
        jobs = -1
        cv = KFold(n_splits=5, shuffle=True, random_state=13)
        grid_dict = constant.GRID_DICT

        return jobs, cv, grid_dict

    def model_save(self, pipe, params, jobs, cv, grid_dict):
        model_rmse, model_r2, model_best_params, model_fit_times, model_res = {}, {}, {}, {}, {}

        for idx, (param, model) in enumerate(zip(params, pipe)):
            start_time = time.time()

            search = GridSearchCV(model, param, scoring="neg_mean_squared_error",
                                  cv=cv, n_jobs=jobs, verbose=-1)
            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)

            model_rmse[grid_dict.get(idx)] = np.sqrt(mse(y_test, y_pred))
            model_r2[grid_dict.get(idx)] = r2(y_test, y_pred)
            model_best_params[grid_dict.get(idx)] = search.best_params_
            model_fit_times[grid_dict.get(idx)] = time.time() - start_time

            joblib.dump(search, f'../models/{grid_dict.get(idx)}.pkl')

        print("------- all Model Saved -------")

        return model_rmse, model_r2, model_best_params, model_fit_times

    # Modeling 결과 시각화
    def model_res_barchart(self, res_df):
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.set(font_scale=2)
        ax = sns.barplot(y="Model", x="R2", data=res_df)

        return plt.show()

    # Modeling 결과 데이터프레임 저장
    def model_res_df(self, model_r2, model_rmse, model_fit_times):
        output = pd.DataFrame([model_r2.keys(), model_r2.values(), model_rmse.values(), model_fit_times.values()],
                              index=["Model", "R2", "RMSE", "Fit_times"]).T
        output.sort_values(["R2"], ascending=False, inplace=True)
        output['R2'] = [float(_) for _ in output['R2']]
        output['RMSE'] = [float(_) for _ in output['RMSE']]

        return output

class Preprocessing:
    # 컬럼 추가
    def add_feature(self, original, filename=None):
        path = constant.PATH + "worldbank_"
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
        data.columns = [cols.upper() for cols in data.columns.tolist()]

        if 'STATUS' in data.columns.tolist():
            data = pd.get_dummies(data, columns=['STATUS'], drop_first=True)

        return data

    # corr
    def get_top_features(self, data, drop_n=None):
        if drop_n is None:
            drop_n = len(data.columns)

        # LIFE_EXPECTANCY와 대한 나머지 feature들의 상관관계
        corr_matrix = data.drop(['COUNTRYCODE', 'ISO3166', 'COUNTRY', 'YEAR', 'REGION', 'INCOMEGROUP'], axis=1).corr()
        corr_matrix['LIFE_EXPECTANCY'].sort_values(ascending=False)

        # LIFE_EXPECTANCY와 높은 상관관계를 가지는 피처 순 정렬
        top_corr = abs(corr_matrix['LIFE_EXPECTANCY']).sort_values(ascending=False)[1:drop_n]
        top_features = top_corr.index.tolist()

        return top_features

    # lower fence, upper fence
    def get_fence(self, data, top_features):
        region = data['REGION'].unique().tolist()
        fence = {}

        for r in region:
            fence[r] = {}

            for i, f in enumerate(top_features):
                q1 = np.percentile(data[data['REGION'] == r][top_features[i]].values, 25)
                q3 = np.percentile(data[data['REGION'] == r][top_features[i]].values, 75)
                iqr = q3 - q1

                upper_fence = ((iqr * 1.5) + q3).round(3)
                lower_fence = (q1 - (iqr * 1.5)).round(3)

                fence[r][f] = [lower_fence, upper_fence]

        return fence

    # outlier processing
    def drop_outlier(self, data, fence, top_features):
        region = data['REGION'].unique().tolist()
        drop_list, target_idx = [], []

        for r in region:
            target_df = data[data['REGION'] == r]

            for f in top_features:
                drop_idx = target_df[(target_df[f] < fence[r][f][0]) |
                                     (target_df[f] > fence[r][f][1])].index.tolist()

                drop_list.append(drop_idx)

        # 제거 대상 인덱스
        target_idx = set([idx for lst in drop_list for idx in lst])
        data = data.drop(target_idx, axis=0)

        return data

if __name__ == '__main__':
    # 데이터 전처리
    p = Preprocessing()
    data = p.preprocessing(original)
    top_features = p.get_top_features(data, 5)
    fence = p.get_fence(data, top_features)
    prep_data = p.drop_outlier(data, fence, top_features)
    prep_data.reset_index(inplace=True, drop=True)

    # ======= Model Predict =======
    # 모델 파이프라인
    m = ModelPipeline()
    X = prep_data.drop(['COUNTRYCODE', 'ISO3166', 'COUNTRY', 'YEAR', 'LIFE_EXPECTANCY', 'REGION', 'INCOMEGROUP'],
                       axis=1)
    y = prep_data['LIFE_EXPECTANCY']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=13)
    split_range = [0.5, 0.7, 0.9]
    max_depth = [2, 4, 6, 8]

    pipe = m.pipe_model()
    params = m.grid_params(max_depth, split_range)
    jobs, cv, grid_dict = m.grid_cv(pipe, params)

    model_rmse, model_r2, model_best_params, model_fit_times = m.model_save(pipe, params, jobs, cv, grid_dict)

    # ====== Model Predict Result ======
    res_df = m.model_res_df(model_r2, model_rmse, model_fit_times)  # 모델링 결과 데이터프레임
    m.model_res_barchart(res_df)  # 모델링 결과 시각화