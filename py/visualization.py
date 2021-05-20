import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image as pil
import statsmodels.api as sm
import seaborn as sns
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.io as pio
import pandas as pd

pd.options.plotting.backend = 'plotly'
pio.renderers.default = "browser"
pio.renderers.default = "notebook_connected"

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)


class Preprocessing:
    def __init__(self, original):
        self.original = original

    def add_feature(self, filename=None):
        path = "../datas/worldbank_"
        original.columns = [cols.upper() for cols in original.columns.tolist()]

        if not filename == None:
            df = pd.read_csv(f"{path}{filename}.csv").groupby('Country Code').mean()
            df.drop(columns=['2016', '2017', '2018', '2019', '2020'], axis=1, inplace=True)
            col_name = filename.upper()
            original[col_name] = [df.loc[original['COUNTRYCODE'][i]][str(original['YEAR'][i])] for i in
                                  range(len(original))]

        return original

    def processing(self, data):
        # Nan값 GDP/POP으로 대체
        data["GDPPERCAP"].fillna(data["GDP"] / data["POPULATION"], inplace=True)

        # Developing: 0, developed: 1
        data["STATUS"] = [row.replace("Developing", "0") for row in data["STATUS"].tolist()]
        data["STATUS"] = [row.replace("Developed", "1") for row in data["STATUS"].tolist()]
        data["STATUS"] = [int(row) for row in data["STATUS"].tolist()]

        return data

    def corr_matrix(self, data):
        # 기대수명에 대한 나머지 feature들의 상관관계
        corr_matrix = data.drop(['COUNTRYCODE', 'ISO3166', 'COUNTRY', 'YEAR', 'ISO3166', 'REGION', 'INCOMEGROUP'],
                                axis=1).corr()
        corr_matrix['LIFE_EXPECTANCY'].sort_values(ascending=False)

        # LIFE_EXPECTANCY와 높은 상관관계를 가지는 피처 순 정렬
        top_corr = abs(corr_matrix['LIFE_EXPECTANCY']).sort_values(ascending=False)[:6]
        top_features = top_corr.index.tolist()

        return top_features

    def minmax_scaling(self, data):
        # mix-max scaling
        scaled_data = pd.DataFrame(preprocessing.minmax_scale(data))
        scaled_data.index = data.index
        scaled_data.columns = data.columns.tolist()

        return scaled_data


class Visualization:
    def __init__(self, original):
        self.original = original

    # 전체 컬럼별 평균값에 대한 전체 연도 추이 그래프
    def show_year_cols(self, year_df):
        plt.figure(figsize=(20, 5))
        plt.title("Yearly Features Fluctuation", fontsize=15)
        for a in year_df.columns.tolist():
            plt.plot(year_data.index, preprocessing.minmax_scale(year_df[a]), label=a)

        plt.xlabel("Year")
        plt.legend()

        return plt.show()

    # 대분류 카테고리별로 나눠본 전체 연도별 추이 plotly 그래프
    def show_px_lines(self, scaled_data, category):
        px.line(scaled_data[cat[category]])

    # 각 컬럼별 전체 국가 연도별 평균 추이 plotly 그래프
    def show_year_sep(self, scaled_data):
        # raw data의 연도별 feature 추이
        lower_titles = [scaled_data.columns.tolist()[i].lower().capitalize().replace('_', ' ')
                        for i in range(len(scaled_data.columns))]

        fig = make_subplots(rows=7, cols=3, subplot_titles=lower_titles)
        count = 0

        for i in range(7):
            for j in range(3):
                if count == 19:
                    break
                fig.add_trace(go.Scatter(x=scaled_data.index,
                                         y=scaled_data[scaled_data.columns[count]],
                                         name=scaled_data.columns[count],
                                         line=dict(width=3.5)), row=i + 1, col=j + 1)
                count += 1

        fig.update_layout(title='Life Expectancy Feautre 연도별 추이', font_size=14,
                          width=2500, height=2000, template="plotly_white")
        fig.update_annotations(font_size=18)

        return fig.show()


if __name__ == '__main__':
    PATH = '../datas/'
    # 대분류 카테고리로 컬럼 나누기
    cat = {'economy': ['PERCENTAGE_EXPENDITURE', 'TOTAL_EXPENDITURE', 'GDP', 'POPULATION',
                       'INCOME_COMPOSITION_OF_RESOURCES'],
           'death_rate': ['INFANT_DEATHS', "ADULT_MORTALITY", 'UNDER_FIVE_DEATHS'],
           'illness_rate': ['THINNESS_1_19_YEARS', 'THINNESS_5_9_YEARS', 'MEASLES', 'HIV/AIDS'],
           'vaccine': ['HEPATITIS_B', 'POLIO', 'DIPHTHERIA'],
           'others': ['SCHOOLING', 'BMI', 'ALCOHOL']}

    original = pd.read_csv(PATH + 'life_expectancy_data_fillna.csv')
    original.columns = [cols.upper() for cols in original.columns.tolist()]

    p = Preprocessing(original)
    add_data = p.add_feature("gdppercap")
    pc_data = p.processing(original)
    top_features = p.corr_matrix(pc_data)

    # 연도별로 groupby
    year_data = pc_data.groupby("YEAR").mean()

    v = Visualization(pc_data)
    v.show_year_cols(year_data)  # 전체 컬럼별 평균값에 대한 전체 연도 추이

    year_df = p.minmax_scaling(year_data)
    v.show_px_lines(year_df, 'economy')  # 대분류 카테고리별로 나눠본 전체 연도별 추이 그래프
    v.show_year_sep(year_df) # 각 컬럼별 전체 국가 연도별 평균 추이 plotly 그래프