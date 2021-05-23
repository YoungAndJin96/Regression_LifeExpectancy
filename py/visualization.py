import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import constant

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

        # GDP per capita 데이터 추가
        gdp_percap = pd.read_csv("../datas/worldbank_gdppercap.csv")
        gdp_percap = gdp_percap.groupby('Country Code').mean()
        gdp_percap.drop(columns=['2016', '2017', '2018', '2019', '2020'], axis=1, inplace=True)

        # life_df에 GDP per capita 컬럼 추가
        original["GDP_PERCAP"] = [gdp_percap.loc[original['COUNTRYCODE'][i]][str(original['YEAR'][i])] for i in
                                  range(len(original))]

        original["GDP_PERCAP"].fillna(original["GDP"] / original["POPULATION"], inplace=True)

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

    def set_region_df(self, data, region):
        data = data.drop(['COUNTRYCODE'], axis=1)
        data = data.replace({'REGION': region})

        # region별 dataframe 선언
        regions_df = [pd.DataFrame(data=data[data['REGION'] == i]) for i in range(len(region))]

        # region별 연도에 따라 groupby
        year_merge_df = [regions_df[i].groupby('YEAR').mean().drop(['REGION', 'ISO3166'], axis=1)
                         for i in range(len(region))]

        scaled_region_datas = [pd.DataFrame(data=self.minmax_scaling(year_merge_df[i]))
                                for i in range(len(region))]

        return year_merge_df, scaled_region_datas

    def set_category(self, year_merge_df):
        # 리전별 대분류 카테고리로 구분한 데이터프레임 생성
        cat_region_df, regions_df = [], []

        for i in range(len(year_merge_df)):
            economy_df = year_merge_df[i][constant.CAT['economy']]
            death_df = year_merge_df[i][constant.CAT['death_rate']]
            illness_df = year_merge_df[i][constant.CAT['illness_rate']]
            vaccine_df = year_merge_df[i][constant.CAT['vaccine']]
            others_df = year_merge_df[i][constant.CAT['others']]

            regions_df.append(economy_df)
            regions_df.append(death_df)
            regions_df.append(illness_df)
            regions_df.append(vaccine_df)
            regions_df.append(others_df)

            cat_region_df.append(regions_df)
            regions_df = []

        return cat_region_df

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
        plt.interactive(False)
        px.line(scaled_data[constant.CAT[category]])

        return plt.show()

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

    # 전체 리전에 대한 컬럼 연도별 추이 plotly 그래프
    def show_year_regions(self, target_region, rows, cols, regions_df):
        category = []
        target_region_idx = constant.REGION[target_region] # 타겟 리전 데이터프레임 인덱스

        for i in range(len(constant.REGION)):
            category.append(regions_df[i][target_region_idx])

        lower_titles = [category[target_region_idx].columns.tolist()[i].lower().capitalize().replace('_', ' ')
                        for i in range(len(category[target_region_idx].columns))]

        fig = make_subplots(rows=rows, cols=cols, subplot_titles=lower_titles)
        count = 0
        colors = ['lightsteelblue', 'cornflowerblue', 'slateblue', 'darkviolet', 'plum', 'limegreen', 'mediumturquoise']

        for i in range(rows):
            for j in range(cols):
                if count == len(category[target_region_idx].columns):
                    break

                for k in range(len(constant.REGION)):
                    flag = True if count == 0 else False
                    fig.add_trace(go.Scatter(x=category[target_region_idx].index,
                                             y=category[k][category[k].columns[count]],
                                             name=list(constant.REGION.items())[k][0],
                                             showlegend=flag,
                                             marker=dict(color=colors[k]), line=dict(width=3)), row=i + 1, col=j + 1)

                count += 1

        fig.update_layout(font_size=14, width=2800, height=900, template="plotly_white")
        fig.update_annotations(font_size=19)

        # return fig.show()

    # 기대수명과 1인당 GDP scatter plotly 그래프
    def show_moving_scatter(self, data, size_target, animation_target, facet_col=None):
        px.scatter(data, x="GDP_PERCAP", y="LIFE_EXPECTANCY", animation_frame=animation_target,
                    animation_group="COUNTRY",
                    size=size_target, color="REGION", hover_name="COUNTRY", facet_col=facet_col,
                    log_x=True, size_max=60, range_y=[40, 100])

        # return fig.show()

    # 선진국 / 개발도상국으로 나라별 상관관계 높은 컬럼들의 수치 비교 plotly 그래프
    def show_status_barchart(self, data):
        mean_df = data.groupby(['COUNTRY']).mean().round(3).drop(['YEAR'], axis=1)

        # top corr columns
        cols = ['LIFE_EXPECTANCY', 'INCOME_COMPOSITION_OF_RESOURCES', 'SCHOOLING', 'INFANT_DEATHS',
                'ADULT_MORTALITY']
        lower_cols = [col.lower().capitalize().replace('_', ' ') for col in cols]
        colors = ['Burg', 'Darkmint', 'Purp', 'Teal', 'Magenta']

        for i in range(5):
            hdi_df = mean_df.sort_values(by=cols[i], ascending=False)

            fig = px.bar(hdi_df, x=hdi_df.index, y=hdi_df[cols[i]], color=hdi_df['STATUS'],
                         barmode='group', color_continuous_scale=colors[i])

            fig.update_layout(
                title_text=lower_cols[i],
                height=500,
                width=1000,
                template='plotly_white',
                font_color='grey'
            )

        # return fig.show()

if __name__ == '__main__':

    original = pd.read_csv(constant.PATH + 'life_expectancy_data_fillna.csv')
    original.columns = [cols.upper() for cols in original.columns.tolist()]

    p = Preprocessing(original)
    add_data = p.add_feature("gdppercap")
    pc_data = p.processing(original)
    top_features = p.corr_matrix(pc_data)
    year_merge_df, scaled_region_datas = p.set_region_df(pc_data, constant.REGION)
    cat_regions_df = p.set_category(year_merge_df)

    # 연도별로 groupby
    year_data = pc_data.groupby("YEAR").mean()

    v = Visualization(pc_data)
    v.show_year_cols(year_data)
    year_df = p.minmax_scaling(year_data)
    v.show_year_regions('South Asia', 2, 3, cat_regions_df)
    v.show_moving_scatter(pc_data, 'POPULATION', 'YEAR')
    v.show_status_barchart(pc_data)

