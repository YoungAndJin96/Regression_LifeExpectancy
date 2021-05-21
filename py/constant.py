# 데이터 저장 경로
PATH = '../datas/'

# 대분류 카테고리별 컬럼 구분
CAT = {'economy': ['PERCENTAGE_EXPENDITURE', 'TOTAL_EXPENDITURE', 'GDP', 'POPULATION',
                   'INCOME_COMPOSITION_OF_RESOURCES'],
       'death_rate': ['INFANT_DEATHS', "ADULT_MORTALITY", 'UNDER_FIVE_DEATHS'],
       'illness_rate': ['THINNESS_1_19_YEARS', 'THINNESS_5_9_YEARS', 'MEASLES', 'HIV/AIDS'],
       'vaccine': ['HEPATITIS_B', 'POLIO', 'DIPHTHERIA'],
       'others': ['SCHOOLING', 'BMI', 'ALCOHOL']}

# Region to Integer
REGION = {'South Asia': 0, 'Europe & Central Asia': 1,
          'Middle East & North Africa': 2, 'Sub-Saharan Africa': 3,
          'Latin America & Caribbean': 4, 'East Asia & Pacific': 5, 'North America': 6}
