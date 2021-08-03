import pandas as pd
import numpy as np

pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)

df_og = pd.read_csv('vgames2.csv')

### 연도 장르 퍼블리셔의 결측치 제거와 이상치 치환
df_check = df_og.dropna(subset=['Year']).copy()

def valueToFour(x):
    x = int(x)
    if x <= 20:
        x += 2000
    if x >= 60 and x <= 99:
        x += 1900
    return x

df_check["Year"] = df_check['Year'].apply(valueToFour)

### 장르와 퍼블리셔 Nan값 제거
df_check['Genre'] = df_check['Genre'].fillna('unknown')
df_check['Publisher'] = df_check['Publisher'].fillna('unknown')

### 판매량의 EDA 해준다
def value_to_float(x):
    if 'K' not in x and 'M' not in x and 'B' not in x:
        return float(x) * 1000000
    if 'K' in x:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    if 'B' in x:
        return float(x.replace('B', '')) * 1000000000
    return x

df_check['NA_Sales'] = df_check['NA_Sales'].apply(value_to_float).astype(float)
df_check['EU_Sales'] = df_check['EU_Sales'].apply(value_to_float).astype(float)
df_check['JP_Sales'] = df_check['JP_Sales'].apply(value_to_float).astype(float)
df_check['Other_Sales'] = df_check['Other_Sales'].apply(value_to_float).astype(float)

df_check.to_csv('vGames_check.csv', mode='w')
