import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_row', 10)
pd.set_option('display.max_columns', 10)

df_og = pd.read_csv('vGames_check.csv')

## 인덱스 정리
df_og = df_og.drop(['Unnamed: 0'], axis=1)
df_og.rename(columns = {'Unnamed: 0.1' : 'Index'}, inplace=True)

### 1. 지역에 따라서 선호하는 게임 장르가 다를까 라는 질문에 대답을 하셔야합니다.

### 지역별 게임 비중 파이 그래프
### x 지역 y 장르별 비중 그래프
### 지역과 장르별 판매의 총합 데이터셋

## 데이터 전처리
df_localGenre = pd.DataFrame()

## 게임 장르별 그룹화 함수
def countGenreGames ():
    global df_localGenre
    GenreGroup = df_og['Genre'].value_counts()
    df_localGenre = pd.DataFrame(GenreGroup).reset_index()
    df_localGenre.columns = ['Genre', "Genre_Count"]
    df_localGenre['NA_Sales'] = 0.0
    df_localGenre['EU_Sales'] = 0.0
    df_localGenre['JP_Sales'] = 0.0
    df_localGenre['Other_Sales'] = 0.0

countGenreGames()

df_tmp1 = df_og.groupby('Genre').sum().reset_index()

## 총판매량 함수
def fillTotalSell (genre):
    global df_localGenre
    global df_tmp1
    tmpSellRow = df_tmp1.loc[df_tmp1['Genre'] == genre]
    sellAmount = tmpSellRow['NA_Sales'].values[0]
    df_localGenre.loc[df_localGenre.Genre == genre, 'NA_Sales'] = sellAmount
    sellAmount = tmpSellRow['EU_Sales'].values[0]
    df_localGenre.loc[df_localGenre.Genre == genre, 'EU_Sales'] = sellAmount
    sellAmount = tmpSellRow['JP_Sales'].values[0]
    df_localGenre.loc[df_localGenre.Genre == genre, 'JP_Sales'] = sellAmount
    sellAmount = tmpSellRow['Other_Sales'].values[0]
    df_localGenre.loc[df_localGenre.Genre == genre, 'Other_Sales'] = sellAmount

for genre in df_localGenre['Genre']:
    fillTotalSell(genre)


# ## 파이차트 시각화 (지역별 장르별 판매량)
# df_localGenre_NA = df_localGenre[['Genre','Genre_Count', 'NA_Sales']]
# df_localGenre_NA = df_localGenre_NA.sort_values(by=['NA_Sales'], axis=0, ascending=False)
# df_localGenre_EU = df_localGenre[['Genre','Genre_Count', 'EU_Sales']]
# df_localGenre_EU = df_localGenre_EU.sort_values(by=['EU_Sales'], axis=0, ascending=False)
# df_localGenre_JP = df_localGenre[['Genre','Genre_Count', 'JP_Sales']]
# df_localGenre_JP = df_localGenre_JP.sort_values(by=['JP_Sales'], axis=0, ascending=False)
# df_localGenre_Other = df_localGenre[['Genre','Genre_Count', 'Other_Sales']]
# df_localGenre_Other = df_localGenre_Other.sort_values(by=['Other_Sales'], axis=0, ascending=False)
#
#
# plt.rcParams['figure.figsize'] = (15, 9)
#
# plt.subplot(221)
# plt.title('NA_Sales')
# plt.pie( df_localGenre_NA['NA_Sales'], labels=df_localGenre_NA.Genre, startangle=90, autopct="%.1f%%")
# plt.subplot(222)
# plt.title('EU_Sales')
# plt.pie( df_localGenre_EU['EU_Sales'], labels=df_localGenre_EU.Genre, startangle=90, autopct="%.1f%%")
# plt.subplot(223)
# plt.title('JP_Sales')
# plt.pie( df_localGenre_JP['JP_Sales'], labels=df_localGenre_JP.Genre, startangle=90, autopct="%.1f%%")
# plt.subplot(224)
# plt.title('Other_Sales')
# plt.pie( df_localGenre_Other['Other_Sales'], labels=df_localGenre_Other.Genre, startangle=90, autopct="%.1f%%")
#
#
# plt.show()



df_localGenre_genre = df_localGenre[['Genre']]
df_localGenre_count = df_localGenre[['Genre_Count']].copy()
df_localGenre_count['Total_sell'] = df_localGenre.iloc[:, 3:7].sum(axis=1)
df_localGenre = df_localGenre.drop(['Genre_Count'], axis=1)
df_localGenre_tidy = pd.melt(df_localGenre, ["Genre"], var_name='local', value_name='sell')
# print(df_localGenre_tidy)


## 1차 시각화 (단순 판매량으로 비중 막대 그래프)
fig = plt.figure(figsize=(15, 8))

sns.set_style("whitegrid")

# g = sns.barplot(x='local', y='sell', hue='Genre', dodge=False, data=df_localGenre_tidy, palette='Paired')

# for p in g.patches:
#     g.annotate(format(p.get_height(), '.1f'),
#                    (p.get_x() + p.get_width() / 2., p.get_height()),
#                    ha = 'center', va = 'center',
#                    xytext = (0, -12),
#                    textcoords = 'offset points')



# plt.show()

## 2차 시각화 발매타이틀 갯수와의 상관관계
df_localGenre_merge = pd.concat([df_localGenre_genre, df_localGenre_count], axis= 1)
# print(df_localGenre_merge)
df_localGenre_merge_tidy = pd.melt(df_localGenre_merge, id_vars='Genre', value_vars= ['Genre_Count','Total_sell'])
# print(df_localGenre_merge_tidy)

# ax1 = sns.barplot(data=df_localGenre_merge_tidy, x='Genre', y='value', hue='variable')
# ax1.twinx()
# ax2 = sns.barplot(data=df_localGenre_merge_tidy, x='Genre', y='value', hue='variable')
df_localGenre_merge = df_localGenre_merge.set_index('Genre')

ax = fig.add_subplot(111)
ax2 = ax.twinx()
width = .2

df_localGenre_merge.Genre_Count.plot(kind='bar', color='green', ax=ax, width=width, position=0)
df_localGenre_merge.Total_sell.plot(kind='bar',color='blue', ax=ax2,width = width,position=1)

ax.grid(None)
ax2.grid(None)

ax.set_ylabel('Release Count')
ax2.set_ylabel('Total sell')

ax.set_xlim(-1,13)

plt.show()






