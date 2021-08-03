import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler

pd.set_option('display.max_row', 30)
pd.set_option('display.max_columns', 10)

df_og = pd.read_csv('vGames_check.csv')

## 인덱스 정리
df_og = df_og.drop(['Unnamed: 0'], axis=1)
df_og.rename(columns={'Unnamed: 0.1': 'Index'}, inplace=True)

## -연산자 애러 방지
df_og['Genre'] = df_og['Genre'].str.replace('-','_')

##2. 연도별 게임의 트렌드가 있을까 라는 질문에 대답을 하셔야합니다.

### 연도별 게임장르 판매 선 그래프

## 게임별 총 판매량 구함
df_yearGenre_sell = df_og[['Name', 'Year', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
df_yearGenre_sellTotal = df_yearGenre_sell.copy()
df_yearGenre_sellTotal['Total_sell'] = df_yearGenre_sell.iloc[:, 3:7].sum(axis=1)
df_yearGenre_sellTotal = df_yearGenre_sellTotal.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)

## 정규화
df_scaled = df_yearGenre_sellTotal.copy()
scaler = RobustScaler()
# df_scaled['Total_sell'] = scaler.fit_transform(df_scaled['Total_sell'].to_numpy())
df_scaled['Total_sell'] = pd.DataFrame(scaler.fit_transform(df_scaled[['Total_sell']].values), columns=['Total_Sell'], index=df_scaled.index)


## 장르별 연도순 판매량 구함
df_genre = df_yearGenre_sell['Genre'].unique()
df_genre_dfName = 'df_' + df_genre  ## 데이터프레임 이름 리스트
# print(df_genre)

def divideGenre(genre):
    global df_yearGenre_sellTotal
    df = df_yearGenre_sellTotal[df_yearGenre_sellTotal['Genre'] == genre]
    df = df.sort_values(by=['Year'], axis=0, ascending=True)
    return df

## 동적변수 할당
for genre in df_genre:
    df = divideGenre(genre)
    locals()['df_{}'.format(genre)] = df

print(df_Action)

###-------------------------------------------------------------------------

# 연도별 라인 그래프 개별
sns.set_theme(style="darkgrid")
fig = plt.figure(figsize=(18, 9))


# plt.subplot2grid((3,4), (0,0))
# g1 = sns.lineplot(x=df_Action['Year'], y=df_Action['Total_sell'], ci=None, legend='brief', label='Action', linestyle='-', marker='o', color='#FF0000')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (0,1))
# sns.lineplot(x=df_Adventure['Year'], y=df_Adventure['Total_sell'], ci=None, label='Adventure', linestyle='-', marker='o', color='#B45F04')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (0,2))
# sns.lineplot(x=df_Misc['Year'], y=df_Misc['Total_sell'], ci=None, label='Misc', linestyle='-', marker='o', color='#FFFF00')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (0,3))
# sns.lineplot(x=df_Platform['Year'], y=df_Platform['Total_sell'], ci=None, label='Platform', linestyle='-', marker='o', color='#9AFE2E')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (1,0))
# sns.lineplot(x=df_Sports['Year'], y=df_Sports['Total_sell'], ci=None, label='Sports', linestyle='-', marker='o', color='#00FF00')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (1,1))
# sns.lineplot(x=df_Simulation['Year'], y=df_Simulation['Total_sell'], ci=None, label='Simulation', linestyle='-', marker='o', color='#00FF80')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (1,2))
# sns.lineplot(x=df_Racing['Year'], y=df_Racing['Total_sell'], ci=None, label='Racing', linestyle='-', marker='o', color='#00FFFF')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (1,3))
# sns.lineplot(x=df_Role_Playing['Year'], y=df_Role_Playing['Total_sell'], ci=None, label='Role_Playing', linestyle='-', marker='o', color='#0080FF')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (2,0))
# sns.lineplot(x=df_Puzzle['Year'], y=df_Puzzle['Total_sell'], ci=None, label='Puzzle', linestyle='-', marker='o', color='#0000FF')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (2,1))
# sns.lineplot(x=df_Strategy['Year'], y=df_Strategy['Total_sell'], ci=None, label='Strategy', linestyle='-', marker='o', color='#8000FF')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (2,2))
# sns.lineplot(x=df_Fighting['Year'], y=df_Fighting['Total_sell'], ci=None, label='Fighting', linestyle='-', marker='o', color='#FF00FF')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# plt.subplot2grid((3,4), (2,3))
# sns.lineplot(x=df_Shooter['Year'], y=df_Shooter['Total_sell'], ci=None, label='Shooter', linestyle='-', marker='o', color='#FF0080')
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
#
# # plt.subplot2grid((3,4), (2,1))
# # sns.lineplot(x=df_unknown['Year'], y=df_unknown['Total_sell'], ci=None, label='unknown', linestyle='-', marker='o', color='#585858')
#
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)

# plt.show()

## 한곳에 겹쳐서 보여줌

# sns.lineplot(x=df_Action['Year'], y=df_Action['Total_sell'], ci=None, legend='brief', label='Action', linestyle='-', marker='o', color='#FF0000')
# sns.lineplot(x=df_Adventure['Year'], y=df_Adventure['Total_sell'], ci=None, label='Adventure', linestyle='-', marker='o', color='#B45F04')
# sns.lineplot(x=df_Misc['Year'], y=df_Misc['Total_sell'], ci=None, label='Misc', linestyle='-', marker='o', color='#FFFF00')
# sns.lineplot(x=df_Platform['Year'], y=df_Platform['Total_sell'], ci=None, label='Platform', linestyle='-', marker='o', color='#9AFE2E')
# sns.lineplot(x=df_Sports['Year'], y=df_Sports['Total_sell'], ci=None, label='Sports', linestyle='-', marker='o', color='#00FF00')
# sns.lineplot(x=df_Simulation['Year'], y=df_Simulation['Total_sell'], ci=None, label='Simulation', linestyle='-', marker='o', color='#00FF80')
# sns.lineplot(x=df_Racing['Year'], y=df_Racing['Total_sell'], ci=None, label='Racing', linestyle='-', marker='o', color='#00FFFF')
# sns.lineplot(x=df_Role_Playing['Year'], y=df_Role_Playing['Total_sell'], ci=None, label='Role_Playing', linestyle='-', marker='o', color='#0080FF')
# sns.lineplot(x=df_Puzzle['Year'], y=df_Puzzle['Total_sell'], ci=None, label='Puzzle', linestyle='-', marker='o', color='#0000FF')
# sns.lineplot(x=df_Strategy['Year'], y=df_Strategy['Total_sell'], ci=None, label='Strategy', linestyle='-', marker='o', color='#8000FF')
# sns.lineplot(x=df_Fighting['Year'], y=df_Fighting['Total_sell'], ci=None, label='Fighting', linestyle='-', marker='o', color='#FF00FF')
# sns.lineplot(x=df_Shooter['Year'], y=df_Shooter['Total_sell'], ci=None, label='Shooter', linestyle='-', marker='o', color='#FF0080')
# sns.lineplot(x=df_unknown['Year'], y=df_unknown['Total_sell'], ci=None, label='unknown', linestyle='-', marker='o', color='#585858')
#
# plt.show()

## 최근의 게임 판매량 변화

df_Action_sc = df_Action[df_Action['Year'] >= 2005]
df_Adventure_sc = df_Adventure[df_Adventure['Year'] >= 2005]
df_Misc_sc = df_Misc[df_Misc['Year'] >= 2005]
df_Platform_sc = df_Platform[df_Platform['Year'] >= 2005]
df_Sports_sc = df_Sports[df_Sports['Year'] >= 2005]
df_Simulation_sc = df_Simulation[df_Simulation['Year'] >= 2005]
df_Racing_sc = df_Racing[df_Racing['Year'] >= 2005]
df_Role_Playing_sc = df_Role_Playing[df_Role_Playing['Year'] >= 2005]
df_Puzzle_sc = df_Puzzle[df_Puzzle['Year'] >= 2005]
df_Strategy_sc = df_Strategy[df_Strategy['Year'] >= 2005]
df_Fighting_sc = df_Fighting[df_Fighting['Year'] >= 2005]
df_Shooter_sc = df_Shooter[df_Shooter['Year'] >= 2005]
df_unknown_sc = df_unknown[df_unknown['Year'] >= 2005]


sns.lineplot(x=df_Action_sc['Year'], y=df_Action_sc['Total_sell'], ci=None, legend='brief', label='Action', linestyle='solid', marker='o', color='#FF0000')
sns.lineplot(x=df_Adventure_sc['Year'], y=df_Adventure_sc['Total_sell'], ci=None, label='Adventure', linestyle='dashed', marker='o', color='#B45F04')
sns.lineplot(x=df_Misc_sc['Year'], y=df_Misc_sc['Total_sell'], ci=None, label='Misc', linestyle='dashdot', marker='o', color='#FFFF00')
sns.lineplot(x=df_Platform_sc['Year'], y=df_Platform_sc['Total_sell'], ci=None, label='Platform', linestyle='dotted', marker='o', color='#9AFE2E')
sns.lineplot(x=df_Sports_sc['Year'], y=df_Sports_sc['Total_sell'], ci=None, label='Sports', linestyle='solid', marker='o', color='#00FF00')
sns.lineplot(x=df_Simulation_sc['Year'], y=df_Simulation_sc['Total_sell'], ci=None, label='Simulation', linestyle='dashed', marker='o', color='#00FF80')
sns.lineplot(x=df_Racing_sc['Year'], y=df_Racing_sc['Total_sell'], ci=None, label='Racing', linestyle='dashdot', marker='o', color='#00FFFF')
sns.lineplot(x=df_Role_Playing_sc['Year'], y=df_Role_Playing_sc['Total_sell'], ci=None, label='Role_Playing', linestyle='dotted', marker='o', color='#0080FF')
sns.lineplot(x=df_Puzzle_sc['Year'], y=df_Puzzle_sc['Total_sell'], ci=None, label='Puzzle', linestyle='solid', marker='o', color='#0000FF')
sns.lineplot(x=df_Strategy_sc['Year'], y=df_Strategy_sc['Total_sell'], ci=None, label='Strategy', linestyle='dashed', marker='o', color='#8000FF')
sns.lineplot(x=df_Fighting_sc['Year'], y=df_Fighting_sc['Total_sell'], ci=None, label='Fighting', linestyle='dashdot', marker='o', color='#FF00FF')
sns.lineplot(x=df_Shooter_sc['Year'], y=df_Shooter_sc['Total_sell'], ci=None, label='Shooter', linestyle='dotted', marker='o', color='#FF0080')
sns.lineplot(x=df_unknown_sc['Year'], y=df_unknown_sc['Total_sell'], ci=None, label='unknown', linestyle='--', marker='o', color='#585858')
#
plt.show()


### 연도별 발매한 장르와 판매량과의 공분산을 계산해 본다
### Fighting, Action,

