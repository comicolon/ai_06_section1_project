import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

pd.set_option('display.max_row', 50)
pd.set_option('display.max_columns', 10)

df_og = pd.read_csv('vGames_check.csv')

## 인덱스 정리
df_og = df_og.drop(['Unnamed: 0'], axis=1)
df_og.rename(columns={'Unnamed: 0.1': 'Index'}, inplace=True)

## -연산자 애러 방지
df_og['Genre'] = df_og['Genre'].str.replace('-','_')

### 3. 출고량이 높은 게임에 대한 분석 및 시각화 프로세스가 포함되어야 합니다.

## 게임별 총 판매량 구함
df_yearGenre_sell = df_og[['Name', 'Year', 'Genre', 'Publisher',  'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
df_yearGenre_sellTotal = df_yearGenre_sell.copy()
df_yearGenre_sellTotal['Total_sell'] = df_yearGenre_sell.iloc[:, 3:7].sum(axis=1)
df_yearGenre_sellTotal = df_yearGenre_sellTotal.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)
df_totalSell_rank = df_yearGenre_sellTotal.sort_values(by=['Total_sell'], axis=0, ascending=False)

## 장르별 판매량 순위 구함
df_genre = df_yearGenre_sell['Genre'].unique()
df_genre_dfName = 'df_' + df_genre  ## 데이터프레임 이름 리스트
# print(df_genre)

def divideGenre(genre):
    global df_yearGenre_sellTotal
    df = df_yearGenre_sellTotal[df_yearGenre_sellTotal['Genre'] == genre]
    df = df.sort_values(by=['Total_sell'], axis=0, ascending=False)
    return df

## 동적변수 할당
for genre in df_genre:
    df = divideGenre(genre)
    locals()['df_{}'.format(genre)] = df

df_totalSell_rank = df_totalSell_rank.reset_index()
df_totalSell_rankMost = df_totalSell_rank.iloc[:300]

### 게임 상위권 장르별 합
df_genreTotalRank = df_totalSell_rankMost.groupby('Genre').sum().reset_index()
# print(df_genreTotalRank)

plt.rcParams['figure.figsize'] = (15, 9)
sns.set_style("whitegrid")
#
# g = sns.barplot(x='Genre', y='Total_sell', hue='Genre', dodge=False, data=df_genreTotalRank, palette='Paired')
# plt.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
#
# plt.show()

## Action, Platform, Role_Playing, Shooter


### 게임 상위권 연도별 분포
# df_totalSell_rankMost = df_totalSell_rankMost.sort_values(by=['Year'], axis=0, ascending=True)
# ax = sns.scatterplot(x="Year", y='Total_sell', hue='Genre', data=df_totalSell_rankMost)
#
# plt.show()

# ### 상위 4개 장르 연도별 변화
df_totalSell_rankMost = df_totalSell_rankMost.sort_values(by=['Year'], axis=0, ascending=True)
df_totalSell_rankMost4 = df_totalSell_rankMost[(df_totalSell_rankMost['Genre'] == 'Action')\
                                               | (df_totalSell_rankMost['Genre'] == 'Platform')\
                                               | (df_totalSell_rankMost['Genre'] == 'Role_Playing')\
                                               | (df_totalSell_rankMost['Genre'] == 'Shooter')]
# ax = sns.lineplot(x="Year", y='Total_sell', hue='Genre', data=df_totalSell_rankMost4)

# plt.show()

### 게임 타이틀에 대한 분석
## shooter 1981 / platform 1985 2006 2009 / Role_Playing 1996 / Action 2013

row_shooter1 = df_totalSell_rankMost4[df_totalSell_rankMost4['Year'] == '1984']
# print(row_shooter1)
## 6   4691  Duck Hunt  1984  Shooter  Nintendo    27840000
row_platform1 = df_totalSell_rankMost4[df_totalSell_rankMost4['Year'] == '1985']
# print(row_platform1)
## 1   8623  Super Mario Bros.  1985  Platform  Nintendo    39470000
row_platform2 = df_totalSell_rankMost4[df_totalSell_rankMost4['Year'] == '2006']
# print(row_platform2)
## 20    6054           Pokemon Diamond/Pokemon Pearl  2006  Role_Playing   Nintendo    16980000
## 7     8779                   New Super Mario Bros.  2006      Platform   Nintendo    27110000
row_platform3 = df_totalSell_rankMost4[df_totalSell_rankMost4['Year'] == '2009']
# print(row_platform3)
## 8     9263                       New Super Mario Bros. Wii  2009   Platform                     Nintendo    26350000
## 33    4375                  Call of Duty: Modern Warfare 2  2009   Shooter                   Activision    12230000
row_rpg1 = df_totalSell_rankMost4[df_totalSell_rankMost4['Year'] == '1996']
# print(row_rpg1)
## 3     5813  Pokemon Red/Pokemon Blue  1996  Role_Playing  Nintendo    30380000
row_action1 = df_totalSell_rankMost4[df_totalSell_rankMost4["Year"] == '2013']
# print(row_action1)
## 19   13784          Grand Theft Auto V  2013        Action    Take-Two Interactive    17250000


## 그 시기에 게임이 많이 나와서 대작들도 나오지 않았을까?
df_yearGenre_sellTotal = df_yearGenre_sellTotal.sort_values(by='Year', axis=0, ascending=True)

df_yearGenre_shooter = df_yearGenre_sellTotal[df_yearGenre_sellTotal['Genre'] == 'Shooter']
df_yearGenre_shooter_group = df_yearGenre_shooter['Year'].value_counts().reset_index()
df_yearGenre_shooter_group.columns = ['Year', 'Count']
df_yearGenre_shooter_group = df_yearGenre_shooter_group.sort_values(by='Year', axis=0, ascending=True)
df_yearGenre_shooter_group = df_yearGenre_shooter_group.reset_index()

df_yearGenre_platform = df_yearGenre_sellTotal[df_yearGenre_sellTotal['Genre'] == 'Platform']
df_yearGenre_platform_group = df_yearGenre_shooter['Year'].value_counts().reset_index()
df_yearGenre_platform_group.columns = ['Year', 'Count']
df_yearGenre_platform_group = df_yearGenre_platform_group.sort_values(by='Year', axis=0, ascending=True)
df_yearGenre_platform_group = df_yearGenre_platform_group.reset_index()

df_yearGenre_rpg = df_yearGenre_sellTotal[df_yearGenre_sellTotal['Genre'] == 'Role_Playing']
df_yearGenre_rpg_group = df_yearGenre_rpg['Year'].value_counts().reset_index()
df_yearGenre_rpg_group.columns = ['Year', 'Count']
df_yearGenre_rpg_group = df_yearGenre_rpg_group.sort_values(by='Year', axis=0, ascending=True)
df_yearGenre_rpg_group = df_yearGenre_rpg_group.reset_index()

df_yearGenre_action = df_yearGenre_sellTotal[df_yearGenre_sellTotal['Genre'] == 'Action']
df_yearGenre_action_group = df_yearGenre_shooter['Year'].value_counts().reset_index()
df_yearGenre_action_group.columns = ['Year', 'Count']
df_yearGenre_action_group = df_yearGenre_action_group.sort_values(by='Year', axis=0, ascending=True)
df_yearGenre_action_group = df_yearGenre_action_group.reset_index()


## 스캐터플롯으로 시각화
# ax1 = sns.scatterplot(x='Year', y='Total_sell', data=df_yearGenre_shooter, color='r')
# ax2 = ax1.twinx()
# ax2 = sns.scatterplot(x='Year', y ='Count', data=df_yearGenre_shooter_group, color= 'g')
#
# ax1 = sns.scatterplot(x='Year', y='Total_sell', data=df_yearGenre_platform, color='r')
# ax2 = ax1.twinx()
# ax2 = sns.scatterplot(x='Year', y ='Count', data=df_yearGenre_platform_group, color= 'g')

# ax1 = sns.scatterplot(x='Year', y='Total_sell', data=df_yearGenre_rpg, color='r')
# ax2 = ax1.twinx()
# ax2 = sns.scatterplot(x='Year', y ='Count', data=df_yearGenre_rpg_group, color= 'g')
#
# ax1 = sns.scatterplot(x='Year', y='Total_sell', data=df_yearGenre_action, color='r')
# ax2 = ax1.twinx()
# ax2 = sns.scatterplot(x='Year', y ='Count', data=df_yearGenre_action_group, color= 'g')

# plt.show()

### 공분산과 상관계수 계산
### 판매량 추출을 위해 Systematic Sampling 사용

# df_yearSell_rpg =? df_Role_Playing['Year'].value_counts(ascending=True)

## 연도별 평균을 구해서 행을 추가 하는 함수
def yearSellMean (df_genre_group, df_genre):
    for year, i in zip(df_genre_group['Year'], df_genre_group.iterrows()):
        selectYear = df_genre[df_genre['Year'] == year]
        sellMean = selectYear['Total_sell'].mean()
        df_genre_group.loc[[i[0]],['YearMean']] = sellMean

df_yearGenre_shooter_group['YearMean'] = 0.0
yearSellMean(df_yearGenre_shooter_group, df_yearGenre_shooter)

df_yearGenre_platform_group['YearMean'] = 0.0
yearSellMean(df_yearGenre_platform_group, df_yearGenre_platform)

df_yearGenre_rpg_group['YearMean'] = 0.0
yearSellMean(df_yearGenre_rpg_group, df_yearGenre_rpg)

df_yearGenre_action_group['YearMean'] = 0.0
yearSellMean(df_yearGenre_action_group, df_yearGenre_action)

# print(df_yearGenre_rpg_group)

## 4가지 장르의 시각화와 공분산, 상관계수를 계산해 준다.

## 슈터
# ax1 = sns.scatterplot(df_yearGenre_shooter_group['Year'], df_yearGenre_shooter_group['YearMean'], color='r')
# ax1 = ax1.twinx()
# ax2 = sns.scatterplot(df_yearGenre_shooter_group['Year'], df_yearGenre_shooter_group['Count'], color='b')
# plt.show()

cov_rpg = np.cov(df_yearGenre_shooter_group['Count'], df_yearGenre_shooter_group['YearMean'])
print(cov_rpg)
corr_rpg = np.corrcoef(df_yearGenre_shooter_group['Count'], df_yearGenre_shooter_group['YearMean'])
print(corr_rpg)

## 플래폼
# ax1 = sns.scatterplot(df_yearGenre_platform_group['Year'], df_yearGenre_platform_group['YearMean'], color='r')
# ax1 = ax1.twinx()
# ax2 = sns.scatterplot(df_yearGenre_platform_group['Year'], df_yearGenre_platform_group['Count'], color='b')
# plt.show()

cov_rpg = np.cov(df_yearGenre_rpg_group['Count'], df_yearGenre_rpg_group['YearMean'])
print(cov_rpg)
corr_rpg = np.corrcoef(df_yearGenre_rpg_group['Count'], df_yearGenre_rpg_group['YearMean'])
print(corr_rpg)

## 롤플레잉
# ax1 = sns.scatterplot(df_yearGenre_rpg_group['Year'], df_yearGenre_rpg_group['YearMean'], color='r')
# ax1 = ax1.twinx()
# ax2 = sns.scatterplot(df_yearGenre_rpg_group['Year'], df_yearGenre_rpg_group['Count'], color='b')
# plt.show()

cov_rpg = np.cov(df_yearGenre_rpg_group['Count'], df_yearGenre_rpg_group['YearMean'])
print(cov_rpg)
corr_rpg = np.corrcoef(df_yearGenre_rpg_group['Count'], df_yearGenre_rpg_group['YearMean'])
print(corr_rpg)

## Action
# ax1 = sns.scatterplot(df_yearGenre_action_group['Year'], df_yearGenre_action_group['YearMean'], color='r')
# ax1 = ax1.twinx()
# ax2 = sns.scatterplot(df_yearGenre_action_group['Year'], df_yearGenre_action_group['Count'], color='b')
# plt.show()

cov_rpg = np.cov(df_yearGenre_action_group['Count'], df_yearGenre_action_group['YearMean'])
print(cov_rpg)
corr_rpg = np.corrcoef(df_yearGenre_action_group['Count'], df_yearGenre_action_group['YearMean'])
print(corr_rpg)


