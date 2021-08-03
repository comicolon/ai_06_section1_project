import pandas as pd

import numpy as np



X = np.arange(30).reshape(15, 2)

y = np.arange(15)



df = pd.DataFrame(np.column_stack((X, y)), columns=['X1','X2', 'y'])

df['grp'] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1004)



for train_idx, test_idx in split.split(df, df["grp"]):

    df_strat_train = df.loc[train_idx]

    df_strat_test = df.loc[test_idx]

print(df_strat_test)
