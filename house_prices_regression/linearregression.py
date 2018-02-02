import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

raw_train = pd.read_csv('./house_prices_data/train.csv')
raw_test = pd.read_csv('./house_prices_data/test.csv')

print(raw_train.shape)

columns = raw_train.columns.values
for i in columns:
    print(i,raw_train[i].dropna().count())
df = []
for i in columns:
    if raw_train.dropna().loc[0, [i]].values.isdigit:
        df[:, [i]] = raw_train[:, [i]]
    else:
        df[:, [i]] = raw_train[:, [i]]

sns.heatmap(raw_train.fillna(0))
plt.savefig('b.png')

continuousdf = raw_train[['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2'
                          , 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                          'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                          'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                          'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                          'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']]
sns.pairplot(continuousdf.dropna())
plt.savefig('a.png')