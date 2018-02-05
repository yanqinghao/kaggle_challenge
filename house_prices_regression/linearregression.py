import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

raw_train = pd.read_csv('./house_prices_data/train.csv')
raw_test = pd.read_csv('./house_prices_data/test.csv')
train = raw_train.drop('Id',axis=1)
train = train.fillna({'LotFrontage': np.inf, 'Alley': 'None', 'MasVnrType': train['MasVnrType'].value_counts().index.values[0],
                      'MasVnrArea': train['MasVnrArea'].mean(),'BsmtQual': 'None', 'BsmtCond': 'None',
                      'BsmtExposure': 'None', 'BsmtFinType1': 'None', 'Electrical': train['Electrical'].value_counts().index.values[0],
                      'FireplaceQu': 'None', 'GarageType': 'None', 'GarageYrBlt': 0, 'GarageFinish': 'None',
                      'GarageQual': 'None', 'GarageCond': 'None', 'PoolQC': 'None', 'Fence': 'None'})
le = LabelEncoder()
le.fit(train.loc[:,['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities'
                    , 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                    'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageArea',
                    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
                    'SaleCondition']].values)
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