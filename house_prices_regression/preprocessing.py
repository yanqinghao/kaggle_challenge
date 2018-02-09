import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,RobustScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer

raw_train = pd.read_csv('./house_prices_data/train.csv')
raw_test = pd.read_csv('./house_prices_data/test.csv')
train = raw_train.drop('Id',axis=1)
Id = raw_test.ix[:,0]
test = raw_test.drop('Id',axis=1)

neighbor = train.groupby('Neighborhood')['LotFrontage'].median()
train['LotFrontage'] = list(map(lambda x,y: neighbor[y] if np.isnan(x) else x,train['LotFrontage'].values,train['Neighborhood'].values))
test['LotFrontage'] = list(map(lambda x,y: neighbor[y] if np.isnan(x) else x,test['LotFrontage'].values,test['Neighborhood'].values))

# train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
#     lambda x: x.fillna(x.median()))
mostfrq = [train['MSZoning'].value_counts().index.values[0],train['Utilities'].value_counts().index.values[0],
           train['Exterior1st'].value_counts().index.values[0],train['Exterior2nd'].value_counts().index.values[0],
           train['MasVnrType'].value_counts().index.values[0],train['Electrical'].value_counts().index.values[0],
           train['KitchenQual'].value_counts().index.values[0],train['Functional'].value_counts().index.values[0],
           train['SaleType'].value_counts().index.values[0],train['BsmtFullBath'].value_counts().index.values[0],
           train['BsmtHalfBath'].value_counts().index.values[0],train['GarageCars'].value_counts().index.values[0]]
meanval = [train['MasVnrArea'].mean(),train['GarageArea'].mean(),train['BsmtFinSF2'].mean(),train['BsmtUnfSF'].mean(),train['TotalBsmtSF'].mean(),train['BsmtFinSF1'].mean(),train['GarageYrBlt'].mean()]
train = train.fillna({'Alley': 'None', 'MSZoning': mostfrq[0], 'Utilities': mostfrq[1], 'Exterior1st': mostfrq[2], 'Exterior2nd': mostfrq[3], 'MasVnrType': mostfrq[4],
                      'MasVnrArea': meanval[0],'BsmtQual': 'None', 'BsmtCond': 'None',
                      'BsmtExposure': 'None', 'BsmtFinType1': 'None', 'BsmtFinType2': 'None', 'BsmtFinSF2': meanval[2], 'BsmtUnfSF': meanval[3], 'TotalBsmtSF': meanval[4], 'BsmtFinSF1': meanval[5],
                      'Electrical': mostfrq[5], 'BsmtFullBath': mostfrq[9], 'BsmtHalfBath': mostfrq[10], 'KitchenQual': mostfrq[6], 'Functional': mostfrq[7], 'FireplaceQu': 'None',
                      'GarageType': 'None', 'GarageCars': mostfrq[11], 'GarageYrBlt': meanval[6], 'GarageFinish': 'None', 'GarageQual': 'None', 'GarageArea': meanval[1],
                      'GarageCond': 'None', 'PoolQC': 'None', 'Fence': 'None', 'MiscFeature': 'None', 'SaleType': mostfrq[8]})
test = test.fillna({'Alley': 'None', 'MSZoning': mostfrq[0], 'Utilities': mostfrq[1], 'Exterior1st': mostfrq[2], 'Exterior2nd': mostfrq[3], 'MasVnrType': mostfrq[4],
                      'MasVnrArea': meanval[0],'BsmtQual': 'None', 'BsmtCond': 'None',
                      'BsmtExposure': 'None', 'BsmtFinType1': 'None', 'BsmtFinType2': 'None', 'BsmtFinSF2': meanval[2], 'BsmtUnfSF': meanval[3], 'TotalBsmtSF': meanval[4], 'BsmtFinSF1': meanval[5],
                      'Electrical': mostfrq[5], 'BsmtFullBath': mostfrq[9], 'BsmtHalfBath': mostfrq[10], 'KitchenQual': mostfrq[6], 'Functional': mostfrq[7], 'FireplaceQu': 'None',
                      'GarageType': 'None', 'GarageCars': mostfrq[11], 'GarageYrBlt': meanval[6], 'GarageFinish': 'None', 'GarageQual': 'None', 'GarageArea': meanval[1],
                      'GarageCond': 'None', 'PoolQC': 'None', 'Fence': 'None', 'MiscFeature': 'None', 'SaleType': mostfrq[8]})
labellist = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities'
                    , 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                    'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType',
                    'GarageFinish', 'MiscFeature', 'SaleType', 'SaleCondition']
lelist = []
for i in labellist:
    le = LabelEncoder()
    le.fit(np.concatenate((train.loc[:,[i]].values,test.loc[:, [i]].values)))
    train.loc[:, [i]] = le.transform(train.loc[:,[i]].values)
    test.loc[:, [i]] = le.transform(test.loc[:, [i]].values)
    lelist.append(le)

deflist = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
           'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
           'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

eva_mapping = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0}
exposure_mapping = {'Gd': 4,'Av': 3,'Mn': 2,'No': 1,'None': 0}
living_mapping = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1,'None': 0}
pave_mapping = {'Y': 2,'P': 1,'N': 0}
fence_mapping = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2,'MnWw': 1,'None': 0}
mapping = [eva_mapping, exposure_mapping, living_mapping, pave_mapping, fence_mapping]
for i in deflist:
    for j in mapping:
        if len(list(set(j.keys()).intersection(set(train[i].unique())))) > 2:
            train[i] = train[i].map(j)

train['YearBuilt'] = train['YrSold']-train['YearBuilt']
train['YearRemodAdd'] = train['YrSold']-train['YearRemodAdd']
train['GarageYrBlt'] = train['YrSold']-train['GarageYrBlt']
test['YearBuilt'] = test['YrSold']-test['YearBuilt']
test['YearRemodAdd'] = test['YrSold']-test['YearRemodAdd']
test['GarageYrBlt'] = test['YrSold']-test['GarageYrBlt']

forest = RandomForestRegressor(n_estimators=400, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(train.ix[:,:-1], train.ix[:,-1])
print(forest.score(train.ix[:,:-1], train.ix[:,-1]))

features = np.row_stack((train.columns[:-1], forest.feature_importances_))
imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
imp_df.to_csv('imp.csv')
sorted_df = imp_df.sort_values('importances',ascending=False)
namelst = list(sorted_df['Names'].values[0:30])
namelst.append('SalePrice')
train_fil = train.ix[:,namelst]
test_fil = test.ix[:,namelst[:-1]]

forest_range = range(400, 1100, 100)
parameters_forest = [{'n_estimators': forest_range}]
gs_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid=parameters_forest, scoring='r2', cv=3, n_jobs=-1)
gs_forest.fit(train_fil.ix[:, :-1], train_fil.ix[:, -1])
print(gs_forest.best_score_)
print(gs_forest.best_params_)