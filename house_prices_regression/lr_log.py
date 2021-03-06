import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,RidgeCV,LassoCV,ElasticNetCV
from sklearn.model_selection import KFold, train_test_split,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from scipy.stats import skew
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor
import xgboost as xgb

raw_train = pd.read_csv('./house_prices_data/train.csv')
raw_test = pd.read_csv('./house_prices_data/test.csv')

train = raw_train.drop('Id',axis=1)
Id = raw_test.ix[:,0]
test = raw_test.drop('Id',axis=1)

train['YearBuilt'] = train['YrSold']-train['YearBuilt']
train['YearRemodAdd'] = train['YrSold']-train['YearRemodAdd']
train['GarageYrBlt'] = train['YrSold']-train['GarageYrBlt']
test['YearBuilt'] = test['YrSold']-test['YearBuilt']
test['YearRemodAdd'] = test['YrSold']-test['YearRemodAdd']
test['GarageYrBlt'] = test['YrSold']-test['GarageYrBlt']

train['SalePrice'] = np.log1p(train['SalePrice'])

alldata = pd.concat((raw_train.iloc[:,1:-1], raw_test.iloc[:,1:]))
numeric_feats = alldata.dtypes[alldata.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

train[skewed_feats] = np.log1p(train[skewed_feats])
test[skewed_feats] = np.log1p(test[skewed_feats])

neighbor = train.groupby('Neighborhood')['LotFrontage'].median()
train['LotFrontage'] = list(map(lambda x,y: neighbor[y] if np.isnan(x) else x,train['LotFrontage'].values,train['Neighborhood'].values))
test['LotFrontage'] = list(map(lambda x,y: neighbor[y] if np.isnan(x) else x,test['LotFrontage'].values,test['Neighborhood'].values))

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
            test[i] = test[i].map(j)

forest = RandomForestRegressor(n_estimators=400, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(train.ix[:,:-1], train.ix[:,-1])
print(forest.score(train.ix[:,:-1], train.ix[:,-1]))

features = np.row_stack((train.columns[:-1], forest.feature_importances_))
imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
sorted_df = imp_df.sort_values('importances',ascending=False)

score_df = pd.DataFrame()
namelst = list(sorted_df['Names'].values[0:60])
namelst.append('SalePrice')
train_fil = train.ix[:, namelst]
test_fil = test.ix[:, namelst[:-1]]

indexoh = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
               'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
               'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
               'Foundation','Heating','Electrical','Functional','GarageType','GarageFinish',
               'MiscFeature','SaleType','SaleCondition']
indexoh_fil = list(set(namelst).intersection(set(indexoh)))

indexstd = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
             '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
             'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearBuilt','YearRemodAdd',
             'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
             'TotRmsAbvGrd','GarageYrBlt','GarageCars','YrSold']
indexstd_fil = list(set(namelst).intersection(set(indexstd)))

indexmm = ['OverallQual','OverallCond','ExterQual','ExterCond',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'HeatingQC','CentralAir','KitchenQual','Fireplaces','FireplaceQu',
             'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MoSold',]
indexmm_fil = list(set(namelst).intersection(set(indexmm)))

if len(indexoh_fil)>0:
    ohscaler = OneHotEncoder()
    onehottrain = ohscaler.fit_transform(train_fil.loc[:, indexoh_fil].values.reshape(-1, len(indexoh_fil)))
    onehottest = ohscaler.transform(test_fil.loc[:, indexoh_fil].values.reshape(-1, len(indexoh_fil)))

if len(indexstd_fil) > 0:
    stdscaler = StandardScaler()
    train_fil.loc[:,indexstd_fil] = stdscaler.fit_transform(train_fil.loc[:,indexstd_fil].values.reshape(-1,len(indexstd_fil)))
    test_fil.loc[:,indexstd_fil] = stdscaler.transform(test_fil.loc[:,indexstd_fil].values.reshape(-1,len(indexstd_fil)))

if len(indexmm_fil) > 0:
    mmscaler = MinMaxScaler()
    train_fil.loc[:,indexmm_fil] = mmscaler.fit_transform(train_fil.loc[:,indexmm_fil].values.reshape(-1,len(indexmm_fil)))
    test_fil.loc[:,indexmm_fil] = mmscaler.transform(test_fil.loc[:,indexmm_fil].values.reshape(-1,len(indexmm_fil)))

stdscalerout = StandardScaler()
train_fil.loc[:,['SalePrice']] = stdscalerout.fit_transform(train_fil.loc[:,['SalePrice']].values.reshape(-1,1))

index_all = indexstd_fil+indexmm_fil
if len(indexoh_fil)>0:
    x = np.column_stack((train.ix[:,index_all], onehottrain.toarray()))
    x_test = np.column_stack((test.ix[:, index_all], onehottest.toarray()))
else:
    x = train.ix[:, index_all]
    x_test = test.ix[:, index_all]

elastic = ElasticNet(alpha=0.01,l1_ratio=0.01)
elastic.fit(x, train_fil.ix[:, -1])
print(elastic.score(x, train_fil.ix[:,-1]))

y = elastic.predict(x_test)
submit = stdscalerout.inverse_transform(y)
submit = np.expm1(submit)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("elastic.csv",index=False)

ridge = Ridge(alpha=10)
ridge.fit(x, train_fil.ix[:, -1])
print(ridge.score(x, train_fil.ix[:,-1]))

y = ridge.predict(x_test)
submit = stdscalerout.inverse_transform(y)
submit = np.expm1(submit)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("ridge.csv",index=False)

bagridge = BaggingRegressor(Ridge(alpha=10),n_estimators=8, random_state=0)
bagridge.fit(x, train_fil.ix[:, -1])
print(bagridge.score(x, train_fil.ix[:,-1]))

y = bagridge.predict(x_test)
submit = stdscalerout.inverse_transform(y)
submit = np.expm1(submit)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("bagridge.csv",index=False)

xgbmdl = xgb.XGBRegressor(n_estimators=700, max_depth=2, learning_rate=0.1, silent=1)
xgbmdl.fit(x, train_fil.ix[:, -1])
print(xgbmdl.score(x, train_fil.ix[:,-1]))

y = xgbmdl.predict(x_test)
submit = stdscalerout.inverse_transform(y)
submit = np.expm1(submit)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("xgb.csv",index=False)

gbr = GradientBoostingRegressor(n_estimators=700, max_depth=2, learning_rate=0.1, loss='huber')
gbr.fit(x, train_fil.ix[:, -1])
print(gbr.score(x, train_fil.ix[:,-1]))

y = gbr.predict(x_test)
submit = stdscalerout.inverse_transform(y)
submit = np.expm1(submit)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("gbr.csv",index=False)