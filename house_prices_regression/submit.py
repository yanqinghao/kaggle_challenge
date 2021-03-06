import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,OneHotEncoder,RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,RidgeCV,LassoCV,ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, train_test_split,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from scipy.stats import skew
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor
import xgboost as xgb
from sklearn.svm import SVR
import StackModels

raw_train = pd.read_csv('./house_prices_data/train.csv')
raw_test = pd.read_csv('./house_prices_data/test.csv')

train = raw_train.drop('Id',axis=1)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
Id = raw_test.ix[:,0]
test = raw_test.drop('Id',axis=1)

train['YearBuilt'] = train['YrSold']-train['YearBuilt']
train['YearRemodAdd'] = train['YrSold']-train['YearRemodAdd']
train['GarageYrBlt'] = train['YrSold']-train['GarageYrBlt']
test['YearBuilt'] = test['YrSold']-test['YearBuilt']
test['YearRemodAdd'] = test['YrSold']-test['YearRemodAdd']
test['GarageYrBlt'] = test['YrSold']-test['GarageYrBlt']

train["TotalHouse"] = train["TotalBsmtSF"] + train["1stFlrSF"] + train["2ndFlrSF"]
test["TotalHouse"] = test["TotalBsmtSF"] + test["1stFlrSF"] + test["2ndFlrSF"]
train["TotalArea"] = train["TotalBsmtSF"] + train["1stFlrSF"] + train["2ndFlrSF"] + train["GarageArea"]
test["TotalArea"] = test["TotalBsmtSF"] + test["1stFlrSF"] + test["2ndFlrSF"] + test["GarageArea"]

train['SalePrice'] = np.log1p(train['SalePrice'])

namecol = list(train.columns[:-3])+list(train.columns[-2:])
namecol.append(train.columns[-3])
train = train[namecol]

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

# forest = RandomForestRegressor(n_estimators=400, criterion='mse', random_state=1, n_jobs=-1)
# forest.fit(train.ix[:,:-1], train.ix[:,-1])
#
# features = np.row_stack((train.columns[:-1], forest.feature_importances_))
# imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
# sorted_df = imp_df.sort_values('importances',ascending=False)

# score_df = pd.DataFrame()
# namelst = list(sorted_df['Names'].values[0:79])
# namelst.append('SalePrice')
train_fil = train
test_fil = test

indexoh = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',
               'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
               'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
               'Foundation','Heating','Electrical','Functional','GarageType','GarageFinish',
               'MiscFeature','SaleType','SaleCondition']
indexoh_fil = indexoh

indexstd = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
             '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
             'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearBuilt','YearRemodAdd',
             'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
             'TotRmsAbvGrd','GarageYrBlt','GarageCars','YrSold']
indexstd_fil = indexstd

indexmm = ['OverallQual','OverallCond','ExterQual','ExterCond',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'HeatingQC','CentralAir','KitchenQual','Fireplaces','FireplaceQu',
             'GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MoSold',]
indexmm_fil = indexmm

if len(indexoh_fil)>0:
    ohscaler = OneHotEncoder()
    onehottrain = ohscaler.fit_transform(train_fil.loc[:, indexoh_fil].values.reshape(-1, len(indexoh_fil)))
    onehottest = ohscaler.transform(test_fil.loc[:, indexoh_fil].values.reshape(-1, len(indexoh_fil)))

index_all = indexstd_fil+indexmm_fil
if len(indexoh_fil)>0:
    x = np.column_stack((train.ix[:,index_all], onehottrain.toarray()))
    x_test = np.column_stack((test.ix[:, index_all], onehottest.toarray()))
else:
    x = train.ix[:, index_all]
    x_test = test.ix[:, index_all]

scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)
x_test_scaled = scaler.transform(x_test)

def rmsle_cv(model, n_folds):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_scaled)
    rmse= np.sqrt(-cross_val_score(model, x_scaled, train_fil.ix[:,-1].values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# forest = RandomForestRegressor(n_estimators=400, criterion='mse', random_state=1, n_jobs=-1)
# forest.fit(x, train.ix[:,-1])
# print('RandomForestRegressor:',(rmsle_cv(forest,5)).mean(),' ',(rmsle_cv(forest,5)).std())

kr = KernelRidge(kernel='polynomial', alpha=0.1, coef0=1.2, degree=2)
kr.fit(x_scaled, train_fil.ix[:, -1])
print('KernelRidge:',(rmsle_cv(kr,5)).mean(),' ',(rmsle_cv(kr,5)).std())

y = kr.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("kr.csv",index=False)

svr = SVR(C = 19, gamma = 0.0005)
svr.fit(x_scaled, train_fil.ix[:, -1])
print('SVR:',(rmsle_cv(svr,5)).mean(),' ',(rmsle_cv(svr,5)).std())

y = svr.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("svr.csv",index=False)


elastic = ElasticNet(alpha=0.002,l1_ratio=0.2)
elastic.fit(x_scaled, train_fil.ix[:, -1])
print('ElasticNet:',(rmsle_cv(elastic,5)).mean(),' ',(rmsle_cv(elastic,5)).std())

y = elastic.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("elastic.csv",index=False)

ridge = Ridge(alpha=12)
ridge.fit(x_scaled, train_fil.ix[:, -1])
print('Ridge:',(rmsle_cv(ridge,5)).mean(),' ',(rmsle_cv(ridge,5)).std())

y = ridge.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("ridge.csv",index=False)

lasso = Lasso(alpha=0.0005)
lasso.fit(x_scaled, train_fil.ix[:, -1])
print('Lasso:',(rmsle_cv(lasso,5)).mean(),' ',(rmsle_cv(lasso,5)).std())

y = lasso.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("lasso.csv",index=False)

bagridge = BaggingRegressor(Ridge(alpha=9),n_estimators=14, random_state=0)
bagridge.fit(x_scaled, train_fil.ix[:, -1])
print('BaggingRegressor:',(rmsle_cv(bagridge,5)).mean(),' ',(rmsle_cv(bagridge,5)).std())

y = bagridge.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("bagridge.csv",index=False)

xgbmdl = xgb.XGBRegressor(n_estimators=600, max_depth=2, learning_rate=0.1, silent=1, gamma=0.00006)
xgbmdl.fit(x_scaled, train_fil.ix[:, -1])
print('XGBRegressor:',(rmsle_cv(xgbmdl,5)).mean(),' ',(rmsle_cv(xgbmdl,5)).std())

y = xgbmdl.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("xgb.csv",index=False)

gbr = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.1, loss='huber', alpha=0.8)
gbr.fit(x_scaled, train_fil.ix[:, -1])
print('GradientBoostingRegressor:',(rmsle_cv(gbr,5)).mean(),' ',(rmsle_cv(gbr,5)).std())

y = gbr.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("gbr.csv",index=False)

modellst = [gbr, lasso, kr, svr]
for i in range(len(modellst)):
    mdls = modellst.copy()
    mdls.pop(i)
    # print(mdls)
    averaged_models = StackModels.AveragingModels(models = mdls)
    averaged_models.fit(x_scaled, train_fil.ix[:, -1])
    print('AveragingModels:',(rmsle_cv(averaged_models,5)).mean(),' ',(rmsle_cv(averaged_models,5)).std())
    y = averaged_models.predict(x_test_scaled)
    submit = np.expm1(y)
    results = pd.Series(submit,name="SalePrice")
    submission = pd.concat([Id,results],axis = 1)
    submission.to_csv("averaged_models.csv",index=False)

stacked_averaged_models = StackModels.StackingAveragedModels(base_models = (gbr, bagridge, ridge, xgbmdl, elastic, lasso, kr, svr),
                                                 meta_model = ridge)
stacked_averaged_models.fit(x_scaled, train_fil.ix[:, -1].values)
print('StackingAveragedModels:',(rmsle_cv(stacked_averaged_models,5)).mean(),' ',(rmsle_cv(stacked_averaged_models,5)).std())
y = stacked_averaged_models.predict(x_test_scaled)
submit = np.expm1(y)
results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("stacked_averaged_models.csv",index=False)