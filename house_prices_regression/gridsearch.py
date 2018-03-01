import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from scipy.stats import skew
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor
import xgboost as xgb

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
# labellist = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities'
#                     , 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
#                     'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#                     'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
#                     'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
#                     'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
#                     'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
#                     'SaleCondition']

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

train_fil = train
test_fil = test
y = train_fil.ix[:, -1].values
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
else:
    x = train.ix[:, index_all]

forest = RandomForestRegressor(n_estimators=400, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(x, train.ix[:,-1])

features = np.row_stack((range(x.shape[1]), forest.feature_importances_))
imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
sorted_df = imp_df.sort_values('importances',ascending=False)

# select_dec = [10,20,30,40,50,60,70]
# corr = train.corr()
# sale_corr = corr.ix[:,-1].to_frame('SalePrice')
# select_dec = [int(x.shape[1]*0.5), int(x.shape[1]*0.6), int(x.shape[1]*0.57), int(x.shape[1]*0.8), int(x.shape[1]*0.9),x.shape[1]]
select_dec = [int(x.shape[1])]
score_df = pd.DataFrame()
for i in select_dec:
    list_score = []
    list_para = []

    namelst = np.array(list(sorted_df['Names'].values[0:i])).astype('int')
    x_train = x[:, namelst]
    # namelst.append('SalePrice')
    # train_fil = train.ix[:, namelst]
    # test_fil = test.ix[:, namelst[:-1]]

    # select_corr = sale_corr[(sale_corr['SalePrice']>i) | (sale_corr['SalePrice']<-i)]
    # train_fil = train.ix[:,list(select_corr.index.values)]
    # test_fil = test.ix[:,list(select_corr.index.values[:-1])]

    # tree_range = range(1, 10, 1)
    # parameters_tree = [{'max_depth': tree_range}]
    # gs_tree = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=parameters_tree, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    # gs_tree.fit(x_train, y)
    # list_score.append(np.sqrt(-gs_tree.best_score_))
    # list_para.append(gs_tree.best_params_)
    #
    # # tree = DecisionTreeRegressor(max_depth=3)
    # forest_range = range(400,1100,100)
    # forest_range1 = range(2, 7, 1)
    # parameters_forest = [{'n_estimators': [600], 'max_depth': forest_range1}]
    # gs_forest = GridSearchCV(estimator=RandomForestRegressor(criterion='mse', random_state=1), param_grid=parameters_forest, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    # gs_forest.fit(x_train, y)
    # list_score.append(np.sqrt(-gs_forest.best_score_))
    # list_para.append(gs_forest.best_params_)

    # pca = PCA(n_components=25)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # test_pca = pca.transform(test)
    # gs_lr = GridSearchCV(estimator=LinearRegression(), scoring='r2', cv=3, n_jobs=-1)
    # gs_lr.fit(train_fil.ix[:, :-1], train_fil.ix[:, -1])
    # print(gs_lr.best_score_)
    # print(gs_lr.best_params_)
    # scores = cross_val_score(estimator=LinearRegression(),X=x_train, y=y,scoring='neg_mean_squared_error',cv=3)
    # list_score.append(np.sqrt(-scores.mean()))
    # list_para.append('None')

    ridge_range = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    parameters_ridge = [{'alpha': ridge_range}]
    gs_ridge = GridSearchCV(estimator=Ridge(), param_grid=parameters_ridge, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs_ridge.fit(x_train, y)
    list_score.append(np.sqrt(-gs_ridge.best_score_))
    list_para.append(gs_ridge.best_params_)

    adb_range = range(2, 15, 1)
    parameters_ridge = [{'n_estimators': adb_range, 'base_estimator__alpha': ridge_range}]
    adbrg = BaggingRegressor(Ridge(), random_state=0)
    gs_ridge = GridSearchCV(estimator=adbrg, param_grid=parameters_ridge, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs_ridge.fit(x_train, y)
    list_score.append(np.sqrt(-gs_ridge.best_score_))
    list_para.append(gs_ridge.best_params_)

    lasso_range = [0.0001, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
    parameters_lasso = [{'alpha': lasso_range}]
    gs_lasso = GridSearchCV(estimator=Lasso(), param_grid=parameters_lasso, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs_lasso.fit(x_train, y)
    list_score.append(np.sqrt(-gs_lasso.best_score_))
    list_para.append(gs_lasso.best_params_)

    elastic_range_1 = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07]
    elastic_range_2 = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07]
    parameters_elastic = [{'alpha': elastic_range_1, 'l1_ratio':elastic_range_2}]
    gs_elastic = GridSearchCV(estimator=ElasticNet(), param_grid=parameters_elastic, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs_elastic.fit(x_train, y)
    list_score.append(np.sqrt(-gs_elastic.best_score_))
    list_para.append(gs_elastic.best_params_)

    xgb_range_n = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    xgb_range_d = [2, 3]
    xgb_range_l = [0.0001, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.005]
    # parameters_xgb = [{'n_estimators': xgb_range_n, 'max_depth': xgb_range_d, 'gamma': xgb_range_l, 'reg_alpha': xgb_range_l, 'reg_lambda': xgb_range_l}]
    parameters_xgb = [{'n_estimators': [600], 'max_depth': [2], 'gamma': xgb_range_l}]
    gs_xgb = GridSearchCV(estimator=xgb.XGBRegressor(learning_rate=0.1, silent=1), param_grid=parameters_xgb, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs_xgb.fit(x_train, y)
    list_score.append(np.sqrt(-gs_xgb.best_score_))
    list_para.append(gs_xgb.best_params_)

    gbr_range_n = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    gbr_range_d = [2, 3]
    gbr_range_a = [0.5, 0.6, 0.7, 0.8, 0.9]
    parameters_gbr = [{'n_estimators': [700], 'max_depth': [2], 'alpha': [0.6]}]
    gs_gbr = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1, loss='huber'), param_grid=parameters_gbr,
                          scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    gs_gbr.fit(x_train, y)
    list_score.append(np.sqrt(-gs_gbr.best_score_))
    list_para.append(gs_gbr.best_params_)

    score_df[str(i) + 'score'] = list_score
    score_df[str(i) + 'para'] = list_para
    print(i)

print(score_df)
score_df.to_csv('score.csv')