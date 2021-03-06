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

forest = RandomForestRegressor(n_estimators=400, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(train.ix[:,:-1], train.ix[:,-1])
print(forest.score(train.ix[:,:-1], train.ix[:,-1]))

features = np.row_stack((train.columns[:-1], forest.feature_importances_))
imp_df = pd.DataFrame(columns=['Names', 'importances'], data=features.T)
sorted_df = imp_df.sort_values('importances',ascending=False)

select_dec = [10,20,30,40,50,60,70]
# corr = train.corr()
# sale_corr = corr.ix[:,-1].to_frame('SalePrice')
# select_dec = [0.2,0.3,0.4,0.5,0.6]
score_df = pd.DataFrame()
for i in select_dec:
    list_score = []
    list_para = []

    namelst = list(sorted_df['Names'].values[0:i])
    namelst.append('SalePrice')
    train_fil = train.ix[:, namelst]
    test_fil = test.ix[:, namelst[:-1]]

    # select_corr = sale_corr[(sale_corr['SalePrice']>i) | (sale_corr['SalePrice']<-i)]
    # train_fil = train.ix[:,list(select_corr.index.values)]
    # test_fil = test.ix[:,list(select_corr.index.values[:-1])]

    tree_range = range(1, 10, 1)
    parameters_tree = [{'max_depth': tree_range}]
    gs_tree = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=parameters_tree, scoring='r2', cv=3, n_jobs=-1)
    gs_tree.fit(train_fil.ix[:, :-1], train_fil.ix[:, -1])
    list_score.append(gs_tree.best_score_)
    list_para.append(gs_tree.best_params_)

    # tree = DecisionTreeRegressor(max_depth=3)
    forest_range = range(400,1100,100)
    parameters_forest = [{'n_estimators': forest_range}]
    gs_forest = GridSearchCV(estimator=RandomForestRegressor(), param_grid=parameters_forest, scoring='r2', cv=3, n_jobs=-1)
    gs_forest.fit(train_fil.ix[:,:-1], train_fil.ix[:,-1])
    list_score.append(gs_forest.best_score_)
    list_para.append(gs_forest.best_params_)
    # tree.fit(train.ix[:,:-1], train.ix[:,-1])
    # print(tree.score(train.ix[:,:-1], train.ix[:,-1]))

    # y = tree.predict(test)
    #
    # results = pd.Series(y,name="SalePrice")
    # submission = pd.concat([Id,results],axis = 1)
    # submission.to_csv("tree.csv",index=False)

    # forest = RandomForestRegressor(n_estimators=900,, criterion='mse', random_state=1, n_jobs=-1)
    # forest.fit(train_fil.ix[:,:-1], train_fil.ix[:,-1])
    # print(forest.score(train_fil.ix[:,:-1], train_fil.ix[:,-1]))
    #
    # y = forest.predict(test_fil)
    #
    # results = pd.Series(y,name="SalePrice")
    # submission = pd.concat([Id,results],axis = 1)
    # submission.to_csv("forest.csv",index=False)

    # sns.heatmap(train.corr())
    # plt.savefig('b.png')

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
    else:
        x = train.ix[:, index_all]
    # X_train, X_test, y_train, y_test = train_test_split(train.ix[:,:-1], train.ix[:,-1], test_size=0.2, random_state=0, shuffle=True)

    # pca = PCA(n_components=25)
    # X_train_pca = pca.fit_transform(X_train)
    # X_test_pca = pca.transform(X_test)
    # test_pca = pca.transform(test)
    # gs_lr = GridSearchCV(estimator=LinearRegression(), scoring='r2', cv=3, n_jobs=-1)
    # gs_lr.fit(train_fil.ix[:, :-1], train_fil.ix[:, -1])
    # print(gs_lr.best_score_)
    # print(gs_lr.best_params_)
    scores = cross_val_score(estimator=LinearRegression(),X=x, y=train_fil.ix[:, -1],scoring='r2',cv=3)
    list_score.append(scores.mean())
    list_para.append('None')
    # lr = LinearRegression().fit(train_fil.ix[:,:-1], train_fil.ix[:,-1])
    # print(lr.score(train_fil.ix[:,:-1], train_fil.ix[:,-1]))
    #
    # y = lr.predict(test_fil)
    # submit = stdscalerout.inverse_transform(y)
    #
    # results = pd.Series(submit,name="SalePrice")
    # submission = pd.concat([Id,results],axis = 1)
    # submission.to_csv("lr.csv",index=False)
    ridge_range = [0.01, 0.1, 1, 10, 100, 1000]
    parameters_ridge = [{'alpha': ridge_range}]
    gs_ridge = GridSearchCV(estimator=Ridge(), param_grid=parameters_ridge, scoring='r2', cv=3, n_jobs=-1)
    gs_ridge.fit(x, train_fil.ix[:, -1])
    list_score.append(gs_ridge.best_score_)
    list_para.append(gs_ridge.best_params_)

    adb_range = range(6, 15, 1)
    parameters_ridge = [{'n_estimators': adb_range, 'base_estimator__alpha': ridge_range}]
    adbrg = BaggingRegressor(Ridge(), random_state=0)
    gs_ridge = GridSearchCV(estimator=adbrg, param_grid=parameters_ridge, scoring='r2', cv=3, n_jobs=-1)
    gs_ridge.fit(x, train_fil.ix[:, -1])
    list_score.append(gs_ridge.best_score_)
    list_para.append(gs_ridge.best_params_)

    # ridge = Ridge(alpha=1.0).fit(train_fil.ix[:,:-1], train_fil.ix[:,-1])
    # print(ridge.score(train_fil.ix[:,:-1], train_fil.ix[:,-1]))
    #
    # y = ridge.predict(test)
    # submit = stdscalerout.inverse_transform(y)
    #
    # results = pd.Series(submit,name="SalePrice")
    # submission = pd.concat([Id,results],axis = 1)
    # submission.to_csv("ridge.csv",index=False)
    lasso_range = [0.01, 0.1, 1, 10, 100, 1000]
    parameters_lasso = [{'alpha': lasso_range}]
    gs_lasso = GridSearchCV(estimator=Lasso(), param_grid=parameters_lasso, scoring='r2', cv=3, n_jobs=-1)
    gs_lasso.fit(x, train_fil.ix[:, -1])
    list_score.append(gs_lasso.best_score_)
    list_para.append(gs_lasso.best_params_)

    # lasso = Lasso(alpha=1.0).fit(train_fil.ix[:,:-1], train_fil.ix[:,-1])
    # print(lasso.score(train_fil.ix[:,:-1], train_fil.ix[:,-1]))
    #
    # y = lasso.predict(test)
    # submit = stdscalerout.inverse_transform(y)
    #
    # results = pd.Series(submit,name="SalePrice")
    # submission = pd.concat([Id,results],axis = 1)
    # submission.to_csv("lasso.csv",index=False)
    elastic_range_1 = [0.01, 0.1, 1, 10, 100, 1000]
    elastic_range_2 = [0.01, 0.1, 1, 10, 100, 1000]
    parameters_elastic = [{'alpha': elastic_range_1, 'l1_ratio':elastic_range_2}]
    gs_elastic = GridSearchCV(estimator=ElasticNet(), param_grid=parameters_elastic, scoring='r2', cv=3, n_jobs=-1)
    gs_elastic.fit(x, train_fil.ix[:, -1])
    list_score.append(gs_elastic.best_score_)
    list_para.append(gs_elastic.best_params_)

    xgb_range_n = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    xgb_range_d = [2, 3]
    # xgb_range_l = [0.01, 0.05, 0.1, 0.5, 1, 10]
    # parameters_xgb = [{'n_estimators': xgb_range_n, 'max_depth': xgb_range_d, 'gamma': xgb_range_l, 'reg_alpha': xgb_range_l, 'reg_lambda': xgb_range_l}]
    parameters_xgb = [{'n_estimators': xgb_range_n, 'max_depth': xgb_range_d}]
    gs_xgb = GridSearchCV(estimator=xgb.XGBRegressor(learning_rate=0.1, silent=1), param_grid=parameters_xgb, scoring='r2', cv=3, n_jobs=-1)
    gs_xgb.fit(x, train_fil.ix[:, -1])
    list_score.append(gs_xgb.best_score_)
    list_para.append(gs_xgb.best_params_)

    gbr_range_n = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    gbr_range_d = [2, 3]
    parameters_gbr = [{'n_estimators': gbr_range_n, 'max_depth': gbr_range_d}]
    gs_gbr = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.1, loss='huber'), param_grid=parameters_gbr,
                          scoring='r2', cv=3, n_jobs=-1)
    gs_gbr.fit(x, train_fil.ix[:, -1])
    list_score.append(gs_gbr.best_score_)
    list_para.append(gs_gbr.best_params_)

    # svr_range_1 = [0.01, 0.1, 1, 10, 100, 1000]
    # # svr_range_2 = [0.01, 0.1, 1, 10, 100, 1000]
    # parameters_svr = [{'C': svr_range_1}]
    # gs_svr = GridSearchCV(estimator=SVR(kernel='linear'), param_grid=parameters_svr, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    # gs_svr.fit(x, train_fil.ix[:, -1])
    # list_score.append(gs_svr.best_score_)
    # list_para.append(gs_svr.best_params_)

    # elastic = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(train_fil.ix[:,:-1], train_fil.ix[:,-1])
    # print(elastic.score(train_fil.ix[:,:-1], train_fil.ix[:,-1]))
    #
    # y = elastic.predict(test)
    # submit = stdscalerout.inverse_transform(y)
    #
    # results = pd.Series(submit,name="SalePrice")
    # submission = pd.concat([Id,results],axis = 1)
    # submission.to_csv("elastic.csv",index=False)
    score_df[str(i) + 'score'] = list_score
    score_df[str(i) + 'para'] = list_para
    print(i)

print(score_df)
score_df.to_csv('score.csv')