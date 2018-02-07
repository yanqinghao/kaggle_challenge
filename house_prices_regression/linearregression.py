import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
meanval = [train['MasVnrArea'].mean(),train['MasVnrArea'].mean(),train['BsmtFinSF2'].mean(),train['BsmtUnfSF'].mean(),train['TotalBsmtSF'].mean(),train['BsmtFinSF1'].mean(),train['GarageYrBlt'].mean()]
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
                    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                    'KitchenQual','Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
                    'SaleCondition']
lelist = []
for i in labellist:
    le = LabelEncoder()
    le.fit(np.concatenate((train.loc[:,[i]].values,test.loc[:, [i]].values)))
    train.loc[:, [i]] = le.transform(train.loc[:,[i]].values)
    test.loc[:, [i]] = le.transform(test.loc[:, [i]].values)
    lelist.append(le)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(train.ix[:,:-1], train.ix[:,-1])
print(tree.score(train.ix[:,:-1], train.ix[:,-1]))

y = tree.predict(test)

results = pd.Series(y,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("tree.csv",index=False)

forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(train.ix[:,:-1], train.ix[:,-1])
print(forest.score(train.ix[:,:-1], train.ix[:,-1]))

y = forest.predict(test)

results = pd.Series(y,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("forest.csv",index=False)

# sns.heatmap(train.corr())
# plt.savefig('b.png')

stdscaler = StandardScaler()
train.loc[:,['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
             '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
             'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearBuilt','YearRemodAdd',
             'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
             'TotRmsAbvGrd','GarageYrBlt','GarageCars','YrSold']] = stdscaler.fit_transform(train.loc[:,['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
             '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
             'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearBuilt','YearRemodAdd',
             'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
             'TotRmsAbvGrd','GarageYrBlt','GarageCars','YrSold']].values.reshape(-1,31))
test.loc[:,['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
             '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
             'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearBuilt','YearRemodAdd',
             'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
             'TotRmsAbvGrd','GarageYrBlt','GarageCars','YrSold']] = stdscaler.transform(test.loc[:,['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
             '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF',
             'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearBuilt','YearRemodAdd',
             'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
             'TotRmsAbvGrd','GarageYrBlt','GarageCars','YrSold']].values.reshape(-1,31))

mmscaler = MinMaxScaler()
train.loc[:,['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour',
             'Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
             'Condition2','BldgType','HouseStyle','OverallQual','OverallCond',
             'RoofStyle','RoofMatl','Exterior1st',
             'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'Heating','HeatingQC','CentralAir','Electrical',
             'KitchenQual','Functional','Fireplaces','FireplaceQu',
             'GarageType','GarageFinish','GarageQual',
             'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold',
             'SaleType','SaleCondition']] = mmscaler.fit_transform(train.loc[:,['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour',
             'Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
             'Condition2','BldgType','HouseStyle','OverallQual','OverallCond',
             'RoofStyle','RoofMatl','Exterior1st',
             'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'Heating','HeatingQC','CentralAir','Electrical',
             'KitchenQual','Functional','Fireplaces','FireplaceQu',
             'GarageType','GarageFinish','GarageQual',
             'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold',
             'SaleType','SaleCondition']].values.reshape(-1,48))
test.loc[:,['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour',
             'Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
             'Condition2','BldgType','HouseStyle','OverallQual','OverallCond',
             'RoofStyle','RoofMatl','Exterior1st',
             'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'Heating','HeatingQC','CentralAir','Electrical',
             'KitchenQual','Functional','Fireplaces','FireplaceQu',
             'GarageType','GarageFinish','GarageQual',
             'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold',
             'SaleType','SaleCondition']] = mmscaler.transform(test.loc[:,['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour',
             'Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
             'Condition2','BldgType','HouseStyle','OverallQual','OverallCond',
             'RoofStyle','RoofMatl','Exterior1st',
             'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'Heating','HeatingQC','CentralAir','Electrical',
             'KitchenQual','Functional','Fireplaces','FireplaceQu',
             'GarageType','GarageFinish','GarageQual',
             'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','MoSold',
             'SaleType','SaleCondition']].values.reshape(-1,48))

stdscalerout = StandardScaler()
train.loc[:,['SalePrice']] = stdscalerout.fit_transform(train.loc[:,['SalePrice']].values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(train.ix[:,:-1], train.ix[:,-1], test_size=0.2, random_state=0, shuffle=True)

pca = PCA(n_components=25)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

test_pca = pca.transform(test)
y = lr.predict(test)
submit = stdscalerout.inverse_transform(y)

results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("lr.csv",index=False)

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
print(ridge.score(X_train, y_train))
print(ridge.score(X_test, y_test))

y = ridge.predict(test)
submit = stdscalerout.inverse_transform(y)

results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("ridge.csv",index=False)

lasso = Lasso(alpha=1.0).fit(X_train, y_train)
print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))

y = lasso.predict(test)
submit = stdscalerout.inverse_transform(y)

results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("lasso.csv",index=False)

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train, y_train)
print(elastic.score(X_train, y_train))
print(elastic.score(X_test, y_test))

y = elastic.predict(test)
submit = stdscalerout.inverse_transform(y)

results = pd.Series(submit,name="SalePrice")
submission = pd.concat([Id,results],axis = 1)
submission.to_csv("elastic.csv",index=False)