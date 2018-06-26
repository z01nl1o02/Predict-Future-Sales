from time import time
import pandas as pd
import os,sys,pdb
import numpy as np
from collections import defaultdict
import gc

dataroot = 'data/'
outroot = './'
ME_K = 2000
ME_F = 1

startTime = time()


def show_time(info):
    t = time() - startTime
    print("%.2f min %s"%(t/60.0,info ))
    return

def check_nan(df,dfname):
    for col in df.columns:
        if df[col].hasnans:
            print '{} {} with nan? {}'.format(dfname,col,df[col].hasnans)


    
#get item_cnt_month
trainset = pd.read_csv(os.path.join(dataroot,'sales_train_v2.csv'))
df = trainset.groupby('shop_id,item_id,date_block_num'.split(','))['item_cnt_day'].agg(['sum']).reset_index().rename(columns={'sum':'item_cnt_month'})
trainset = pd.merge(trainset, df, on = 'shop_id,item_id,date_block_num'.split(','), how = 'left')
trainset.drop('item_cnt_day',axis=1, inplace = True)
trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)

#date feature
if 0:
    trainset['date'] = pd.to_datetime(trainset['date'],format='%d.%m.%Y')
    trainset['year'] = trainset['date'].dt.year - 2013
    trainset['month'] = trainset['date'].dt.month
    trainset['dow'] = trainset['date'].dt.dayofweek
trainset.drop('date,item_price'.split(','),axis=1,inplace=True)
print 'before drop_duplicates() ',len(trainset)
trainset = trainset.drop_duplicates()
print 'after drop_duplicates() ',len(trainset)

#insert null samples (samples not found in train.csv)
from itertools import product
grid = []
for month in trainset['date_block_num'].drop_duplicates():
    shop = trainset[trainset.date_block_num == month]['shop_id'].drop_duplicates()
    item = trainset[trainset.date_block_num == month]['item_id'].drop_duplicates()
    grid.append( np.asarray(   list( product( *[shop,item,[month]] ) )    )  )
cols = ['shop_id','item_id','date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = cols, dtype=np.int32)
trainset = pd.merge(grid,trainset, on = cols, how = 'left').fillna(0)
print trainset.columns
show_time("clean trainset done")


testset = pd.read_csv(os.path.join(dataroot,'test.csv')).drop('ID',axis=1,inplace=False)
testset['date_block_num'] = 34
testset['item_cnt_month'] = 0
print testset.columns


print 'before move shop only in trainset ',len(trainset)
trainset = trainset.merge(testset[['shop_id']].drop_duplicates(),how='right')
print 'after move shop only in trainset ',len(trainset)

show_time("clean testset done")



#check nan
check_nan(trainset, 'trainset')
check_nan(testset, 'testset')
show_time("check nan done")


#add categatory/item
items = pd.read_csv(os.path.join(dataroot,'items.csv'))
items.drop('item_name',inplace=True, axis=1)
trainset = trainset.merge(items, on=['item_id'], how = 'left').fillna(0)
testset = testset.merge(items, on=['item_id'],how='left').fillna(0)

print 'trainset columns:',trainset.columns
print 'testset columns:',testset.columns
print 'train set date_block_num min/max:{},{}'.format(trainset['date_block_num'].min(), trainset['date_block_num'].max())
print 'train set item_cnt_month min/max:{},{}'.format(trainset['item_cnt_month'].min(), trainset['item_cnt_month'].max())
show_time("merge categ done")


# mean encoding
# targetCol = 'item_cnt_month'
# baseCols = 'shop_id,item_id,item_category_id'.split(',')
# def mean_encoding(df,colX, colY):
#     pr0 = df[colY].mean()
#     pr = df.groupby(colX)[colY].mean()
#     cnt = df.groupby(colX)[colX].count()
#     df['pr0'] = pr0
#     df['pr'] = df[colX].map(pr)
#     df['cnt'] = df[colX].map(cnt)
#     df['w'] = 1.0 / ( 1 + np.exp(  -1*(df['cnt'] - ME_K)/ME_F  ) )
#     df['w'] = df['w'].clip(0,1)
#     df[colX+"_ME"] = df.apply(lambda x : x['pr'] * x['w'] + x['pr0'] * (1-x['w']), axis=1 )
#     mapdata = (pr0, pr, cnt)
#     df.drop('pr0,pr,cnt,w'.split(','),axis=1,inplace=True)
#     return df, mapdata
#
# colmaps = defaultdict(str)
# for col in baseCols:
#     trainset, colmaps[col] = mean_encoding(trainset, col,targetCol)
#     print trainset.columns
# trainset.to_csv('trainset_with_me.csv',index=False)
# show_time("mean encoding done")

# lag features
alldata = pd.concat([trainset, testset],sort=False)
print alldata.columns
print alldata.describe()

keyCols = ['shop_id','item_id','date_block_num','item_category_id']
lagCols = ['item_cnt_month']

for offset in [1,2,3,4,12]:
    shiftdata = alldata[keyCols + lagCols].copy()
    shiftdata['date_block_num'] += offset
    old2new = {}
    for col in lagCols:
        old2new[col]='{}_lag_{}'.format(col,offset)
    shiftdata.rename( columns = old2new, inplace=True )
    alldata = alldata.merge(shiftdata,on = keyCols, how = 'left' ).fillna(0)
print alldata.columns
print alldata['date_block_num'].describe()

#remove train set before 2014
alldata = alldata[alldata.date_block_num >= 12]

#alldata.to_csv('all_with_lag.csv',index=False)
show_time("generate lag feature done")

#copy some to be feature
for col in ['shop_id','item_id','item_category_id','date_block_num']:
    alldata[col + '_ft'] = alldata[col]

#scaling
from sklearn.preprocessing import StandardScaler
dateCol = 'date_block_num'
trainset = alldata[alldata[dateCol] != alldata[dateCol].max()].copy()
testset = alldata[alldata[dateCol] == alldata[dateCol].max()].copy()

targetCols = ['item_cnt_month']
keyCols = ['shop_id','item_id','date_block_num']
featureCols = alldata.columns.difference(targetCols + keyCols)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

trainset.loc[:,featureCols] = ss.fit_transform(trainset[featureCols])
testset.loc[:,featureCols] = ss.transform(testset[featureCols])

alldata = pd.concat([trainset,testset],axis=0)

del trainset
del testset
gc.collect()

show_time("scale done")
#regression
#level 1
regressors = []
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

def calc_error(testY, predY,name):
    error = np.sqrt(  mean_squared_error(testY, predY)  )
    log = "{} error: {}".format(name, error)
    return name,error,log

from tqdm import tqdm
L2XY = []
errors = []

lgb_params = {

    'feature_fraction': 0.75,

    'metric': 'rmse',

    'nthread': 1,

    'min_data_in_leaf': 2 ** 7,

    'bagging_fraction': 0.75,

    'learning_rate': 0.03,

    'objective': 'mse',

    'bagging_seed': 2 ** 7,

    'num_leaves': 2 ** 7,

    'bagging_freq': 1,

    'verbose': -1,
    


}

for month in tqdm(range(27, 34+1)):
    trainX = alldata[alldata.date_block_num < month][featureCols].copy()
    trainY = alldata[alldata.date_block_num < month][targetCols].values.flatten()

    testX = alldata[alldata.date_block_num == month][featureCols].copy()
    testY = alldata[alldata.date_block_num == month][targetCols].values.flatten()


    L2Y = []

    if 1:
        clf = LogisticRegression()
        clf.fit(trainX,trainY)
        predY = clf.predict(testX)
        L2Y.append(predY)
        errors.append(calc_error(testY,predY,"level 1 logisticRegression"))

    clf = SGDRegressor()
    clf.fit(trainX, trainY)
    predY = clf.predict(testX)
    L2Y.append(predY)
    errors.append(calc_error(testY,predY,"level 1 SGDRegression"))


    if 0:
        clf = RandomForestRegressor(n_jobs=4)
        clf.fit(trainX, trainY)
        predY = clf.predict(testX)
        L2Y.append(predY)
        errors.append(calc_error(testY, predY, "level 1 RandomForestRegressor"))

    if 1:
        clf = lgb.train(lgb_params,lgb.Dataset(trainX,label=trainY),300)
        predY = clf.predict(testX)
        L2Y.append(predY)
        errors.append(calc_error(testY, predY, "level 1 lightBGM"))

    L2Y = [np.reshape(d,(-1,1)) for d in L2Y]
    L2Y = np.concatenate(L2Y,axis=1)
    L2XY.append((testY, L2Y))

for error in errors:
    print error[2]

show_time("L1 training done")

import cPickle
with open('L2XY.pkl','wb') as f:
    cPickle.dump((alldata,L2XY),f)


#train L2 clf
trainXY = L2XY[0:-2]
validXY = L2XY[-2:-1]
testXY = [L2XY[-1]]


trainX = np.concatenate([X[1] for X in trainXY],axis=0)
trainY = np.concatenate([X[0] for X in trainXY],axis=0)

validX = np.concatenate([X[1] for X in validXY])
validY = np.concatenate([X[0] for X in validXY])


testX = np.concatenate([X[1] for X in testXY])
testY = np.concatenate([X[0] for X in testXY])

clf = LinearRegression(n_jobs = 4)

clf.fit(trainX, trainY)
predY = clf.predict(validX)

print calc_error(validY,predY,"L2 valid ")[2]

predY = clf.predict(testX)

df = pd.read_csv('data/test.csv')
df['date_block_num'] = 34


targetCol = ['item_cnt_month']
keyCols = ['shop_id','item_id','date_block_num']
df = df.merge(alldata[keyCols + targetCol], on=keyCols,how='left')
df['item_cnt_month'] = predY
df[['ID','item_cnt_month']].to_csv("df.test.csv",index=False)

submission = alldata[alldata.date_block_num == alldata.date_block_num.max()].copy()
submission.loc[:,'item_cnt_month'] = predY.clip(0,20)

#submission = submission['shop_id,ID,item_id,item_cnt_month'.split(',')].merge(df['ID'],on='ID'.split(','), how='right')

submission['ID,item_cnt_month'.split(',')].to_csv('submission.csv',index=False)












