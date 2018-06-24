import os,sys,pdb
import numpy as np
import pandas as pd


df = pd.read_csv('samples.csv')

train = df[ df['ID'] < 0]
test = df[ df['ID'] >= 0]

train31 = train[ train['date_block_num'] == 31 ]['shop_id,item_id,item_price'.split(',')]
train33 = train[ train['date_block_num'] == 33 ]['shop_id,item_id,item_price'.split(',')]

newtrain = train33.merge( train31, how='outer',on = 'shop_id,item_id'.split(','), suffixes='33,31'.split(','))


def get_mean(X):
    a = X['item_price33']
    b = X['item_price31']
    if np.isnan(a):
        a = b
    elif np.isnan(b):
        b = a
    return (a+b)/2

newtrain['item_price'] = newtrain.apply(get_mean,axis=1)

newtrain = newtrain.drop('item_price33,item_price31'.split(','), axis=1)

def get_price(X):
    a = X['item_pricetrain']
    if np.isnan(a):
        a = 0
    return a

def get_has_price(X):
    a = X['item_pricetrain']
    if np.isnan(a):
        return 0
    return 1


print '# test before merge ',len(test)
newtest = test.merge( newtrain, how = 'left', on = 'shop_id,item_id'.split(','), suffixes='test,train'.split(','))
newtest['item_price'] = newtest.apply(get_price, axis=1)
newtest['has_price'] = newtest.apply(get_has_price,axis=1)
print '# test after merge ',len(newtest)

newtest = newtest.drop('item_pricetest,item_pricetrain'.split(','), axis=1)

train = df[ df['ID'] < 0].copy()
train['has_price'] = 1
df = pd.concat([train,newtest],sort=False)
df.to_csv('samples-1.csv',index=False)



