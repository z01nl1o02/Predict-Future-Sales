import os,sys,pdb
import pandas as pd
import numpy as np
from collections import defaultdict

def check_number(strList):
    res = [float(x) for x in strList]
    res = np.asarray(res)
    print res.min(), res.max(), res.mean()

def merge_file(left,right_path,join_keys):
    item = pd.read_csv(right_path)
    left = left.merge(item,on=join_keys,how='inner')
    return left

def group_by_month(trainset):
    result_dict = defaultdict(list)
    for name,group in trainset.groupby( 'shop_id,item_id,date_block_num'.split(',')):
        #shop_id,item_id,date_block_num = group[shop_id],group[item_id],group['date_block_num']
        shop_id,item_id,date_block_num = name #group[shop_id],group[item_id],group['date_block_num']
        print name
        item_price = group['item_price'].mean()
        item_cnt_day = group['item_cnt_day'].sum()
        result_dict['shop_id'].append( shop_id )
        result_dict['item_id'].append( item_id )
        result_dict['date_block_num'].append( date_block_num )
        result_dict['item_price'].append( item_price )
        result_dict['item_cnt_day'].append( item_cnt_day )
    return pd.DataFrame( result_dict )


paths = [os.path.join('data',x) for x in 'items.csv,shops.csv,item_categories.csv'.split(',')]
keys = 'item_id,shop_id,item_category_id'.split(',')

trainset = pd.read_csv('data/sales_train_v2.csv')
print 'trainset size:',len(trainset)

#remove invalidate row
print '# total train set ',len(trainset)
trainset = trainset[ trainset['item_price'] > 0] 
trainset = trainset[ trainset['item_cnt_day'] >= 0]
print '# valid trainset ',len(trainset)

#group to month
trainset = group_by_month(trainset)
print '# after group by month ',len(trainset)

testset = pd.read_csv('data/test.csv')
testset['date_block_num']=32 
testset['item_price'] = 0
print 'testset size:',len(testset)

for path,key in zip(paths,keys):
    print 'merge ',key,' from ',path
    trainset = merge_file(trainset,path,key)
    testset = merge_file(testset, path, key)

print 'trainset size:',len(trainset)
print 'testset size:',len(testset)

strcols = [x for x in trainset.columns.values if x.find('name') >= 0]
print 'remove columns ',
for col in strcols:
    print col,',',
print ''
trainset = trainset.drop(columns=strcols)
testset = testset.drop(columns=strcols)

#merge train/test into one file
trainset['ID'] = -1
testset['item_cnt_day'] = -1

testcols =  testset.columns.values 
trainset = trainset[testcols]

allset = pd.concat([trainset,testset])

allset.to_csv('samples.csv',index=False)


