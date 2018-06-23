import os,sys,pdb
import numpy as np
import pandas as pd


try:
    os.makedirs('shops')
except Exception,e:
    pass


df = pd.read_csv('data/sales_train_v2.csv')

for shop,shop_group in df.groupby('shop_id'):
    items = []
    months = []
    item_cnt_month = []
    for item, item_group in shop_group.groupby('item_id'):
        for month,month_group in item_group.groupby('date_block_num'):
            items.append( item )
            months.append( month )
            item_cnt_month.append( month_group['item_cnt_day'].sum() )
    pd.DataFrame(data = {'item':items, 'month':months,'cnt':item_cnt_month} ).to_csv('shops/%d.csv'%shop,index=False)





