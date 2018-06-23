import os,sys,pdb
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt

df = pd.read_csv('samples-1.csv')
df = df[df.ID < 0]['shop_id,date_block_num,item_cnt_day'.split(',')]

res = defaultdict(dict)
for shop,shop_group in df.groupby('shop_id'):
    dates = []
    sales = []
    for date,date_group in shop_group.groupby('date_block_num'):
        sales.append( date_group['item_cnt_day'].sum() )
        dates.append( date )
    plt.plot(dates, sales, label = 'shop %d'%shop)

plt.legend()
plt.show()



