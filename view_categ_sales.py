import os,sys,pdb
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt

df = pd.read_csv('samples-1.csv')
df = df[df.ID < 0]['item_category_id,date_block_num,item_cnt_day'.split(',')]

res = defaultdict(dict)
num = 0
for item,item_group in df.groupby('item_category_id'):
    num += 1
    dates = []
    sales = []
    for date,date_group in item_group.groupby('date_block_num'):
        sale = date_group['item_cnt_day'].sum()
        sales.append( sale  )
        dates.append( date )
    plt.ylim((0,300))
    plt.plot(dates, sales, label = 'item %d'%item)

plt.legend()
plt.show()



