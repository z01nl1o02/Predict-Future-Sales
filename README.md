# Predict-Future-Sales
kaggle competions

[数据源](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)


用法:
* 下载数据源到当前目录data目录下
* python main.py 
* 运行结束后,得到df.test.csv,就是submission.txt, error = 0.99

2016-06-26:
好奇如何使error小于1,参考了几个kernel后,得到main.py. 其实和ML/feature engining没什么关系....

让error小于1.0的关键是在训练集中增加销售量为0的样本, 以下是关键代码
-
-```
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
```
