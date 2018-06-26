# Predict-Future-Sales
kaggle competions


让error小于1.0的关键是在训练集中增加销售量为0的样本, 以下是关键代码

```
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