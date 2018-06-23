import pandas as pd
import numpy as np



df = pd.read_csv('samples-1.csv')
df = df[ df.shop_id == 20 ]

df.to_csv('samples-2.csv', index=False)

