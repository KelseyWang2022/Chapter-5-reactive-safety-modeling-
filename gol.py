import pandas as pd
import numpy as np

df = pd.read_csv('crash_data (2).csv')
#查看特征grav唯一取值
print(df['grav'].unique())
#查看grav统计数量
print(df['grav'].value_counts())