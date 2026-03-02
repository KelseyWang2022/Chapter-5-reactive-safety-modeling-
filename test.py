import pandas as pd
import numpy as np

df = pd.read_csv('cleaned_data_recoded.csv')
#查看特征prof.plan和atm和surf的唯一取值


#将这四个特征#重新编码，1保持为1，其他的全部编码为0
df['prof'] = df['prof'].apply(lambda x: 1 if x == 1 else 0)
df['plan'] = df['plan'].apply(lambda x: 1 if x == 1 else 0)
df['atm'] = df['atm'].apply(lambda x: 1 if x == 1 else 0)
df['surf'] = df['surf'].apply(lambda x: 1 if x == 1 else 0)
df['obs']= df['obs'].apply(lambda x: 0 if x == 0 else 1)
df['obsm'] = df['obsm'].apply(lambda x: 0 if x == 0 else 1)
df['secu'] = df['secu'].apply(lambda x: 0 if x in [92, 22, 12, 32, 42] else x)
#删除取值不为0，1，2，3的行
df = df[df['secu'].isin([0, 1, 2, 3])]


print(df['secu'].unique())
print(df['prof'].unique())
print(df['plan'].unique())
print(df['atm'].unique())
print(df['surf'].unique())
print(df['obs'].unique())
print(df['obsm'].unique())


df.to_csv('cleaned_data_recoded_final.csv', index=False)
