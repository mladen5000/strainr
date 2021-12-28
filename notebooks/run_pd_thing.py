import pickle
import glob
import pandas as pd
import time

t0 = time.time()
with open("/home/mladen/Strainr/new_databases/823/pdist_complete_refseq_31.pkl",'rb') as ph:
    db = pickle.load(ph)
 
t1 = time.time()
print(t1-t0)
db2 = {k:list(v) for k,v in db.items() if len(v) > 1}
t2 = time.time()
print(t2-t1)
print(len(db2))
df = pd.Series(db2)
t2 = time.time()
print(df.head())
df2 = pd.DataFrame( df.values.tolist(),index=df.index,)
print(df2)
df3 = pd.get_dummies(df2)
print(df3)
t3 = time.time()
print(t3-t2)
print(df3.sum(axis=0))
print(df3.sum(axis=1))