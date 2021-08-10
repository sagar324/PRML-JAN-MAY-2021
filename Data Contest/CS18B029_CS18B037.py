#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import lightgbm as ltb
import catboost as cb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

paths = ['song_labels.csv','songs.csv','save_for_later.csv','train.csv','test.csv']

        
for p in paths:
    if 'song_labels' in p:
        data1 = pd.read_csv(p)
    if 'songs' in p:
        data3 = pd.read_csv(p)
    if 'save_for_later' in p:
        data2 = pd.read_csv(p)
    if 'train' in p:
        train = pd.read_csv(p)
    if 'test.csv' in p:
        test = pd.read_csv(p)


datasfl = data2
data1.loc[data1['count'] < 0, 'count'] = 0
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df1 = data1.groupby(["platform_id"])['count'].transform(max) == data1['count']
data1 = data1[df1]
df1 = data1.groupby(["platform_id"])['label_id'].transform(max) == data1['label_id']
data1 = data1[df1]


# In[ ]:


#df1 = data1.groupby(["platform_id"])['count'].transform(max) == data1['count']
#data1 = data1[df1]
#df1 = data1.groupby(["platform_id"])['label_id'].transform(max) == data1['label_id']
#data1 = data1[df1]
#data1.info()
#data1.head()


# In[ ]:


data1["number_of_reviews"] = data1["count"]
data1 = data1.drop(['count'],axis=1)
data3 = pd.merge(data1,data3,on="platform_id",how="right")
data3['released_year'].fillna(0, inplace=True)


# In[ ]:


lst = data3['released_year'].tolist()
for i in range(len(lst)):
    if lst[i]<100:
        lst[i]=0
data3 = data3.drop(['released_year'],axis=1)
data3.insert(loc=4, column='released_year', value=lst)
data3['released_year'] = data3['released_year'].replace(0,1981)


# In[ ]:


data3['released_year'].fillna(1981, inplace=True)
data3['number_of_comments'].fillna((data3['number_of_comments'].mean()), inplace=True)
data3['language'].fillna('eng', inplace=True)

data3['language'] = data3['language'].replace(['en-US'],'eng')
data3['language'] = data3['language'].replace(['en-GB'],'eng')
data3['language'] = data3['language'].replace(['en-CA'],'eng')
data3['language'] = data3['language'].replace(['en'],'eng')
data3 = data3.drop(['platform_id'], axis=1)


# In[ ]:


datanew = train.merge(data3,on="song_id",how='left')
datanew['released_year'].fillna(1981, inplace=True)
datanew['number_of_reviews'].fillna((data3['number_of_reviews'].mean()), inplace=True)
datanew['language'].fillna("eng", inplace=True)
datanew['number_of_comments'].fillna((data3['number_of_comments'].mean()), inplace=True)
datanew['label_id'].fillna(30574, inplace=True)


# In[ ]:


data2['sfl']=1
datanew2 = pd.merge(data2,datanew,on=['customer_id','song_id'],how='right')
datanew2['sfl'].fillna(0, inplace=True)


# In[ ]:


testnew = test.merge(data3,on="song_id",how='left')
testnew['released_year'].fillna(1981, inplace=True)
testnew['number_of_reviews'].fillna((data3['number_of_reviews'].mean()), inplace=True)
testnew['language'].fillna("eng", inplace=True)
testnew['number_of_comments'].fillna((data3['number_of_comments'].mean()), inplace=True)
testnew['label_id'].fillna(30574, inplace=True)

testnew2 = pd.merge(data2,testnew,on=['customer_id','song_id'],how='right')
testnew2['sfl'].fillna(0, inplace=True)


# In[ ]:


datanew2 = datanew2.drop(['sfl'],axis=1)
testnew2 = testnew2.drop(['sfl'],axis=1)


# In[ ]:


train = datanew2
test = testnew2
temp = datasfl
temp2 = data2

users = train['customer_id'].unique().tolist()
encodedvals = []
dicto = {}
for i in range(len(users)):
    encodedvals.append(i)
    dicto[users[i]] = i
userscolumntrain = train['customer_id'].tolist()
userscolumntest = test['customer_id'].tolist()
userscolumntemp = temp['customer_id'].tolist()
userscolumntemp2 = temp2['customer_id'].tolist()
newuserscolumntrain = []
newuserscolumntest = []
newuserscolumntemp = []
newuserscolumntemp2 = []
for i in range(len(userscolumntrain)):
    newuserscolumntrain.append(dicto[userscolumntrain[i]])
for i in range(len(userscolumntest)):
    newuserscolumntest.append(dicto[userscolumntest[i]])
for i in range(len(userscolumntemp)):
    newuserscolumntemp.append(dicto[userscolumntemp[i]])
for i in range(len(userscolumntemp2)):
    newuserscolumntemp2.append(dicto[userscolumntemp2[i]])
train = train.drop(['customer_id'],axis=1)
test = test.drop(['customer_id'],axis=1)
temp = temp.drop(['customer_id'],axis=1)
temp2 = temp2.drop(['customer_id'],axis=1)


# In[ ]:


train.insert(loc=1, column='customer_id', value=newuserscolumntrain)
test.insert(loc=1, column='customer_id', value=newuserscolumntest)
temp.insert(loc=1, column='customer_id', value=newuserscolumntemp)
temp2.insert(loc=1, column='customer_id', value=newuserscolumntemp2)


# In[ ]:


users = train['language'].unique().tolist()
encodedvals = []
dicto2 = {}
for i in range(len(users)):
    encodedvals.append(i)
    dicto2[users[i]] = i
userscolumntrain = train['language'].tolist()
userscolumntest = test['language'].tolist()
newuserscolumntrain = []
newuserscolumntest = []
for i in range(len(userscolumntrain)):
    newuserscolumntrain.append(dicto2[userscolumntrain[i]])
for i in range(len(userscolumntest)):
    newuserscolumntest.append(dicto2[userscolumntest[i]])
train = train.drop(['language'],axis=1)
test = test.drop(['language'],axis=1)


# In[ ]:


train.insert(loc=3, column='language', value=newuserscolumntrain)
test.insert(loc=3, column='language', value=newuserscolumntest)


# In[ ]:


datanew2=train
testnew2=test
datasfl = temp
data2 = temp2


# In[ ]:


lst2 = testnew2['label_id'].tolist()
testnew2 = testnew2.drop(['label_id'],axis=1)
testnew2.insert(3,column='label_id', value=lst2)


# In[ ]:




find_avg = datanew2

find_avg = find_avg.drop(['customer_id'],axis=1)
find_avg = find_avg.drop(['language'],axis=1)
find_avg = find_avg.drop(['label_id'],axis=1)
#find_avg = find_avg.drop(['number_of_reviews'],axis=1)
find_avg = find_avg.drop(['released_year'],axis=1)
find_avg = find_avg.drop(['number_of_reviews'],axis=1)
find_avg = find_avg.drop(['number_of_comments'],axis=1)
find_avg['freq']=1

find_avg = find_avg.groupby(['song_id']).sum()
find_avg['avg'] = find_avg['score']/find_avg['freq']
find_avg = find_avg.drop(['score'],axis=1)
find_avg = find_avg.drop(['freq'],axis=1)

lst2 = []
for i in find_avg.itertuples():
    lst2.append(i[1])


    
find_avg = datanew2

find_avg = find_avg.drop(['song_id'],axis=1)
find_avg = find_avg.drop(['language'],axis=1)
find_avg = find_avg.drop(['label_id'],axis=1)
#find_avg = find_avg.drop(['number_of_reviews'],axis=1)
find_avg = find_avg.drop(['released_year'],axis=1)
find_avg = find_avg.drop(['number_of_reviews'],axis=1)
find_avg = find_avg.drop(['number_of_comments'],axis=1)
find_avg['freq']=1

find_avg = find_avg.groupby(['customer_id']).sum()
find_avg['avg'] = find_avg['score']/find_avg['freq']
find_avg = find_avg.drop(['score'],axis=1)
find_avg = find_avg.drop(['freq'],axis=1)

lst_2 = []
for i in find_avg.itertuples():
    lst_2.append(i[1])


# In[ ]:


lst3 = []
for i in datanew2.itertuples():
    lst3.append(lst2[i[1]-1])
lst_3 = []
for i in datanew2.itertuples():
    lst_3.append(lst_2[i[2]])
datanew2['song_rat_avg'] = lst3
datanew2['user_rat_avg'] = lst_3

#modify test
lst3 = []
for i in testnew2.itertuples():
    lst3.append(lst2[i[1]-1])
lst_3 = []
for i in testnew2.itertuples():
    lst_3.append(lst_2[i[2]])
testnew2['song_rat_avg'] = lst3
testnew2['user_rat_avg'] = lst_3


# In[ ]:


temporary = datasfl


# In[ ]:


songs_df = datanew2
songs_df3 = songs_df
songs_df = songs_df.drop(['customer_id','score','released_year','number_of_comments','user_rat_avg','song_rat_avg'],axis=1)
songs_df = songs_df.groupby('song_id').max()
songs_df = songs_df.reset_index()

songs_df2 = songs_df


# In[ ]:


datasfl = temporary
songs_df = songs_df.drop(['language'],axis=1)
datasfl = pd.merge(temporary,songs_df,on="song_id",how="left")
datasfl = datasfl.drop(['sfl','song_id'],axis=1)
datasfl['freq'] = 1
datasfl = datasfl.groupby(['customer_id','label_id']).sum()
datasfl = datasfl.reset_index()


df1 = datasfl.groupby(["customer_id"])['freq'].transform(max) == datasfl['freq']
datasfl = datasfl[df1]
df1 = datasfl.groupby(["customer_id"])['label_id'].transform(max) == datasfl['label_id']
datasfl = datasfl[df1]

datasfl = datasfl.drop(['freq'],axis=1)


# In[ ]:


avg_liked_label = []
for i in range(14053):    
    avg_liked_label.append(30574)
for i in datasfl.itertuples():
    avg_liked_label[i[1]] = i[2]

avg_liked_label_train = []
avg_liked_label_test = []

for i in datanew2.itertuples():
    avg_liked_label_train.append(avg_liked_label[i[2]])
for i in testnew2.itertuples():
    avg_liked_label_test.append(avg_liked_label[i[2]])
datanew2['avg_liked_label'] = avg_liked_label_train
testnew2['avg_liked_label'] = avg_liked_label_test


# In[ ]:


songs_df2 = songs_df2.drop(['label_id'],axis=1)
datasfl = pd.merge(temporary,songs_df2,on="song_id",how="left")
datasfl = datasfl.drop(['sfl','song_id'],axis=1)
datasfl['freq'] = 1
datasfl = datasfl.groupby(['customer_id','language']).sum()
datasfl = datasfl.reset_index()


df1 = datasfl.groupby(["customer_id"])['freq'].transform(max) == datasfl['freq']
datasfl = datasfl[df1]
df1 = datasfl.groupby(["customer_id"])['language'].transform(max) == datasfl['language']
datasfl = datasfl[df1]

datasfl = datasfl.drop(['freq'],axis=1)


# In[ ]:


avg_liked_label = []
for i in range(14053):    
    avg_liked_label.append(0)
for i in datasfl.itertuples():
    avg_liked_label[i[1]] = i[2]

avg_liked_label_train = []
avg_liked_label_test = []

for i in datanew2.itertuples():
    avg_liked_label_train.append(avg_liked_label[i[2]])
for i in testnew2.itertuples():
    avg_liked_label_test.append(avg_liked_label[i[2]])
datanew2['avg_liked_language'] = avg_liked_label_train
testnew2['avg_liked_language'] = avg_liked_label_test


# In[ ]:


datasfl = temporary
datasfl['freq']=1

part1 = datasfl
part2 = datasfl
lst01 = []
lst10 = []

for i in range(14053):
    lst10.append(0)
for i in range(10000):
    lst01.append(0)

part1 = part1.drop(['customer_id'],axis=1)
part1 = part1.groupby('song_id').sum()
part1 = part1.reset_index()

part2 = part2.drop(['song_id'],axis=1)
part2 = part2.groupby('customer_id').sum()
part2 = part2.reset_index()

for i in part1.itertuples():
    lst01[i[1]-1] = (i[2]*1.0)/140.53 
for i in part2.itertuples():
    lst10[i[1]] = (i[2]*1.0)/100

user_save_train = []
song_save_train = []
for i in datanew2.itertuples():
    user_save_train.append(lst10[i[2]])
    song_save_train.append(lst01[i[1]-1])

datanew2['user_save_freq'] = user_save_train
datanew2['song_save_freq'] = song_save_train

user_save_test = []
song_save_test = []
for i in testnew2.itertuples():
    user_save_test.append(lst10[i[2]])
    song_save_test.append(lst01[i[1]-1])

testnew2['user_save_freq'] = user_save_test
testnew2['song_save_freq'] = song_save_test


# In[ ]:


datanew2 = pd.merge(datanew2,data2,on=['song_id','customer_id'],how = "left")
datanew2['sfl'].fillna(-1, inplace=True)
testnew2 = pd.merge(testnew2,data2,on=['song_id','customer_id'],how = "left")
testnew2['sfl'].fillna(-1, inplace=True)


# In[ ]:


songs_df = datanew
songs_df = songs_df.drop(['customer_id','score','number_of_reviews','language','label_id','number_of_comments'],axis=1)
songs_df['freq']=1
songs_df = songs_df.groupby('song_id').sum()
songs_df = songs_df.reset_index()
songs_df ['released_year'] = songs_df['released_year']/songs_df['freq']
songs_df = songs_df.drop(['freq'],axis=1)

#

likey = pd.merge(data2,songs_df,on = "song_id",how = "left")
likey = likey.drop(['sfl','song_id'],axis=1)
likey['freq']=1
likey = likey.groupby(['customer_id']).sum()
likey = likey.reset_index()
likey['avg_liked_year'] = likey['released_year']/likey['freq']
likey = likey.drop(['released_year','freq'],axis=1)

#

avglikedyear = []
constant = likey['avg_liked_year'].mean()
for i in range(14053):
    avglikedyear.append(constant)
train_likeys = []
test_likeys = []
count=0
for i in likey.itertuples():
    avglikedyear[i[1]] = i[2]
for i in datanew2.itertuples():
    train_likeys.append(avglikedyear[i[2]])
for i in testnew2.itertuples():
    test_likeys.append(avglikedyear[i[2]])
datanew2['avg_liked_year'] = train_likeys
testnew2['avg_liked_year'] = test_likeys


# In[ ]:


songs_df = datanew
songs_df = songs_df.drop(['customer_id','score','number_of_reviews','language','label_id','released_year'],axis=1)
songs_df['freq']=1
songs_df = songs_df.groupby('song_id').sum()
songs_df = songs_df.reset_index()
songs_df ['number_of_comments'] = songs_df['number_of_comments']/songs_df['freq']
songs_df = songs_df.drop(['freq'],axis=1)

#

likey = pd.merge(data2,songs_df,on = "song_id",how = "left")
likey = likey.drop(['sfl','song_id'],axis=1)
likey['freq']=1
likey = likey.groupby(['customer_id']).sum()
likey = likey.reset_index()
likey['avg_noc'] = likey['number_of_comments']/likey['freq']
likey = likey.drop(['number_of_comments','freq'],axis=1)

#

avglikedyear = []
constant = likey['avg_noc'].mean()
for i in range(14053):
    avglikedyear.append(constant)
train_likeys = []
test_likeys = []
count=0
for i in likey.itertuples():
    avglikedyear[i[1]] = i[2]
for i in datanew2.itertuples():
    train_likeys.append(avglikedyear[i[2]])
for i in testnew2.itertuples():
    test_likeys.append(avglikedyear[i[2]])
datanew2['avg_noc'] = train_likeys
testnew2['avg_noc'] = test_likeys


# In[ ]:


datanew2.label_id = datanew2.label_id.astype('int64')
testnew2.label_id = testnew2.label_id.astype('int64')
datanew2.avg_liked_label = datanew2.avg_liked_label.astype('int64')
testnew2.avg_liked_label = testnew2.avg_liked_label.astype('int64')


# In[ ]:


#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#testnew2['released_year'] = min_max_scaler.fit_transform(np.array(testnew2['released_year']).reshape(-1,1))
#testnew2['number_of_comments'] = min_max_scaler.fit_transform(np.array(testnew2['number_of_comments']).reshape(-1,1))
#testnew2['song_rat_avg'] = min_max_scaler.fit_transform(np.array(testnew2['song_rat_avg']).reshape(-1,1))
#testnew2['user_rat_avg'] = min_max_scaler.fit_transform(np.array(testnew2['user_rat_avg']).reshape(-1,1))
#testnew2['user_save_freq'] = min_max_scaler.fit_transform(np.array(testnew2['user_save_freq']).reshape(-1,1))
#testnew2['song_save_freq'] = min_max_scaler.fit_transform(np.array(testnew2['song_save_freq']).reshape(-1,1))
#testnew2['avg_liked_year'] = min_max_scaler.fit_transform(np.array(testnew2['avg_liked_year']).reshape(-1,1))
#datanew2['released_year'] = min_max_scaler.fit_transform(np.array(datanew2['released_year']).reshape(-1,1))
#datanew2['number_of_comments'] = min_max_scaler.fit_transform(np.array(datanew2['number_of_comments']).reshape(-1,1))
#datanew2['song_rat_avg'] = min_max_scaler.fit_transform(np.array(datanew2['song_rat_avg']).reshape(-1,1))
#datanew2['user_rat_avg'] = min_max_scaler.fit_transform(np.array(datanew2['user_rat_avg']).reshape(-1,1))
#datanew2['user_save_freq'] = min_max_scaler.fit_transform(np.array(datanew2['user_save_freq']).reshape(-1,1))
#datanew2['song_save_freq'] = min_max_scaler.fit_transform(np.array(datanew2['song_save_freq']).reshape(-1,1))
#datanew2['avg_liked_year'] = min_max_scaler.fit_transform(np.array(datanew2['avg_liked_year']).reshape(-1,1))


# In[ ]:


datanew2 = datanew2.drop(['sfl'],axis=1)
testnew2 = testnew2.drop(['sfl'],axis=1)


# In[ ]:


X = datanew2
y = datanew2['score']
X = X.drop(['score'],axis=1)


# In[ ]:



model = cb.CatBoostRegressor(cat_features = [0,1,2,3,9,10])
model.fit(X, y)
#predict target
y_pred = model.predict(testnew2)


# 

# In[ ]:


lst2 = y_pred.tolist()
lst1 = []
for i in range(len(lst2)):
    lst1.append(i)
dfsubmit = pd.DataFrame(np.array(lst1),columns=['test_row_id'])
dfsubmit['score'] = lst2
dfsubmit.to_csv('CS18B029_CS18B037.csv', index = False, header = True)

