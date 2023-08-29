#%%
import numpy as np
import os
from tqdm import tqdm

data = []
for f in tqdm(os.listdir("data/cropped/Data TrigG1 new Code 6")):
    data.append([f, np.load("data/cropped/Data TrigG1 new Code 6/"+f)['y']])

import pandas as pd

df = pd.DataFrame(data=data, columns=['name','y'])
df['axles'] = df['y'].apply(np.sum)/10

df.axles.value_counts()[df.axles.unique()].sort_index().plot(kind='bar')


# print("train/val: ", vals[0], " test: ", vals[1:].sum())
# %%
import pickle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

names = df.name.to_numpy()
y = df.axles.to_numpy()

splits = 6
ind = df.axles.value_counts().index

# if y equals a value of ind, then change value to 0
for i in ind[df.axles.value_counts() < splits]:
    y[y==i] = 0

names = names[y!=0]
y = y[y!=0]

df = pd.DataFrame(data={'name':names, 'axles':y})
vals = df.axles.value_counts().to_numpy()
thresh1, thresh2 = 1, 5
print("train: ", vals[:thresh1].sum(), int(vals[:thresh1]/vals.sum()*100), 
      "% val: ", vals[thresh1:thresh2].sum(), int(vals[thresh1:thresh2].sum()/vals.sum()*100),
      "% test: ", vals[thresh2:].sum(), int(vals[thresh2:].sum()/vals.sum()*100), "%")

test_names, test_test_index = [], []
for val in ind[thresh1:]:
    test_names.extend(names[y==val].tolist())
    test_test_index.extend(np.argwhere(y==val).flatten().tolist())
print(len(test_names),test_names[:10])
with open(f"data/test_names.ob", "wb") as fp:
    pickle.dump(test_names, fp)
np.savetxt(f"data/test_names.csv", np.array(test_names), fmt='%s')

train_names, train_y = [], []
for val in ind[:thresh1]:
    train_names.extend(names[y==val].tolist())
    train_y.extend(y[y==val].tolist())
train_names = np.array(train_names)
train_y = np.array(train_y)

plt.close('all')
kf = KFold(n_splits=splits-1, shuffle=True, random_state=99)
for i, (train_index, test_index) in enumerate(kf.split(train_names)):
    os.makedirs(f"data/fold{i}", exist_ok=True)
    
    data = np.hstack((train_y[train_index], 
                      train_y[test_index],
                      y[test_test_index]
                      )).astype(int)
    label = ['train_names']*len(train_index) + ['train_y']*len(test_index) + ['test_names']*len(test_test_index)
    _df = pd.DataFrame(data={'data':data, 'label':label})
    print(_df[_df.label!='test_names'].data.sum(), len(_df[_df.label!='test_names'].data), ' | ',
          _df[_df.label=='test_names'].data.sum(), len(_df[_df.label=='test_names'].data))
    plt.figure()
    ax = sns.countplot(data=_df, x='data', hue='label')
    plt.xlabel('length in axles') 
    plt.ylabel('trains') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
    ax.spines['bottom'].set_visible(False)     
    # plt.title(f'Fold {i}')
    plt.legend(['training', 'validation', 'test'])  
    plt.savefig(f'fold{i}.png', dpi=600)
    plt.close('all')
    
    with open(os.path.join(f'data/fold{i}/train_names.ob'), "wb") as fp:
        pickle.dump(train_names[train_index], fp)
    np.savetxt(f"data/fold{i}/train_names.csv", train_names[train_index], fmt='%s')
    
    with open(os.path.join(f'data/fold{i}/val_names.ob'), "wb") as fp:
        pickle.dump(train_names[test_index], fp)
    np.savetxt(f"data/fold{i}/val_names.csv", train_names[test_index], fmt='%s')

#%%


X_train, X_test, y_train, y_test = train_test_split(names, y, test_size=1/splits, stratify=y, shuffle=True, random_state=99)

# check how many elemnts are equal in both splits
# print(np.intersect1d(X_test, X_test2).shape)


with open(f"data/test_names00.ob", "wb") as fp:
    pickle.dump(X_test.tolist(), fp)
np.savetxt(f"data/test_names00.csv", X_test, fmt='%s')

skf = StratifiedKFold(n_splits=splits-1, shuffle=True, random_state=99)
for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    os.makedirs(f"data/fold{i+1}00", exist_ok=True)
    
    
    data = np.hstack((y_train[train_index], 
                      y_train[test_index],
                      y_test
                      )).astype(int)
    label = ['train']*len(train_index) + ['val']*len(test_index) + ['test']*len(y_test)
    _df = pd.DataFrame(data={'data':data, 'label':label})
    print(_df[_df.label!='test'].data.sum(), len(_df[_df.label!='test'].data), ' | ',
          _df[_df.label=='test'].data.sum(), len(_df[_df.label=='test'].data))
    plt.figure()
    ax = sns.countplot(data=_df, x='data', hue='label') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)   
    ax.spines['bottom'].set_visible(False)     
    plt.xlabel('length in axles') 
    plt.ylabel('trains') 
    # plt.title(f'Fold {i}')
    plt.legend(['training', 'validation', 'test'])  
    plt.savefig(f'fold{i} stratified.png', dpi=600)
    plt.close('all')
    
    with open(os.path.join(f'data/fold{i+1}00/train_names.ob'), "wb") as fp:
        pickle.dump(X_train[train_index], fp)
    np.savetxt(f"data/fold{i+1}00/train_names.csv", X_train[train_index], fmt='%s')
    
    with open(os.path.join(f'data/fold{i+1}00/val_names.ob'), "wb") as fp:
        pickle.dump(X_train[test_index], fp)
    np.savetxt(f"data/fold{i+1}00/val_names.csv", X_train[test_index], fmt='%s')
    
#%%
# with open(os.path.join('val_names.ob'), "wb") as fp:
#     val_names = []
#     for val in ind[thresh1:thresh2]:
#         val_names.extend(df.name[df.axles==val].to_list())
#     print(len(val_names),val_names[:10])
#     pickle.dump(val_names, fp)

# %%
