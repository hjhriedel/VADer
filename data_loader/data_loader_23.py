#%%
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

#%%
# Data in Memory! One file per train

class DataLoader():
    def __init__(self, config, timestamp): #, batch_size=16, shuffle=False): 
        self.config = config
        self.POOLING = 2**self.config.model.pooling_steps
        self.batch_size = self.config.trainer.batch_size
        self.shuffle = self.config.trainer.shuffle
        self.fold = self.config.data_loader.fold
        self.shape = None
        self.timestamp = timestamp
        self.trainDS, self.valDS, self.testDS = self.preprocess()
        
    def preprocess(self, val_size=0.2, test_size=0.1): 
        nameTrain = np.loadtxt(f"data/fold{self.fold}/train_names.csv", dtype=str).tolist()
        nameVal = np.loadtxt(f"data/fold{self.fold}/val_names.csv", dtype=str).tolist()
        if self.fold > 99:
            print('stratified')
            nameTest = np.loadtxt(f"data/test_names00.csv", dtype=str).tolist()
        else:
            nameTest = np.loadtxt(f"data/test_names.csv", dtype=str).tolist()

        Xtrain, Ytrain = self._loadSequential(nameTrain)
        Xval, Yval = self._loadSequential(nameVal)
        Xtest, Ytest = self._loadSequential(nameTest)
        self.train_len = len(Xtrain)
        self.val_len = len(Xval)
        print(self.train_len, self.val_len)
        self.shape = Xtrain[0].shape
        print(self.shape)

        trainDS = self._list_to_dataset(Xtrain, Ytrain, self._getBuckets(Xtrain))#.repeat()
        valDS = self._list_to_dataset(Xval, Yval, self._getBuckets(Xval))#.repeat()
        testDS = self._list_to_dataset(Xtest, Ytest, self._getBuckets(Xtest))
        return trainDS, valDS, testDS
        
    def get_train_data(self,shuffle=True):
        if shuffle:
            return self.trainDS.shuffle(buffer_size=50, reshuffle_each_iteration=True)
        else:
            return self.trainDS
        
    def get_validation_data(self,shuffle=False):
        if shuffle: 
            return self.valDS.shuffle(buffer_size=50, reshuffle_each_iteration=True)
        else:
            return self.valDS

    def get_testing_data(self):
        return self.testDS

    def get_shape(self):
        for element in self.trainDS.as_numpy_iterator():
            temp = element[0].shape
            break
        return temp

    def _loadExample(self, name):
        data = np.load(self.config.data_loader.data_dir + name)
        _x, _y = data['x'], data['y']
        data.close
        return _x, _y
    
    def _loadSelection(self, _x, _y):        
        if self.config.data_loader.acc and not self.config.data_loader.str:
            _x, _y = _x[...,:self.config.data_loader.acc_num], _y[...,:self.config.data_loader.acc_num]
        elif self.config.data_loader.str and not self.config.data_loader.acc:
            _x, _y = _x[...,self.config.data_loader.acc_num:], _y[...,self.config.data_loader.acc_num:]
        elif not self.config.data_loader.str and not self.config.data_loader.acc:
            return None 
        
        if self.config.data_loader.sl and not self.config.data_loader.wl:
            _x = _x[:,:,:self.config.data_loader.sl_num]
        elif self.config.data_loader.wl and not self.config.data_loader.sl:
            _x = _x[:,:,self.config.data_loader.sl_num:]
        elif not self.config.data_loader.wl and not self.config.data_loader.sl:
            return None
        return _x, _y

    def _loadData(self, name):
        _x, _y = self._loadExample(name)
        _x, _y = self._loadSelection(_x, _y)
        return _x, _y

    def _loadSequential(self, names):
        X, Y = [], []
        for name in tqdm(names):
            _x, _y = self._loadData(name)
            X.extend([__x.T.astype(np.float32) for __x in _x.T])
            Y.extend([__y.astype(np.float32) for __y in _y.T])
        return X, Y

    def _loadParallel(self, names): # Achtung funtioniert nicht als deamon subprocess und benötigt deutlich mehr RAM
        x, y = zip(*Parallel(n_jobs=2)(delayed(self._loadData)(name) for name in tqdm(names)))
        X, Y = [], []
        for _x, _y in zip(x, y):
            X.extend([__x.T.astype(np.float32) for __x in _x.T])
            Y.extend([__y.astype(np.int16) for __y in _y.T])
        return X, Y


    def _list_to_dataset(self, X, Y, buckets):
        """
        Method used to set up the tensorflow dataset
        """
        ds = tf.data.Dataset.from_generator(lambda: zip(X, Y),
                (tf.float32, tf.int16),
                (tf.TensorShape([None,self.POOLING,self.shape[-1]]), 
                 tf.TensorShape([None])))
        ds = ds.bucket_by_sequence_length(
                pad_to_bucket_boundary=True,
                element_length_func=self._element_length_fn,
                bucket_boundaries=buckets,
                bucket_batch_sizes=[self.batch_size,]*(len(buckets)+1)).prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds     
          
    def _element_length_fn(self, x, y):
        return tf.shape(x)[0]
    
    def _getBuckets(self, X):
        lengths = []
        for x in X:
            lengths.append(len(x))
        lengths = np.array(lengths)
        return np.arange(np.floor(lengths.min()/128)*128,
                        np.ceil(lengths.max()/128)*128+128,
                        128).astype(np.int16)+1
