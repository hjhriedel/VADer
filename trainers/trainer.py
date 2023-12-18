from utils.logger import CometLogger
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import math
from tqdm import tqdm
import pickle

import numpy as np


class ModelTrainer():
    def __init__(self, run, path_to_config, model, data_train, data_validate, testDS, config, timestamp):
        self.model = model
        self.train_data = data_train
        self.val_data = data_validate
        self.config = config
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.logger = None
        self.run = run
        self.path_to_config = path_to_config
        self.testDS = testDS
        self.timestamp = timestamp
        self.logger = CometLogger(self.run, self.path_to_config)
        self.init_callbacks()

    def init_callbacks(self):
        # self.callbacks.append(
        #     ModelCheckpoint(
        #         filepath=os.path.join(self.config.callbacks.checkpoint_dir, self.timestamp),
        #         monitor='val_f1',
        #         mode=self.config.callbacks.checkpoint_mode,
        #         save_best_only=self.config.callbacks.checkpoint_save_best_only,
        #         verbose=self.config.callbacks.checkpoint_verbose,
        #         save_weights_only=self.config.callbacks.save_weights_only
        #     )
        # )
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor=self.config.callbacks.checkpoint_monitor,
                factor=0.3,
                patience=int(self.config.callbacks.patience/2),
                min_lr=0.000001,
                verbose=1,
                mode=self.config.callbacks.checkpoint_mode,
                cooldown=int(self.config.callbacks.patience/2),
                min_delta=0.001
            )
        )
        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.callbacks.checkpoint_monitor,
                patience=self.config.callbacks.patience,
                restore_best_weights=True,
                mode=self.config.callbacks.checkpoint_mode,
                min_delta=0.001
            )
        )

    def train(self):        
        self.model.fit(self.train_data,
            workers=4, 
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=self.val_data,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            max_queue_size=50
            )
        
        self.model.save(os.path.join(self.config.callbacks.checkpoint_dir, self.timestamp, self.run))

        evaluation = self.model.evaluate(self.testDS, verbose=1, return_dict=True)
        with self.logger.experiment.test():
            self.logger.experiment.log_metrics(evaluation)

    
    def loadRaw(self, PATH):
        data = np.load(PATH, allow_pickle=True)
        x, y = data['x'], data['y']
        data.close        
        #x = (x + x.min())/(x.max()-x.min())
        #try:
        #    x = x[:,::self.config.data_loader.every_nth]
        #except:
        #    pass
        while x.ndim < 4:
            x = np.expand_dims(x, axis=1)  
        pad_size = int(self.config.model.pooling_size**self.config.model.pooling_steps)
        width = math.ceil(len(y)/pad_size)*pad_size
        x_padded, y_padded = np.zeros((width,x.shape[1],x.shape[2],x.shape[3])), np.zeros((width,x.shape[3]))
        x_padded[:len(y)], y_padded[:len(y)] = x, y
        x_padded = x_padded[:,:,self.config.data_loader.sl_num:]
        return x_padded[:,:,:, :10], y_padded
    
    def test(self):        
        if self.config.data_loader.fold > 99:
            print('stratified')
            nameTest = np.loadtxt(f"data/test_names00.csv", dtype=str).tolist()
        else:
            nameTest = np.loadtxt(f"data/test_names.csv", dtype=str).tolist()
        vPATH = "data/neue Geschwindigkeiten"
        results = []
        for name in tqdm(nameTest):
            vl = np.vstack(np.genfromtxt(f"{vPATH}/v_idxl_acc{name[:-4]}.txt", delimiter=","))
            vr = np.vstack(np.genfromtxt(f"{vPATH}/v_idxr_acc{name[:-4]}.txt", delimiter=","))
            V = np.hstack((vl,vr)) # axle speed in m/s
            measurement, label = self.loadRaw(PATH = self.config.data_loader.data_dir + name)  
            i = 0
            predictions = self.model.predict(np.moveaxis(measurement, -1, 0), workers=4, verbose=0)
            for p, true_values, v in zip(predictions, label.T, V.T):
                results.append([name, i, p, true_values, v])
                i += 1

        with open('final/' + self.timestamp + self.run + 'test.bin', "wb") as output:
            pickle.dump(results, output)

    def end(self):
        self.logger.end()
