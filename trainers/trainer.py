from utils.logger import CometLogger
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import math

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
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, self.timestamp),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=self.config.callbacks.checkpoint_verbose
            )
        )
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=self.config.callbacks.patience,
                verbose=0,
            )
        )

    def train(self):
        self.logger = CometLogger(self.run, self.path_to_config)
        
        self.model.fit(self.train_data,
            steps_per_epoch=self.config.trainer.steps_per_epoch,
            validation_steps=math.ceil(self.config.trainer.steps_per_epoch * self.config.trainer.validation_percentage), 
            workers=tf.data.AUTOTUNE, 
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=self.val_data,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            max_queue_size=2000
            )
        
        self.model.evaluate(self.testDS, verbose=2)
    
    def test(self):
        prediction = []
        label = []
        self.model.save_weights(os.path.join(self.config.callbacks.checkpoint_dir, self.timestamp, 'manual_weights/'), overwrite=False)
        for element in self.testDS.as_numpy_iterator():
            prediction.append(self.model.predict(element[0]))
            label.append(element[1])
        np.savez(os.path.join(self.config.callbacks.checkpoint_dir, self.timestamp, 'test.npz'), x=prediction, y=label, allow_pickle=True)
        
    def end(self):
        self.logger.end()
