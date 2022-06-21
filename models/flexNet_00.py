# from base.base_model import BaseModel
import models.metric_axles as ma
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Reshape, Input, BatchNormalization, MaxPool2D, Concatenate, Add, Conv2DTranspose
from tensorflow.keras.models import Model

# metrics = [ma.acc, ma.mean_fs, ma.std_fs, ma.percent_neg, ma.percent_null, ma.percent_pos, ma.avg_neg, ma.avg_pos, ma.recall, ma.precision]
metrics = [ma.f1, ma.recall, ma.precision, ma.f1_7, ma.recall7, ma.precision7, ma.f1_3, ma.recall3, ma.precision3, ma.mean_fs, ma.std_fs]

class Model():
    def __init__(self, config, shape):
        self.config = config
        self.model = None
        self.build_model(shape)
 
    def _conv2d_block(self, input_tensor, n_filters, kernel_size = 3):
        if self.config.model.batchnorm:
            x = BatchNormalization()(input_tensor)  
            x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(x)
        else:
            x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(input_tensor)
        if self.config.model.batchnorm:
            x = BatchNormalization()(x)
        x = Conv2D(filters = n_filters, kernel_size = kernel_size, padding = 'same', activation = 'relu')(x)
        if self.config.model.batchnorm:
            x = BatchNormalization()(x)  
        x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(x)
        
        if self.config.model.batchnorm:
            input_tensor = BatchNormalization()(input_tensor)  
        skip = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(input_tensor)

        if self.config.model.concat:
            return Concatenate()([x, skip])
        else:
            return Add()([x, skip])

    def build_model(self, shape): 
        pooling_steps = min(self.config.model.pooling_steps, 6)
        n_filters = self.config.model.n_filters
        input_data = Input(shape=[None, 2**pooling_steps, shape[-1]]) 
        print('Required amount of scales: ', 2**pooling_steps)    
        
        bn = BatchNormalization()(input_data)
        c1 = Conv2D(n_filters, kernel_size = 3, padding="same", activation='relu')(bn)
        
        for _ in range(self.config.model.encode_layer):
            c1 = self._conv2d_block(c1, n_filters)
        n_filters *= 2
        c2 = MaxPool2D()(c1)

        if pooling_steps >= 2:
            for _ in range(self.config.model.encode_layer):
                c2 = self._conv2d_block(c2, n_filters)
            n_filters *= 2
            c3 = MaxPool2D()(c2)
            c7 = c3 if pooling_steps == 2 else None

        if pooling_steps >= 3:
            for _ in range(self.config.model.encode_layer):
                c3 = self._conv2d_block(c3, n_filters)
            n_filters *= 2
            c4 = MaxPool2D()(c3)
            c7 = c4 if pooling_steps == 3 else None

        if pooling_steps >= 4:
            for _ in range(self.config.model.encode_layer):
                c4 = self._conv2d_block(c4, n_filters)
            n_filters *= 2
            c5 = MaxPool2D()(c4)
            c7 = c5 if pooling_steps == 4 else None
        
        if pooling_steps >= 5:
            for _ in range(self.config.model.encode_layer):
                c5 = self._conv2d_block(c5, n_filters)
            n_filters *= 2
            c6 = MaxPool2D()(c5)
            c7 = c6 if pooling_steps == 5 else None
        
        if pooling_steps == 6:
            for _ in range(self.config.model.encode_layer):
                c6 = self._conv2d_block(c6, n_filters)
            n_filters *= 2
            c7 = MaxPool2D()(c6)
        
        for _ in range(self.config.model.bottleneck_layer):
            c7 = self._conv2d_block(c7, n_filters)
        n_filters /= 2
        
        if pooling_steps >= 6:
            c7 = Conv2DTranspose(n_filters, kernel_size = (3,1), strides = (2,1), padding = 'same')(c7)
            c6 = Reshape((-1,1,c6.shape[-1]*c6.shape[-2]))(c6)
            c6 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c6)
            c7 = Concatenate()([c7, c6])
            for _ in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters)
            n_filters /= 2
        
        if pooling_steps >= 5:
            c7 = Conv2DTranspose(n_filters, kernel_size = (3,1), strides = (2,1), padding = 'same')(c7)
            c5 = Reshape((-1,1,c5.shape[-1]*c5.shape[-2]))(c5)
            c5 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c5)
            c7 = Concatenate()([c7, c5])
            for _ in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters)
            n_filters /= 2
        
        if pooling_steps >= 4:
            c7 = Conv2DTranspose(n_filters, kernel_size = (3,1), strides = (2,1), padding = 'same')(c7)
            c4 = Reshape((-1,1,c4.shape[-1]*c4.shape[-2]))(c4)
            c4 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c4)
            c7 = Concatenate()([c7, c4])
            for _ in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters)
            n_filters /= 2
        
        if pooling_steps >= 3:
            c7 = Conv2DTranspose(n_filters, kernel_size = (3,1), strides = (2,1), padding = 'same')(c7)
            c3 = Reshape((-1,1,c3.shape[-1]*c3.shape[-2]))(c3)
            c3 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c3)
            c7 = Concatenate()([c7, c3])
            for _ in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters)
            n_filters /= 2

        if pooling_steps >= 2:
            c7 = Conv2DTranspose(n_filters, kernel_size = (3,1), strides = (2,1), padding = 'same')(c7)
            c2 = Reshape((-1,1,c2.shape[-1]*c2.shape[-2]))(c2)
            c2 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c2)
            c7 = Concatenate()([c7, c2])
            for _ in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters)
            n_filters /= 2
        
        c7 = Conv2DTranspose(n_filters, kernel_size = (3,1), strides = (2,1), padding = 'same')(c7)
        c1 = Reshape((-1,1,c1.shape[-1]*c1.shape[-2]))(c1)
        c1 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c1)
        c7 = Concatenate()([c7, c1])
        for _ in range(self.config.model.decode_layer):
            c7 = self._conv2d_block(c7, n_filters)

        c7 = BatchNormalization()(c7)
        c8 = Conv2D(n_filters, kernel_size = 3, padding="same", activation='relu')(c7)
        outputs = Conv2D(1, kernel_size = 3, padding="same", activation='sigmoid')(c8)
        if not self.config.model.label_as_img:
            outputs = Flatten()(outputs)

        self.model = tf.keras.models.Model(inputs=[input_data], outputs=[outputs])
        self.model.compile(
            loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=self.config.model.gamma), 
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.model.lr), 
            metrics = metrics,
            run_eagerly=True)
        
    def printModel(self):
        print(self.model.summary())
        
    def plotModel(self):
        _ = tf.keras.utils.plot_model(self.model, to_file = 'flatnet_shape.png', show_layer_names=False, show_shapes=True, rankdir='TB')
        
