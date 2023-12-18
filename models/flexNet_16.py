# from base.base_model import BaseModel
import models.metric_axles as ma
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Reshape, Input, BatchNormalization, MaxPool2D, Concatenate, Add, Conv2DTranspose, GaussianNoise, GroupNormalization
from tensorflow.keras.models import Model

# metrics = [ma.acc, ma.mean_fs, ma.std_fs, ma.percent_neg, ma.percent_null, ma.percent_pos, ma.avg_neg, ma.avg_pos, ma.recall, ma.precision]
metrics = [ma.f1]#, ma.recall, ma.precision, ma.mean_fs, ma.std_fs]#, ma.f1_7, ma.recall7, ma.precision7, ma.f1_3, ma.recall3, ma.precision3]

class Model():
    def __init__(self, config, shape):
        self.config = config
        self.model = None
        self.build_model(shape)
 
    def _conv2d_block(self, input_tensor, n_filters, i, kernel_size = 3):
        if self.config.model.batchnorm:
            x = GroupNormalization(int(n_filters/16))(input_tensor)  
            x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(x)
        else:
            x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(input_tensor)
        if self.config.model.batchnorm:
            x = GroupNormalization(int(n_filters/16))(x)
        x = Conv2D(filters = n_filters, kernel_size = kernel_size, padding = 'same', activation = 'relu')(x)
        if self.config.model.batchnorm:
            x = GroupNormalization(int(n_filters/16))(x)  
        x = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(x)
        
        if self.config.model.batchnorm:
            input_tensor = GroupNormalization(int(n_filters/16))(input_tensor)  
        if i == 0:
            skip = Conv2D(filters = n_filters, kernel_size = 1, padding = 'same', activation = 'relu')(input_tensor)
        else:
            skip = input_tensor

        if self.config.model.concat:
            return Concatenate()([x, skip])
        else:
            return Add()([x, skip])

    def build_model(self, shape): 
        pooling_steps = min(self.config.model.pooling_steps, 6)
        n_filters = self.config.model.n_filters
        kernel_mid = self.config.model.kernel_mid
        pool = self.config.model.pooling_size
        input_data = Input(shape=[None, shape[-2], shape[-1]]) 
        print('Required amount of scales: ', 2**pooling_steps, shape[-2])    
        
        bn = GroupNormalization(1)(input_data)
        kernel = 1 if shape[-2] == 1 else self.config.model.kernel_start
        c1 = Conv2D(n_filters, kernel_size = (self.config.model.kernel_start,kernel), padding="same", activation='relu')(bn)
        
        for i in range(self.config.model.encode_layer):
            c1 = self._conv2d_block(c1, n_filters, i, kernel_size=(kernel_mid,kernel))
        n_filters *= 2
        c2 = MaxPool2D()(c1) if shape[-2] > 2**0 else MaxPool2D(pool_size=(pool,1))(c1)

        kernel = 1 if kernel == 1 else kernel//2
        if pooling_steps >= 2:
            for i in range(self.config.model.encode_layer):
                c2 = self._conv2d_block(c2, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters *= 2
            c3 = MaxPool2D()(c2) if shape[-2] > 2**1 else MaxPool2D(pool_size=(pool,1))(c2)
            c7 = c3 if pooling_steps == 2 else None

        kernel = 1 if kernel == 1 else kernel//2
        if pooling_steps >= 3:
            for i in range(self.config.model.encode_layer):
                c3 = self._conv2d_block(c3, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters *= 2
            c4 = MaxPool2D()(c3) if shape[-2] > 2**2 else MaxPool2D(pool_size=(pool,1))(c3)
            c7 = c4 if pooling_steps == 3 else None

        kernel = 1 if kernel == 1 else kernel//2
        if pooling_steps >= 4:
            for i in range(self.config.model.encode_layer):
                c4 = self._conv2d_block(c4, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters *= 2
            c5 = MaxPool2D()(c4) if shape[-2] > 2**3 else MaxPool2D(pool_size=(pool,1))(c4)
            c7 = c5 if pooling_steps == 4 else None
        
        kernel = 1 if kernel == 1 else kernel//2
        if pooling_steps >= 5:
            for i in range(self.config.model.encode_layer):
                c5 = self._conv2d_block(c5, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters *= 2
            c6 = MaxPool2D()(c5) if shape[-2] > 2**4 else MaxPool2D(pool_size=(pool,1))(c5)
            c7 = c6 if pooling_steps == 5 else None
        
        kernel = 1 if kernel == 1 else kernel//2
        if pooling_steps == 6:
            for i in range(self.config.model.encode_layer):
                c6 = self._conv2d_block(c6, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters *= 2
            c7 = MaxPool2D()(c6) if shape[-2] > 2**5 else MaxPool2D(pool_size=(pool,1))(c6)
        
        kernel = 1 if kernel == 1 else kernel//2
        for i in range(self.config.model.bottleneck_layer):
            c7 = self._conv2d_block(c7, n_filters, i, kernel_size=(kernel_mid,kernel))
        n_filters /= 2
        
        if pooling_steps >= 6:
            c7 = Conv2DTranspose(n_filters, kernel_size = (kernel_mid,1), strides=(pool,1), padding = 'same')(c7)
            c6 = Reshape((-1,1,c6.shape[-1]*c6.shape[-2]))(c6)
            c6 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c6)
            c7 = Concatenate()([c7, c6])
            for i in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters /= 2
        
        if pooling_steps >= 5:
            c7 = Conv2DTranspose(n_filters, kernel_size = (kernel_mid,1), strides=(pool,1), padding = 'same')(c7)
            c5 = Reshape((-1,1,c5.shape[-1]*c5.shape[-2]))(c5)
            c5 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c5)
            c7 = Concatenate()([c7, c5])
            for i in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters /= 2
        
        if pooling_steps >= 4:
            c7 = Conv2DTranspose(n_filters, kernel_size = (kernel_mid,1), strides=(pool,1), padding = 'same')(c7)
            c4 = Reshape((-1,1,c4.shape[-1]*c4.shape[-2]))(c4)
            c4 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c4)
            c7 = Concatenate()([c7, c4])
            for i in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters /= 2
        
        if pooling_steps >= 3:
            c7 = Conv2DTranspose(n_filters, kernel_size = (kernel_mid,1), strides=(pool,1), padding = 'same')(c7)
            c3 = Reshape((-1,1,c3.shape[-1]*c3.shape[-2]))(c3)
            c3 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c3)
            c7 = Concatenate()([c7, c3])
            for i in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters /= 2

        if pooling_steps >= 2:
            c7 = Conv2DTranspose(n_filters, kernel_size = (kernel_mid,1), strides=(pool,1), padding = 'same')(c7)
            c2 = Reshape((-1,1,c2.shape[-1]*c2.shape[-2]))(c2)
            c2 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c2)
            c7 = Concatenate()([c7, c2])
            for i in range(self.config.model.decode_layer):
                c7 = self._conv2d_block(c7, n_filters, i, kernel_size=(kernel_mid,kernel))
            n_filters /= 2
        
        c7 = Conv2DTranspose(n_filters, kernel_size = (kernel_mid,1), strides=(pool,1), padding = 'same')(c7)
        c1 = Reshape((-1,1,c1.shape[-1]*c1.shape[-2]))(c1)
        c1 = Conv2D(n_filters, kernel_size = 1, padding="same", activation='relu')(c1)
        c7 = Concatenate()([c7, c1])
        for i in range(self.config.model.decode_layer):
            c7 = self._conv2d_block(c7, n_filters, i, kernel_size=(kernel_mid,kernel))

        c7 = BatchNormalization()(c7)
        c8 = Conv2D(n_filters, kernel_size = (self.config.model.kernel_end,1), padding="same", activation='relu')(c7)
        outputs = Conv2D(1, kernel_size = (self.config.model.kernel_end,1), padding="same", activation='sigmoid')(c8)
        if not self.config.model.label_as_img:
            outputs = Flatten()(outputs)

        self.model = tf.keras.models.Model(inputs=[input_data], outputs=[outputs])
        self.model.compile(
            loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=self.config.model.gamma), 
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.model.lr), 
            metrics = metrics)
        
    def printModel(self):
        print(self.model.summary())
        
    def plotModel(self):
        _ = tf.keras.utils.plot_model(self.model, to_file = 'flatnet_shape.png', show_layer_names=False, show_shapes=True, rankdir='TB')
        
