# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import superlet as sl
import wavelet as wl
import model as m

from sklearn.model_selection import train_test_split
import numpy as np
import pywt
from scipy.stats import gaussian_kde
from tqdm import tqdm

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, save_img
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
AUTOTUNE = tf.data.experimental.AUTOTUNE

NAME = 'wltot'
POOLING = 32
# a = gaussian_kde(np.argwhere(y[0])[:,0],bw_method=0.003)
# b = np.array([[a(np.arange(len(y[0])))]]).T
# b[b<b.max()*0.1] = 0
# save_img('kde.jpg',b)

def loadPaths(Xdir = 'png/example', Ydir = 'png/label'):      
    Xpaths = []
    ypaths = []
    for dir in os.listdir(Xdir):
        Xpaths.append(os.path.join(Xdir,dir))
    for dir in os.listdir(Ydir):
        ypaths.append(os.path.join(Ydir,dir))
    return np.array(Xpaths), np.array(ypaths)

def loadData(file):
    data  = np.load(file, allow_pickle=True)
    arrays = data['X']
    names = data['y']
    data.close
    return arrays, names

X, y = loadData(f'{NAME}.npz')
channels = X[0].shape[2]
#%%
# def distribute(y):
#     kde = gaussian_kde(np.argwhere(y)[:,0],bw_method=0.003)
#     result = kde(np.arange(len(y)))
#     result[result<result.max()*0.1] = 0
#     return result/result.max()

# for i in tqdm(range(len(y))):
#     y[i] = distribute(y[i]).astype(np.float16)
# np.savez_compressed('kde.npz',X=X,y=y)
#%%
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, shuffle=True)
X, y = None, None
#%%
# class DataLoaderRAM():
#     def __init__(self, Xpaths, ypaths, batch_size=16, shuffle=False):                
#         self.Xpaths = Xpaths
#         self.ypaths = ypaths
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.idx = np.arange(len(self.Xpaths))   #index array for shuffling data
        
#     def __len__(self):  #number of batches in an epoch        
#         return int(np.ceil(len(self.Xpaths)/float(self.batch_size)))
    
#     def _pad_images(self,img,shape):    #pad images to match largest image in batch        
#         return np.pad(img, (((shape-img.shape[0])//2, ((shape-img.shape[0])//2) + ((shape-img.shape[0])%2)),
#                           (0,0),(0,0)), mode='constant', constant_values=0.)    
#     def _pad_label(self,img,shape):    #pad images to match largest image in batch        
#         return np.pad(img, ((shape-img.shape[0])//2, ((shape-img.shape[0])//2) + ((shape-img.shape[0])%2)), mode='constant', constant_values=0.)

#     def __call__(self):    
#         if self.shuffle:    #shuffle index
#             np.random.shuffle(self.idx)        
        
#         for batch in range(len(self)):  #generate batches

#             batch_images = self.Xpaths[self.idx[batch*self.batch_size:(batch+1)*self.batch_size]]
#             batch_labels = self.ypaths[self.idx[batch*self.batch_size:(batch+1)*self.batch_size]]

#             max_resolution = (np.ceil(np.array(max([img.shape[0] for img in batch_images])) / POOLING) * POOLING).astype(int)

#             batch_images = np.array([self._pad_images(image,max_resolution) for image in batch_images])
#             batch_labels = np.array([self._pad_label(labels,max_resolution) for labels in batch_labels])

#             yield batch_images, batch_labels

# BATCH = 6

# train_generator = DataLoaderRAM(xtrain, ytrain, batch_size=BATCH, shuffle=True)
# test_generator = DataLoaderRAM(xtest, ytest, batch_size=BATCH)

# #convert generators into tf.data.Dataset objects for optimization with keras model fit method

# train_dataset = tf.data.Dataset.from_generator(train_generator,
#      (tf.float32, tf.int32),
#     (tf.TensorShape([None,None,16,channels]), tf.TensorShape([None,None]))).prefetch(buffer_size=AUTOTUNE).repeat()

# test_dataset = tf.data.Dataset.from_generator(test_generator,
#      (tf.float32, tf.int32),
#     (tf.TensorShape([None,None,16,channels]), tf.TensorShape([None,None]))).prefetch(buffer_size=AUTOTUNE).repeat()
# #%%
# BATCH = 6
# def getBuckets(X):
#     lengths = []
#     for x in tqdm(X):
#         lengths.append(len(x))
#     lengths = np.array(lengths)
#     return np.arange(np.ceil(lengths.min()/128)*128,
#                      np.floor(lengths.max()/128)*128+128,
#                      128).astype(int)+1
# buckets = getBuckets(X)

# def _element_length_fn(x, y):
#     return tf.shape(x)[0]
# trainDS = tf.data.Dataset.from_generator(lambda: zip(xtrain,ytrain),
#      (tf.float32, tf.int32),
#     (tf.TensorShape([None,16,channels]), tf.TensorShape([None]))).prefetch(buffer_size=AUTOTUNE).repeat()
# trainDS = trainDS.bucket_by_sequence_length(
#     pad_to_bucket_boundary=True,
#     element_length_func=_element_length_fn,
#     bucket_boundaries=buckets,
#     bucket_batch_sizes=[BATCH,]*(len(buckets)+1))

# testDS = tf.data.Dataset.from_generator(lambda: zip(xtest,ytest),
#      (tf.float32, tf.int32),
#     (tf.TensorShape([None,16,channels]), tf.TensorShape([None]))).prefetch(buffer_size=AUTOTUNE).repeat()
# testDS = testDS.bucket_by_sequence_length(
#     pad_to_bucket_boundary=True,
#     element_length_func=_element_length_fn,
#     bucket_boundaries=buckets,
#     bucket_batch_sizes=[BATCH,]*(len(buckets)+1))
#%%
model = tf.keras.models.load_model(f'model/{NAME}')

#%%

import math
idxs = np.random.choice(len(xtrain), 100, replace=False)
xtrain = xtrain[idxs]
ytrain = ytrain[idxs]
max_len = math.ceil(max([i.shape[0]for i in xtrain]) / 64)*64
xtrain = np.array([np.pad(i, ((0,max_len-i.shape[0]),(0,0),(0,0)), mode='constant', constant_values=0.) for i in xtrain])
ytrain = np.array([np.pad(i, ((0,max_len-i.shape[0])), mode='constant', constant_values=0.) for i in ytrain])
#%%
import matplotlib.pyplot as plt

def getBaseline(img, i):
    image = tf.convert_to_tensor(img, tf.float32)
    # baseline = np.zeros_like(xtrain[0])
    baseline = img # np.random.uniform(0,1,xtrain[0].shape)
    # baseline[:,:,i] = np.random.uniform(0,1,xtrain[0].shape)[:,:,i]
    baseline[:,:,i] = 0.
    baseline = tf.convert_to_tensor(baseline, tf.float32)
    return image, baseline


def interpolate_images(baseline,
                    image,
                    alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = logits[:, target_class_idx] #tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)
    
def prob_grad_over_alpha(idx, img, i = 0):
    image, baseline = getBaseline(img, i)
    
    m_steps=30
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

    interpolated_images = interpolate_images(
        baseline=baseline,
        image=image,
        alphas=alphas)

    path_gradients = compute_gradients(
        images=interpolated_images,
        target_class_idx=idx)

    pred = model(interpolated_images)
    pred_proba = pred[:, idx] #tf.nn.softmax(pred, axis=-1)[:, idx]

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(alphas, pred_proba)
    ax1.set_title('Target class predicted probability over alpha')
    ax1.set_ylabel('model p(target class)')
    ax1.set_xlabel('alpha')
    ax1.set_ylim([0, 1])

    ax2 = plt.subplot(1, 2, 2)
    # Average across interpolation steps
    average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
    # Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
    average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
    ax2.plot(alphas, average_grads_norm)
    ax2.set_title('Average pixel gradients (normalized) over alpha')
    ax2.set_ylabel('Average pixel gradients')
    ax2.set_xlabel('alpha')
    ax2.set_ylim([0, 1])
#%%
j = 1
for i in range(xtrain[j].shape[-1]):
    idx = np.argwhere(ytrain[j] > 0)[0,0]
    prob_grad_over_alpha(idx, xtrain[j][idx-128:idx+128], i)

#%%
def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=8):
    # Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Collect gradients.    
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
        gradient_batches.append(gradient_batch)

    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    return gradient_batch

j = 1
for i in range(xtrain[j].shape[-1]):
    idx = np.argwhere(ytrain[j] > 0)[0,0]
    # prob_grad_over_alpha(idx, xtrain[j][idx-128:idx+128], i)
    
    image, baseline = getBaseline(xtrain[j], i)

    ig_attributions = integrated_gradients(baseline=baseline*0.0,
                                       image=image,
                                       target_class_idx=idx,
                                       m_steps=240)
    print(ig_attributions.numpy().max())
#%%
def plot_img_attributions(baseline,
                            image,
                            target_class_idx,
                            m_steps=50,
                            cmap=None,
                            overlay_alpha=0.4):

    attributions = integrated_gradients(baseline=baseline*0.,
                                        image=image,
                                        target_class_idx=target_class_idx,
                                        m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(15, 5))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline[:,:,5].numpy().T, aspect="auto")
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image[:,:,5].numpy().T, aspect="auto")
    axs[0, 1].axis('off')
    print(attribution_mask.shape)
    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask.numpy().T, cmap=cmap, aspect="auto")
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask.numpy().T, cmap=cmap, aspect="auto")
    axs[1, 1].imshow(image[:,:,5].numpy().T, alpha=overlay_alpha, aspect="auto")
    axs[1, 1].axis('off')

    fig.tight_layout()
    plt.savefig(f'ig.png',bbox_inches='tight',dpi=2000)

plot_img_attributions(image=image,
                          baseline=baseline,
                          target_class_idx=idx,
                          m_steps=240,
                          cmap=plt.cm.inferno,
                          overlay_alpha=0.4)
#%%
# #%%
# import shap
# import math
# idxs = np.random.choice(len(xtrain), 100, replace=False)
# xtrain = xtrain[idxs]
# ytrain = ytrain[idxs]
# max_len = math.ceil(max([i.shape[0]for i in xtrain]) / 64)*64
# xtrain = np.array([np.pad(i, ((0,max_len-i.shape[0]),(0,0),(0,0)), mode='constant', constant_values=0.) for i in xtrain])
# ytrain = np.array([np.pad(i, ((0,max_len-i.shape[0])), mode='constant', constant_values=0.) for i in ytrain])
# #%%
# e = shap.GradientExplainer(model, xtrain[:6]) # explain predictions of the model on three images
# #%%
# # ...or pass tensors directly
# # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
# shap_values = e.shap_values(xtest[0])


# # since we have two inputs we pass a list of inputs to the explainer
# explainer = shap.Explainer(model, [xtrain, xtrain])

# # we explain the model's predictions on the first three samples of the test set
# shap_values = explainer.shap_values([xtest[:3], xtest[:3]])

#%%

# model = tf.keras.models.load_model('model/wltot')
# model.save_weights('weights/')
# model = m.getDeepFlatNet(label_as_img=False,channels=channels)
# model.load_weights('weights/')
# model.summary()
# model = tf.keras.models.load_model('model/')
# model.save_weights('weights/')
#%%
# steps_per_epoch = len(X) // (BATCH * 10)
# validation_steps = len(X) // (BATCH * 10)
# callbacks = [EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto', baseline=None, restore_best_weights=True),
#              ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=15, verbose=0, mode='auto', cooldown=0, min_lr=0.000001),
#              ModelCheckpoint(f'model/{NAME}/',save_best_only=True)]

# model.fit(trainDS, 
#           steps_per_epoch=steps_per_epoch,
#           validation_data=testDS,
#           verbose=1, 
#           workers=12, 
#           epochs=30,
#           validation_steps=validation_steps, 
#           callbacks=callbacks,
#           use_multiprocessing=True,
#           max_queue_size=2000)
# # model.save('model/')

# #%%
# import matplotlib.pyplot as plt

# model = tf.keras.models.load_model(f'model/{NAME}')

# for i in range(10):
#     # test_X = img_to_array( load_img(pathx, color_mode='rgb', interpolation='nearest') ).transpose(1,0,2) / 255
#     test_X = xtest[i]
#     shape = (np.ceil(np.array(test_X.shape[0]) / POOLING) * POOLING).astype(int)
#     test_X = np.pad(test_X, (((shape-test_X.shape[0])//2, ((shape-test_X.shape[0])//2) + ((shape-test_X.shape[0])%2)),
#                           (0,0),(0,0)), mode='constant', constant_values=0.)    
#     test_y = ytest[i]
#     test_y = np.pad(test_y, ((shape-test_y.shape[0])//2, ((shape-test_y.shape[0])//2) + ((shape-test_y.shape[0])%2)), mode='constant', constant_values=0.)
#     test = model.predict(np.array([test_X]))

#     x = np.arange(len(test.flatten()))
#     fig, ax = plt.subplots(figsize=(15, 5))
#     ax.plot(x, test.flatten(),zorder=2, lw=0.5)
#     # ax.plot([x[test_y>0],x[test_y>0]], [0,1], color='r',zorder=0, lw=0.5)
#     ax.plot(x, test_y.flatten(),zorder=0, color='r',lw=0.5)
#     ax.fill_between(x, test_y.flatten(), alpha=0.3, color='r')
#     plt.savefig(f'{NAME}{i}.png',bbox_inches='tight',dpi=2000)