"""
    File name: train_script.py
    Author: Corinne Dumais
    Last modified: May 2022
"""

import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import tensorflow as tf

from unet import UNetRiverBands, Dice, DiceLoss, categorical_crossentropy_mask, combined
from Utils.dataset import RiverBandsDataset
from unet_attention import UNetRiverBandsAttention

print(tf.config.experimental.list_physical_devices('GPU'))

input_train_dir = 'DataBR/Patches/Train/img_patches'
target_train_dir = 'DataBR/Patches/Train/mask_patches'
input_eval_dir = 'DataBR/Patches/Eval/img_patches'
target_eval_dir = 'DataBR/Patches/Eval/mask_patches'

train_img_paths = sorted([os.path.join(input_train_dir, file) for file in os.listdir(input_train_dir) if file.endswith(".npy")])
train_target_paths = sorted([os.path.join(target_train_dir, file) for file in os.listdir(target_train_dir) if file.endswith(".npy")])

eval_img_paths = sorted([os.path.join(input_eval_dir, file) for file in os.listdir(input_eval_dir) if file.endswith(".npy")])
eval_target_paths = sorted([os.path.join(target_eval_dir, file) for file in os.listdir(target_eval_dir) if file.endswith(".npy")])

assert len(train_img_paths) == len(train_target_paths), 'Number of train targets and images must be equal'
assert len(eval_img_paths) == len(eval_target_paths), 'Number of eval targets and images must be equal'

batch_size = 256
img_size = (32, 32)
target_size = (32, 32)
epochs = 100
channels = 32
reg = 0.0000
bn = True
drop = 0.0

model_name = f'may_ATT_lr4_{channels}c_bs{batch_size}_{epochs}ep_BN{bn}_reg{reg}_drop{drop}.h5'

random.Random(1339).shuffle(train_img_paths)
random.Random(1339).shuffle(train_target_paths)

random.Random(144).shuffle(eval_img_paths)
random.Random(144).shuffle(eval_target_paths)

# Verify shapes
print("Number of samples for training:", len(train_img_paths))
print("Number of samples for validation:", len(eval_target_paths))

# Compute class weights
# y_train = np.array([np.argmax(np.load(i), axis=-1) for i in train_target_paths])
# y_train = np.ndarray.flatten(y_train)
# weights = []
# for i in range(7):
#     number_class = np.count_nonzero(y_train == i)
#     weights.append(number_class)
#
# sum_W = np.sum(weights)
# for i in range(7):
#     weights[i] = weights[i]/sum_W
# print(weights)

# Instantiate data Sequences for each batch
train_gen = RiverBandsDataset(batch_size, img_size, target_size, train_img_paths, train_target_paths)
val_gen = RiverBandsDataset(1, img_size, target_size, eval_img_paths, eval_target_paths)

# Define callbacks to use during training
callbacks = [keras.callbacks.ModelCheckpoint(os.path.join('Trained_models', model_name), save_best_only=True),
             keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}')]

# model = UNetRiverBands(input_size=(32, 32, 3), output_classes=7, kernel_size=(3, 3), channels=channels,
#                        reg=reg, batchnorm=bn, drop=drop).build()
model = UNetRiverBandsAttention(input_shape=(32, 32, 3), num_classes=7, channels=channels, l2_reg=reg)
# model.fit(train_gen, batch_size=batch_size, epochs=epochs, validation_data=val_gen, shuffle=True, callbacks=callbacks)

model = keras.models.load_model(os.path.join('Trained_models', model_name), custom_objects={'Dice': Dice,
                                                                                            'DiceLoss': DiceLoss,
                                                                                            'loss': categorical_crossentropy_mask,
                                                                                            'combined': combined})
preds = model.predict(val_gen, batch_size=1)

y_test = np.array([np.argmax(np.load(i), axis=-1) for i in eval_target_paths])
y_preds = np.array([np.argmax(pred, axis=-1) for pred in preds])
matrix = tf.math.confusion_matrix(np.ndarray.flatten(y_test), np.ndarray.flatten(y_preds))

array = np.array(matrix)/np.sum(np.array(matrix))*100
import pandas as pd
df = pd.DataFrame(array).T
df.to_excel(excel_writer="cm.xlsx")

cmaplist = ['#000000', '#003300', '#994c00', '#A16BA1', '#ffcc99', '#DBC820', '#331900']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)
bounds = np.linspace(0, 7, 8)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

paths = eval_target_paths
for i in range(0, len(paths), 5):
    im1 = np.argmax(preds[i], axis=-1)
    im2 = np.argmax(preds[i+1], axis=-1)
    im3 = np.argmax(preds[i+2], axis=-1)
    im4 = np.argmax(preds[i+3], axis=-1)
    im5 = np.argmax(preds[i+4], axis=-1)

    fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(4, 8))
    ax = axes.ravel()

    ax[0].imshow(np.argmax(np.load(paths[i]), axis=-1), cmap=cmap, norm=norm)
    ax[1].imshow(im1, cmap=cmap, norm=norm)
    ax[2].imshow(np.argmax(np.load(paths[i+1]), axis=-1), cmap=cmap, norm=norm)
    ax[3].imshow(im2, cmap=cmap, norm=norm)
    ax[4].imshow(np.argmax(np.load(paths[i+2]), axis=-1), cmap=cmap, norm=norm)
    ax[5].imshow(im3, cmap=cmap, norm=norm)
    ax[6].imshow(np.argmax(np.load(paths[i+3]), axis=-1), cmap=cmap, norm=norm)
    ax[7].imshow(im4, cmap=cmap, norm=norm)
    ax[8].imshow(np.argmax(np.load(paths[i+4]), axis=-1), cmap=cmap, norm=norm)
    ax[9].imshow(im5, cmap=cmap, norm=norm)

    for j in range(10):
        ax[j].axis('off')

    ax[0].set_title('Vérité terrain')
    ax[1].set_title(f'Prédiction {i}-{i+4}')

    plt.tight_layout()
    plt.show()
    # fig.savefig(f'preds{i}.png', dpi=600)
