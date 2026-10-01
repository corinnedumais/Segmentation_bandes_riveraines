"""
    File name: inference_script.py
    Author: Corinne Dumais
    Last modified: May 2022
"""

from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import numpy as np

from unet import Dice, categorical_crossentropy_mask, DiceLoss, combined


def full_prediction(trained_model, input_path, output_path, ground_truth, patch_size):
    # Read tif file as a numpy array
    with rasterio.open(input_path, 'r') as ds:
        image = np.moveaxis(ds.read(), 0, -1)  # [1:-1, 1:-1, :]
        transform = ds.transform
        crs = ds.crs
        count = ds.count
        w, h = ds.width, ds.height

    with rasterio.open(ground_truth, 'r') as ds:
        gt = np.squeeze(np.moveaxis(ds.read(), 0, -1))

    # Predict each patch and reassemble full image
    segm_img = np.zeros(image.shape[:2])  # Array with zeros to be filled with segmented values
    X = ((image.shape[0] - patch_size) // patch_size) * patch_size
    Y = ((image.shape[1] - patch_size) // patch_size) * patch_size
    for i in range(0, image.shape[0], patch_size):
        for j in range(0, image.shape[1], patch_size):
            if i + patch_size > image.shape[0]:
                diff = i + patch_size - image.shape[0]
                i -= diff
            if j + patch_size > image.shape[1]:
                diff = j + patch_size - image.shape[1]
                j -= diff
            single_patch = image[i:i + patch_size, j:j + patch_size]

            if np.all(single_patch != -1):
                # Normalize with respect to the entire dataset
                b0 = single_patch[:, :, 0] / 31.13
                b1 = (single_patch[:, :, 1] - np.amin(image)) / (np.amax(image) - np.amin(image))
                b2 = single_patch[:, :, 2] / 24.

                single_patch = np.stack([b2, b1, b0], axis=-1)

                single_patch_prediction = (np.argmax(
                    trained_model.predict(np.expand_dims(single_patch, axis=0), verbose=0), axis=-1)).astype(np.uint8)[
                                          0, :, :]
                segm_img[i:i + patch_size, j:j + patch_size] = single_patch_prediction

    segm_img[gt == 255] = 0
    segm_img[gt == 10] = 0

    with rasterio.open(output_path, "w", driver='GTiff',
                       count=count, dtype=image.dtype,
                       width=w, height=h, transform=transform,
                       crs=crs) as raschip:
        raschip.write(segm_img, 1)

    return segm_img


model_file = 'Trained_models/may_ATT_lr4_32c_bs256_100ep_BNTrue_reg0.0_drop0.0.h5'
farm = "TicouapeFerme1Composite.tif"
path = 'DataBR/Composites/Eval/'
i_path = path + farm
o_path = "Results/" + farm
gt_path = 'DataBR/VeriteTerrain/' + farm[:-13] + 'Verite.tif'

model = keras.models.load_model(model_file, custom_objects={'Dice': Dice,
                                                            'DiceLoss': DiceLoss,
                                                            'loss': categorical_crossentropy_mask,
                                                            'combined': combined})

cmaplist = ['#ffffff', '#003300', '#994c00', '#A16BA1', '#ffcc99', '#DBC820', '#331900']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)
bounds = np.linspace(0, 7, 8)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

pred = full_prediction(model, i_path, o_path, gt_path, patch_size=32)
plt.figure(figsize=(10, 10))
plt.imshow(pred, cmap=cmap, norm=norm)
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{farm[:-4]}.png", dpi=500)
plt.show()
