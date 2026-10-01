"""
    File name: make_tiles.py
    Author: Corinne Dumais
    Last modified: May 2022
"""

import os
import numpy as np
import rasterio as rs

number_of_farms = 0
number_of_unique_tiles = 0


def read_data(directory, mode='Train', augment=False):
    global number_of_farms
    patch_size = 32
    patch_id = 1
    for file in os.listdir(os.path.join(directory, 'Composites', mode)):
        if file.endswith('.tif'):
            number_of_farms += 1
            with rs.open(os.path.join(directory, 'Composites', mode, file), 'r') as ds:
                im = ds.read()
                transform = ds.transform
                crs = ds.crs
                count = ds.count

            with rs.open(os.path.join(directory, 'VeriteTerrain', f'{file[:-13]}Verite.tif'), 'r') as ds:
                mask = np.squeeze(ds.read())

            for i in (range(0, im.shape[1], patch_size)):
                for j in (range(0, im.shape[2], patch_size)):
                    img_patch = im[:, i:i + patch_size, j:j + patch_size]
                    lbl_patch = mask[i:i + patch_size, j:j + patch_size]
                    x_coord, y_coord = (j * transform[0] + transform[2]), (transform[5] + i * transform[4])
                    transf = [transform[0], 0, x_coord, 0, transform[4], y_coord]

                    saved = save_training_data(np.moveaxis(img_patch, 0, -1), np.moveaxis(lbl_patch, 0, -1), patch_id, mode, augment)

                    if saved:
                        with rs.open("DataBR/Patches/" + mode + "/img_patches/" + str(patch_id) + "_img.tif", "w", driver='GTiff',
                                     count=count, dtype=im.dtype,
                                     width=patch_size, height=patch_size, transform=transf,
                                     crs=crs) as raschip:
                            raschip.write(img_patch)
                        with rs.open("DataBR/Patches/" + mode + "/mask_patches/" + str(patch_id) + "_mask.tif", "w", driver='GTiff', count=1,
                                     dtype=mask.dtype,
                                     width=patch_size, height=patch_size, transform=transform,
                                     crs=crs) as lblchip:
                            lblchip.write(lbl_patch, 1)

                        if augment:
                            patch_id += 6
                        else:
                            patch_id += 1


def save_training_data(im_saved, mask_saved, patch_id, mode, augment):
    global number_of_unique_tiles
    directory = 'DataBR'
    if np.all(mask_saved != 255) and not np.all(mask_saved == 6) \
            and im_saved.shape[0] == 32 and im_saved.shape[1] == 32 and mask_saved.shape[0] == 32 \
            and mask_saved.shape[1] == 32:
        number_of_unique_tiles += 1
        # Change classes 9 and 10, as well as no_data (255) to 0. This class will be excluded from performance metrics.
        mask_saved[mask_saved == 9] = 0
        mask_saved[mask_saved == 10] = 0
        mask_saved[mask_saved == 6] = 5  # combine paturage et culture
        mask_saved[mask_saved == 7] = 6
        mask_saved[mask_saved == 255] = 0

        assert len(np.unique(mask_saved) <= 9)

        im_saved[im_saved < 0] = 0

        # Normalize MHC and TWI with respect to the entire dataset
        # Normalize MNT with respect to each image
        b0 = im_saved[:, :, 0] / 31.13
        b1 = (im_saved[:, :, 1] - np.amin(im_saved[:, :, 1])) / (np.amax(im_saved[:, :, 1]) - np.amin(im_saved[:, :, 1]))
        b2 = im_saved[:, :, 2] / 24.

        im_saved = np.stack([b2, b1, b0], axis=-1)
        mask_categorical = np.zeros(shape=(32, 32, 7))
        for k in range(7):
            mask_categorical[:, :, k] = (mask_saved == k).astype(np.uint8)
        mask_categorical = np.rot90(np.flipud(mask_categorical), 3)

        if mask_categorical.shape != (32, 32, 7):
            print(mask_categorical.shape)
        assert im_saved.shape == (32, 32, 3)

        # Write to memory
        np.save(os.path.join(directory, 'Patches', mode, 'img_patches', f'{patch_id:05}'), im_saved)
        np.save(os.path.join(directory, 'Patches', mode, 'mask_patches', f'{patch_id:05}'), mask_categorical)

        if augment:
            np.save(os.path.join(directory, 'Patches', mode, 'img_patches', f'{patch_id + 1:05}'), np.rot90(im_saved, 1))
            np.save(os.path.join(directory, 'Patches', mode, 'mask_patches', f'{patch_id + 1:05}'), np.rot90(mask_categorical, 1))

            np.save(os.path.join(directory, 'Patches', mode, 'img_patches', f'{patch_id + 2:05}'), np.rot90(im_saved, 2))
            np.save(os.path.join(directory, 'Patches', mode, 'mask_patches', f'{patch_id + 2:05}'), np.rot90(mask_categorical, 2))

            np.save(os.path.join(directory, 'Patches', mode, 'img_patches', f'{patch_id + 3:05}'), np.rot90(im_saved, 3))
            np.save(os.path.join(directory, 'Patches', mode, 'mask_patches', f'{patch_id + 3:05}'), np.rot90(mask_categorical, 3))

            np.save(os.path.join(directory, 'Patches', mode, 'img_patches', f'{patch_id + 4:05}'), np.fliplr(im_saved))
            np.save(os.path.join(directory, 'Patches', mode, 'mask_patches', f'{patch_id + 4:05}'), np.fliplr(mask_categorical))

            np.save(os.path.join(directory, 'Patches', mode, 'img_patches', f'{patch_id + 5:05}'), np.flipud(im_saved))
            np.save(os.path.join(directory, 'Patches', mode, 'mask_patches', f'{patch_id + 5:05}'), np.flipud(mask_categorical))
        return True
    else:
        return False


print('Generating training data...')
read_data('DataBR', mode='Train', augment=True)
train_farms = number_of_farms
train_tiles = number_of_unique_tiles
print(f"Training data: {train_farms} farms - {train_tiles} unique tiles - {train_tiles * 4} tiles with data augmentation")

# print('Generating eval data...')
# read_data('DataBR', mode='Eval', augment=False)
# eval_farms = number_of_farms - train_farms
# eval_tiles = number_of_unique_tiles - train_tiles
# print(f"Training data: {eval_farms} farms - {eval_tiles} unique tiles - {eval_tiles * 4} tiles with data augmentation")