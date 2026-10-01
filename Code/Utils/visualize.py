import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rs

# cmaplist = ['#000000', '#003300', '#994c00', '#A16BA1', '#ffcc99', '#6DA16B', '#DBC820', '#331900']
cmaplist = ['#ffffff', '#003300', '#994c00', '#A16BA1', '#ffcc99', '#DBC820', '#331900']
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, 7)
bounds = np.linspace(0, 7, 8)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# for i in range(1, 3100, 4):
#     file = f'{i:05}.npy'
#     fig, axes = plt.subplots(ncols=4, figsize=(10, 6))
#     ax = axes.ravel()
#
#     img = np.load(os.path.join('DataBR', 'Patches', 'Train', 'img_patches', file))
#
#     ax[0].imshow(img[:, :, 2])
#     ax[1].imshow(img[:, :, 1])
#     ax[2].imshow(img[:, :, 0])
#     ax[3].imshow(np.argmax(np.load(os.path.join('DataBR', 'Patches', 'Train', 'mask_patches', file)), axis=-1), cmap=cmap, norm=norm)
#
#     plt.suptitle(file)
#     plt.show()


def display_dataset(dir_name='DataBR', mode='Train'):
    for file in os.listdir(os.path.join(dir_name, 'Composites', mode)):
        if file.endswith('.tif'):
            with rs.open(os.path.join(dir_name, 'Composites', mode, file), 'r') as ds:
                img = ds.read()
            with rs.open(os.path.join(dir_name, 'VeriteTerrain', f'{file[:-13]}Verite.tif'), 'r') as ds:
                mask = ds.read()

            mask[mask == 9] = 0
            mask[mask == 10] = 0
            mask[mask == 6] = 5  # combine paturage et culture
            mask[mask == 7] = 6
            mask[mask == 255] = 0

            fig, axes = plt.subplots(ncols=4, figsize=(14, 5))
            ax = axes.ravel()

            fig.suptitle(file)

            ax[0].imshow(img[0, :, :])
            ax[1].imshow(img[1, :, :])
            ax[2].imshow(img[2, :, :])
            ax[3].imshow(np.squeeze(mask), cmap=cmap, norm=norm)

            for i in range(4):
                ax[i].axis('off')

            plt.suptitle(file)
            plt.tight_layout()
            fig.savefig(f'DataBR/Bands/{mode}/' + file[:-4] + '.tif')
            plt.show()


display_dataset(mode='Eval')
