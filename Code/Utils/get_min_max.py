"""
    File name: get_min_max.py
    Author: Corinne Dumais
    Last modified: May 2022
"""

import os

import numpy as np
import rasterio as rs


def display_min_max(dir_name='DataBR/Composites/Train'):
    """
    Utility function to print min and max value for each band within the dataset. Useful for normalization.

    Parameters:
        dir_name (string): path to the dataset
    """
    max_list_mhc = []
    min_list_mhc = []
    max_list_mnt = []
    min_list_mnt = []
    max_list_twi = []
    min_list_twi = []

    for file in os.listdir(dir_name):
        if file.endswith('.tif'):
            with rs.open(os.path.join(dir_name, file), 'r') as ds:
                im = ds.read()
            im[im < 0] = 255
            max_list_mhc.append(np.amax(im[0, :, :]))
            min_list_mhc.append(np.amin(im[0, :, :]))
            max_list_mnt.append(np.amax(im[1, :, :]))
            min_list_mnt.append(np.amin(im[1, :, :]))
            max_list_twi.append(np.amax(im[2, :, :]))
            min_list_twi.append(np.amin(im[2, :, :]))

    print('MHC')
    print(f'Min: {np.amin(min_list_mhc)}    Max: {np.amax(max_list_mhc)}')
    print('MNT')
    print(f'Min: {np.amin(min_list_mnt)}    Max: {np.amax(max_list_mnt)}')
    print('TWI')
    print(f'Min: {np.amin(min_list_twi)}    Max: {np.amax(max_list_twi)}')

display_min_max()