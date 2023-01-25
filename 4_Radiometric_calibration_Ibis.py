#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IBIS radiometric calibration / QA file, 2022-11-22

@author: rramsay
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show, title, subplot, colorbar
from spectral import open_image 
import pandas as pd
import scipy as sp
from arsf_envi_reader import envi_header

#%% Read in frame, take the auto dark statrt line from the metadata

#IBIS image
ibis_image = open_image("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/IBIS_QA_11-2022/N30IBIS5101_2022-11-21_14-28-13/capture/N30IBIS5101_2022-11-21_14-28-13.hdr")
dark_frame_start = int(int(ibis_image.metadata['autodarkstartline']))   #takes the frame number at which the dark frames begin
tint = float(float(ibis_image.metadata['tint']))

#%% Conduct statistical treatment, and subtract dark frame from light frame

ibis_dark_mean = np.average(ibis_image[(dark_frame_start+1):,:,:],axis=0)
ibis_dark_std = np.std(ibis_image[(dark_frame_start+1):,:,:],axis=0,ddof=1)
ibis_dark_snr = np.average(ibis_image[(dark_frame_start+1):,:,:],axis=0)/np.std(ibis_image[dark_frame_start:,:,:],axis=0,ddof=1)

ibis_light_mean = np.average(ibis_image[:dark_frame_start,:,:],axis=0)
ibis_light_std = np.std(ibis_image[:dark_frame_start,:,:],axis=0,ddof=1)
ibis_light_snr = np.average(ibis_image[:dark_frame_start,:,:],axis=0)/np.std(ibis_image[:dark_frame_start,:,:],axis=0,ddof=1)

ibis_frame = np.subtract(ibis_light_mean, ibis_dark_mean)
ibis_std = np.subtract(ibis_light_std, ibis_dark_std)
ibis_snr = np.subtract(ibis_light_snr, ibis_light_snr)

plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.figsize'] = [10, 15]
plt.rcParams['image.cmap']='plasma'

subplot(131)
title("Frame DCC average")
imshow(ibis_frame)
colorbar()

subplot(132)
title("Frame σ DCC average")
imshow(ibis_std)
colorbar()

subplot(133)
title("Frame SNR DCC average")
imshow(ibis_snr)
colorbar()

show()


#%% VNIR  and SWIR frame with errors as percetange

ibis_uncertainity = (ibis_std / ibis_frame) * 100

imshow(ibis_uncertainity)
colorbar()


#%% Import Helios 350 -- 2500 nm radiance data

#Need to change this to the ARF sphere calibration file
ibis_IT = tint
ARF_lamp_4X = pd.read_csv("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/ARF_Relamp_Feb2022_SpectralRadianceCal_Extrapolate.csv")


#%% Call in wavlength calibration values

ibis_header_dict = envi_header.read_hdr_file("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/IBIS_QA_11-2022/N30IBIS5101_2022-11-21_14-28-13/capture/N30IBIS5101_2022-11-21_14-28-13.hdr")
ibis_wavelengths = ibis_header_dict['wavelength'].split(',')
ibis_wavelengths = [float(l) for l in ibis_wavelengths]


#%% Interpolate ARF_lamp_4X radiance data to the wavelength region

def f(x):
    x_points = ARF_lamp_4X["Wavelength"]
    y_points = ARF_lamp_4X["Spectral Radiance (mW/m2-sr-nm)"]

    tck = sp.interpolate.splrep(x_points, y_points)
    return sp.interpolate.splev(x, tck)

interpolated_ibis_radiance = (f(ibis_wavelengths))

#Normalize the IBIS image by dividing by it's integration time, tint
ibis_frame_normalized = ibis_frame / tint  


#%% Transpose interpolated radiance, then divide the respective averaged frame on row per row basis with transposed frame

#VNIR
interpolated_ibis_radiance = interpolated_ibis_radiance.T
Calibrated_ibis = ibis_frame_normalized / interpolated_ibis_radiance 

plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['image.cmap']='plasma'

title("Calibration Coefficents IBIS")
imshow(Calibrated_ibis)
colorbar()
show()

#np.savetxt("IBIS_Calibration_Coefficents_Radiance.csv", Calibrated_ibis, delimiter=",")


#%% To do --
#Need to clear those bad pixels ASAP. Bad pixel map required.
#SNR and stability across frame

#Bad pixel mapping -- how to determine...
#Take the SNR