#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:35:32 2022

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

#%% Read in light and dark frames

#VNIR
VNIR_dark = open_image("/home/rramsay/Documents/EOES/Equipment/Headwall/Calibration/2022-08-03/100357_VNIR_RadCal2_220803_dark_2015_06_04_22_02_32/raw_0.hdr")
VNIR_light = open_image("/home/rramsay/Documents/EOES/Equipment/Headwall/Calibration/2022-08-03/100358_VNIR_RadCal2_220803_2015_06_04_22_04_00/raw_0.hdr")

#SWIR
SWIR_dark = open_image("/home/rramsay/Documents/EOES/Equipment/Headwall/Calibration/2022-08-03/100353_SWIR_RadCal220308_dark_2019_03_29_21_30_07/raw_0.hdr")
SWIR_light = open_image("/home/rramsay/Documents/EOES/Equipment/Headwall/Calibration/2022-08-03/100354_SWIR_RadCal220308_2019_03_29_21_31_20/raw_0.hdr")

#%% Conduct statistical treatment, and subtract dark frame from light frame

#VNIR
VNIR_dark_mean = np.average(VNIR_dark[:,:],axis=0)
VNIR_dark_std = np.std(VNIR_dark[:,:],axis=0,ddof=1)
VNIR_dark_snr = np.average(VNIR_dark[:,:],axis=0)/np.std(VNIR_dark[:,:],axis=0,ddof=1)

VNIR_light_mean = np.average(VNIR_light[:,:],axis=0)
VNIR_light_std = np.std(VNIR_light[:,:],axis=0,ddof=1)
VNIR_light_snr = np.average(VNIR_light[:,:],axis=0)/np.std(VNIR_light[:,:],axis=0,ddof=1)

VNIR_frame = np.subtract(VNIR_light_mean, VNIR_dark_mean)
VNIR_std = np.subtract(VNIR_light_std, VNIR_dark_std)
VNIR_snr = np. subtract(VNIR_light_snr, VNIR_light_snr)

#SWIR
SWIR_dark_mean = np.average(SWIR_dark[:,:],axis=0)
SWIR_dark_std = np.std(SWIR_dark[:,:],axis=0,ddof=1)
SWIR_dark_snr = np.average(SWIR_dark[:,:],axis=0)/np.std(SWIR_dark[:,:],axis=0,ddof=1)

SWIR_light_mean = np.average(SWIR_light[:,:],axis=0)
SWIR_light_std = np.std(SWIR_light[:,:],axis=0,ddof=1)
SWIR_light_snr = np.average(SWIR_light[:,:],axis=0)/np.std(SWIR_light[:,:],axis=0,ddof=1)

SWIR_frame = np.subtract(SWIR_light_mean, SWIR_dark_mean)
SWIR_std = np.subtract(SWIR_light_std, SWIR_dark_std)
SWIR_snr = np. subtract(SWIR_light_snr, SWIR_dark_snr)

plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.figsize'] = [10, 15]
plt.rcParams['image.cmap']='binary'

subplot(131)
title("VNIR DCC average")
imshow(VNIR_frame)
colorbar()

subplot(132)
title("VNIR σ DCC average")
imshow(VNIR_std)
colorbar()

subplot(133)
title("VNIR SNR DCC average")
imshow(VNIR_snr)
colorbar()

show()

subplot(131)
title("SWIR DCC average")
imshow(VNIR_frame)
colorbar()

subplot(132)
title("SWIR σ DCC average")
imshow(SWIR_std)
colorbar()

subplot(133)
title("SWIR SNR DCC average")
imshow(SWIR_snr)
colorbar()

show()

#%% VNIR  and SWIR frame with errors as percetange

VNIR_uncertainity = (VNIR_std / VNIR_frame) * 100
SWIR_uncertainity = (SWIR_std / SWIR_frame) * 100

imshow(VNIR_uncertainity)
colorbar()

#%% Applying integration time correction factor

#Define function to get integration time from settings files

def getintegrationtime(x):
    f = open(input("Enter .txt filename: "))
    file = f.readlines()
    for item in file:
        content = item.split("=")
        
        if content[0]==x:
            f.close()
            return content[1]
        else:
            continue
    f.close()
       
VNIR_IT = getintegrationtime("Exposure (ms) ")
VNIR_IT  = float(VNIR_IT)  #remember to convert it from string to float  

SWIR_IT = getintegrationtime("Exposure (ms) ")
SWIR_IT  = float(SWIR_IT)  #remember to convert it from string to float 

#%% Import Helios 350 -- 2500 nm radiance data

Helios_2500 = pd.read_csv("/home/rramsay/Documents/EOES/Equipment/Headwall/Helios_Calibration/Helios_2500_Interpolated.csv")
Helios_2500["Spectral Radiance (mW/m2-sr-nm-)"] = Helios_2500["Spectral Radiance (uW/cm2-sr-nm)"] * 10

Helios_2500_Uncertainities = pd.read_csv("/home/rramsay/Documents/EOES/Equipment/Headwall/Helios_Calibration/Helios_Uncertainity.csv")


#%% Call in wavlength calibration values

#VNIR
VNIR_header_dict = envi_header.read_hdr_file("/home/rramsay/Documents/EOES/Equipment/Headwall/Calibration/2022-08-03/100357_VNIR_RadCal2_220803_dark_2015_06_04_22_02_32/raw_0.hdr")
VNIR_wavelengths = VNIR_header_dict['wavelength'].split(',')
VNIR_wavelengths = [float(l) for l in VNIR_wavelengths]

#SWIR
SWIR_header_dict = envi_header.read_hdr_file("/home/rramsay/Documents/EOES/Equipment/Headwall/Calibration/2022-08-03/100353_SWIR_RadCal220308_dark_2019_03_29_21_30_07/raw_0.hdr")
SWIR_wavelengths = SWIR_header_dict['wavelength'].split(',')
SWIR_wavelengths = [float(l) for l in SWIR_wavelengths]


#%% Interpolate Helios_2500 radiance data to the wavelength region, then multiply by integration time

#VNIR
def f(x):
    x_points = Helios_2500["Wavelength (nm)"]
    y_points = Helios_2500["Spectral Radiance (mW/m2-sr-nm-)"]

    tck = sp.interpolate.splrep(x_points, y_points)
    return sp.interpolate.splev(x, tck)

interpolated_VNIR_radiance = (f(VNIR_wavelengths))                        # * 10 added to convert from uW cm-2 sr-1 nm-1 to mW m-2 sr-1 nm-1
interpolated_VNIR_radiance_IT = interpolated_VNIR_radiance * VNIR_IT

#SWIR
interpolated_SWIR_radiance = (f(SWIR_wavelengths))
interpolated_SWIR_radiance_IT = interpolated_SWIR_radiance * SWIR_IT

# Interpolate uncertainities

def g(x):
    x_points = Helios_2500_Uncertainities["Wavelength (nm)"]
    y_points = Helios_2500_Uncertainities["Uc"]

    tck = sp.interpolate.splrep(x_points, y_points)
    return sp.interpolate.splev(x, tck)

interpolated_VNIR_Uncertainity = (g(VNIR_wavelengths))  
interpolated_SWIR_Uncertainity = (g(SWIR_wavelengths))  


#%% Exporting uncertainities


np.savetxt("VNIR_Uncertainity_RadianceSphere.csv", interpolated_VNIR_Uncertainity, delimiter=",")
np.savetxt("VNIR_Uncertainity_Headwall.csv", VNIR_uncertainity, delimiter=",")

np.savetxt("SWIR_Uncertainity_RadianceSphere.csv", interpolated_SWIR_Uncertainity, delimiter=",")
np.savetxt("SWIR_Uncertainity_Headwall.csv", SWIR_uncertainity, delimiter=",")

#%% Combining Errors

#VNIR

VNIR_Radiance_Uncertainity = np.genfromtxt("/home/rramsay/Documents/EOES/Equipment/Headwall/VNIR_Uncertainity_RadianceSphere.csv", delimiter=",")
VNIR_Combined_Uncertainity = np.sqrt(np.square(VNIR_uncertainity) + np.square(VNIR_Radiance_Uncertainity))

plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.figsize'] = [10, 15]
plt.rcParams['image.cmap']='plasma'

title("Radiometric Calibration Coefficients Uc VNIR cAVS-120 k=2 %")
imshow(VNIR_Combined_Uncertainity)
colorbar()
show()
np.savetxt("VNIR_Uc_Calibration.csv", VNIR_Combined_Uncertainity, delimiter=",")


#SWIR
SWIR_Radiance_Uncertainity = np.genfromtxt("/home/rramsay/Documents/EOES/Equipment/Headwall/SWIR_Uncertainity_RadianceSphere.csv", delimiter=",")
SWIR_Combined_Uncertainity = np.sqrt(np.square(SWIR_uncertainity) + np.square(SWIR_Radiance_Uncertainity))

title("Radiometric Calibration Coefficients Uc SWIR cAVS-120 k=2 %")
imshow(SWIR_Combined_Uncertainity)
colorbar()
show()
np.savetxt("SWIR_Uc_Calibration.csv", SWIR_Combined_Uncertainity, delimiter=",")

#%% Measurement Error
#VNIR

VNIR_Measurement_Uncertainity = np.genfromtxt("/home/rramsay/Documents/EOES/Equipment/Headwall/Base_VNIR_Measurement_Error.csv", delimiter=",")
VNIR_Oban_Radiance_Uncertainity = np.sqrt(np.square(VNIR_Combined_Uncertainity) + np.square(VNIR_Measurement_Uncertainity ))
np.savetxt("VNIR_Uc_Radiance.csv", VNIR_Oban_Radiance_Uncertainity, delimiter=",")
#SWIR

SWIR_Measurement_Uncertainity = np.genfromtxt("/home/rramsay/Documents/EOES/Equipment/Headwall/Base_SWIR_Measurement_Error.csv", delimiter=",")
SWIR_Oban_Radiance_Uncertainity = np.sqrt(np.square(SWIR_Combined_Uncertainity) + np.square(SWIR_Measurement_Uncertainity))
np.savetxt("SWIR_Uc_Radiance.csv", SWIR_Oban_Radiance_Uncertainity, delimiter=",")

#%% Transpose interpolated radiance, then divide the respective averaged frame on row per row basis with transposed frame

#VNIR
interpolated_VNIR_radiance = interpolated_VNIR_radiance.T
Calibrated_VNIR =  VNIR_frame / interpolated_VNIR_radiance 

plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.figsize'] = [10, 15]
plt.rcParams['image.cmap']='plasma'

title("Calibration Coefficents VNIR")
imshow(Calibrated_VNIR)
colorbar()
show()

np.savetxt("VNIR_Calibration_Coefficents_Radiance.csv", Calibrated_VNIR, delimiter=",")


#SWIR
interpolated_SWIR_radiance = interpolated_SWIR_radiance.T
Calibrated_SWIR = interpolated_SWIR_radiance/SWIR_frame 

plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.figsize'] = [10, 15]
plt.rcParams['image.cmap']='plasma'

title("Calibration Coefficents SWIR")
imshow(Calibrated_SWIR)
colorbar()
show()

np.savetxt("SWIR_Calibration_Coefficents_Radiance.csv", Calibrated_SWIR, delimiter=",")

#%% To do --
#Need to clear those bad pixels ASAP. Bad pixel map required.
#SNR and stability across frame