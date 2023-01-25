#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 19:43:11 2022

@author: rramsay
"""

#%% Libraries

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show, title, subplot, colorbar
from spectral import open_image 
import pandas as pd
from astropy.modeling import models
import astropy.units as u
from specutils import Spectrum1D, SpectralRegion
from arsf_envi_reader import envi_header
from specutils.fitting import find_lines_threshold, find_lines_derivative, fit_continuum, estimate_line_parameters
from specutils.manipulation import noise_region_uncertainty, extract_region
import warnings
from specutils.fitting import fit_lines

#%% Load the Hg-Ar and Ar light and dark frames for VNIR, average, combine, then take average over the 640 spatial channels

# Plotting options
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['image.cmap']='binary'

#Hg-Ar
hg_ar_dark_op = open_image("100361_VNIR_LambdaCal_HgAr_dark_2015_06_04_22_37_27/raw_0.hdr")
hg_ar_light_op = open_image("100362_VNIR_LambdaCal_HgAr_2015_06_04_22_38_17/raw_0.hdr")

#Ar
ar_dark_op = open_image("100363_VNIR_LambdaCal_Ar_dark_2015_06_04_22_55_57/raw_0.hdr")
ar_light_op = open_image("100364_VNIR_LambdaCal_Ar_2015_06_04_22_56_49/raw_0.hdr")


hgar_dark_mean = np.average(hg_ar_dark_op[:,:],axis=0)
hgar_dark_std = np.std(hg_ar_dark_op[:,:],axis=0,ddof=1)
hgar_dark_snr = np.average(hg_ar_dark_op[:,:],axis=0)/np.std(hg_ar_dark_op[:,:],axis=0,ddof=1)

hgar_light_mean = np.average(hg_ar_light_op[:,:],axis=0)
hgar_light_std = np.std(hg_ar_light_op[:,:],axis=0,ddof=1)
hgar_light_snr = np.average(hg_ar_light_op[:,:],axis=0)/np.std(hg_ar_light_op[:,:],axis=0,ddof=1)

hgar_frame = np.subtract(hgar_light_mean, hgar_dark_mean)
hgar_std = np.subtract(hgar_light_std, hgar_dark_std)

#Ar frame
ar_dark_mean = np.average(ar_dark_op[:,:],axis=0)
ar_dark_std = np.std(ar_dark_op[:,:],axis=0,ddof=1)
ar_dark_snr = np.average(ar_dark_op[:,:],axis=0)/np.std(ar_dark_op[:,:],axis=0,ddof=1)

ar_light_mean = np.average(ar_light_op[:,:],axis=0)
ar_light_std = np.std(ar_light_op[:,:],axis=0,ddof=1)
ar_light_snr = np.average(ar_light_op[:,:],axis=0)/np.std(ar_light_op[:,:],axis=0,ddof=1)

ar_frame = np.subtract(ar_light_mean, ar_dark_mean)
ar_std = np.subtract(ar_light_std, ar_dark_std)

VNIR_frame = np.add(hgar_frame, ar_frame)

VNIR_frame_Channel_Average = VNIR_frame.mean(axis=0)

#%% Using only one frame at a time, so as to see if continuum removal can be resolved
#Use the centre line of the frame
hgar_centreline_frame = hgar_frame[319]

flux_pixel = hgar_centreline_frame.tolist()
flux_pixel = flux_pixel * u.dimensionless_unscaled
wavelengths_pixel = list(range(1, 269)) * u.m
hgar_frame_cal = Spectrum1D(spectral_axis=wavelengths_pixel, flux=flux_pixel)

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    hg_ar_pixel_fit = fit_continuum(hgar_frame_cal)
    
hg_ar_pixel_continuum_fitted = hg_ar_pixel_fit(wavelengths_pixel)

hg_ar_pixel_normalized = hgar_frame_cal / hg_ar_pixel_continuum_fitted

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    pixel_lines = find_lines_derivative(hgar_frame_cal, flux_threshold=100.0)
    pixel_lines = pixel_lines.to_pandas()
    
amplitude_value = []
mean_value = []
std_value = []
fwhm = []
def gaussian_fit_parameters_pixel(x):
    sub_region = SpectralRegion((x-3)*u.m, (x+3)*u.m)
    sub_spectrum = extract_region(hgar_frame_cal, sub_region)
    result = (estimate_line_parameters(sub_spectrum, models.Gaussian1D()))
    amplitude_value.append(result.amplitude.value)
    mean_value.append(result.mean.value)
    std_value.append(result.stddev.value)
    fwhm.append(result.fwhm.value)
    return mean_value, std_value, fwhm

iterate_parameters = [gaussian_fit_parameters_pixel(x) for x in pixel_lines.line_center]
VNIR_lines_pixel_parameters = pd.DataFrame(list(zip(amplitude_value, mean_value, std_value, fwhm)),
                                     columns =['Amplitude', 'Centre Line', 'STD', 'FWHM'])


         
g1_435_init = models.Gaussian1D(amplitude=775.449 * u.dimensionless_unscaled, mean=15.249497*u.m, stddev=1.207624*u.m)     
g2_546_init = models.Gaussian1D(amplitude=2348.586* u.dimensionless_unscaled, mean=64.054908*u.m, stddev=1.195153*u.m)
g3_576_init = models.Gaussian1D(amplitude=669.399* u.dimensionless_unscaled, mean=78.200824*u.m, stddev=1.306486*u.m)

pixel_fit_435 = g1_435_init(wavelengths_pixel)
pixel_fit_546 = g2_546_init(wavelengths_pixel)
pixel_fit_576 = g3_576_init(wavelengths_pixel)

plt.plot(hgar_frame_cal.spectral_axis, hgar_frame_cal.flux, '#241F1F', linewidth=3.5)
plt.plot(hgar_frame_cal.spectral_axis, pixel_fit_435, '#BD687B', linewidth=3.5, linestyle='dashed')
plt.plot(hgar_frame_cal.spectral_axis, pixel_fit_546, '#E3E36B', linewidth=3.5, linestyle='dashed')
plt.plot(hgar_frame_cal.spectral_axis, pixel_fit_576, '#86B565', linewidth=3.5, linestyle='dashed')
plt.axis([0,267, 0, 2500])
plt.xlabel('Pixel #')
plt.ylabel('DN')
plt.grid(True)
plt.legend(['Original spectrum (average of 640 spatial channels, continuum removed ', 'Fit result, Gaussian 1D'])
plt.tight_layout()
plt.savefig("2.png", dpi = 300)
plt.show()


#%% Process the data for the frame with current lambda calibration to determine peak correspondence to true wavelength

#reading in lambda cal current
header_dict = envi_header.read_hdr_file("100362_VNIR_LambdaCal_HgAr_2015_06_04_22_38_17/raw_0.hdr")
wavelengths = header_dict['wavelength'].split(',')
wavelengths = [float(l) for l in wavelengths]

# creating spectra
flux_lambda = VNIR_frame_Channel_Average.tolist()
flux_lambda = flux_lambda * u.dimensionless_unscaled
wavelengths_lambda = wavelengths * u.nm
VNIR_Cal_Spectrum_lambda = Spectrum1D(spectral_axis=wavelengths_lambda, flux=flux_lambda)

#generating continuum
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    lambda_fit = fit_continuum(VNIR_Cal_Spectrum_lambda)
    
lambda_continuum_fitted = lambda_fit(wavelengths_lambda)
    
VNIR_Cal_Lambda_Corrected = VNIR_Cal_Spectrum_lambda / lambda_continuum_fitted

# Generate emission line frame
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    lambda_lines = find_lines_derivative(VNIR_Cal_Lambda_Corrected, flux_threshold=2.0)
    lambda_lines = lambda_lines.to_pandas()
    
#%% Process the data for pixel range, find lines, create Gaussian fit, refind lines, take the new peak center as the sub-pixel that matches to wavelength
# Use the index, i.e. the spectral pixel number, as the wavelengths input
flux_pixel = VNIR_frame_Channel_Average.tolist()
flux_pixel = flux_pixel * u.dimensionless_unscaled
wavelengths_pixel = list(range(1, 269)) * u.m
VNIR_Cal_Spectrum_Pixel = Spectrum1D(spectral_axis=wavelengths_pixel, flux=flux_pixel)

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    pixel_fit = fit_continuum(VNIR_Cal_Spectrum_Pixel)
    
pixel_continuum_fitted = pixel_fit(wavelengths_pixel)

pixel_normalized = VNIR_Cal_Spectrum_Pixel / pixel_continuum_fitted

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    pixel_lines = find_lines_derivative(pixel_normalized, flux_threshold=2.0)
    pixel_lines = pixel_lines.to_pandas()
    
amplitude_value = []
mean_value = []
std_value = []
fwhm = []
def gaussian_fit_parameters_pixel(x):
    sub_region = SpectralRegion((x-3)*u.m, (x+3)*u.m)
    sub_spectrum = extract_region(pixel_normalized, sub_region)
    result = (estimate_line_parameters(sub_spectrum, models.Gaussian1D()))
    amplitude_value.append(result.amplitude.value)
    mean_value.append(result.mean.value)
    std_value.append(result.stddev.value)
    fwhm.append(result.fwhm.value)
    return mean_value, std_value, fwhm

iterate_parameters = [gaussian_fit_parameters_pixel(x) for x in pixel_lines.line_center]
VNIR_lines_pixel_parameters = pd.DataFrame(list(zip(amplitude_value, mean_value, std_value, fwhm)),
                                     columns =['Amplitude', 'Centre Line', 'STD', 'FWHM'])

g1_435_init = models.Gaussian1D(amplitude=6.746530 * u.dimensionless_unscaled, mean=15.605889*u.m, stddev=1.247983*u.m)     
g2_546_init = models.Gaussian1D(amplitude=23.731818 * u.dimensionless_unscaled, mean=64.335825*u.m, stddev=1.229297*u.m)
g3_576_init = models.Gaussian1D(amplitude=4.967622 * u.dimensionless_unscaled, mean=78.464109*u.m, stddev=1.361161*u.m)
g4_692_init = models.Gaussian1D(amplitude=2.182420 * u.dimensionless_unscaled, mean=131.103709*u.m, stddev=1.418286*u.m)
g5_763_init = models.Gaussian1D(amplitude=6.144672 * u.dimensionless_unscaled, mean=160.562854*u.m, stddev=1.505035*u.m) 
g6_811_init = models.Gaussian1D(amplitude=4.772345 * u.dimensionless_unscaled, mean=181.188459*u.m, stddev=1.731401*u.m)
g7_840_init = models.Gaussian1D(amplitude=2.801979 * u.dimensionless_unscaled, mean=195.338626*u.m, stddev=1.398777*u.m)
g8_912_init = models.Gaussian1D(amplitude=3.904525 * u.dimensionless_unscaled, mean=226.579145*u.m, stddev=1.327337*u.m)   
g9_965_init = models.Gaussian1D(amplitude=3.864702 * u.dimensionless_unscaled, mean=250.757416*u.m, stddev=1.346275*u.m)

pixel_total_fit = fit_lines(pixel_normalized,
                        g1_435_init+g2_546_init+g3_576_init+g4_692_init+g7_840_init+g8_912_init+g9_965_init,)
pixel_fit_total = pixel_total_fit(wavelengths_pixel)
             

#Creating data frame with values to use
Regression_VNIR = pd.DataFrame(
    {
     "Lambda_Actual": [435.84, 546.08, 576.96, 696.54, 763.51, 840.82, 912.30, 965.80],
     "Centre_Pixel": [15.606, 64.336, 78.461, 131.104, 160.563, 195.338, 226.579, 250.757],
     "x_err": [1.248, 1.229, 1.361, 1.418, 1.505, 1.399, 1.327, 1.346],
     "y_err": [2, 2, 2, 2, 2, 2, 2, 2],
    },
    index=[0, 1, 2, 3, 4, 5, 6, 7],
)

Regression_VNIR.to_csv("Output_for_Regression.csv")

#%% Polynomial model
np.polyfit(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual, 3, full = True)

p, cov = np.polyfit(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual, 3, cov = True)
q = np.diag(cov)
r = np.sqrt(q)

Polyfit_Output = pd.DataFrame(
    {
     "Coefficents": [4.00858120e+02, 2.24129321e+00, 1.66284872e-04, -4.67832147e-07],
     "Standard Error Coefficents Matrix": [9.34189041e-01, 3.00524752e-02, 2.61041467e-04, 6.44654840e-07],
    },
    index=[0, 1, 2, 3],
)

Polyfit_Output.to_csv("OPolyfit_Output.csv")

mymodel = np.poly1d(np.polyfit(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual, 3))


myline = np.linspace(1, 268, 268)

plt.plot(myline, mymodel(myline))
plt.show() 

# model output is -4.67832147e-07,  1.66284872e-04,  2.24129321e+00,  4.00858120e+02])
#Step here is to take the coefficcents of the poly fit, and then create a covariance matrix from which the error can be calculated. 


# %% Plots for presentation

# Plotting Paramters
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = [12, 8]


# Original spectra continuum removed and averaged, wavelength, with Hg-Ar peaks
HgAr_lines = pd.DataFrame(
    {
     "Lambda": [435.84, 546.08, 576.96, 696.54, 763.51, 840.82, 912.30, 965.80],
     "Value": [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    },
    index=[0, 1, 2, 3, 4, 5, 6, 7],
)


plt.plot(VNIR_Cal_Lambda_Corrected.spectral_axis, VNIR_Cal_Lambda_Corrected.flux, '#241F1F', linewidth=3.5)
plt.bar(HgAr_lines.Lambda, HgAr_lines.Value, width=5, color = '#5392AD')
plt.axis([400, 1000, 0, 30])
plt.xlabel('λ (nm)')
plt.ylabel('DN')
plt.grid(True)
plt.legend(['Original spectrum (average of 640 spatial channels, continuum removed ', 'Hg-Ar Emission Lines'])
plt.tight_layout()
plt.savefig("1.png", dpi = 300)
plt.show()

# Original spectra with the curve fitting

plt.plot(pixel_normalized.spectral_axis, pixel_normalized.flux, '#241F1F', linewidth=3.5)
plt.plot(pixel_normalized.spectral_axis, pixel_fit_total, '#BD687B', linewidth=3.5, linestyle='dashed')
plt.axis([0,267, 0, 25])
plt.xlabel('Pixel #')
plt.ylabel('DN')
plt.grid(True)
plt.legend(['Original spectrum (average of 640 spatial channels, continuum removed ', 'Fit result, Gaussian 1D'])
plt.tight_layout()
plt.savefig("2.png", dpi = 300)
plt.show()


# Regression fitting
myline = np.linspace(1, 268, 268)


bbox = dict(boxstyle ="square", fc ="1.0")

plt.axis([0,268, 400, 1000])
plt.plot(myline, mymodel(myline), '#241F1F', linewidth=3.5, linestyle='dashed')
plt.errorbar(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual,
             xerr=Regression_VNIR.x_err,
             yerr=Regression_VNIR.y_err,
             ls='none',
             marker='o', mfc='#86B565',
             mec='#241F1F', ms=12, mew=4
             )  
plt.ylabel('λ (nm)')
plt.xlabel('Pixel #')
plt.annotate("y = 400.8 + 2.24$x$ + 1.662E-4$x^2$ - 4.676E-7$x^3$ \n y = λ (nm) \n x = Pixel # ", xy =(20,850), bbox = bbox)
plt.grid(True)
plt.tight_layout()
plt.savefig("3.png", dpi = 300)
plt.show()



#Error highlight
plt.subplot(2, 2, 1)   # Define 3 rows, 2 column, Activate subplot 1. 
plt.axis([62, 66, 541, 549])
plt.plot(myline, mymodel(myline), '#241F1F', linewidth=3.5, linestyle='dashed')
plt.errorbar(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual,
             xerr=Regression_VNIR.x_err,
             yerr=Regression_VNIR.y_err,
             ls='none',
             marker='o', mfc='#BD687B',
             mec='#241F1F', ms=12, mew=4,
             ecolor = '#5E5E5E', 
             elinewidth = 2.5
             )  
plt.ylabel('λ (nm)')
plt.xlabel('Pixel #')
plt.grid(True)

plt.subplot(2, 2, 2)   # 3 rows, 2 column, Activate subplot 2.
plt.axis([76, 80, 572, 580])
plt.plot(myline, mymodel(myline), '#241F1F', linewidth=3.5, linestyle='dashed')
plt.errorbar(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual,
             xerr=Regression_VNIR.x_err,
             yerr=Regression_VNIR.y_err,
             ls='none',
             marker='o', mfc='#E3E36B',
             mec='#241F1F', ms=12, mew=4,
             ecolor = '#5E5E5E', 
             elinewidth = 2.5
             )  
plt.ylabel('λ (nm)')
plt.xlabel('Pixel #')
plt.grid(True)


plt.subplot(2, 2, 3)   # 3 rows, 2 column, Activate subplot 3.
plt.axis([158, 162, 758, 766])
plt.plot(myline, mymodel(myline), '#241F1F', linewidth=3.5, linestyle='dashed')
plt.errorbar(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual,
             xerr=Regression_VNIR.x_err,
             yerr=Regression_VNIR.y_err,
             ls='none',
             marker='o', mfc='#86B565',
             mec='#241F1F', ms=12, mew=4,
             ecolor = '#5E5E5E', 
             elinewidth = 2.5
             )  
plt.ylabel('λ (nm)')
plt.xlabel('Pixel #')
plt.grid(True)



plt.subplot(2, 2, 4)   # 3 rows, 2 column, Activate subplot 3.
plt.axis([193, 197, 835, 843])
plt.plot(myline, mymodel(myline), '#241F1F', linewidth=3.5, linestyle='dashed')
plt.errorbar(Regression_VNIR.Centre_Pixel, Regression_VNIR.Lambda_Actual,
             xerr=Regression_VNIR.x_err,
             yerr=Regression_VNIR.y_err,
             ls='none',
             marker='o', mfc='#5392AD',
             mec='#241F1F', ms=12, mew=4,
             ecolor = '#5E5E5E', 
             elinewidth = 2.5
             )  
plt.ylabel('λ (nm)')
plt.xlabel('Pixel #')
plt.grid(True)


# to Prevent subplots overlap
plt.tight_layout()  
plt.savefig("4.png", dpi = 300)
plt.show()


