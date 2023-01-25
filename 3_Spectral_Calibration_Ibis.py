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
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['image.cmap']='binary'


#Ar
ar_ibis = open_image("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/IBIS_QA_11-2022/N30IBIS56221_2022-11-21_15-17-09/capture/N30IBIS56221_2022-11-21_15-17-09.hdr")
ar_dark_frame_start = int(int(ar_ibis.metadata['autodarkstartline']))   #takes the frame number at which the dark frames begin
tint_ar = float(float(ar_ibis.metadata['tint']))

#Read the IBIS wavelengths from the Ar .hdr file. Applicable to all spectral line lamp processing steps. 
header_dict = envi_header.read_hdr_file("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/IBIS_QA_11-2022/N30IBIS56221_2022-11-21_15-17-09/capture/N30IBIS56221_2022-11-21_15-17-09.hdr")
wavelengths = header_dict['wavelength'].split(',')
wavelengths = [float(l) for l in wavelengths]

ibis_ar_mean_dark = np.average(ar_ibis[(ar_dark_frame_start+1):,:,:],axis=0)
ibis_ar_std_dark = np.std(ar_ibis[(ar_dark_frame_start+1):,:,:],axis=0,ddof=1)
ibis_ar_snr_dark = np.average(ar_ibis[(ar_dark_frame_start+1):,:,:],axis=0)/np.std(ar_ibis[ar_dark_frame_start:,:,:],axis=0,ddof=1)

ibis_ar_mean_light = np.average(ar_ibis[:ar_dark_frame_start,:,:],axis=0)
ibis_ar_std_light = np.std(ar_ibis[:ar_dark_frame_start,:,:],axis=0,ddof=1)
ibis_ar_snr_light = np.average(ar_ibis[:ar_dark_frame_start,:,:],axis=0)/np.std(ar_ibis[:ar_dark_frame_start,:,:],axis=0,ddof=1)

ibis_ar_frame = np.subtract(ibis_ar_mean_light, ibis_ar_mean_dark)
ibis_ar_std = np.subtract(ibis_ar_std_light, ibis_ar_std_dark)
ibis_ar_snr = np.subtract(ibis_ar_snr_light, ibis_ar_snr_dark)

#Kr
kr_ibis = open_image("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/IBIS_QA_11-2022/N30IBIS53223_2022-11-21_15-28-38/capture/N30IBIS53223_2022-11-21_15-28-38.hdr")
kr_dark_frame_start = int(int(kr_ibis.metadata['autodarkstartline']))   #takes the frame number at which the dark frames begin
tint_kr = float(float(kr_ibis.metadata['tint']))

ibis_kr_mean_dark = np.average(kr_ibis[(kr_dark_frame_start+1):,:,:],axis=0)
ibis_kr_std_dark = np.std(kr_ibis[(kr_dark_frame_start+1):,:,:],axis=0,ddof=1)
ibis_kr_snr_dark = np.average(kr_ibis[(kr_dark_frame_start+1):,:,:],axis=0)/np.std(kr_ibis[kr_dark_frame_start:,:,:],axis=0,ddof=1)

ibis_kr_mean_light = np.average(kr_ibis[:kr_dark_frame_start,:,:],axis=0)
ibis_kr_std_light = np.std(kr_ibis[:kr_dark_frame_start,:,:],axis=0,ddof=1)
ibis_kr_snr_light = np.average(kr_ibis[:kr_dark_frame_start,:,:],axis=0)/np.std(kr_ibis[:kr_dark_frame_start,:,:],axis=0,ddof=1)

ibis_kr_frame = np.subtract(ibis_kr_mean_light, ibis_kr_mean_dark)
ibis_kr_std = np.subtract(ibis_kr_std_light, ibis_kr_std_dark)
ibis_kr_snr = np.subtract(ibis_kr_snr_light, ibis_kr_snr_dark)

#Ne
ne_ibis = open_image("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/IBIS_QA_11-2022/N30IBIS53225_2022-11-21_15-40-21/capture/N30IBIS53225_2022-11-21_15-40-21.hdr")
ne_dark_frame_start = int(int(ne_ibis.metadata['autodarkstartline']))   #takes the frame number at which the dark frames begin
tint_kr = float(float(ne_ibis.metadata['tint']))

ibis_ne_mean_dark = np.average(ne_ibis[(ne_dark_frame_start+1):,:,:],axis=0)
ibis_ne_std_dark = np.std(ne_ibis[(ne_dark_frame_start+1):,:,:],axis=0,ddof=1)
ibis_ne_snr_dark = np.average(ne_ibis[(ne_dark_frame_start+1):,:,:],axis=0)/np.std(ne_ibis[ne_dark_frame_start:,:,:],axis=0,ddof=1)

ibis_ne_mean_light = np.average(ne_ibis[:ne_dark_frame_start,:,:],axis=0)
ibis_ne_std_light = np.std(ne_ibis[:ne_dark_frame_start,:,:],axis=0,ddof=1)
ibis_ne_snr_light = np.average(ne_ibis[:ne_dark_frame_start,:,:],axis=0)/np.std(ne_ibis[:ne_dark_frame_start,:,:],axis=0,ddof=1)

ibis_ne_frame = np.subtract(ibis_ne_mean_light, ibis_ne_mean_dark)
ibis_ne_std = np.subtract(ibis_ne_std_light, ibis_ne_std_dark)
ibis_ne_snr = np.subtract(ibis_ne_snr_light, ibis_ne_snr_dark)

#Xe
xe_ibis = open_image("/home/rramsay/Documents/EOES/Projects/2022/2022 ARF FSF Cal Workshop/November/Day One/IBIS/IBIS_QA_11-2022/N30IBIS53228_2022-11-21_15-44-14/capture/N30IBIS53228_2022-11-21_15-44-14.hdr")
xe_dark_frame_start = int(int(xe_ibis.metadata['autodarkstartline']))   #takes the frame number at which the dark frames begin
tint_kr = float(float(xe_ibis.metadata['tint']))

ibis_xe_mean_dark = np.average(xe_ibis[(xe_dark_frame_start+1):,:,:],axis=0)
ibis_xe_std_dark = np.std(xe_ibis[(xe_dark_frame_start+1):,:,:],axis=0,ddof=1)
ibis_xe_snr_dark = np.average(xe_ibis[(xe_dark_frame_start+1):,:,:],axis=0)/np.std(xe_ibis[xe_dark_frame_start:,:,:],axis=0,ddof=1)

ibis_xe_mean_light = np.average(xe_ibis[:xe_dark_frame_start,:,:],axis=0)
ibis_xe_std_light = np.std(xe_ibis[:xe_dark_frame_start,:,:],axis=0,ddof=1)
ibis_xe_snr_light = np.average(xe_ibis[:xe_dark_frame_start,:,:],axis=0)/np.std(xe_ibis[:xe_dark_frame_start,:,:],axis=0,ddof=1)

ibis_xe_frame = np.subtract(ibis_xe_mean_light, ibis_xe_mean_dark)
ibis_xe_std = np.subtract(ibis_xe_std_light, ibis_xe_std_dark)
ibis_xe_snr = np.subtract(ibis_xe_snr_light, ibis_xe_snr_dark)


#%% Take the centre line of each frame. Use this to determine pixels to use in the spectral calibration
# I have currently -not- applied a continuum fit to any of the spectral data
# The Specim IBIS has an extremely low baseline noise. One could probably find the fit accurately without first removing continuum
# Alternatively, refer to the gamma ray photoelectron paper regarding fitting. 
# Will need to apply a flux limit to the pixel identification, as we're picking up lots of tiny lines, especially in the Xe lamp

#Ar centreline analysis
ar_centreline_frame = ibis_ar_frame[384]

#Ar pixel spectrum
flux_pixel = ar_centreline_frame.tolist()
flux_pixel = flux_pixel * u.dimensionless_unscaled
wavelengths_pixel = list(range(1, 1005)) * u.m
ibis_ar_frame_pixel = Spectrum1D(spectral_axis=wavelengths_pixel, flux=flux_pixel)

# Ar lambda aspectrum
flux_lambda = ar_centreline_frame.tolist()
flux_lambda = flux_lambda * u.dimensionless_unscaled
wavelengths_lambda = wavelengths * u.nm
ibis_ar_spectrum_lambda = Spectrum1D(spectral_axis=wavelengths_lambda, flux=flux_lambda)

# Emission line identiation Ar

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    pixel_lines_ar = find_lines_derivative(ibis_ar_frame_pixel, flux_threshold=100.0)
    pixel_lines_ar = pixel_lines_ar.to_pandas()
    
amplitude_value = []
mean_value = []
std_value = []
fwhm = []
def gaussian_fit_parameters_pixel(x):
    sub_region = SpectralRegion((x-3)*u.m, (x+3)*u.m)
    sub_spectrum = extract_region(ibis_ar_frame_pixel, sub_region)
    result = (estimate_line_parameters(sub_spectrum, models.Gaussian1D()))
    amplitude_value.append(result.amplitude.value)
    mean_value.append(result.mean.value)
    std_value.append(result.stddev.value)
    fwhm.append(result.fwhm.value)
    return mean_value, std_value, fwhm

iterate_parameters = [gaussian_fit_parameters_pixel(x) for x in pixel_lines_ar.line_center]
ar_lines_pixel_parameters = pd.DataFrame(list(zip(amplitude_value, mean_value, std_value, fwhm)),
                                     columns =['Amplitude', 'Centre Line', 'STD', 'FWHM'])


#Kr centreline analysis
kr_centreline_frame = ibis_kr_frame[384]

#Kr pixel spectrum
flux_pixel = kr_centreline_frame.tolist()
flux_pixel = flux_pixel * u.dimensionless_unscaled
wavelengths_pixel = list(range(1, 1005)) * u.m
ibis_kr_frame_pixel = Spectrum1D(spectral_axis=wavelengths_pixel, flux=flux_pixel)

# Kr lambda aspectrum
flux_lambda = kr_centreline_frame.tolist()
flux_lambda = flux_lambda * u.dimensionless_unscaled
wavelengths_lambda = wavelengths * u.nm
ibis_kr_spectrum_lambda = Spectrum1D(spectral_axis=wavelengths_lambda, flux=flux_lambda)

# Emission line identiation Kr

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    pixel_lines_kr = find_lines_derivative(ibis_kr_frame_pixel, flux_threshold=100.0)
    pixel_lines_kr = pixel_lines_kr.to_pandas()
    
amplitude_value = []
mean_value = []
std_value = []
fwhm = []
def gaussian_fit_parameters_pixel(x):
    sub_region = SpectralRegion((x-3)*u.m, (x+3)*u.m)
    sub_spectrum = extract_region(ibis_kr_frame_pixel, sub_region)
    result = (estimate_line_parameters(sub_spectrum, models.Gaussian1D()))
    amplitude_value.append(result.amplitude.value)
    mean_value.append(result.mean.value)
    std_value.append(result.stddev.value)
    fwhm.append(result.fwhm.value)
    return mean_value, std_value, fwhm

iterate_parameters = [gaussian_fit_parameters_pixel(x) for x in pixel_lines_kr.line_center]
kr_lines_pixel_parameters = pd.DataFrame(list(zip(amplitude_value, mean_value, std_value, fwhm)),
                                     columns =['Amplitude', 'Centre Line', 'STD', 'FWHM'])

# Ne centreline analysis
ne_centreline_frame = ibis_ne_frame[384]

# Ne pixel spectrum
flux_pixel = ne_centreline_frame.tolist()
flux_pixel = flux_pixel * u.dimensionless_unscaled
wavelengths_pixel = list(range(1, 1005)) * u.m
ibis_ne_frame_pixel = Spectrum1D(spectral_axis=wavelengths_pixel, flux=flux_pixel)

# Ne lambda aspectrum
flux_lambda = ne_centreline_frame.tolist()
flux_lambda = flux_lambda * u.dimensionless_unscaled
wavelengths_lambda = wavelengths * u.nm
ibis_ne_spectrum_lambda = Spectrum1D(spectral_axis=wavelengths_lambda, flux=flux_lambda)

# Emission line identiation Ne

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    pixel_lines_ne = find_lines_derivative(ibis_ne_frame_pixel, flux_threshold=100.0)
    pixel_lines_ne = pixel_lines_ne.to_pandas()
    
amplitude_value = []
mean_value = []
std_value = []
fwhm = []
def gaussian_fit_parameters_pixel(x):
    sub_region = SpectralRegion((x-3)*u.m, (x+3)*u.m)
    sub_spectrum = extract_region(ibis_ne_frame_pixel, sub_region)
    result = (estimate_line_parameters(sub_spectrum, models.Gaussian1D()))
    amplitude_value.append(result.amplitude.value)
    mean_value.append(result.mean.value)
    std_value.append(result.stddev.value)
    fwhm.append(result.fwhm.value)
    return mean_value, std_value, fwhm

iterate_parameters = [gaussian_fit_parameters_pixel(x) for x in pixel_lines_ne.line_center]
ne_lines_pixel_parameters = pd.DataFrame(list(zip(amplitude_value, mean_value, std_value, fwhm)),
                                     columns =['Amplitude', 'Centre Line', 'STD', 'FWHM'])

# Xe centreline analysis
xe_centreline_frame = ibis_xe_frame[384]

# Ne pixel spectrum
flux_pixel = xe_centreline_frame.tolist()
flux_pixel = flux_pixel * u.dimensionless_unscaled
wavelengths_pixel = list(range(1, 1005)) * u.m
ibis_xe_frame_pixel = Spectrum1D(spectral_axis=wavelengths_pixel, flux=flux_pixel)

# Ne lambda aspectrum
flux_lambda = xe_centreline_frame.tolist()
flux_lambda = flux_lambda * u.dimensionless_unscaled
wavelengths_lambda = wavelengths * u.nm
ibis_xe_spectrum_lambda = Spectrum1D(spectral_axis=wavelengths_lambda, flux=flux_lambda)

# Emission line identiation Ne

with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    pixel_lines_xe = find_lines_derivative(ibis_xe_frame_pixel, flux_threshold=100.0)
    pixel_lines_xe = pixel_lines_xe.to_pandas()
    
amplitude_value = []
mean_value = []
std_value = []
fwhm = []
def gaussian_fit_parameters_pixel(x):
    sub_region = SpectralRegion((x-3)*u.m, (x+3)*u.m)
    sub_spectrum = extract_region(ibis_xe_frame_pixel, sub_region)
    result = (estimate_line_parameters(sub_spectrum, models.Gaussian1D()))
    amplitude_value.append(result.amplitude.value)
    mean_value.append(result.mean.value)
    std_value.append(result.stddev.value)
    fwhm.append(result.fwhm.value)
    return mean_value, std_value, fwhm

iterate_parameters = [gaussian_fit_parameters_pixel(x) for x in pixel_lines_xe.line_center]
xe_lines_pixel_parameters = pd.DataFrame(list(zip(amplitude_value, mean_value, std_value, fwhm)),
                                     columns =['Amplitude', 'Centre Line', 'STD', 'FWHM'])

# Next steps -- generate one pandas file of all the pen lamps. Concetnate the data frames. 

#%% Gaussian plotting
# Goal -- to have a function which will run through each of the "XX_lines_pixel_parameters" and generate a curve 
# Then, plot each of the XX curves.
# This is not really neccessary I think. It's only to plot the Gaussian curves and see if anything strange exists. 
         
g1_435_init = models.Gaussian1D(amplitude=775.449 * u.dimensionless_unscaled, mean=15.249497*u.m, stddev=1.207624*u.m)     
g2_546_init = models.Gaussian1D(amplitude=2348.586* u.dimensionless_unscaled, mean=64.054908*u.m, stddev=1.195153*u.m)
g3_576_init = models.Gaussian1D(amplitude=669.399* u.dimensionless_unscaled, mean=78.200824*u.m, stddev=1.306486*u.m)

pixel_fit_435 = g1_435_init(wavelengths_pixel)
pixel_fit_546 = g2_546_init(wavelengths_pixel)
pixel_fit_576 = g3_576_init(wavelengths_pixel)



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


