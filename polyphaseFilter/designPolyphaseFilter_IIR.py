%load_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from polyphaseFilter.filterFactory import IIR_NthBand_PolyphaseFilter
from polyphaseFilter.filterFactory import IIR_Halfband_PolyphaseFilter
from polyphaseFilter.filterFactory import utilities
################################################################################
###############N-th band linear IIR polyphase filter############################
#%% Use Nth band linear IIR polyphase filter with a pre filter (simple low pass)
#with wp at 2*(np.pi-wp) to suppress no care bands
#Interpolator specification:
L = 9                 #Interpolation factor
As = -130             #degrees of freedom
wp = 0.8*np.pi        #Passband edge

#Pre-filter specification:
N=8
type='butterworth'
fs = 2*np.pi
wp_prefilter = 0.75*np.pi
#get filter coefficients
b_pp, a_pp = IIR_NthBand_PolyphaseFilter.nth_band_IIR_polyphaseFilter(L, wp, As)
b_pre, a_pre = IIR_NthBand_PolyphaseFilter.peakCancelling_filter(N, wp_prefilter, fs, type)
IIR_NthBand_PolyphaseFilter.plot_IIR_NthBand_filter(b_pp, a_pp, True, b_pre, a_pre)
Nm_pp, Na_pp = IIR_NthBand_PolyphaseFilter.computationalComplexity(b_pp, a_pp)
Nm_pre, Na_pre = utilities.computationalComplexity(b_pre, a_pre)
print("Multiplications Nth-Band IIR Filter: {}, Additions Nth-Band IIR Filter: {}".format(Nm_pp, Na_pp))
print("Multiplications Prefilter: {}, Additions Prefilter: {}".format(Nm_pre, Na_pre))
#%%
###############Halfband IIR polyphase filter############################
#We use an elliptic halfband IIR filter for interpolation. This is possible
#as equiripple passband and stopband characteristic can be preserved by elliptic
#filter. (Elliptic filter preserves symmertry constraint in regard to stopband and
#and passband edge and stopband and passband attenuation)
#Filter specification:
wp = 0.4*np.pi
As = 90

b, a = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False)
Nm, Na = IIR_Halfband_PolyphaseFilter.computationalComplexity(IIR_Halfband_PolyphaseFilter.minimum_ellipFilterOrder(wp, As))
gd = IIR_Halfband_PolyphaseFilter.groupDelay(b,a,'avW')
print("Multiplications: {}, Additions: {}".format(Nm, Na))
print("Group Delay: {}".format(gd))
#%%
###############FIR Halfband Polyphase Filter############################
#Specification:
As = 90
wd = 0.1
fc = 0.5
b = utilities.FIR_lowpass(As, wd, fc)
#convert filter to polyphase filter
b = utilities.FIR_polyphase_filter(b,L=2)
utilities.plot_FIR_polyphase_filter(b)
Nm, Na = utilities.computationalComplexity_polyphaseFilters(b,np.ones((L,1)))
print("Multiplications: {}, Additions: {}".format(Nm, Na))

#%%
######################Tuneable IIR Halfband Filter##############################
#%% Compare adjusted Halfband Filter with Prototype Halfband Filter
wp = 0.1*np.pi
As = 90
wc = 0.5*np.pi
b, a = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False, True)
b_new, a_new, delayTF = IIR_Halfband_PolyphaseFilter.adjust_EMQF_halfband_filter(b,a,wc,False, True)
Nm, Na = IIR_Halfband_PolyphaseFilter.computationalComplexity(IIR_Halfband_PolyphaseFilter.minimum_ellipFilterOrder(wp, As),True)
print("Multiplications: {}, Additions: {}".format(Nm, Na))
b_new[0][::2]
a_new[0][::2]
#%%Characteristic Curves
wp = 0.4*np.pi
As = 90
b, a = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False, False)
#Adjust Prototype Filter
wc = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])*np.pi
IIR_Halfband_PolyphaseFilter.plot_characteristicCurves_Adjustable_EMQF_halfband_filter(b,a,wc)










#%%
