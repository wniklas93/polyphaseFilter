%load_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from polyphaseFilter.filterFactory import IIR_Halfband_PolyphaseFilter
from polyphaseFilter.filterFactory import polyphaseFilter_api
from polyphaseFilter.filterFactory import utilities
###############################################################################
#Filter specification:
wp = 0.4*np.pi
As = 90

b, a = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False, False)

#Signal Specification
dur = 10                                     #Duration in [sec]
fs = 48000                                   #Sampling rate in [Hz]
f0 = 1000                                    #Frequency of sinusoid in [Hz]
t = np.linspace(0,dur,fs*dur)
s = np.sin(2*f0*np.pi*t)
#Upsampling:
s_up = np.zeros((len(s)*2))
zi = sp.signal.lfilter_zi(b[0][::2], a[0][::2]) * 0  #no energy in the system at the beginning
s_up[0::2], _ = sp.signal.lfilter(b[0][::2], a[0][::2], s, zi=zi)

zi = sp.signal.lfilter_zi(b[1][::2], a[1][::2]) * 0  #no energy in the system at the beginning
s_up[1::2], _ = sp.signal.lfilter(b[1][::2], a[1][::2], s, zi=zi)

gd_av = IIR_Halfband_PolyphaseFilter.groupDelay(b,a,'av')
gd_avw = IIR_Halfband_PolyphaseFilter.groupDelay(b,a,'avW')

fig = plt.figure(figsize=(10,5))
plt.plot(s_up[0:20], label="Upsampled Signal")
plt.axvline(gd_av, label="Averaged Group Delay", linestyle="--", color="b")
plt.axvline(gd_avw, label="Weighted Averaged Group Delay", linestyle="--", color="g")
plt.legend()


#%%
###############################################################################
#########################Use Polyphase API for IIR Resampler###################
p_filter = polyphaseFilter_api.polyphaseFactory.get_polyphaseFilter(L=16, polyphaseType='IIR')

output = p_filter.process(s)
gd_av = p_filter.gd

fig = plt.figure(figsize=(10,5))
plt.plot(output[0:1000], label="Upsampled Signal by IIR")
plt.axvline(gd_av, label="Averaged Group Delay", linestyle="--", color="b")
plt.legend()
#%%
################################################################################
#########################Use Polyphase API for FIR Resampler####################
L = 10
fs = 48000
wc = 2*np.pi*18000/fs/L
wd = 2*np.pi*6000/fs/L

p_filter = polyphaseFilter_api.polyphaseFactory.get_polyphaseFilter(L=L, polyphaseType='FIR', wc=wc, wd=wd, fs=fs)
gd_av = p_filter.gd
output = p_filter.process(s)
fig = plt.figure(figsize=(10,5))
plt.plot(output[:1000], label="Upsampled Signal by FIR")
plt.legend()

Y = np.fft.rfft(output)
fig = plt.figure(figsize=(10,5))
plt.plot(20*np.log(np.abs(Y)), label="Upsampled Signal by FIR")
