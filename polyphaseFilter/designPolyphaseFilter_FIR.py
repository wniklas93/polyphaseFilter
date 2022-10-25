%load_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


from polyphaseFilter.filterFactory import utilities
from polyphaseFilter.filterFactory import polyphaseFilter_api
from polyphaseFilter.filterFactory import filter
################################################################################
#Signal Specification
dur = 10                                     #Duration in [sec]
fs = 48000                                   #Sampling rate in [Hz]
f0 = 1000                                    #Frequency of sinusoid in [Hz]
t = np.linspace(0,dur,fs*dur)
s = np.sin(2*f0*np.pi*t)
#Filter Spec:
L = 10
fs = 48000
wc = 2*np.pi*18000/fs/L
wd = 2*np.pi*6000/fs/L

pp_filter = polyphaseFilter_api.polyphaseFactory.get_polyphaseFilter('FIR', wc=wc, wd=wd, fs=fs, L=L)
wBins, H = sp.signal.freqz(pp_filter.b_prototype,[1])

fig = plt.figure(figsize=(15,5))
plt.plot(wBins, 20*np.log10(np.abs(H)))
#%%
################################################################################
#Block by Block processing:
fir_filter = filter.FIR_Filter(pp_filter.b_prototype)
B = 128
numBlocks = len(s) // B
y = np.zeros((B*numBlocks))
for b in range(numBlocks):
    y[b*B:(b+1)*B] = fir_filter.process(s[B*b:B*(b+1)])


fig = plt.figure(figsize=(15,5))
plt.plot(y[8090:8100])
fig = plt.figure(figsize=(15,5))
Y = np.fft.rfft(y)
plt.plot(20*np.log10(np.abs(Y)))


#%%
