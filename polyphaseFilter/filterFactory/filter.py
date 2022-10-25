import scipy as sp
import numpy as np

################################################################################
class Filter():
    '''
    Filter:
    The Filter class provides a useful functions in order to filter signals
    with the underlying filter. It is possible to change the filter coefficients
    while the filter state is preserved.

    Functions:
    _________
    process:            Filter signal with the underlying filter

    '''
    def __init__(self, b, a):
        '''
        Initializer/Constructor for Filter class.

        Paramters:
        __________
        b:                      Numerator polynomial (descending order, negative powers)
        a:                      Denominator polynomial (descending order, negative powers)

        return:
        _______
        '''
        self._b = b
        self._a = a
        self._zi = sp.signal.lfilter_zi(self._b, self._a) * 0  #no energy in the system at the beginning


    def process(self, input):
        '''
        Filter the given input samples with the underlying filter.

        Parameter:
        __________
        input:              To be processed samples

        return:
        _______
        output:             Processed samples
        '''
        output, self._zi = sp.signal.lfilter(self._b, self._a, input, zi=self._zi)
        return output

    #getter
    @property
    def b(self):
        return self._b

    @property
    def a(self):
        return self._a

    @property
    def zi(self):
        return self._zi

    #setter
    @b.setter
    def b(self, b):
        self._b = b

    @a.setter
    def a(self, a):
        self._a = a
################################################################################
class FIR_Filter:
    '''

    '''
    def __init__(self,b):
        self._b = b                             #Filter taps
        self._zi = np.zeros((len(b)))           #Internal filter state
        self._m = len(b)                        #Length of FIR filters

    def process(self, input):
        '''
        This function convolves the given input with the underlying fir filter
        response. The underlying algorithm used is overlap and save method.

        Parameters:
        ___________
        input:                  To be processed samples

        return:
        _______
        output                  Convolution result
        '''

        N = len(input)
        x = np.zeros((N+2*self._m-1))
        h = np.zeros_like(x)

        x[:N+self._m] = np.concatenate((self._zi,input))
        h[:self._m] = self._b

        #convolve in frequency domain:
        X = np.fft.rfft(x)
        H = np.fft.rfft(h)

        Y = X * H
        y = np.fft.irfft(Y, len(x))

        #update filter state:
        self._zi = np.concatenate((self._zi,input))[-self._m:]
        # print("x: {}, Y: {}, y: {}, Start: {}, End: {}, Out: {}".format(len(x), len(Y), len(y,),self._m,N+self._m, len(y[self._m:N+self._m])))
        return y[self._m:N+self._m]


    @property
    def b(self):
        return self._b

    @property
    def zi(self):
        return self._zi

    @b.setter
    def b(self, b):
        self._b = b

    @zi.setter
    def zi(self, zi):
        self._zi = zi




#%%
