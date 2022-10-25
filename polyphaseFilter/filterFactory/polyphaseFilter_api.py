
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod, ABCMeta

from polyphaseFilter.filterFactory import IIR_Halfband_PolyphaseFilter
from polyphaseFilter.filterFactory import IIR_NthBand_PolyphaseFilter
from polyphaseFilter.filterFactory import utilities
from polyphaseFilter.filterFactory import filter
################################################################################
class PolyphaseFilter(metaclass=ABCMeta):
    '''
    This polyphase filter class provides the api for the different types of avail-
    able polyphase filter (IIR/FIR).
    '''

    def __init__(self,L):
        self._L = L
        return

    @abstractmethod
    def process(self, input, **kwargs):
        pass

    @property
    @abstractmethod
    def gd(self):
        pass

################################################################################
class IIR_polyphaseFilter(PolyphaseFilter):
    '''
    This class defines polyPhaseFilter for a given interpolation/decimation
    factor. Additionally, it provides a prefilter which can be used to adjust the
    cutoff frequency of the IIR polyphase filter.
    '''

    def __init__(self, L, wc=None):
        '''
        This function initializes the class instance.

        Parameter:
        __________
        L:                      Interpolation/Decimation factor
        wc:                     Normalized circular cutoff frequency
        '''
        super().__init__(L)
        #Initialize prototype filter (used for prefilter and for polyphase block)
        wp = 0.4*np.pi
        As = 90
        b, a = IIR_Halfband_PolyphaseFilter.EMQF_Halfband_filter(wp, As, False, False)
        self._prototype_b = b
        self._prototype_a = a

        #Initialize prefilter
        self._preFilter = False
        self._aa_branch0, self._aa_branch1, self._aa_branch1_delayTF = None, None, None
        if wc != None:
            self._preFilter = True
            b, a, delayTF = IIR_Halfband_PolyphaseFilter.adjust_EMQF_halfband_filter(b,a,wc,False,False)
            self._aa_branch0 = filter.Filter(b[0],a[0])                         #1. Branch of EMQF halfband filter
            self._aa_branch1 = filter.Filter(b[1],a[1])                         #2. Branch of EMQF halfband filter
            self._aa_branch1_delayTF = filter.Filter(delayTF[0],delayTF[1])     #Transformed delay element of 2. Branch

        #Initialize EMQF Polyphase Filter (if L = 1 --> empty polyphase filter block)
        self._pp_filters0 = []
        self._pp_filters1 = []
        if self._L != 1:
            assert np.ceil(np.log2(L)) == np.floor(np.log2(L)), "Interpolation factor must be power of 2 and greater than 1!"
            self._subL = 2
            Nstages = int(np.round(np.log2(L),1))

            self._pp_filters0 = [filter.Filter(self._prototype_b[0][::2],self._prototype_a[0][::2]) for i in range(Nstages)]       #1. Branch of EMQF halfband filter
            self._pp_filters1 = [filter.Filter(self._prototype_b[1][::2],self._prototype_a[1][::2]) for i in range(Nstages)]       #2. Branch of EMQF halfband filter


        #Determine group delay
        nPrototype = len(self._pp_filters0)                   #number of used prototype filters
        if self._preFilter:
            nPrototype += 1

        self._gd = nPrototype * IIR_Halfband_PolyphaseFilter.groupDelay(self._prototype_b, self._prototype_a, 'av')

        return

    def process(self, input):
        '''
        This function filters the given input samples by the prefilter if instantiated and the polyphase filter block.

        Parameters:
        ___________
        input:                      To be filtered input samples
        '''

        #Apply prefilter
        B = len(input)
        output = np.zeros((B*self._L))
        if self._preFilter:
            output[::self._L] = 0.5*( self._aa_branch0.process(input) + self._aa_branch1.process(self._aa_branch1_delayTF.process(input)))
        else:
            output[::self._L] = input

        stride = self._L
        for filter0, filter1 in zip(self._pp_filters0, self._pp_filters1):
            stride //=self._subL
            output_ptr = output[::stride]                                                   #Note: Slicing is based on call by reference
            output_ptr[1::self._subL] = filter1.process(output_ptr[::self._subL])           #sets the inserted zeros
            output_ptr[::self._subL] = filter0.process(output_ptr[::self._subL])            #resets the given values

        return output

    def update_prefilter(self, wc):
        '''
        Update cutoff frequency of prefilter.

        Parameters:
        ___________
        wc:                     New normalized circular cutoff frequency
        '''

        assert self._preFilter == True, "Prefilter not instantiated!"

        b, a, delayTF = IIR_Halfband_PolyphaseFilter.adjust_EMQF_halfband_filter(self._prototype_b,self._prototype_a,wc,False,False)
        #fill new filter values into filter
        self._aa_branch0.b = b[0]
        self._aa_branch0.a = a[0]
        self._aa_branch1.b = b[1]
        self._aa_branch1.a = a[1]
        self._aa_branch1_delayTF.b = delayTF[0]
        self._aa_branch1_delayTF.a = delayTF[1]

        return

    @property
    def gd(self):
        return self._gd

################################################################################
class FIR_polyphaseFilter(PolyphaseFilter):
    '''
    This class defines polyphase filter for a given interpolation/decimation
    factor. This polyphase filter is based on a FIR prototype filter.
    '''

    def __init__(self, L, wc, **kwargs):
        super().__init__(L)

        #prototype filter
        self._b_prototype = utilities.FIR_lowpass(wc=wc, **kwargs)
        #transform FIR filter into polyphase FIR filter
        b_poly = utilities.FIR_polyphase_filter(self._b_prototype,L)

        #initialize parallel filters:
        self._pp_filters = [filter.FIR_Filter(b_poly[i,:]) for i in range(L)]

        self._gd = utilities.FIR_filter_groupDelay(self._b_prototype, 'av')

    def process(self, input, switch=None):
        '''
        This function upsamples/downsamples the given input by using the underlying
        lowpass filter given in polyphase structure. If additionally switch boundaries
        are given the input is divided into subinput, where each subinput is mapped to
        one subfilter. If the number of switch boundaries is greater than the number of
        filters the mapping starts again from the beginning. This mode is used for inter
        polation. If no switch boundaries are given, the filters are running parallel.

        Parameters:
        ___________
        input:                      To be filtered input
        switch:                     Order of used filters [0,...,end]

        return:
        _______
        output:                     Processed samples
        '''

        if switch==None:
            output = np.zeros((self._L*len(input)))
            for i,filter in enumerate(self._pp_filters):
                output[i::self._L] = self._L*filter.process(input)
        else:
            output = np.zeros((len(input)))
            for start, end in zip(switch[0:-1],switch[1,:]):
                #get current filter state
                output[start:end] = self._pp_filters.process(input)

        return output

    @property
    def gd(self):
        return self._gd

    @property
    def b_prototype(self):
        return self._b_prototype



################################################################################
class polyphaseFactory():
    @staticmethod
    def get_polyphaseFilter(polyphaseType, **kwargs):
        if polyphaseType=='FIR':
            #create FIR polyphase filter
            return FIR_polyphaseFilter(**kwargs)
        elif polyphaseType=='IIR':
            #create IIR polyphase filter
            return IIR_polyphaseFilter(**kwargs)
        else:
            raise Exception('Polyphase filter type not supported!')







#%%
