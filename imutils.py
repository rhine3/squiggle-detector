'''
imutils.py
Tessa Rhinehart

Image-processing utilities for spectrograms.
All functions take an initial spectrogram and return 
the processed version of that spectrogram.
'''


from skimage.morphology import remove_small_objects
from scipy import signal, ndimage
import numpy as np

def spectrogram_bandpass(spectrogram, frequencies, low_freq, high_freq):
    '''
    Perform a "bandpass" filter on a spectrogram

    Performs a "bandpass" filter on a spectrogram using the 
    array of frequencies associated with each sample on the spectrogram.
    Finds the indices of the desired frequencies with respect
    to the array of frequencies. Returns the new spectrogram and
    frequencies, if a change to class variables is desired, 
    this must be specified like in the example as follows:

    Inputs:
        spectrogram: the spectrogram image
        frequencies: sample frequencies
        low_freq (float): the lowest frequency (Hz) to keep
        high_freq (float): the highest frequency (Hz) to keep

    Returns:
        a new spectrogram
        a new array of sample frequencies
    '''

    frequency_filter_indices = np.argwhere(
        (frequencies >= low_freq) & 
        (frequencies <= high_freq)
    ).flatten()
    new_frequencies = frequencies[frequency_filter_indices]
    new_spectrogram = spectrogram[frequency_filter_indices, :]

    return new_spectrogram, new_frequencies


def box_binary(spectrogram):
    '''
    Identify boxes in a binary spectrogram
    
    Inputs:
        spectrogram: a binary spectrogram
    '''
    # Label sub-segments on binary spectrogram
    binary_labeled, num_bin_segments = ndimage.label(spectrogram)
    
    # Put a box around each labeled sub-segment
    bounding_boxes = ndimage.find_objects(binary_labeled)

    # Convert boxes, which are slices, to nice lists
    for idx, slice_box in enumerate(bounding_boxes):
        bounding_boxes[idx] = [slice_box[0].start, slice_box[0].stop, slice_box[1].start, slice_box[1].stop]
    
    '''
    verbose = True
    if verbose:
        plotter(spectrogram = binary_labeled, title = f'{num_bin_segments} segments identified:')
    '''
    
    return bounding_boxes


######### GENERIC FUNCTIONS #########
'''
 All functions below have this format:
        
    def func(spectrogram, *kwargs):
        <do some processing>
        
        return processed_spectrogram
'''

def flip_spect(spectrogram):
    '''
    Flip spectrogram across vertical axis.
    Note this does not flip the associated freqs array
    '''
    return np.flip(spectrogram, 0)

def normalize_spect(spectrogram):
    '''
    Complete a simple value normalization of a spectrogram
    '''

    # Simple Normalization
    spectrogram_max = spectrogram.max()
    spectrogram = spectrogram / spectrogram_max
    
    return spectrogram

def binarize_by_median(spectrogram, multiplier=9.0):
    '''
    Zeroes out spectrogram values below median * multiplier
    '''

    # removeNoisePerFreqBandAndTimeFrame
    def _filter_by_scaled_median(arr, factor=multiplier):
        _temp = np.copy(arr)
        median = factor * np.median(_temp)
        for i, val in enumerate(_temp):
            if val < median:
                _temp[i] = 0.0
        return _temp

    row_median_filtered = np.apply_along_axis(_filter_by_scaled_median, 0, spectrogram)
    column_median_filtered = np.apply_along_axis(_filter_by_scaled_median, 1, spectrogram)

    return np.logical_and(row_median_filtered, column_median_filtered)


def binary_closing(spectrogram, size):
    '''
    Binary close a spectrogram
    
    Inputs:
        spectrogram: a spectrogram
        size (tuple of ints (x, y)): structure of the filter
    
    Returns: processed spectrogram
    '''
    return ndimage.morphology.binary_closing(
        spectrogram,
        structure = np.ones(size)
    )

def binary_dilation(spectrogram, size):
    '''
    Binary dilate a spectrogram
    
    Inputs:
        spectrogram: a spectrogram
        size (tuple of ints (x, y)): structure of the filter
    
    Returns: processed spectrogram
    '''
    
    return ndimage.morphology.binary_dilation(
        spectrogram,
        structure = np.ones(size)
    )

def median_filter(spectrogram, size):
    '''
    Median filter a spectrogram
        
    Inputs:
        spectrogram: a spectrogram
        size (tuple of ints (x, y)): size of the filter
    
    Returns: processed spectrogram
    '''
    return ndimage.median_filter(spectrogram, size = median_filter_size)

def small_objects(spectrogram, size):
    '''
    Remove small objects in a binary spectrogram
    
    Inputs:
        spectrogram: a binary spectrogram
        size (int): a size limit for objects to be removed
    
    Returns: processed spectrogram
    '''
    return remove_small_objects(spectrogram, size)
