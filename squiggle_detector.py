import numpy as np
from scipy import signal, ndimage
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from librosa import load, to_mono
from librosa.output import write_wav
from librosa.util.exceptions import ParameterError
import noisereduce as nr
from skimage.morphology import remove_small_objects
from more_itertools import consecutive_groups
import os
from scipy.signal import butter, lfilter
from librosa.core import power_to_db
from collections import OrderedDict
import csv

def plotter(
    spectrogram,
    title=None,
    upside_down = False,
    db=False, #db transform the spect
    fig_size=(15, 15), #Without this, just plots without a figsize
):
    return 
    
    # Plot, flip the y-axis
    if fig_size:
        fig, ax = plt.subplots(1, figsize=fig_size)
    else:
        fig, ax = plt.subplots(1)#, figsize=(10, 10))
    if db:
        ax.imshow(power_to_db(spectrogram), cmap=plt.get_cmap("gray_r"))
    else:
        ax.imshow(spectrogram, cmap=plt.get_cmap("gray_r"))
    if upside_down:
        ax.set_ylim(ax.get_ylim()[::-1])
    if title:
        ax.set_title(title)
    ax.set_aspect(spectrogram.shape[1] / (3*spectrogram.shape[0]))

    #plt.show()

def load_file(filename, sample_rate=22050):
    '''
    Load samples from an audio file
    
    Inputs:
        filename: path to audio file from which to make spectrogram (optional)
        sample_rate: rate at which to resample audio
    
    Returns:
        samples: the samples from the wav file
        sample_rate: the sample rate from the wav file
    '''
    
    samples, sample_rate = load(
        filename,
        mono=False,  # Don't automatically load as mono, so we can warn if we force to mono
        sr=sample_rate, # Resample
        res_type='kaiser_best',
    )
    
    # Force to mono if wav has multiple channels
    if samples.ndim > 1:
        samples = to_mono(samples)
        #print(
        #    f"WARNING: Multiple-channel file detected ({filename}). Automatically mixed to mono."
        #)
        
    return samples, int(sample_rate)


def make_spect(samples, samples_per_seg, overlap_percent, sample_rate=22050):
    '''
    Make spectrogram from an audio file
    
    If filename is provided, uses librosa to load samples from filename. Else,
    preloaded_samples must be provided; will generate a spectrogram from these samples
    
    Inputs:
        samples: mono samples loaded from an audio file
        samples_per_seg: window size for spectrogram
        overlap_percent: overlap percent for spectrogram (between 0 and 1)
        sample_rate: sample rate for audio
        preloaded_samples: (optional) already-loaded samples
    
    Returns:
        frequencies - sample frequencies
        times - time for each segment
        spectrogram - spectrogram values
    '''

    
    overlap_per_seg = samples_per_seg * overlap_percent

    frequencies, times, spectrogram = signal.spectrogram(
        samples,
        sample_rate,
        window='hann',
        nperseg=samples_per_seg,
        noverlap=overlap_per_seg,
        nfft=512)
    
    return frequencies, times, spectrogram


def spectrogram_bandpass(spectrogram, frequencies, low_freq, high_freq):
    '''
    Perform a "bandpass" filter on a spectrogram
    
    Performs a "bandpass" filter on a spectrogram using the 
    array of frequencies associated with each sample on the spectrogram.
    Finds the indices of the desired frequencies with respect
    to the array of frequencies.
    
    
    Inputs:
        spectrogram: the spectrogram image
        frequencies: sample frequencies
        low_freq: the lowest frequency to keep
        high_freq: the highest frequency to keep
        
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




def normalize_spect(spectrogram):
    '''
    Complete a simple value normalization of a spectrogram
    '''
    

    # Simple Normalization (this also flips the image)
    spectrogram_max = spectrogram.max()
    spectrogram = spectrogram / spectrogram_max
    
    # Flip image back to correct orientation
    spectrogram = np.flip(spectrogram, 0)
    
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


def image_processing_dict(
    spectrogram, 
    params = OrderedDict({
    'binary_closing':(6, 10),
    'binary_dilation':(3, 5), 
    'median_filter':(5, 3),
    'small_objects':25,
}),
    plot_func = None):
    
    '''
    Use 4 noise reduction algorithms
    
    Inputs: (defaults are as in Barry Moore's original code)
        spectrogram: the spectrogram
        params: OrderedDict where the order of the keys specifies the order 
            of the functions, and the values specify the parameters for the functions.
        plot_func:  function to use to display spectrograms.
            Should take keyword arguments `spectrogram`, `title`
            If not provided, spectrograms will not be plotted
    
    Defaults
    '''
    from collections import OrderedDict
    
    if plot_func == None:
        def plot_func(*args, **kwargs):
            return None
    
    # Create wrapper functions to supply params as simple ordered args
    def binary_closing(spect, struct):
        return ndimage.morphology.binary_closing(spect, structure = np.ones(struct))
    def binary_dilation(spect, struct):
        return ndimage.morphology.binary_dilation(spect, structure = np.ones(struct))
    def median_filter(spect, size):
        return ndimage.median_filter(spect, size = median_filter_size)
    def small_objects(spect, size):
        return remove_small_objects(spect, size)
    
    # Translate keys to actual functions
    funcs = {
        'binary_closing':binary_closing,
        'binary_dilation':binary_dilation,
        'median_filter':median_filter,
        'small_objects':small_objects
    }
    
    # Input validation
    assert(set(params.keys()).issubset(funcs.keys()))
    
    # Go through the steps in order (ordered b/c params is an OrderedDict)
    result = spectrogram
    for step in params.keys():
        func = funcs[step]
        result = func(result, params[step])
        title = f'{step}, {params[step]}'
        plot_func(spectrogram = result, title = title, fig_size=(10, 10))
    return result


def box_binary(spectrogram_binary, verbose = False):
    '''
    Identify boxes in a binary spectrogram
    '''
    # Label sub-segments on binary spectrogram
    binary_labeled, num_bin_segments = ndimage.label(spectrogram_binary)
    
    # Put a box around each labeled sub-segment
    bounding_boxes = ndimage.find_objects(binary_labeled)

    # Convert boxes, which are slices, to nice lists
    for idx, slice_box in enumerate(bounding_boxes):
        bounding_boxes[idx] = [slice_box[0].start, slice_box[0].stop, slice_box[1].start, slice_box[1].stop]
    
    if verbose:
        plotter(spectrogram = binary_labeled, title = f'{num_bin_segments} segments identified:')
    
    return bounding_boxes

def wav_writer(samples, sample_rate, suffix, orig, newdir=None, subdir=None, verbose=True):
    '''
    Saves a wav in same place as original .wav file
    
    Inputs:
        samples: new samples to save
        orig: original filename of .wav file
        suffix: suffix for the new filename
        newdir: a new directory to use instead of the original wav file's path
        subdir: name of a subdirectory to make in the original or new directory
        verbose: whether or not to print filename
        
    Returns:
        the new filename
    '''
    
    filesplit = os.path.split(orig)
    
    # Get the path in which to save the wav
    if newdir:
        base_path = newdir
    else:
        base_path = filesplit[0] #Same path as the original file
    
    if subdir:
        base_path = os.path.join(base_path, subdir)
    
    # Make path if necessary
    try: 
        os.mkdir(base_path)
    except FileExistsError:
        pass
    
    # Get the name by which to save the wav
    file_name = filesplit[1]
    base_name = f'{os.path.splitext(file_name)[0]}_{suffix}.wav'
    
    # Full path & filename by which wav should be saved
    file_path = os.path.join(base_path, base_name)
    
    try: write_wav(file_path, np.array(samples), sample_rate)
    except ParameterError: # librosa.util.exceptions.ParameterError
        print(f'Skipping {file_path} due to ParameterError')

    if verbose: print(f'Saved files to {file_path}')
    
    return file_path


def box_to_fs(box, freqs, times, sr):
    '''
    Convert an np box to freq/sample
    
    Inputs:
        box (array): [low_freq_np, high_freq_np, start_time, end_time]
        times (array): array of times where times[start_time] = the desired start time
        freqs (array): array of freqs where freqs[len(freqs) - low_freq_np] = the desired frequency
            i.e. frequencies are a reversed list of what frequency each window in the np spectrogram belongs to
    
    Returns:
        array: [low_freq, high_freq, start_sample, end_sample]
    '''
    
    low_freq = freqs[len(freqs)-box[0]-1]
    high_freq = freqs[len(freqs)-box[1]-1]
    
    # account for rounding error in boxing
    start_time = box[2]-1
    end_time = box[3]-1
    if start_time > len(times)-1:
        start_time = len(times)-1
    if end_time > len(times) - 1:
        end_time = len(times) - 1
    start_sample = int(round(times[start_time] * sr))
    end_sample = int(round(times[end_time] * sr)) 
    
    return [low_freq, high_freq, start_sample, end_sample]
    
    
def save_noise_and_detections_files(
    binary_spectrogram,
    bounding_boxes,
    original_filename,
    samples,
    freqs,
    times,
    sr,
    newdir=None,
    subdir=None
):
    '''
    Save a file containing the noise
    
    Inputs:
        binary_spectrogram: a 0/1 spectrogram
        bounding_boxes: boxes around detections in binary spect
        original_filename: original filename of the file
        samples: samples from original file
        freqs: frequency list for the original spect used to generate bounding boxes
        times: time list for the original spect used to generate bounding boxes
        sr: sample rate
        newdir: a place to save new files 
            if newdir == None (default), saves in same dir as original_filename
        subdir: a subdirectory in which to save new files 
            if subdir == None (default), defaults to behavior of newdir, not creating a subdirectory
            if subdir == 'orig', saves in a directory named after original_filename
            if subdir is something else, creates a subdir under newdir
        
    Returns:
        new_filename (string): a path to the filename of the noise file
    '''
    
    # Convert bounding boxes to frequency/sample # boxes
    fs_boxes = []
    for box in bounding_boxes:
        fs_boxes.append(box_to_fs(box, freqs, times, sr))

    # Create list of samples identified as noise (True) or not (False)
    use_as_noise_samples = np.full(samples.shape[0], True, dtype=bool)
    for fs_box in fs_boxes:
        time_start = fs_box[2]
        time_end = fs_box[3]
        use_as_noise_samples[time_start:time_end] = False

    # Create list of x boundaries for noisy spots
    data = np.where(use_as_noise_samples)[0] #indices where noise == True
    start_end_samples = []
    for group in consecutive_groups(data):
        group_list = list(group)
        start_end_samples.append([group_list[0], group_list[-1]])
    
    
    # Write noise to file
    noise_data = []
    detection_data = []
    for idx in range(len(start_end_samples)):
        # Append samples from the current noise
        noise_start = int(start_end_samples[idx][0])
        noise_stop = int(start_end_samples[idx][1])
        noise_data.extend(samples[noise_start:noise_stop])

        # Append samples from the previous detection
        if noise_start > 0:
            # For sounds at the beginning of the file
            if idx == 0:
                detection_data.extend(samples[0:noise_start])

            # For sounds between two noises
            else:
                sound_start = int(start_end_samples[idx-1][1])
                sound_end = noise_start
                detection_data.extend(samples[sound_start:sound_end])

    # Write wavs and store their filenames
    if subdir=='orig':
        subdir = os.path.splitext(os.path.basename(original_filename))[0]
    noise_filename = wav_writer(noise_data, sr, suffix='noise', orig = original_filename, subdir=subdir, newdir=newdir)
    detections_filename = wav_writer(detection_data, sr, suffix='detections', orig = original_filename, subdir=subdir, newdir=newdir)
    
    return noise_filename



def audacity_noise_reduce(noise_file, audio_samples, verbose=False):
    '''
    Uses a sample file of noise to noise-reduce like Audacity does
    
    Inputs:
        noise_file: path to noise file
        audio_samples: samples to be noise-reduced
        verbose: whether or not to print graphs
    
    Returns: 
        noise-reduced samples.
        for some reason makes a smaller spectrogram? #TODO
    '''
    
    noise_samples, sample_rate = load(
        noise_file,
        mono=False,  # Don't automatically load as mono, so we can warn if we force to mono
        sr=22050.0, # Resample
        res_type='kaiser_best',
    )
    # perform noise reduction
    reduced_noise_samples = nr.reduce_noise(audio_clip=audio_samples, noise_clip=noise_samples, verbose=verbose)

    
    return reduced_noise_samples


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    Wn = [low, high]
    if low == 0:
        low = 0.00001
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    

def identify_segments_audio(
    filename,
    samples,
    bounding_boxes,
    sample_rate, 
    freqs,
    times,
    basepath,
    template_dir,
    csv_path
):
    
    '''
    Break audio into segments
    
    Inputs:
        filename (string): filename of the original file, 
            used to give each segment a name
        samples: samples for the noise-reduced file
        bounding_boxes (list of lists):
            where each sublist is a detection box of the form:
            [low_freq_np, high_freq_np, start_time, end_time]
        freqs_nr (list): frequencies of the noise-reduced spectrogram
        sample_rate (int): sample rate
        times (list): times of the noise-reduced spectrogram
        basepath (path): path to put the detections
        template_dir (string): subdirectory of basepath in which to save templates
        csv_path (string): path to file to append detections to. does NOT add csv header.
    '''
    
    if os.path.exists(csv_path):
        mode = 'a' # append if already exists
    else:
        mode = 'w+' # make a new file if not
    
    open_file = open(csv_path, mode)
    writer = csv.writer(open_file)

    for idx, box in enumerate(bounding_boxes):
        # convert box, which is in terms of numpy array, to sample number and frequency number
        high_freq, low_freq, start_sample, end_sample = box_to_fs(box, freqs, times, sample_rate)

        # extract those samples from the audio
        segment_samples = samples[start_sample: end_sample]

        # bandpass filter the samples above and below the box limits
        filtered_samples = butter_bandpass_filter(segment_samples, low_freq, high_freq, sample_rate)

        # save samples
        detection_filename = wav_writer(
            filtered_samples,
            sample_rate,
            f'detection{idx}',
            orig = filename,
            newdir = basepath,
            subdir = template_dir)
        
        # write information to csv
	# account for rounding error in boxing
        if box[2] > len(times) - 1:
            start_time = times[-1]
        else:
            start_time = times[box[2]]
        if box[3] > len(times) - 1:
            end_time = times[-1]
        else:
            end_time = times[box[3]]
        duration = end_time - start_time
        writer.writerow([detection_filename, duration, low_freq, high_freq])
        
    open_file.close()
    
    
def identify_segments_image(
    spectrogram_original,
    spectrogram_binary,
    bounding_boxes,
    margin = 2,
    method = None,
    plot_func = None
):
    '''
    Break spectrogram into segments
    
    Inputs:
        spectrogram_original: the original raw spectrogram from which to pull segments
        spectrogram_binary: a binarized, noise-reduced spectrogram
        bounding_boxes: 
        margin: approximate pixel margin around each detection;
            detections closer than this margin will be considered one segment
        method: 'min' or None. 
            - if 'min', will replace non-needed values with a minimum 
              (return a spect that blends in to background)
            - otherwise, will return a spect where detections don't blend in 
              with background (background will be np.nans)
        plot_func: function to use to display spectrograms.
            Should take keyword arguments `spectrogram`, `title`
            If not provided, spectrograms will not be plotted
        
    '''  

    # Iteratively add boxes to spect
    box_image = np.full(spectrogram_binary.shape, 0)
    for b in bounding_boxes:
        y_min = b[0] - margin
        y_max = b[1] + margin
        x_min = b[2] - margin
        x_max = b[3] + margin
        box_image[y_min:y_max, x_min:x_max] = 1
        
    # Label unique boxes on boxed spectrogram   
    box_labeled, num_box_segs = ndimage.label(box_image)
    
    
    # Extract original spect values using binary spectrogram
    # Replace all "black" (1s) with the value from the original spectrogram
    # Replace all "white" (0s) with either minimum_value or np.nan
    if method == 'min': # 
        valid_values = spectrogram_original[spectrogram_binary]
        minimum_value = np.min(valid_values)*100000
        extracted_squiggles = np.where(spectrogram_binary, spectrogram_original, minimum_value)
    else: # 
        extracted_squiggles = np.where(spectrogram_binary, spectrogram_original, np.nan)

    segs = []
    # For each labeled box extent
    for num in range(1,num_box_segs+1):
        # Just get the box for this label
        box_num = np.where(box_labeled, box_labeled == num, 0)
        
        # Grab values from the original spectrogram within this box
        if method == 'min':
            segs.append(np.where(box_num, extracted_squiggles, minimum_value))
        else:
            segs.append(np.where(box_num, extracted_squiggles, 0))#np.nan))

    
    steps = {
        f'{len(bounding_boxes)} boxes with margin of {margin}:':box_image,
        f'{num_box_segs} segments identified:':box_labeled,
        'extracted squiggles to segment (logged)':np.log(extracted_squiggles),
    }
    
    if plot_func:
        for key in steps.keys():
            plot_func(spectrogram = steps[key], title = key)
    
    return segs



def cropper(spectrogram):
    '''
    Crop a segment
    
    Only works with `min` option
    '''
    
    min_val = spectrogram[0][0]
    non_empty_columns = np.where(spectrogram.max(axis=0) != min_val)[0]
    non_empty_rows = np.where(spectrogram.max(axis=1) != min_val)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    cropped_spectrogram = spectrogram[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    return cropped_spectrogram
