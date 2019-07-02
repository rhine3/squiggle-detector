import numpy as np
from sys import platform as sys_pf

# For mac
#if sys_pf == 'darwin':
#    import matplotlib
#    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patches as patch

# Audio utils
from librosa import load, to_mono
from librosa.output import write_wav
from librosa.util.exceptions import ParameterError
from librosa.core import power_to_db
import noisereduce as nr
from more_itertools import consecutive_groups

# I/O utils
import os
import csv

# Image processing utils
from scipy import signal, ndimage
import imutils

# Misc. utils
from collections import OrderedDict
import warnings
from datetime import datetime

# More misc. utils
def plotter(
    spectrogram,
    title=None,
    upside_down = False,
    db=False, #db transform the spect
    fig_size=(15, 15), #Without this, just plots without a figsize
    boxes=None,
):

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
    if boxes:
        for box in boxes:
            rect = patch.Rectangle(
                xy = (box[2], box[0]),
                width = box[3]-box[2],
                height = box[1]-box[0],
                fill=None)
            ax.add_patch(rect)

    ax.set_aspect(spectrogram.shape[1] / (3*spectrogram.shape[0]))

    #return fig, ax
    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    Wn = [low, high]
    if low == 0:
        low = 0.00001
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)

    # Remove nans to attempt to fix librosa ParameterError
    where_are_NaNs = np.isnan(y)
    y[where_are_NaNs] = 0

    return y



########## CLASSES ##########

class Spectrogram():
    '''
    A spectrogram and its associated data.

    Inputs:
        s: spectrogram itself
        f: array of frequencies associated with each window in the spectrogram
        t: array of times associated with each window in the spectrogram
        pixel_boxes: boxes generated by performing some form of object
            detection on a spectrogram. len(pixel_boxes) == 4 with the
            following meanings:
                [low_freq, high_freq, start_time, end_time]
        freq_samp_boxes: a conversion of pixel_boxes to the below:
                [low_freq, high_freq, start_sample, end_sample]
    '''

    def __init__(self, s, f, t, pixel_boxes = None, freq_samp_boxes = None):
        self.spect = s
        self.freqs = f
        self.times = t
        self.pixel_boxes = pixel_boxes
        self.freq_samp_boxes = freq_samp_boxes



class Audio():
    def __init__(
        self,
        filename,
#        species = '',
#        helper_files_path,
#        templates_path,
#        csvs_dir,

        # Spectrogram parameters
        sample_rate = 22050.0,
        samples_per_seg = 512,
        overlap_percent = 0.75,
#        low_frequency_thresh = None,
#        high_frequency_thresh = None,

        # Noise reduction parameters
#        median_binarize_multiplier = 9.0,
#        small_obj_size = 1000.0,
#        binary_closing_size = (5, 5),
#        binary_dilation_size = (3, 5),

        # Other things
        verbosity = 1,
        audio_or_image = 'audio'
    ):
        '''
        Load samples of an object, `filename`.

        Inputs:
            filename (absolute path): the path of the file to be considered
            sample_rate (int): the sample rate at which audio should be loaded
            samples_per_seg (int): window size; how many samples to consider
                for each spectrogram window
            overlap_percent (float between 0 and 1): percentage of overlap
                between each spectrogram window
            verbosity (0, 1, 2): how much information to print
        '''

        # Set input variables
        self.filename = filename
        self.sample_rate = int(sample_rate)
        self.samples_per_seg = samples_per_seg
        self.overlap_percent = overlap_percent
        self.verbosity = verbosity
        self.mode = audio_or_image

        # Load file
        self.samples = self.load_file()
        self.samples_dn = None #For denoised samples
        self.noise_samples = None #For pure noise samples
        self.detection_samples = None #For pure noise samples

        # These will be Spectrogram objects
        self.raw = None
        self.denoised = None #denoised
        self.normalized = None
        self.binarized = None
        self.bandpassed = None
        self.processed = None

        # Other variables to be used
        self.bounding_boxes = None
        self.noise_filename = None
        self.dn_filename = None

        # File-saving-related variables to be used
        self.templates_path = None,
        self.helper_files_path = None,
        self.species = None
        self.author = None, # Your name

    ########## FUNCTIONS FOR GETTING AND SETTING CLASS ATTRS ##########

    def check_spect(self, label):
        '''
        Check that self.label and all its attributes exist


        Options for label are the same as all
        spectrogram-related instance variables:
            'raw'
            'denoised'
            'normalized'
            'binarized'
            'processed'
            'bandpassed'

        Returns:
            if exists, returns the spectrogram array
            if doesn't exist, returns False
        '''


        try:
            assert label in ['raw', 'denoised', 'normalized', 'binarized', 'processed', 'bandpassed']
        except AssertionError:
            return None

        spect = getattr(self, label)

        # Assert that all the necessary components of the spect are here
        try:
            assert spect.spect.any()
            assert spect.freqs.any()
            assert spect.times.any()
            return spect
        except AssertionError:
            return None

    def get_samples(self, label):
        '''
        Return the values of a labeled sample, self.label
        '''

        assert label in ['samples', 'samples_dn', 'noise_samples', 'detection_samples']
        return getattr(self, label)


    def get_spect(self, label):
        '''
        Return the values of self.label
        '''

        spect = self.check_spect(label)

        if (not spect) and (self.verbosity > 0):
            warnings.warn(f'self.{label} does not exist.')

        return spect


    def set_spect(
        self,
        label,
        spect=None,
        freqs=None,
        times=None,
        pixel_boxes=None,
        freq_samp_boxes=None
    ):

        '''
        Set self.label to a new Spectrogram object

        Inputs:
            label: label of spectrogram variable to set

            Variables that are optional if self.label already exists:
            spect: the desired spect to set
            freqs: the frequency array
            times: the times array
            pixel_boxes: the list of boxes in terms of pixels
            freq_samp_boxes: the list of frequency/sample boxes
        '''

        # If they're not provided, make sure they exist in the source spect
        if (spect is None) or (freqs is None) or (times is None):
            if (self.check_spect(label) == None) and (verbosity > 0):
                warnings.warn(f'self.{label} is not pre-existing. \
                    Must provide spect, frequency, and time arrays')

        # If a frequency and time array are not provided,
        # use the current frequency and time array in self.label
        if spect is None:
            spect = getattr(self, label).spect
        if freqs is None:
            freqs = getattr(self, label).freqs
        if times is None:
            times = getattr(self, label).times

        new_spect = Spectrogram(
            s=spect,
            f=freqs,
            t=times,
            pixel_boxes = pixel_boxes,
            freq_samp_boxes = freq_samp_boxes
        )

        setattr(self, label, new_spect)



    ######## FUNCTIONS FOR LOADING AUDIO/MAKING SPECTS ########

    def load_file(self):
        '''
        Load samples from an audio file

        Uses:
            self.filename: path to audio file from which to make spectrogram (optional)
            self.sample_rate: rate at which to resample audio

        Returns:
            samples: the samples from the wav file
        '''

        samples, sample_rate = load(
            self.filename,
            mono = False,  # Don't automatically load as mono, so we can warn if we force to mono
            sr = self.sample_rate, # Resample
            res_type = 'kaiser_fast',
        )

        # Force to mono if wav has multiple channels
        if samples.ndim > 1:
            samples = to_mono(samples)
            if self.verbosity > 1:
                print(
                f"WARNING: Multiple-channel file detected ({filename}). Automatically mixed to mono.")

        return samples


    def make_spect(self, source_samples='samples', dest_spect='raw'):
        '''
        Make spectrogram from an audio file

        If filename is provided, uses librosa to load samples from filename.
        Else, preloaded_samples must be provided; will generate a spectrogram
        from these samples

        Inputs:
            source_samples: class attribute for the samples. options:
                'samples', 'samples_nr', 'noise_samples', 'detection_samples'
                By default, just uses 'samples', the samples directly from
                the original audio file in self.filename
            dest_spect: class attribute for the destination spectrogram

        Returns:
            frequencies - sample frequencies
            times - time for each segment
            spectrogram - spectrogram values
        '''


        samples = self.get_samples(source_samples)
        overlap_per_seg = self.samples_per_seg * self.overlap_percent

        freqs, times, spect = signal.spectrogram(
            samples,
            self.sample_rate,
            window = 'hann',
            nperseg = self.samples_per_seg,
            noverlap = overlap_per_seg,
            nfft = 512)

        self.set_spect(dest_spect, spect = spect, freqs = freqs, times = times)

        self.flip(dest_spect)



    ######## FUNCTIONS FOR PROCESSING SPECTROGRAMS ########

    def make_bandpassed(
        self, low_freq, high_freq, source_label, dest_label='bandpassed'):
        '''
        Bandpass a spectrogram

        Inputs:
            source_label (str): the label of the source spectrogram, e.g. 'raw'
                to bandpass the self.raw spectrogram
            dest_label (str): the label to save the new bandpassed spectrogram
                to. By default, destination = 'bandpassed' so the spectrogram
                is saved to self.bandpassed
            low_freq (float): the lowest frequency (Hz) to keep
            high_freq (float): the highest frequency (Hz) to keep
        '''


        source = self.get_spect(source_label)

        new_spect, new_freqs = imutils.spectrogram_bandpass(
            spectrogram = source.spect,
            frequencies = source.freqs,
            low_freq = low_freq,
            high_freq = high_freq
        )

        self.set_spect(
            label = dest_label,
            spect = new_spect,
            freqs = new_freqs
        )

    def flip(self, source_label):
        '''
        Flip a spectrogram on the frequency axis.

        Flip the spectrogram associated with self.source_label
        across the frequency axis.
        Also flip the spectrogram's `freqs` array.
        Replaces the source Spectrogram object with a
        flipped version (i.e. flips in place)

        Inputs:
            source_label (str): label of the class attribute
                for the source spectrogram, e.g. self.raw
        '''

        source = self.get_spect(source_label)

        new_spect = imutils.flip_spect(source.spect)

        self.set_spect(
            label = source_label,
            spect = new_spect,
            freqs = np.flip(source.freqs, 0)
        )

    def normalize(self, source_label = 'raw', dest_label = 'normalized'):
        '''
        Normalize a spectrogram

        Complete a value normalization of a source
        spectrogram and save it as the destination
        spectrogram. By default, saves the new
        spectrogram in self.normalized


        Inputs:
            source_label (str): label of the class attribute
                for the source spectrogram, e.g. self.raw
        '''

        source = self.get_spect(source_label)
        new_spect = imutils.normalize_spect(source.spect)

        self.set_spect(
            label = dest_label,
            spect = new_spect,
            freqs = source.freqs,
            times = source.times
        )

    def binarize(
        self,
        source_label = 'normalized',
        dest_label = 'binarized',
        value = 9.0
    ):

        '''
        Binarize a spectrogram

        Binarize the spectrogram in self.source_label
        by median. By default, saves the new spectrogram in self.binarized.


        Inputs:
            source_label (str): label of the class attribute
                for the source spectrogram, e.g. self.normalized
            dest_label (str): label of the class attribute where
                the destination spectrogram should be saved
        '''

        source = self.get_spect(source_label)
        new_spect = imutils.binarize_by_median(source.spect, multiplier=value)

        self.set_spect(
            label = dest_label,
            spect = new_spect,
            freqs = source.freqs,
            times = source.times
        )

    def close(self, source_label, dest_label = 'processed', size=(6, 10)):
        '''
        Binary close a binary spectrogram

        Binary-closes spectrogram at self.source_label,
        saving it at self.dest_label. For instance,
        if used as a step in processing, this function
        can be used to update the self.processed spectrogram:

            audio.close(source_label = 'processed', dest_label = 'processed')


        Inputs:
            source_label (str): label of the class attribute
                for the source spectrogram, e.g. self.normalized
            dest_label (str): label of the class attribute where
                the destination spectrogram should be saved.
            size (tuple of ints (x, y)): structure of the filter
        '''

        source = self.get_spect(source_label)
        new_spect = imutils.binary_closing(source.spect, size = size)

        self.set_spect(
            label = dest_label,
            spect = new_spect,
            freqs = source.freqs,
            times = source.times
        )


    def dilate(self, source_label, dest_label = 'processed', size=(3, 5)):
        '''
        Binary dilate a binary spectrogram

        Binary-dilates spectrogram at self.source_label,
        saving it at self.dest_label. For instance,
        if used as a step in processing, this function
        can be used to update the self.processed spectrogram:

            audio.dilate(source_label = 'processed', dest_label = 'processed')


        Inputs:
            source_label (str): label of the class attribute
                for the source spectrogram, e.g. self.normalized
            dest_label (str): label of the class attribute where
                the destination spectrogram should be saved.
            size (tuple of ints (x, y)): structure of the filter
        '''

        source = self.get_spect(source_label)
        new_spect = imutils.binary_dilation(source.spect, size = size)

        self.set_spect(
            label = dest_label,
            spect = new_spect,
            freqs = source.freqs,
            times = source.times
        )


    def filter(self, source_label, dest_label = 'processed', size=(5, 3)):
        '''
        Median filter a binary spectrogram

        Median-filters spectrogram at self.source_label,
        saving it at self.dest_label. For instance,
        if used as a step in processing, this function
        can be used to update the self.processed spectrogram:

            audio.filter(source_label = 'processed', dest_label = 'processed')


        Inputs:
            source_label (str): label of the class attribute
                for the source spectrogram, e.g. self.normalized
            dest_label (str): label of the class attribute where
                the destination spectrogram should be saved.
            size (tuple of ints (x, y)): structure of the filter
        '''

        source = self.get_spect(source_label)
        new_spect = imutils.median_filter(source.spect, size = size)

        self.set_spect(
            label = dest_label,
            spect = new_spect,
            freqs = source.freqs,
            times = source.times
        )



    def smallobj(self, source_label, dest_label = 'processed', size=25):
        '''
        Remove small objects from a binary spectrogram

        Removes small objects from spectrogram at self.source_label,
        saving it at self.dest_label. For instance,
        if used as a step in processing, this function
        can be used to update the self.processed spectrogram:

            audio.smallobj(source_label = 'processed', dest_label = 'processed')


        Inputs:
            source_label (str): label of the class attribute
                for the source spectrogram, e.g. self.normalized
            dest_label (str): label of the class attribute where
                the destination spectrogram should be saved.
            size (int): object size to remove
        '''

        source = self.get_spect(source_label)
        new_spect = imutils.small_objects(source.spect, size = size)

        self.set_spect(
            label = dest_label,
            spect = new_spect,
            freqs = source.freqs,
            times = source.times
        )

    ########## BOXING/NOISE REDUCTION FUNCTIONS ##########
    def box(self, box_from, box_on):
        '''
        Box a spectrogram

        Get boxes from a binary spectrogram, and
        associate them with another spectrogram.
        Generates boxes in terms of frequency and
        time boundaries; also converts boxes in terms
        of frequency and sample boundaries.

        Inputs:
            box_from: will box based on self.box_from.spect
            box_on: will add box boundaries to the
                self.box_on Spectrogram object
        '''

        source = self.get_spect(box_from)

        # Box the binary spectrogram
        px_boxes = imutils.box_binary(source.spect)
        freqs = source.freqs
        times = source.times

        # Convert the pixel boxes to frequency/sample boxes
        # (see Spectrogram class documentation)
        fs_boxes = []
        for box in px_boxes:

            # This is the same between both representations
            low_freq = freqs[len(freqs)-box[0]-1]
            high_freq = freqs[len(freqs)-box[1]-1]

            # account for rounding error in boxing
            start_time = box[2]-1
            end_time = box[3]-1

            if start_time > len(times)-1: start_time = len(times)-1
            if end_time > len(times) - 1: end_time = len(times) - 1

            start_sample = int(round(times[start_time] * self.sample_rate))
            end_sample = int(round(times[end_time] * self.sample_rate))

            # Add to list of fs boxes
            fs_boxes.append([low_freq, high_freq, start_sample, end_sample])

        # Update the desired spect with these limits
        self.set_spect(
            box_on,
            pixel_boxes = px_boxes,
            freq_samp_boxes = fs_boxes
        )

    def save_noise_and_detections_files(self, source_label):
        '''
        Save noise and detections files for a boxed file


        '''

        source = self.get_spect(source_label)
        fs_boxes = source.freq_samp_boxes

        # Validate: make sure that file save destinations are set
        try:
            assert self.helper_files_path
            assert self.species
        except AssertionError as e:
            raise ValueError(
'Must use Audio.set_save_dests(helper_files_path = <>, species = <>) \
before saving noise and detection files')

        # Validate: make sure that fs_boxes are calculated
        if fs_boxes is None:
            raise ValueError(
'Noise and detection files can only be saved for Spectrogram objects with \
frequency and sample boxes computed with Audio.box()')


        # Create list of samples identified as noise (True) or not (False)
        use_as_noise_samples = np.full(self.samples.shape[0], True, dtype=bool)
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


        # Create arrays of samples of noise and detections
        noise_data = []
        detection_data = []
        for idx in range(len(start_end_samples)):

            # Append samples from the current noise
            noise_start = int(start_end_samples[idx][0])
            noise_stop = int(start_end_samples[idx][1])
            noise_data.extend(self.samples[noise_start:noise_stop])

            # Append samples from the previous detection
            if noise_start > 0:
                # For sounds at the beginning of the file
                if idx == 0:
                    detection_data.extend(self.samples[0:noise_start])

                # For sounds between two noises
                else:
                    sound_start = int(start_end_samples[idx-1][1])
                    sound_end = noise_start
                    detection_data.extend(self.samples[sound_start:sound_end])

        # Write wavs and store their filenames
        path_for_helper_files = os.path.join(self.helper_files_path, self.species)

        self.noise_filename = self.save_samples_as_wav(
            samples = noise_data,
            path_to_save = path_for_helper_files,
            filename_to_save = 'noise')

        detections_filename = self.save_samples_as_wav(
            samples = detection_data,
            path_to_save = path_for_helper_files,
            filename_to_save = 'detections')

    def audacity_noise_reduce(self):
        '''
        Noise-reduce self.samples and save file.

        Noise-reduce self.samples using Audacity's algorithm.
        Requires that self.noise_samples is already computed.
        '''

        # Ensure that noise reduction happened
        try:
            assert self.noise_filename is not None
        except AssertionError:
            raise ValueError(
'The noise-reduced samples (Audio.samples_nr) must be loaded first; \
use Audio.save_noise_and_detections_files')

        # Set self.noise_samples if not already loaded
        if self.noise_samples is None:
            self.noise_samples, sample_rate = load(
                self.noise_filename,
                mono = True,
                sr = self.sample_rate, # Resample
                res_type='kaiser_best',
            )

        # Noise-reduce self.samples
        self.samples_dn = nr.reduce_noise(
            audio_clip = self.samples,
            noise_clip = self.noise_samples,
            #verbose = bool(self.verbosity)
        )


        # Save denoised files
        path_for_helper_files = os.path.join(self.helper_files_path, self.species)
        self.dn_filename = self.save_samples_as_wav(
            samples = self.samples_dn,
            path_to_save = path_for_helper_files,
            filename_to_save = 'denoised')


    ########## I/O RELATED FUNCTIONS ##########

    def set_save_dests(
        self,
        templates_path = None,
        helper_files_path = None,
        species = None,
        author = None, # Your name
    ):
        '''
        Set one or multiple save destinations

        Sets the save destinations for whatever kwargs
        are provided. If the kwargs are not provided,
        then will not change the current save
        destination.

        The keywords to this argument are exactly
        the name of the class attributes, e.g.
            aud.set_save_dets(species = 'EATO')
        will set attribute self.species = 'EATO'.

        The file save structure is as follows:

            helper_files_path/
            |---species/
            |   |---cat_num_detections.wav
            |   |---cat_num_noise.wav
            |   |---cat_num_denoised.wav


            templates_path/
            |---species/
            |   |---species_0.wav
            |   |---species_1.wav
            |   |---species_2.wav
            |   |---...
            |   |---species.csv

        where species.csv has the format:
            FILENAME,TYPE,DURATION_s,LOW-BOUND,HIGH-BOUND,BOUND-UNIT,SOURCE-FILE,EXTRACTOR,EXTRACTION-METHOD,DATE,DEPRECATED,CALL-TYPE
            species_0.wav,audio,<seconds>,<Hz>,<Hz>,Hz,<path-to-source-file>,author,autoboxer,<date>,0,unknown
        '''
        # TODO: what to do about cat_num?

        # Path to save templates
        if templates_path: self.templates_path = templates_path

        # Path to save helper files
        if helper_files_path: self.helper_files_path = helper_files_path

        # Species folder within templates folder and helper files folder
        if species: self.species = species

        # Extractor for CSV field
        if author: self.author = author


    def save_samples_as_wav(self, samples, path_to_save, filename_to_save):
        '''
        Saves a wav in the given path

        Inputs:
            samples: samples to save
                (assumed to have same sample rate as self.sample_rate)
            path_to_save: folder in which to save file
            filename_to_save: filename with which to save file,
                e.g. 'detections' will be saved as 'detections.wav'

        Returns:
            file_path: the path to the saved file
        '''

        # Make path if necessary
        try:
            os.makedirs(path_to_save)
        except FileExistsError:
            pass


        # Full path & filename by which wav should be saved
        file_path = os.path.join(path_to_save, f'{filename_to_save}.wav')

        try:
            write_wav(file_path, np.array(samples), self.sample_rate)
        except ParameterError: # librosa.util.exceptions.ParameterError
            print(f'Skipping {file_path} due to ParameterError')
            return None

        if self.verbosity: print(f'Saved files to {file_path}')

        return file_path


    def identify_segments_audio(
        self,
        box_source, # spectrogram
        sample_source, #samples
    ):

        '''
        Break audio into segments

        Inputs:
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


        CSV format:
            FILENAME, TYPE,DURATION_s, LOW-BOUND, HIGH-BOUND, BOUND-UNIT, SOURCE-FILE, EXTRACTOR, EXTRACTION-METHOD, DATE, DEPRECATED, CALL-TYPE
            species_0.wav,audio,<seconds>,<Hz>,<Hz>,Hz,<path-to-source-file>,author,autoboxer,<date>,0,unknown
        '''
        full_templates_path = os.path.join(self.templates_path, self.species)
        try:
            os.makedirs(full_templates_path)
        except FileExistsError:
            pass

        assert(self.author)

        # Open .csv to save segment information
        csv_path = os.path.join(full_templates_path, f'{self.species}.csv')
        if os.path.exists(csv_path):
            mode = 'a' # append if already exists
        else:
            mode = 'w+' # make a new file if not
        open_file = open(csv_path, mode)
        writer = csv.writer(open_file)
        if mode == 'w+': # add a header
            writer.writerow(['FILENAME', 'TYPE', 'DURATION_s', 'LOW-BOUND', 'HIGH-BOUND', 'BOUND-UNIT', 'SOURCE-FILE', 'EXTRACTOR', 'EXTRACTION-METHOD', 'DATE', 'DEPRECATED', 'CALL-TYPE'])

        # Get bounding boxes and times
        box_source = self.get_spect(box_source)
        bounding_boxes = box_source.freq_samp_boxes
        print('Freq_samp_boxes:', bounding_boxes)
        times = box_source.times
        try:
            assert bounding_boxes is not None
        except AssertionError:
            raise ValueError(f'self.{source_label} does not have bounding boxes')

        # Get samples
        samples = self.get_samples(sample_source)

        now = datetime.now()
        date = now.strftime("%Y%m%d")
        # Extract samples for each box
        segment_number = 0
        for idx, box in enumerate(bounding_boxes):
            # Get individual values from box
            high_freq, low_freq, start_sample, end_sample = box

            # Extract those samples from the audio
            segment_samples = samples[start_sample: end_sample]

            # Bandpass filter the samples above and below the box limits
            filtered_samples = butter_bandpass_filter(
                segment_samples,
                low_freq,
                high_freq,
                self.sample_rate
            )

            # save samples
            filename_to_save = f'{self.species}_{segment_number}'
            segment_number += 1
            detection_filename = self.save_samples_as_wav(
                filtered_samples,
                full_templates_path,
                filename_to_save
            )

            # This only happens if the detection was successfully saved
            if detection_filename is not None:
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

                writer.writerow(
                    [f'{filename_to_save}.wav',
                     'audio',
                     duration,
                     low_freq,
                     high_freq,
                     'Hz',
                     self.filename,
                     self.author,
                     'autoboxer',
                     date,
                     0,
                     'unknown'])

        open_file.close()





# TODO: implement these

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