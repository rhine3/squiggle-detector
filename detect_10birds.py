import os
import random
from squiggle_detector import *
from collections import OrderedDict

def squiggle_detect(
    # File I/O parameters
    filename,
    species,
    helper_files_path,
    templates_path,
    csvs_dir,
    
    # Spectrogram parameters
    sample_rate,
    samples_per_seg,
    overlap_percent,
    low_frequency_thresh = None,
    high_frequency_thresh = None,
    
    # Noise reduction parameters
    median_binarize_multiplier = 9.0,
    small_obj_size = 1000.0,
    binary_closing_size = (5, 5),
    binary_dilation_size = (3, 5),
    
    # Other things
    verbosity = 0,
    audio_or_image = 'audio'
):
    '''
    Squiggle-detect on a spectrogram
    
    Inputs:
        filename (str): the path of the file to squiggle-detect on
        sample_rate (int): sample rate for audio processing
        save_dir (str): 
        species (str): the species of this file
            used to organize results into specices
        cat_num (int): the catalogue number of this file
            unique number identifying the file, used to
            organize results by source file
    
    Results:
        Puts files into the following directory structure:
        
            helper_files_path/
            |---species/
            |   |---cat_num_detections.wav
            |   |---cat_num_noise.wav
            |   |---cat_num_denoised.wav



            templates_path/
            |---species/
            |   |---cat_num1.wav
            |   |---cat_num2.wav
            |   |---...
            |
            |---csvs_dir/
            |   |---species.csv
        
        where species.csv has the format:
            Filename,Duration,LowFreq,HighFreq
            cat_num1.wav,<seconds>,<Hz>,<Hz>
    '''
    
    # Load file
    samples, sample_rate = load_file(
        filename = filename,
        sample_rate = sample_rate)

    # Make spectrogram
    freqs, times, spect = make_spect(
        samples = samples,
        samples_per_seg = samples_per_seg,
        overlap_percent = overlap_percent,
        sample_rate = sample_rate)
    
    if verbosity > 0:
        plotter(spect, upside_down=True, title=f'Initial spectrogram for {filename}', db=True)

    # Remove undesired frequencies
    #low_frequency_thresh = 173
    #high_frequency_thresh = 10033
    #spect, f = spectrogram_bandpass(spect, f, low_frequency_thresh, high_frequency_thresh)
    #if verbosity > 1:
    #    plotter(power_to_db(spect), upside_down=True, title='Frequencies removed')

    ### STEP 1 ###
    # Normalize spectrogram
    normalized_spect = normalize_spect(spect)

    # Binarize by median filtering
    binary_spect = binarize_by_median(normalized_spect, multiplier = median_binarize_multiplier)
    
    # Image processing noise reduction
    desired_steps = OrderedDict({
        'binary_closing':binary_closing_size,
        'binary_dilation':binary_dilation_size,
        'small_objects':small_obj_size
    })
    if verbosity > 1:
        binary_processed = image_processing_dict( 
            spectrogram = binary_spect,
            params = desired_steps,
            plot_func = plotter
        )
    else:
        binary_processed = image_processing_dict( 
            spectrogram = binary_spect,
            params = desired_steps
            #plot_func = plotter
        )

    # Find bounding boxes in processed spect
    bounding_boxes = box_binary(binary_processed, verbose = verbosity)

    # Use bounding boxes to save noise and detections files
    noise_filename = save_noise_and_detections_files(
        binary_spectrogram=binary_processed,
        bounding_boxes=bounding_boxes,
        original_filename=filename,
        samples = samples,
        freqs = freqs,
        times = times,
        sr = sample_rate,
        newdir = helper_files_path,
        subdir = species
    )

    # Noise-reduce the samples
    samples_nr = audacity_noise_reduce(noise_file=noise_filename, audio_samples=samples)
    freqs_nr, times_nr, spect_nr = make_spect(
        samples_nr, samples_per_seg=samples_per_seg, overlap_percent=overlap_percent)
    spect_nr = normalize_spect(spect_nr)
    if verbosity > 1:
        plotter(spect_nr, db=True, title='noise-reduced')

    # Save denoised sound file
    wav_writer(
        samples = samples_nr,
        sample_rate=sample_rate,
        suffix='denoised',
        orig=filename,
        newdir=helper_files_path,
        subdir=species
    )
    
    ## STEP 2 ##
    # If using Audacity noise reduction and binary_spect was not made from 
    # the noise-reduced spectrogram, it will be slightly larger than the 
    # noise-reduced spect. Make them the same size by just selecting the 
    # first part of the binary_spect:
    if spect_nr.shape != binary_spect.shape:
        shape_difference = np.subtract(spect_nr.shape, binary_spect.shape)
        new_dims = np.add(binary_spect.shape, shape_difference)
        # TODO: centering this might work better than just taking the beginning
        binary_spect_to_use = binary_spect[0:new_dims[0], 0:new_dims[1]]
    else:
        binary_spect_to_use = binary_spect

    # Save each extracted segment
    csv_folder = os.path.join(templates_path, csvs_dir)
    try: os.mkdir(csv_folder)
    except: pass
    csv_path = os.path.join(csv_folder, f'{species}.csv')
    
    if audio_or_image == 'audio':
        identify_segments_audio(
            filename = filename,
            samples = samples_nr,
            bounding_boxes = bounding_boxes,
            sample_rate = sample_rate, 
            freqs = freqs_nr,
            times = times_nr,
            basepath = templates_path,
            template_dir = species,
            csv_path = csv_path)

    # Identify segments on spectrogram
    else:
        if verbosity > 0:
            segs = identify_segments(
                spect_nr, binary_spect_to_use, bounding_boxes, plot_func=plotter, margin = 3)#, method='min')
        else:
            segs = identify_segments(
                spect_nr, binary_spect_to_use, bounding_boxes, margin = 3)#, plot_func=plotter, method='min')
        
        return segs

# TODO: add headers to csvs




import csv
assess_file = 'app/assess_short.csv'
count = 0
use_files = []
with open(assess_file, 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        if line[1] == 'accept':
            use_files.append(line[0])
len(use_files)

helper_files_path = '/Users/tessa/Code/detect-towhee/squiggle-detector/detections/helper_files'
templates_path = '/Users/tessa/Code/detect-towhee/squiggle-detector/detections/templates'
csvs_dir = 'foreground_csvs'
samples_per_seg = 512
overlap_percent = 0.75
sample_rate = 22050.0
debug=False

for file in use_files:
    species = file.split('/')[-2]
    
    squiggle_detect(
        # File I/O parameters
        filename = file,
        species = species,
        helper_files_path = helper_files_path,
        templates_path = templates_path,
        csvs_dir = csvs_dir,

        # Spectrogram parameters
        sample_rate = sample_rate,
        samples_per_seg = samples_per_seg,
        overlap_percent = overlap_percent,
        low_frequency_thresh = None,
        high_frequency_thresh = None,

        # Noise reduction parameters
        median_binarize_multiplier = 9.0,
        small_obj_size = 1000.0,
        binary_closing_size = (5, 5),
        binary_dilation_size = (3, 5),

        # Other things
        verbosity = 2,
        audio_or_image = 'audio'
    )
