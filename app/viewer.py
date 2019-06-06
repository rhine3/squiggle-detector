import tkinter as tk
from tkinter import messagebox
import numpy as np

from squiggle_detector import load_file, make_spect

from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

root= tk.Tk() # create window

width = 800
height = 350
canvas1 = tk.Canvas(root, width = width, height = height)
canvas1.pack()

specs = ['/Volumes/seagate3/xeno-canto/20181207/pipilo-erythrophthalmus/mp3s/320359.mp3',
 '/Volumes/seagate3/xeno-canto/20181207/hylocichla-mustelina/mp3s/101365.mp3',
 '/Volumes/seagate3/xeno-canto/20181207/cardinalis-cardinalis/mp3s/125284.mp3',
 '/Volumes/seagate3/xeno-canto/20181207/vireo-olivaceus/mp3s/20985.mp3',
 '/Volumes/seagate3/xeno-canto/20181207/seiurus-aurocapilla/mp3s/142342.mp3']
spec_id = 0

def make_spec():
	global specs
	global spec_id
	
	sample_rate = 22050
	samples_per_seg = 512
	overlap_percent = 0.75

	try:
		print(f'making {specs[spec_id]}')
	except:
		print('No more specs!')
		root.destroy()
	
	# Load file
	samples, sample_rate = load_file(
		filename = specs[spec_id],
		sample_rate = sample_rate)

	# Make spectrogram
	freqs, times, spect = make_spect(
		samples = samples,
		samples_per_seg = samples_per_seg,
		overlap_percent = overlap_percent,
		sample_rate = sample_rate)

	
def wipe_and_continue():
	global specs
	global spec_id

	print(f'wiping {specs[spec_id]}')
	spec_id += 1
	make_spec()

def add_to_list():
	global specs
	global spec_id

	print (f'we like {specs[spec_id]}')
	wipe_and_continue()
	pass

button1 = tk.Button (root, text='yes',command = add_to_list)
button2 = tk.Button (root, text='no',command = wipe_and_continue)
canvas1.create_window(width/2-20, 2*height/3, window=button1)
canvas1.create_window(width/2+20, 2*height/3, window=button2)
  
root.mainloop()
