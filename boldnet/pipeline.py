from __future__ import absolute_import, division, print_function, unicode_literals
import os, csv, shutil, time, random, psutil
import numpy as np
import nibabel as nib
from glob import glob
from scipy.stats import zscore

class wrangler:
	def __init__(self, config):
		self.config = config
		return
	
	def wrangle(self, subject_pool, subjects = [], count = 0, session = '*', fold_size = 10, shuffle = False, resolution = np.float16, activation = 'linear',  exclude_trained = False):
		# Wrangle subjects into training and validation batches, using the fold size
		# to determine how many subjects to put in the training set compared to validation
		#
		# NOTE: This function is not typically used for grabing testing subjects, consider
		# Using the load subject function directly for testing

		print(f"Wrangling {count} subjects data (session {session}) with fold of size {fold_size}.")

		print(f"Training portion: {round(((fold_size - 2)/fold_size)*100, 1)}%\nValidation portion: {round((1/fold_size)*100, 1)}%\nTesting portion: {round((1/fold_size)*100, 1)}%\n")
	
		# Handle misformatted subjects
		if isinstance(subjects, list) == False:
			if isinstance(subjects, str):
				subjects = subjects.split(',')
			else:
				subjects = [str(subjects)]
		
		# Update count if not passed in
		if count == 0: 
			count = len(subjects)
			if count == 0: # Check that enough subjects were passed in
				print("Not enough subjects passed into wrangle, must pass in at least 1 subject or a count of 1 specified...")
				return 

		# Define the subject count found
		subject_count = 0
		
		# Run through each subject and load data
		self.training_batch = []
		
		# Define variables to hold subject data
		images = np.array([])
		labels = np.array([])

		train_indices = []
		val_indices = []

		# While we haven't reached out subject count goal (minimum of 2)
		while subject_count < count:

			# If we've run out of subjects but need more
			if subjects == []: 
				subject = self.config.subject_pool.pop(random.randint(0, len(self.config.subject_pool) - 1))
			else: # Grab the first subject in our passes subject list
				subject = subjects.pop(0)

			# Check if subject is in excluded list
			if subject in self.config.excluded_pool: 
				print(f'Excluding subject {subject} from run...\n')
				continue

			# If subject previously run and we're not retraining\
			if exclude_trained == True and sum([subject == trained_subject for trained_subject in self.config.trained_pool]) > 0: # check is subject previously run
				print(f"Subject {subject} previously run, skipping subject...\n")
				continue

			# Assess subject position in fold
			viewed_count = len(self.config.trained_pool) + len(self.config.validation_pool) + len(self.config.test_pool) + len(self.training_batch)
			position = viewed_count % fold_size
			if subject in self.config.test_pool: # If requesting a test pool subject
				position = -1 # Mark as test subject load

			if position < (fold_size - 1) or position == -1: # If not a test subject during training or a requested test subject
				image, label = self.load_subject(subject, session, shuffle, resolution, activation)
				
				# Replace any NaN in BOLD images
				nan_indices = np.isnan(image) # Find all NaN values
				image[nan_indices] = 1e-6 # Replace with a small value/epsilon

				# Remove images with irrelavent labels
				nan_indices = np.isnan(label)
				label = label[~nan_indices]
				image = image[~nan_indices]

				if len(label) == 0: # If no images available
					print(f"Skipping {subject}, no labels...\n")
					continue # Skip subjects
				if position != 0: # Append images
					try: # appending the images to the image object
						images = np.append(images, image, axis = 0)
						labels = np.append(labels, label)
					except: # initialize a new image object
						images = image
						labels = label

			# Move through the fold
			if position == 0: # Grab the validation image
				self.validation_image = image
				self.validation_labels = label
				self.config.validation_pool.append(subject)
				subject_count += 1
				continue # Continue without incrementing subject count

			elif position == (fold_size - 1): # If a test subject fold position
				self.config.test_pool.append(subject) # Add to test pool
				print(f"Subject {subject} added to test pool...\n")
				subject_count += 1
				continue

			elif position == -1: # If requested test subject(s) "position"
				return images, labels # Return all images and labels

			else: # Grab training and validation indices indices
				train_indices += [ind for ind in range(len(train_indices), len(train_indices) + len(label) - 1 )]
				
				val_start = int(round(self.validation_image.shape[0] * ((position-1)/8), 0))

				val_end = int(round(self.validation_image.shape[0] * (position/8), 0))
				if val_end > self.validation_image.shape[0]: val_end = self.validation_image.shape[0]

				val_indices += [ind for ind in range(val_start, val_end)]

				self.training_batch.append(subject)

			subject_count += 1

		# Check if images loaded had any volumes in then
		if images.shape[0] == 0:
			return False
		
		if position >= 0: # If a training batch, split and return using indices
			print(f'Train indices count: {len(train_indices)}\nValidation indices count: {len(val_indices)}\nLength of labels: {len(labels)}\n')
			return images[train_indices,:,:,:], labels[train_indices], self.validation_image[val_indices,:,:,:], self.validation_labels[val_indices]
		
		elif position == -1: # If test batch
			return images, labels # Return all images and labels loaded


	def load_subject(self, subject, session, shuffle = False, resolution = np.float16, activation = 'linear',  load_affine = False, load_header = False):
		
		print(f"Loading subject {subject}...")
	
		image_file = self.load_image(subject, session, memory_map = True)
	
		# Grab images header/meta data
		header = image_file.header 
		self.config.header = header

        # Grab data
		images = np.array(image_file.get_fdata(dtype=resolution))

		# Reshape image to have time dimension as first dimension and add channel dimension
		images = images.reshape((images.shape[3], images.shape[0], images.shape[1], images.shape[2], 1))
		self.config.data_shape = images.shape[1:-1]
		print(f"Subject {subject} data shape: {images.shape}")

        # Normalize image data
		images = self.normalize(images)

		# Load labels
		labels = self.load_labels(subject, session, resolution)
		labels = np.array(labels)
		print(f'Subject {subject} label shape: {labels.shape}\n')
        
        # If no labels ended up being loaded, exit
		if labels.shape[0] == 0 or labels.shape[0] != images.shape[0]:
			print(f"WARNING: {subject} labels are empty or labels length does not match image length...\n Image shape - {image.shape}\n Label shape - {labels.shape})\n")
			return [], []

        #  Shuffle images and labels if configured
		if subject not in self.config.test_pool:
			if self.config.shuffle == True or shuffle == True:
				images, labels = self.shuffle(images, labels)
		return images, labels
		
	def load_image(self, subject, session, memory_map = False):
		# look for all relavent image file
		if self.config.tool == "default":
			image_filenames = glob(f"{self.config.bids_directory}/{subject}/ses-{session}/func/{self.config.bold_identifier}")
		else:
			image_filenames = glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.config.bold_identifier}")
		if image_filenames: #  if no image files found, return empty handed
			if len(image_filenames) > 1:
				print(f"WARNING: Multiple BOLD files found for subject {subject} session {session} using BOLD identifier {self.config.bold_identifier} provided, using first file found which could cause instability later in NeuroNet...")
			print(f"Loading BOLD image {image_filenames[0]}")
			return nib.load(image_filenames[0], mmap=memory_map) # Load images
		else:
			print(f'No images found for {subject} ses-{session} using BOLD identifier {self.config.bold_identifier}')
			return False
	
	def load_labels(self, subject, session, resolution):
		# Grab relavent label filenames
		if self.config.tool == "default":
			label_filenames = glob(f"{self.config.bids_directory}/{subject}/ses-{session}/func/{self.config.label_identifier}")
		else:
			label_filenames = glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/{subject}/ses-{session}/func/{self.config.label_identifier}")
		if label_filenames: # If we grabbed more/less than one label file, exit
			if len(label_filenames) > 1:
				print(f"WARNING: Mutliple labels found for {subject} session {session}, grabbing first label file found however this may cause instability in NeuroNet...")
			label_filename = label_filenames[0]
		else: # Alert of no labels found...
			label_filenames = '\n'.join(label_filenames)
			print(f"No label files found for subject {subject} session {session}:\n {label_filenames}")
			return False
		
		print(f"Loading label file {label_filename}")
		# initialize empty labels and read in labels
		labels = []
		with open(label_filename, 'r') as label_file:
            # Handle text file case
			if label_filename[-4:] == '.txt': 
				labels = label_file.readlines()
				labels = [str(label).lower() for label in ''.join(labels).split('\n') if label != '']
            
            # Handle csv file case
			if label_filename[-4:] == '.csv':
				labels = []
				csv_reader = csv.reader(label_filename)
				for row in csv_reader:
					labels.append(str(row).lower())

			# Handle tsv file case
			if label_filename[-4:] == '.tsv':
				tsv_reader = csv.reader(label_filename, '\t')
				for row in tsv_reader:
					labels.append(str(row).lower())
		
		# Configure model for labels provided, i.e. classification or regression models
		nan_texts = [None, 'nan', 'none', 'na']
		if '.' in labels[0]:
			labels = [float(label) if label not in nan_texts else np.nan for label in labels] # format labels as float points
			
			# Modify configuration for regression
			self.config.output_activation = 'linear'
			self.config.kernel_initializer = 'glorot_uniform'
			self.config.loss = 'mse'
			
		else:
			labels = [int(label) if label not in nan_texts else np.nan for label in labels] # Format labels as integers
			# Modify configuration for classification
			self.config.kernel_initializer = 'he_normal'
			if self.config.output_activation == None:
				if len(set(labels)) > 2:
					self.config.loss = 'sparse_categorical_crossentropy'
					self.config.output_activation = 'softmax'
				else:
					self.config.loss = 'binary_crossentropy'
					self.config.output_activation = 'sigmoid'
		
		labels = np.array(labels, dtype=resolution) # Convert to a numpy array

		return labels

	def load_shape(self, subject, session, image = None):
		if image == None:
			image = self.load_image(subject, session)
		return image.header.get_data_shape()

	def shuffle(self, images, labels):
		# Get sample indices
		indices = np.arange(images.shape[0])
		
        # Shuffle indices
		np.random.shuffle(indices)

		# Rearrange images and labels using shuffled indices
		images = images[indices, :, :, :, :]
		
		labels = labels[indices]
		
		return images, labels
	
	def normalize(self, array, norm_min = -1):
		# Z score the BOLD data before normalizing
		array = zscore(array, axis=0)

		# Find array min and max
		array_min = np.min(array)
		array_max = np.max(array)

		if self.config.norm_min == -1:
			array = 2 * (array - array_min) / (array_max - array_min) - 1

		elif self.config.norm_min == 0:
			array = (array - array_min) / (array_max - array_min)

		else:
			print("ERROR: norm_min configuration can only be set to 0 (BOLD normalized ranging from [0, 1]) or -1 (normalized to [-1, 1])")
			return None

		# Pass back normalized array
		return array


	def mean_filter(self, image, filter_size):
		return
	
	def median_filter(self, image, filter_size):
		return

	def gaussian_filter(self, image, filter_size, sigma = 1):
		return
	
	def bilateral_filter(self, image, filter_size):
		return
	
	def laplacian_filter(self, image, filter_size, sign = 'positive'):
		return

	def dilation_filter(self, image, filter_size):
		return

	def erosion_filter(self, image, filter_size):
		return

	def display_memory(self, title):
		# Get the memory information
		memory = psutil.virtual_memory()
		swap = psutil.swap_memory()

		print(f"|------ {title} -------|")

		# Display RAM usage
		print("---- System Memory (RAM) ----")
		print(f"Total RAM: {memory.total / (1024 ** 3):.2f} GB")
		print(f"Available RAM: {memory.available / (1024 ** 3):.2f} GB")
		print(f"Used RAM: {memory.used / (1024 ** 3):.2f} GB")
		print(f"RAM Usage: {memory.percent}%")

		# Display swap memory usage
		print("\n---- Swap Memory ----")
		print(f"Total Swap: {swap.total / (1024 ** 3):.2f} GB")
		print(f"Used Swap: {swap.used / (1024 ** 3):.2f} GB")
		print(f"Free Swap: {swap.free / (1024 ** 3):.2f} GB")
		print(f"Swap Usage: {swap.percent}%")


	def create_dir(self):
		print(self.config.outputs_category)
		first_dir = self.config.outputs_category # Create lists of all directory levels for extraction outputs
		second_dir = [f'Layer_{str(layer)}' for layer in range(1, self.config.convolution_depth)]
		third_dir = ["DeConv_Feature_Maps", "DeConv_CAM"]
		fourth_dir = ["GB", "SM", "SSM"]
		if self.config.rebuild == True:
			if os.path.exists(f'{self.config.project_directory}{self.config.model_directory}/') == True:
				print(f'\nRun directory {self.config.project_directory}{self.config.model_directory} already exists, clearing directory...')
				shutil.rmtree(f'{self.config.project_directory}{self.config.model_directory}')
				time.sleep(1)
		else: # If not resetting model
			if os.path.isdir(f'{self.config.project_directory}{self.config.model_directory}/') == True: # If model exists
				print(f"Run directory already exists for {self.config.model_directory}, if you intended to train a fresh model consider altering run directory or deleting the model saved in {self.config.model_directory} (i.e. setting config.reset_model = True)")
				return

		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/model')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/plots')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/jack_knife')
		os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/jack_knife/probabilities')
		for first in first_dir:
			os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}')
			for second in second_dir:
				os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}')
				for third in third_dir:
					os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}/{third}')
					for fourth in fourth_dir:
						os.mkdir(f'{self.config.project_directory}{self.config.model_directory}/{first}/{second}/{third}/{fourth}')
		print(f'\nResult directories generated for {self.config.model_directory}\n')
