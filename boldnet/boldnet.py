from __future__ import absolute_import, division, print_function, unicode_literals
import os, atexit, psutil, wandb, gc, time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

import pipeline, observer, config

class boldnet:
	
	def __init__(self, network_folder = None):

		self.config = config.build(network_folder) or config.build() # Load default configuration
		
		if self.config == False: # If configuration build failed
			print("Configuration failed, exiting...") 
			return # Exit

		# Initialize wrangler class for piping data
		self.wrangler = pipeline.wrangler(self.config) 

		#Initialize an observer class lens for analyzing network activity
		self.lens = observer.lens(self.config)

		self.model = None # Initialize model variable

		# Initialize dataset variables
		self.x_train, self.y_train ,self.x_val, self.y_val, self.x_test, self.y_test = [np.array(None)]*6

		atexit.register(self.save) # Set up force save model before exiting

		print("\n - BOLDnet Initialized -\n - Process PID - " + str(os.getpid()) + ' -\n')
	
	def orient(self, bids_directory, bold_identifier, label_identifier, exclude_trained = False):
		# Attach orientation variables to object for future use
		self.config.bids_directory = bids_directory
		self.config.bold_identifier = bold_identifier
		self.config.label_identifier = label_identifier

		# Grab all available subjects with fMRIPrep data
		self.subject_pool = []
		print(f"\nOrienting and generating BOLDnet lexicon for bids directory {self.config.bids_directory}...")
		
		# Generate a lexicon of all potential subjects
		if self.config.tool == 'default':
			lexicon = [item for item in glob(f"{self.config.bids_directory}/sub-*/") if os.path.isdir(item)]
		else:
			lexicon = [item for item in glob(f"{self.config.bids_directory}/derivatives/{self.config.tool}/sub-*") if os.path.isdir(item)]

		# Iterate through each subject found
		for subject in lexicon:
			split = subject.split('/')
			if split[-1]:
				subject_id = split[-1]
			else:
				subject_id = split[-2]

			# If subject was previously run, exclude from subject pool
			if exclude_trained == True and subject_id in self.config.trained_pool: 
				continue

			# Grab all available sessions
			sessions = glob(f"{subject}/ses-*/")
			

			# If none found alert about missing data
			if len(sessions) == 0:
				print(f"No sessions found for {subject_id}")
			split = sessions[0].split('/')
			if split[-1]:
				self.example_session = split[-1]
			else:
				self.example_session = split[-2]
			self.example_session = self.example_session.split('-')[-1]

			# Iterate through each session
			for session in sessions:

				# Look if the subject has usable bold file
				bold_filename = glob(f"{session}/func/{self.config.bold_identifier}")
				if len(bold_filename) == 0: # If none found, skip
					continue
				if len(bold_filename) > 1: # If too many found, skip
					bold_filename = '\n'.join(bold_filename)
					print(f"Multiple bold files found for {subject_id}...\n{bold_filename}")
					continue

				# Look if the subject has labels (regressors/classifiers)
				label_filename = glob(f"{session}/func/{self.config.label_identifier}")
				if len(label_filename) == 0: # If none found, skip
					print(f"No labels found for {subject_id}, excluding from analysis...")
					continue
				if len(label_filename) > 1: # If too many found, continue
					label_filename = '\n'.join(label_filename)
					print(f"Multiple label files found for {subject_id}...\n{label_filename}")
					continue
				

				# Add subject to subject pool if it passed criteria
				if subject_id not in self.subject_pool:
					print(f"subject: {subject_id}")
					self.subject_pool.append(subject_id) 

		# Print found subject pool to user
		self.config.subject_pool = self.subject_pool
		subject_pool = '\n'.join(self.subject_pool)
		print(f"\n\nSubject pool available for use...\n{subject_pool}")
		
	def load_data(self, subjects = [], count = 0, session = '*', fold_size = 10, shuffle = False, resolution = np.float32, activation = 'linear',  exclude_trained = True):
		# Adjust count metric if only subject names were passed in 
		if len(subjects) > 0: count = len(subjects)

		# Count how many test subjects have been passed in
		test_count = sum([subject in self.config.test_pool for subject in subjects])
		if test_count != 0 and test_count != count:
			subjects_requested = "\n".join(subjects)
			trained_subjects = "\n".join(self.config.trained_pool)
			validation_subjects = "\n".join(self.config.validation_pool)
			test_subjects = "\n".join(self.config.test_pool)
			print(f"Model test subjects have been passed in, but some subjects passed in aren't apart of the test pool. Please add these subjects into the test pool before loading them in to avoid model testing/training errors. If you did not intend to load test subjects, check you're subject loading scheme to make sure you aren't unintentionally loading in subjects twice.\n\nRequested subjects to load: {subjects_requested}\n\nTest subject pool:\n{test_subjects}\n\nTrained subjects:\n{trained_subjects}\n\nValidation subjects: {validation_subjects}")
			return False

		def clean_mem():
			garbage_count = gc.collect() # Take out the trash - prevents spike in memory consumption while loading new subjects
			time.sleep(5) # Wait some time to give the system time to manage memory

		# Delete old data to make room for new data
		if self.x_train.all() != None:
			self.x_train, self.y_train, self.x_val, self.y_val = [None] * 4
			clean_mem()
			
		if self.x_test.all() != None:
			self.x_test, self.y_test = [None] * 2
			clean_mem()

		results = self.wrangler.wrangle(self.subject_pool, subjects, count, session, fold_size, shuffle, resolution, activation, exclude_trained)
		if results != False:
			if test_count:
				self.x_test, self.y_test = results
				print(f"X test max value: {self.x_test.max()} | min value: {self.x_test.min()}")	
				print(f"y test max value: {self.y_test.max()} | min value: {self.y_test.min()}\n")
			else:
				self.x_train, self.y_train, self.x_val, self.y_val = results
				print(f"x-train shape: {self.x_train.shape}")
				print(f"X train max value: {self.x_train.max()} | min value: {self.x_train.min()}")	
				print(f"y train max value: {self.y_train.max()} | min value: {self.y_train.min()}\n")	
				print(f"X val max value: {self.x_val.max()} | min value: {self.x_val.min()}")	
				print(f"y val max value: {self.y_val.max()} | min value: {self.y_val.min()}")		
		else:
			return False
		return True

	def plan(self):
		print("\nPlanning BOLDnet model structure")
		# initialize an empty list to store layer filter counts
		self.filter_counts = []
		
		#Iterate through each layer and calculate filter counts
		convolution_size = self.config.init_filter_count
		for depth in range(self.config.convolution_depth):
			self.filter_counts.append(convolution_size)
			convolution_size = convolution_size*2 # Double the size each layer

		self.layer_shapes = []
		self.output_layers = []
		conv_shape = [self.config.data_shape[0], self.config.data_shape[1], self.config.data_shape[2]]
		conv_layer = 1
		print(f"Convolution Shape: {conv_shape}")
		for depth in range(self.config.convolution_depth):
			if depth > 0:
				conv_shape = self.calc_conv(conv_shape)
			self.layer_shapes.append(conv_shape)
			self.output_layers.append(conv_layer)
			conv_layer += 1
			if depth < self.config.convolution_depth - 1:
				conv_shape = self.calc_maxpool(conv_shape)

		self.new_shapes = []
		print(f'Layer shapes...\n{self.layer_shapes}')
		for layer_ind, conv_shape in enumerate(self.layer_shapes):
			new_shape = self.calc_convtrans(conv_shape)
			for layer in range(layer_ind,  0, -1):
				new_shape = self.calc_convtrans(new_shape)
				if layer != 1:
					new_shape = self.calc_upsample(new_shape)
			self.new_shapes.append(new_shape)

		for layer, plan in enumerate(zip(self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes)):
			print(f"Layer {layer + 1} ({plan[0]}) | Filter count: {plan[1]} | Layer Shape: {plan[2]} | Deconvolution Output: {plan[3]}")

	def build(self):
		# Create data directory
		self.wrangler.create_dir()

		# Plan and build out model structure based on config
		print('\nConstructing BOLDnet model')
		if self.config.data_shape == None: # Grab x, y, z dimension sizes from first subject 
			self.config.data_shape = self.wrangler.load_shape(self.subject_pool[0], self.example_session)
		
		self.plan() # Call plan function to predict model dimensions

		self.checkpoint_path = f"{self.config.project_directory}{self.config.model_directory}/model/ckpt.weights.h5"
		self.config.current_epoch = 0

		self.model = tf.keras.models.Sequential() # Create first convolutional layer

		print(f"depth: {self.config.convolution_depth}\nkernel stride:{self.config.kernel_size}\nkernel size:{self.config.kernel_stride}")

		# Add in initial input layer
		print(f"build shape: {self.config.data_shape}")
		self.model.add(tf.keras.Input((self.config.data_shape[0], self.config.data_shape[1], self.config.data_shape[2], 1), self.config.batch_size))
		
		for layer in range(self.config.convolution_depth): # Build the layer on convolutions based on config convolution depth indicated
			self.model.add(tf.keras.layers.Conv3D(self.filter_counts[layer], self.config.kernel_size, strides = self.config.kernel_stride, padding = self.config.zero_padding, use_bias = False, kernel_initializer = self.config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(self.config.bias)))
			self.model.add(tf.keras.layers.BatchNormalization())
			self.model.add(tf.keras.layers.LeakyReLU(self.config.negative_slope))
			if layer + 1 < self.config.convolution_depth:
				self.model.add(tf.keras.layers.MaxPooling3D(pool_size = self.config.pool_size, strides = self.config.pool_stride, padding = self.config.zero_padding, data_format = "channels_last"))
			self.model.add(SpatialAttention())
			if self.config.dropout:
				self.model.add(tf.keras.layers.Dropout(self.config.dropout))

		if self.config.multiscale_pooling:
			self.model.add(MultiScalePooling())

		self.model.add(tf.keras.layers.Flatten()) # Create top density layers
		
		for density, dense_dropout in zip(self.config.top_density, self.config.density_dropout):
			self.model.add(tf.keras.layers.Dense(density, use_bias = True, kernel_initializer = self.config.kernel_initializer, bias_initializer = tf.keras.initializers.Constant(self.config.bias))) # Density layer based on population size of V1 based on Full-density multi-scale account of structure and dynamics of macaque visual cortex by Albada et al.
			self.model.add(tf.keras.layers.LeakyReLU(self.config.negative_slope))
			if dense_dropout == True:
				self.model.add(tf.keras.layers.Dropout(self.config.dropout))
		self.model.add(tf.keras.layers.Dense(1, activation = self.config.output_activation)) #Create output layer

		self.model.build()
		self.model.summary()

		if self.config.learning_rate == 'cos':
			self.config.learning_rate = tf.keras.optimizers.schedules.CosineDecay(
				initial_learning_rate=self.config.learning_rate,
				decay_steps=10000  # Total steps to reach the minimum learning rate
			)

		if self.config.optimizer == 'Adam':
			self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.config.learning_rate, epsilon = self.config.epsilon, amsgrad = self.config.use_amsgrad, clipnorm = 1.0)
		elif self.config.optimizer == 'SGD':
			self.optimizer = tf.keras.optimizers.SGD(learning_rate = self.config.learning_rate, momentum = self.config.momentum, nesterov = self.config.use_nestrov, clipnorm = 1.0)
		else:
			self.optimizer = self.config.optimizer
			
		if self.config.output_activation == 'linear': # Compile model for regression task
			self.config.history_types = ['loss']
			self.model.compile(optimizer = self.optimizer, loss = self.config.loss, run_eagerly = True) # Compile model
		else: # Else compile model for classification
			self.config.history_types = ['accuracy']
			self.model.compile(optimizer = self.optimizer, loss = self.config.loss, metrics = self.config.history_types, run_eagerly=True) # Compile mode

		print(f'\nBOLDnet model compiled using {self.config.optimizer} - {self.model.optimizer}')

		# Check if a model already exists and load	
		if self.load_model():
			print('BOLDnet weights and history loaded...')
			
		else: # Else save new weights to checkpoint path
			if os.path.exists(self.checkpoint_path) == False or self.config.rebuild == True:
				self.model.save_weights(self.checkpoint_path)
			self.config.model_history = {}
			for history_type in self.config.history_types:
				self.config.model_history[history_type] = [] 
				self.config.model_history[f"val_{history_type}"] = [] 
			print(f"Model weights reinitialized")

		# Set up Weights and Bias logging
		if self.config.dataset != 'Study':
			self.project_name = f"{self.config.dataset} {self.config.architecture}"

		wandb.init(
			# set the wandb project where this run will be logged
			project = self.project_name,

			# track hyperparameters and run metadata with wandb.config
			config = {
				"epoch":0,
				"accuracy": self.config.history_types,
				"loss": self.config.loss,
				"val_accuracy": self.config.history_types,
				"val_loss": self.config.loss,
				"convolution_depth": self.config.convolution_depth,
				"init_filter_count": self.config.init_filter_count,
				"epochs": self.config.epochs,
				"batch_size": self.config.batch_size,
				"learning_rate": self.config.learning_rate,
				"dropout": self.config.dropout,
				"epsilon": self.config.epsilon,
				"bias": self.config.bias,
				"momentum": self.config.momentum,
				"negative_slope": self.config.negative_slope,
			},

			tags = [self.config.dataset, self.config.architecture, self.config.model_directory, "CNN"]
		)

		# Define learning rate annealing 
		cosine_annealing_callback = CosineAnnealingScheduler(self.config.learning_rate, self.config.learning_rate / 100, self.config.epochs)

		wandb_callback = WandbLogger(self.config)

		# Create a callback to the saved weights for saving model while training
		self.callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = self.checkpoint_path + '',
											save_weights_only = True,
											verbose = 1),
								cosine_annealing_callback,
								wandb_callback]
		
		# Save the configuration for the build
		self.config.save_config()

	def calc_conv(self, shape):
		return [(input_length - filter_length + (2*pad))//stride + 1 for input_length, filter_length, stride, pad in zip(shape, self.config.kernel_size, self.config.kernel_stride, self.config.padding)]

	def calc_maxpool(self, shape):
		return [(input_length - pool_length + (2*pad))//stride + 1 for input_length, pool_length, stride, pad in zip(shape, self.config.pool_size, self.config.pool_stride, self.config.padding)]

	def calc_convtrans(self, shape):
		if self.config.zero_padding == 'valid':
			return [(input_length - 1)*stride + filter_length for input_length, filter_length, stride in zip(shape, self.config.kernel_size, self.config.kernel_stride)]
		else:
			return [(input_length - 1)*stride + filter_length - 2*pad for input_length, filter_length, stride, pad in zip(shape, self.config.kernel_size, self.config.kernel_stride, self.config.padding)]

	def calc_upsample(self, shape):
		return [input_length * self.config.pool_stride[0] for input_length in shape]

	def train(self):
		# Display info about the training set
		print(f"\nx-train: {self.x_train.shape}\ny-train: {self.y_train.shape}\n\nx-val: {self.x_val.shape}\ny-val: {self.y_val.shape}")

		# Fit the model to the training set
		self.history = self.model.fit(self.x_train, self.y_train, epochs = self.config.epochs, batch_size = self.config.batch_size, validation_data = (self.x_val, self.y_val), callbacks = self.callbacks)
		print(f"Train history: {self.history}")
		self.config.current_epoch += self.config.epochs

		# Iterate through all history types and add too run history object
		for history_type in self.config.history_types: # Save training history
			self.config.model_history[history_type] += self.history.history[history_type]
			self.config.model_history[f'val_{history_type}'] += self.history.history[f'val_{history_type}']

		# Add subjects trained on to previously run batch and save
		self.config.trained_pool += self.wrangler.training_batch 
		self.wrangler.training_batch = [] # Reset the training batch
		self.config.save_config()
		self.model.save_weights(self.checkpoint_path) # Save model

	def test(self):
		# Test model by evaluating on test set
		self.history = self.model.evaluate(self.x_test, self.y_test, verbose=2)
		print(f"Test History: {self.history}")
		history_types = self.config.history_types
		if 'accuracy' in history_types:
			history_types = self.config.history_types.append('loss')
		for history_type in self.config.history_types: # Save test history
			self.config.model_history[f"val_{history_type}"] += self.history.history[f"val_{history_type}"]
		self.config.save_config()

	def predict(self, x_range = None, subject_id = ""):
		output = self.model.predict(self.x_test)
		
		if x_range == None:
			x_range = range(self.x_test.shape[0])

		if subject_id != "":
			fileend = f"_{subject_id}.png"
		else:
			fileend = ".png"

		plt.plot(x_range, [output[value] for value in x_range], color = 'orange')
		plt.plot(x_range, [self.y_test[value] for value in x_range], color = 'blue')
		plt.legend()
		plt.xlabel("Sample Volume")
		plt.ylabel("Value")
		plt.title(f"Predictions vs. Actual {self.config.output_descriptor} for fMRI Images")
		plt.savefig(f"{self.config.project_directory}{self.config.model_directory}/plots/prediction_vs_real_output{fileend}")
		plt.close()
		
		plt.scatter(self.y_test, output, color='blue', label='Outcomes')
		plt.plot(self.config.outputs, self.config.outputs, color='gray', linestyle='-', label="Unity Line")
		plt.xlabel(f'Actual {self.config.output_descriptor}')
		plt.ylabel(f'Predicted {self.config.output_descriptor}')
		plt.title(f'Scatter Plot of {self.config.output_descriptor} Predicted vs. Actual Outcomes')
		plt.savefig(f"{self.config.project_directory}{self.config.model_directory}/plots/prediction_vs_real_scatterplot{fileend}")
		plt.close()

	def save(self):
		if self.model != None:
			self.model.save_weights(self.checkpoint_path) # Save model
			self.config.save_config()
			wandb.finish()
	
	def load_model(self):
		if os.path.exists(self.checkpoint_path):
			checkpoint_dir = '/'.join(self.checkpoint_path.split('/')[:-1])
			checkpoints = os.listdir(checkpoint_dir)
			if len(checkpoints) > 0:
				self.model.load_weights(self.checkpoint_path)
				print('BOLDnet loaded successfully')
				return True
			else:
				print('BOLDnet not found...')
				return False
	
	def display_memory(self):
		# Get the memory information
		memory = psutil.virtual_memory()
		swap = psutil.swap_memory()

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

class SpatialAttention(tf.keras.layers.Layer):
	def __init__(self, kernel_size=7, **kwargs):
		super(SpatialAttention, self).__init__(**kwargs)
		self.kernel_size = kernel_size
		self.concat = tf.keras.layers.Concatenate(axis=-1)
		self.multiply = tf.keras.layers.Multiply()
		self.conv = None

	def build(self, input_shape):
		self.conv = tf.keras.layers.Conv3D(
			filters=input_shape[-1],  # Preserve the same number of channels
			kernel_size=(1, self.kernel_size, self.kernel_size),
			strides=1,
			padding="same",
			activation="sigmoid"
		)
		super(SpatialAttention, self).build(input_shape)

	def call(self, inputs):
		# Calculate average-pooling and max-pooling
		avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
		max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

		# Concatenate pooled tensors
		concat = self.concat([avg_pool, max_pool])

		# Compute spatial attention map
		attention = self.conv(concat)

		# Ensure the attention map has the same shape as the input
		attention = tf.broadcast_to(attention, tf.shape(inputs))

		# Scale by attention map (element-wise multiplication)
		output = self.multiply([inputs, attention])
		return output

	def compute_output_shape(self, input_shape):
		batch_size, depth, height, width, channels = input_shape
		return (batch_size, depth, height, width, channels)

	def get_config(self):
		config = super(SpatialAttention, self).get_config()
		config.update({"kernel_size": self.kernel_size})
		return config


class MultiScalePooling(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(MultiScalePooling, self).__init__(**kwargs)
		# Define the pooling layers with different kernel sizes
		self.pool1 = tf.keras.layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding='same')
		self.pool2 = tf.keras.layers.MaxPooling3D(pool_size = (4, 4, 4), strides = (4, 4, 4), padding='same')
		self.pool3 = tf.keras.layers.AveragePooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding='same')
		self.small_upsample = tf.keras.layers.UpSampling3D(size = (2, 2, 2))
		self.large_upsample = tf.keras.layers.UpSampling3D(size = (4, 4, 4))

	def call(self, inputs):
		# Apply the pooling operations
		pooled1 = self.pool1(inputs)
		pooled2 = self.pool2(inputs)
		pooled3 = self.pool3(inputs)

		# Reshape so all pools are the same size
		pooled2_resized = self.large_upsample(pooled2) 
		pooled3_resized = self.small_upsample(pooled3)

		# Concatenate the outputs along the channel axis
		return tf.keras.layers.Concatenate(axis=-1)([pooled1, pooled2_resized, pooled3_resized])
	
	def compute_output_shape(self, input_shape):
		# Compute output shape for the layer
		pooled_shape = (
			input_shape[0],  # Batch size
			input_shape[1] // 2,  # Depth reduced by pool1
			input_shape[2] // 2,  # Height reduced by pool1
			input_shape[3] // 2,  # Width reduced by pool1
			input_shape[4] * 3   # Concatenated channels
			)
		return pooled_shape


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
	def __init__(self, max_lr, min_lr, T_max, reset_every=None):
		super(CosineAnnealingScheduler, self).__init__()
		self.max_lr = max_lr
		self.min_lr = min_lr
		self.T_max = T_max
		self.reset_every = reset_every

	def on_train_begin(self, logs=None):
		# Ensure optimizer is properly set
		if not hasattr(self.model.optimizer, "learning_rate"):
			raise ValueError("Optimizer is not set correctly. Ensure model.compile() is called before model.fit().")

	def on_epoch_begin(self, epoch, logs=None):
		if self.reset_every and epoch % self.reset_every == 0 and epoch > 0:
			print(f"\nEpoch {epoch+1}: Resetting Cosine Annealing")
			epoch = 0  

		# Compute new LR using cosine formula
		lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * (epoch % self.T_max) / self.T_max))

		# Set the new learning rate
		self.model.optimizer.assign(self.model.optimizer.learning_rate, lr)
		print(f"\nEpoch {epoch+1}: Learning rate set to {lr:.6f}")

		# Log the learning rate
		logs = logs or {}
		logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)


class WandbLogger(tf.keras.callbacks.Callback):
	def __init__(self, config):  # Allow passing a model
		super().__init__()
		self.config = config  # Store model reference

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		# Log all metrics to W&B
		wandb.log({
			"epoch": self.config.current_epoch + epoch,
			"train_loss": logs.get("loss", None),
			"val_loss": logs.get("val_loss", None),
			"train_acc": logs.get("accuracy", logs.get("acc", None)) ,
			"val_acc": logs.get("val_accuracy", logs.get("val_acc", None)),
			"learning_rate": self.model.optimizer.learning_rate.numpy(),
			"convolution_depth": self.config.convolution_depth,
			"init_filter_count": self.config.init_filter_count,
			"epochs": self.config.epochs,
			"batch_size": self.config.batch_size,
			"dropout": self.config.dropout,
			"epsilon": self.config.epsilon,
			"bias": self.config.bias,
			"momentum": self.config.momentum,
			"negative_slope": self.config.negative_slope,
		})