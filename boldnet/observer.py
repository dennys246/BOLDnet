from __future__ import absolute_import, division, print_function, unicode_literals
import random	
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
from numpy import asarray
from nilearn import plotting, datasets, surface
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU
from random import randint, randrange


class lens:
	def __init__(self, config):
		self.config = config
		return

	def plot_accuracy(self, i = 1):
		print("\nEvaluating NeuroNet model accuracy & loss...")
		for history_type in self.config.history_types:		# Evaluate the model accuracy and loss
			plt.plot(self.config.model_history[history_type], label=history_type)
			plt.plot(self.config.model_history[f"val_{history_type}"], label = f'validation {history_type}')
			plt.xlabel('Epoch')
			plt.ylabel(history_type)
			plt.legend(loc='upper right')
			plt.ylim([0, 1])
			title = f"~learnig rate: {str(self.config.learning_rate)} ~negative_slope: {str(self.config.negative_slope)} ~bias: {str(self.config.bias)} ~optimizer: {self.config.optimizer}"
			if self.config.optimizer == 'SGD':
				title = f"{title} ~epsilon: {str(self.config.epsilon)}"
			else:
				title = f"{title} ~momentum: {str(self.config.momentum)}"
			plt.title(title)
			plt.savefig(f"{self.config.project_directory}{self.config.model_directory}/model_{history_type}.png")
			plt.close()

	def observe(self, interest):
		self.images = []
		ind = 0
		while self.images == []: # Iterate till you find a subject
			self.images, self.labels, self.header, self.affine = self.load_subject(self.config.subject_pool[ind], session = '1', load_affine = True, load_header = True)
			ind += 1

		self.sample_label = -2
		while self.sample_label <= interest - 0.25 or self.sample_label >= interest + 0.25: # Grab next sampsle that is the other category
			rand_ind = random.randint(0, self.images.shape[0] - 1)
			self.sample_label = self.labels[rand_ind] # Grab sample label
		self.sample = self.images[rand_ind, :, :, :, :] # Grab sample volume
#		self.sample = self.sample.reshape((1, self.sample.shape[0], self.sample.shape[1], self.sample.shape[2], self.sample.shape[3]))

		for category, label in zip(self.config.outputs_category, self.config.outputs):
			if interest == label:
				self.category = category

		print(f"\nObserving {self.category} outcome structure")

		print(f"\nExtracting {interest} answer features from NeuroNet convolutional layers...")
		self.output_layers, self.filter_counts, self.layer_shapes, self.new_shapes
		layer_outputs = [layer.output for layer in self.model.layers[:]]
		layer_names = [layer.name for layer in self.model.layers if layer.name[:6] == 'conv3d']
		print(f'layer names: {layer_names}')
		for self.layer in range(1, (self.config.convolution_depth)): # Build deconvolutional models for each layer
			self.model(tf.keras.Input(self.sample.shape))
			print(f"Model new input: {self.model.input}")
			print(f"Model layer output to be applied to activation {self.model.get_layer(layer_names[self.layer - 1]).output}")
			
			#self.model.input
			self.activation_model = tf.keras.models.Model(inputs = tf.keras.Input(self.sample.shape), outputs = [self.model.get_layer(layer_names[self.layer - 1]).output]) 
			print(f"Model outputs - {self.activation_model.output} \n\n Layer outputs - {layer_outputs[self.output_layers[self.layer - 1]]}")

			self.deconv_model = tf.keras.models.Sequential() # Create first convolutional layer
			print(f"Deconv model shape - {self.layer_shapes[self.layer - 1][0], self.layer_shapes[self.layer - 1][1], self.layer_shapes[self.layer - 1][2]}")
			self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, kernel_size = self.config.kernel_size, strides = self.config.kernel_stride, input_shape = (self.layer_shapes[self.layer - 1][0], self.layer_shapes[self.layer - 1][1], self.layer_shapes[self.layer - 1][2], 1), kernel_initializer = tf.keras.initializers.Ones()))
			for deconv_layer in range(self.layer - 1, 0, -1): # Build the depths of the deconvolution model
				if deconv_layer != 1:
					self.deconv_model.add(tf.keras.layers.UpSampling3D(size = self.config.pool_size, data_format = 'channels_last'))
				self.deconv_model.add(tf.keras.layers.Conv3DTranspose(1, self.config.kernel_size, strides = self.config.kernel_stride, kernel_initializer = tf.keras.initializers.Ones()))
			print(f'Summarizing layer {self.layer} deconvolution model')
			self.deconv_model.build()
			self.deconv_model.summary()
			print(f"Sample shape {self.sample.shape}")
			self.activation_model.summary()
			self.sample = self.sample.reshape((1, self.sample.shape[0], self.sample.shape[1], self.sample.shape[2], self.sample.shape[3]))
			self.feature_maps, predictions = self.activation_model.predict(self.sample) # Grab feature map using single volume
			self.feature_maps = self.feature_maps[0, :, :, : ,:].reshape(self.current_shape[0], self.current_shape[1], self.current_shape[2], self.current_shape[3])

			for map_index in range(self.feature_maps.shape[3]): # Save feature maps in glass brain visualization pictures
				feature_map = (self.feature_maps[:, :, :, map_index].reshape(self.current_shape[0], self.current_shape[1], self.current_shape[2])) # Grab Feature map
				deconv_feature_map = self.deconv_model.predict(self.feature_maps[:, :, :, map_index].reshape(1, self.current_shape[0], self.current_shape[1], self.current_shape[2], 1)).reshape(self.new_shape[0], self.new_shape[1], self.new_shape[2])
				self.plot_all(deconv_feature_map, 'DeConv_Feature_Maps', map_index)
			print(f"\n\nExtracting NeuroNet model class activation maps for layer {self.layer}")
			
			with tf.GradientTape() as gtape: # Create CAM
				conv_output, predictions = self.activation_model(self.sample)
				loss = predictions[:, np.argmax(predictions[0])]
				grads = gtape.gradient(loss, conv_output)
				pooled_grads = K.mean(grads, axis = (0, 1, 2, 3))

			self.heatmap = tf.math.reduce_mean((pooled_grads * conv_output), axis = -1)
			self.heatmap = np.maximum(self.heatmap, 0)
			max_heat = np.max(self.heatmap)
			if max_heat == 0:
				max_heat = 1e-10
			self.heatmap /= max_heat

			# Deconvolute heatmaps and visualize
			self.heatmap = self.deconv_model.predict(self.heatmap.reshape(1, self.current_shape[0], self.current_shape[1], self.current_shape[2], 1)).reshape(self.new_shape[0], self.new_shape[1], self.new_shape[2])
			self.plot_all(self.heatmap, 'CAM', 1)

	def folc(self):
		# Feature output linear correlation analysis
		# Within this function we will correlated feature node 
		# activity to their output value within each layer and 
		# feature.	

		# For each layer
		
		# For each feature map

		# For each node

		# Iterate through a subjects bold images
		# Collect each feature map activity

		# Correlate the feature node activity to the output (NN output or label?)

		# Output 3D r coefficient matrix to their T1/T2 and plot
			# Which slices to use until we build GUI?

		# Convolve feature maps with r coefficient matrix and plot
		return

# How do we incorperate the unique attributes and personality of a child?
# ElasticNet and MVPA integration

	def plot_all(self, data, data_type, map_index):
		self.surf_stat_maps(data, data_type, map_index)
		#self.glass_brains(data, data_type, map_index)
		#self.stat_maps(data, data_type, map_index)

	def prepare_plots(self, data, data_type, map_index, plot_type):
		affine = self.header.get_best_affine()
		max_value, min_value, mean_value, std_value = describe_data(data)
		#-Thresholding could take some more consideration-#
		threshold = 0
		intensity = 0.5
		data = data * intensity
		# ---------------------------------------------- #
		data = nib.Nifti1Image(data, affine = self.affine, header = self.header) # Grab feature map
		title = f"{layer} {data_type} Map {str(map_index)} for  {self.category} Answer"
		output_folder = f"{self.config.project_directory}{self.config.model_directory}{self.catergory}/Layer_{self.layer}/{data_type}/{plot_type}/"
		return data, title, threshold, output_folder

	def glass_brains(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Glass_Brain")
		plotting.plot_glass_brain(stat_map_img = data, black_bg = True, plot_abs = False, display_mode = 'lzry', title = title, threshold = threshold, annotate = True, output_file = (output_folder + 'feature_' + str(map_index) + '-' + self.category + '_category.png')) # Plot feature map using nilearn glass brain - Original threshold = (mean_value + (std_value*2))

	def stat_maps(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Stat_Maps")
		for display, midfix, cut_coord in zip(['z', 'x', 'y'], ['-zview-', '-xview-', '-yview-'], [6, 6, 6]):
			plotting.plot_stat_map(data, bg_img = self.anatomy, display_mode = display, cut_coords = cut_coord, black_bg = True, title = title, threshold = threshold, annotate = True, output_file = (output_folder + 'feature_' + str(map_index) +  midfix + self.category + '_category.png')) # Plot feature map using nilearn glass brain

	def surf_stat_maps(self, data, data_type, map_index):
		data, title, threshold, output_folder = self.prepare_plots(data, data_type, map_index, "Surf_Stat_Maps")
		fsaverage = datasets.fetch_surf_fsaverage()

		texture = surface.vol_to_surf(data, fsaverage.pial_left)
		plotting.plot_surf_stat_map(fsaverage.infl_left, texture, hemi = 'left', view = 'lateral', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_left, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-left-lateral-' + self.category + '_category.png'))
		plotting.plot_surf_stat_map(fsaverage.infl_left, texture, hemi = 'left', view = 'medial', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_left, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-left-medial-' + self.category + '_category.png'))

		texture = surface.vol_to_surf(data, fsaverage.pial_right)
		plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi = 'right', view = 'lateral', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_right, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-right-lateral-' + self.category + '_category.png'))
		plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi = 'right', view = 'medial', title = title, colorbar = True, threshold = threshold, bg_map = fsaverage.sulc_right, bg_on_data = True, cmap='Spectral', output_file = (output_folder + 'feature_' + str(map_index) + '-right-medial-' + self.category + '_category.png'))

	

	def ROC(self):
		self.probabilities = self.model.predict(self.x_test).ravel()
		np.save(f"{self.config.project_directory}{self.config.model_directory}/Jack_Knife/Probabilities/Sub-{str(self.jackknife)}_Volumes_Prob.np", self.probabilities)
		fpr, tpr, threshold = roc_curve(self.y_test, self.probabilities)
		predictions = np.argmax(self.probabilities, axis=-1)
		AUC = auc(fpr, tpr)
		plt.figure()
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr, tpr, label = 'RF (area = {:.3f})'.format(AUC))
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(f'Subject {str(self.jackknife)} ROC Curve')
		plt.legend(loc = 'best')
		plt.savefig(f"{self.config.project_directory}{self.config.model_directory}/Jack_Knife/Sub_{str(self.jackknife)}_ROC_Curve.png")
		plt.close()

