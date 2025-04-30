import json, atexit
from glob import glob

config_template = {
    "dataset": "Study",
    "architecture": "NeuroNet",
    "data_shape": None,
    "subject_pool": [],
    "trained_pool": [],
    "validation_pool": [],
    "test_pool": [],
    "excluded_pool": [],
    "bids_directory": None,
    "bold_identifier": None,
	"label_identifier":None,
    "project_directory": "models/",
    "model_directory": "run_1",
    "checkpoint_path": None,
    "tool": "default",
    "shuffle": True,
    "norm_min": -1,
    "epochs": 100,
    "current_epoch": 0,
    "batch_size": 36,
    "negative_slope": 0.1,
    "epsilon": 1e-6,
    "learning_rate": 0.001,
    "bias": 0,
    "dropout": 0.1,
    "momentum": 0.9,
    "kernel_initializer": "glorot_uniform",
    "convolution_depth": 4,
    "init_filter_count": 8,
    "kernel_size": [2, 2, 2],
    "kernel_stride": [1, 1, 1],
    "zero_padding": "valid",
    "padding": [0, 0, 0],
    "pool_size": [2, 2, 2],
    "pool_stride": [2, 2, 2],
    "multiscale_pooling": False,
    "top_density": [2500, 1000, 400, 200, 100], # Change name, top density seems excessive?
    "density_dropout": [0.2, 0.1, False, False, False],
    "output_activation": None,
    "outputs": [-1.0, 1.0],
    "outputs_category": ["Negative", "Positive"],
    "output_descriptor": "Outcome",
    "output_unit": "Value",
    "history_types": ["accuracy", "loss"],
    "model_history": {"accuracy":[], 'loss':[], 'val_accuracy':[], 'val_loss': []},
    "optimizers": ["SGD", "Adam"],
    "optimizer": "SGD",
    "use_nestrov": True,
    "use_amsgrad": True,
    "loss": "mse",
    "rebuild": False,
    "use_wandb": False
}

class build:
    def __init__(self, config_folder = None):
        config_json = None
        if config_folder: # Try and load config if folder passed in     
            config_files = glob(f"{config_folder}config.json")
            if len(config_files) == 0:
                config_files = glob(f"{config_folder}/model/config.json")
                if len(config_files) == 0:
                    print(f'No config json files found in {config_folder}')
                    return None
            if len(config_files) > 1:
                response = input("Multiple config files found, please enter the number corresponding to the file would you like to use?\n" + "\n".join([f"{ind + 1} - {file}" for ind, file in enumerate(config_files)] + "\nc - cancel\n\n"))
                if response == 'c' or response.isdigit() == False:
                    return None
                else:
                    config_file = config_files[int(response) - 1]
            else:
                config_file = config_files[0]
            config_json = self.load_config(config_file)
            config_json['rebuild'] = False # Set to false if able to load a pre-existing model
        
        config_json = config_json or config_template # Use passed in config or default config

        self.configure(**config_json) # Build configuration

        atexit.register(self.save_config)
        
    def __repr__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.dump.items()])
    
    def save_config(self, config_path = None):
        config_path = config_path or f"{self.project_directory}{self.model_directory}/model/config.json"
        with open(config_path, 'w') as config_file:
            json.dump(self.dump(), config_file, indent = 4)
             
    def load_config(self, config_path = None):
        print(config_path)
        with open(config_path, "r") as config_file:
            config_json = json.load(config_file)
        return config_json

    def configure(self, dataset, architecture, data_shape, subject_pool, trained_pool, validation_pool, test_pool, excluded_pool, bids_directory, bold_identifier, label_identifier, project_directory, model_directory, checkpoint_path, model_history, tool, shuffle, norm_min, epochs, current_epoch, batch_size, negative_slope, epsilon, learning_rate, bias, dropout, momentum, kernel_initializer, convolution_depth, init_filter_count, kernel_size, kernel_stride, zero_padding, padding, pool_size, pool_stride, multiscale_pooling, top_density, density_dropout, output_activation, outputs, outputs_category, output_descriptor, output_unit, history_types, optimizers, optimizer, use_nestrov, use_amsgrad, loss, rebuild, use_wandb):
		#-------------------------------- Model Set-Up -------------------------------#
		#These initial variables are used by NeuroNet and won't need to be set to anything
        self.dataset = dataset
        self.architecture = architecture
        self.data_shape = data_shape
        self.subject_pool = subject_pool
        self.trained_pool = trained_pool
        self.validation_pool = validation_pool
        self.test_pool = test_pool
        self.excluded_pool = excluded_pool
		
        # Folder Structure - These variables are used to desribes where data is stored
		# along with where to store the outputs of NeuroNet. The results directory is
		# a general folder where specific model run folder will be created. The run
		# directory is the specific folder NeuroNet will generate and output results into.
        
        self.bids_directory = bids_directory
        self.bold_identifier = bold_identifier
        self.label_identifier = label_identifier
        self.project_directory = project_directory
        self.model_directory = model_directory
        self.checkpoint_path = checkpoint_path
        self.model_history = model_history
        self.tool = tool

		#-------------------------------- Shuffle Data ------------------------------##
		# Shuffle will zip all images and labels and shuffle the data before assigning
		# them to training and testing sets. Defaults to true, however you might want
		# to change this to false if you think your question has a time dimension to it
		
        self.shuffle = shuffle
        self.norm_min = norm_min

		#------------------------------- Run parameters ------------------------------#
		# Epoch and Batch Size - Both used to describe how the model will run, and
		# for how long. Epochs represents how many times the data will be presented to
		# the model for deep learning. The batch size simply defines how the model will
		# batch the samples into miniature training samples to help plot model performance.
		
        self.epochs = epochs
        self.current_epoch = current_epoch
        self.batch_size = batch_size

		#---------------------------- Model Hyperparameters ---------------------------#
		# Model Hyperparameters - Hyperparameters used within the models algorithms to
		# to learn about the dataset. These values were found while optimizing the model
		# over a simple stroop dataset so consider using the optiimize function within
		# the NeuroNet library to find optimum values. Bias can be a bit tricky to optimize
		# and I would recommend using the NeuroNet.optimum_bias() to find bias when using
		# an inbalanced dataset. Hyperparameter descriptors to be added with GUI.
		
        self.negative_slope = negative_slope # Formally known as alpha 
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.bias = bias
        self.dropout = dropout
        self.momentum = momentum

		# The Kernel initializer - is used to initialize the state of the model. We initialize
		# the model (e.g. weights, biases, etc.) using Xavier (glorot) uniform which randomly selects
		# values from a uniform gaussian distribution. I found this initializer to be superior to others
		# at the time of building the model however you can change the initializer by typing
		# in the tensorflow string code for the initializer here (e.g. 'glorot_uniform')
		
        self.kernel_initializer = kernel_initializer

		# Convolution Depth - NeuroNet is built to use a basic convolution layer structure
		# that is stacked based on how deep the model is indicated her. Having the depth
		# set at 2 meaning will cause the model to build 2 convolutions layers from
		# the convolution template in the NeuroNet.build() function before building the top density
		
        self.convolution_depth = convolution_depth

		# Initial Filter Count - NeuroNet convolution layer filter sizes are calculated
		# within the NeuroNet.plan() function using a common machine learning rule of
		# doubling filter count per convolution layer. init_filter_count is the initial
		# value the filters starts on before doubling.
		
        self.init_filter_count = init_filter_count

		# Kernel Size & Stride  - These variables are used to decide what the convolution
		# kernel size will be along with how it moves across the layer to generate Features.
		# Generally the bigger the kernel stride, the small the output which... could be a good thing?
		
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride

		# Zero Padding - Padding is used to decide if 0's will be added onto the edge of
		# the input to make sure the convolutions don't move outside of the model and crash
		# your script. 'valid' padding means no 0's will be added to the side were 'same'
		# padding means 0's will be added to the edges of the layer input. The padding
		# variable declares, if using same padding, the size of the padding to add on the the edges
		
        self.zero_padding = zero_padding
        self.padding = padding

		# Max Pooling - Max pooling is used to generally reduce the size of the input.
		# These layers will general a pool kernel of size pool_size and move throughout the layers input
		# based on pool_stride. The max pooling layer finds the max value within the kernel's
		# area and pools it all into a smaller space.
		
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        # Multiscale pooling - The Multiscale pooling block can be added after the final 
        # convolutional block to capture spatial feature's on different scales. This can
        # be useful in a number of situations like when analyzing data in non-standardized
        # space or different age groups like infants, children and adults.

        self.multiscale_pooling = multiscale_pooling

		# Top Density Layer(s) - The following variables are used to define the structure
		# of the top density. The top_density variable holds the sizes of each layer were
		# the density_dropout layer defines whether there is dropout moving into that layer.
		# You might notice that the density_dropout variables hold an extra value conpared
		# to the top_density variable and this is to account for the flattening layer that
		# is automatically built within the NeuroNet.build() function. The first value of
		# density_dropout[0] corresponds to dropout applied to the flatterning layer.
		
        self.top_density = top_density
        self.density_dropout = density_dropout


		# Outputs - The output variables are used to help the model process and understand what it
		# is classifying and help it display some of the output better. These variables are
		# also used within NeuroNet.build() to create the output layer. The output activation
		# is used to decide what activation the model will us in it's output layer.
		
        self.output_activation = output_activation
        self.outputs = outputs
        self.outputs_category = outputs_category
        self.output_descriptor = output_descriptor
        self.output_unit = output_unit
        self.history_types = history_types

		# Optimizers - This section is used to help switch between different Optimizers
		# without having to worry about changing code too much. While talking about
		# SGD/Adam and Nestrov/AMSGrad is not within the scope of this config file I
		# would recommend looking up literature to find which would be best for you.
		
        self.optimizers = optimizers
        self.optimizer = optimizer # Set to either 'SGD' or 'Adam'
        self.use_nestrov = use_nestrov # If using SDG optimizer
        self.use_amsgrad = use_amsgrad # If using Adam optimizer

		# Loss - This variable describes the loss calculation used within the model.
		# the standard used while initially building NeuroNet was binary crossentropy
		# however you may need to change this based on the questions you are asking.
        
        self.loss = loss
		
		#--------------------------------- Rebuild Model ----------------------------##
		# This variable defines wether a new model will be build each time NeuroNet is 
		# called. It can be useful to set this to True when initially setting up the model
		# so the model will be rebuilt with new configurations.
		
        self.rebuild = rebuild

        self.use_wandb = use_wandb

    def dump(self):
        config = {
            "dataset": self.dataset,
            "architecture": self.architecture,
            "data_shape": self.data_shape,
            "subject_pool": self.subject_pool,
            "trained_pool": self.trained_pool,
            "validation_pool": self.validation_pool,
            "test_pool": self.test_pool,
            "excluded_pool": self.excluded_pool,
            "bids_directory": self.bids_directory,
            "bold_identifier": self.bold_identifier,
            "label_identifier":self.label_identifier,
            "project_directory": self.project_directory,
            "model_directory": self.model_directory,
            "checkpoint_path": self.checkpoint_path,
            "model_history": self.model_history,
            "tool": self.tool,
            "shuffle": self.shuffle,
            "norm_min": self.norm_min,
            "epochs": self.epochs,
            "current_epoch": self.current_epoch,
            "batch_size": self.batch_size,
            "negative_slope": self.negative_slope,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "bias": self.bias,
            "dropout": self.dropout,
            "momentum": self.momentum,
            "kernel_initializer": self.kernel_initializer,
            "convolution_depth": self.convolution_depth,
            "init_filter_count": self.init_filter_count,
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "zero_padding": self.zero_padding,
            "padding": self.padding,
            "pool_size": self.pool_size,
            "pool_stride": self.pool_stride,
            "multiscale_pooling": self.multiscale_pooling,
            "top_density": self.top_density,
            "density_dropout": self.density_dropout,
            "output_activation": self.output_activation,
            "outputs": self.outputs,
            "outputs_category": self.outputs_category,
            "output_descriptor": self.output_descriptor,
            "output_unit": self.output_unit,
            "history_types": self.history_types,
            "optimizers": self.optimizers,
            "optimizer": self.optimizer,
            "use_nestrov": self.use_nestrov,
            "use_amsgrad": self.use_amsgrad,
            "loss": self.loss,
            "rebuild": self.rebuild,
            "use_wandb": self.use_wandb
            }
        return config
