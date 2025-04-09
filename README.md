# NeuroNet
This repository contains the source code for NeuroNet used for building and training AI on NeuroImaging data previously known as KleinNet. This tool can be used on any fMRI data collected on regression or classification tasks and is compatable with all BIDS formatted data to rapidly load in data and start training an AI in as little as 7 lines of code. NeuroNet at it's core is a TensorFlow wrapper that is entirely customizable to the task at hand with a tried and tested default convolutional neural network structure based off the paper (add citation). The tool incorperates modern neural network architecture's like attention mechanism's and mulit-scalar pooling to learn intricate non-learner spatial relationships in your data. Heatmaps and experimental multi-voxel pattern analysis allow for us to gain insight into the hidden layers of the model and the important patterns the model finds useful for predicting!

<img width="577" alt="image" src="https://github.com/user-attachments/assets/0c4b9833-34f3-438b-bdec-e50b92ee62ba" />

# How to train NeuroNet on your NeuroImaging data
To run NeuroNet first import the library and initialize a model...

```
import neuronet as nn

net = nn.NeuroNet()
```

Once TensorFlow and your default model builder has initialized, you now need to orient your model to you dataset. You can do this by calling the net.orient() function which takes three argumentsl; your BIDS compliant folder containing your neuroimaging data, your BOLD file identifier and your label file identifier. 



```
net.orient('/path/to/BIDS/folder/', '*movie*MNIPediatricAsym*preproc_bold_6mm_smoothed_deconvolved.nii', 'movie?_labels.txt')
```
NOTE: These identifiers use glob to find relavent files for each subject so you can use wildcards (* for multi-character and ? for single-character wildcards) as shown below to flexibly find data you're interested in

Once your model has been oriented to your database, it will list all subject data it found. 

From here you can customize the AI architecture and training process completely by modifying the config variable attached to your network. The config variable is intended to allow for easy saving, loading and transfer learning of your model and stores all information about your model for later reference. The tool is built so you don't need to adjust this config variable at all if you wish, but if you wish you can change any of the following variables to highly modify your AI architecture...

```
config = {
    "dataset": "NeuroImaging Study",
    "architecture": "NeuroNet",
    "shuffle": False,
    "epochs": 10,
    "batch_size": 36,
    "negative_slope": 0.1,
    "epsilon": 1e-6,
    "learning_rate": 0.001,
    "bias": 0,
    "dropout": 0.5,
    "momentum": 0.9,
    "kernel_initializer": "glorot_uniform",
    "convolution_depth": 2,
    "init_filter_count": 8,
    "kernel_size": [2, 2, 2],
    "kernel_stride": [1, 1, 1],
    "zero_padding": "valid",
    "padding": [0, 0, 0],
    "pool_size": [2, 2, 2],
    "pool_stride": [2, 2, 2],
    "multiscale_pooling": False,
    "top_density": [2500, 1000, 400, 200, 100],
    "density_dropout": [0.2, 0.1, False, False, False],
    "outputs": [-1.0, 1.0],
    "outputs_category": ["Negative", "Positive"],
    "output_descriptor": "Outcome",
    "output_unit": "Value",
    "optimizer": "SGD",
    "use_nestrov": True,
    "use_amsgrad": True,
    "loss": "mse",
    "rebuild": False
}
```
For more information on these customizations, review the tutorial page or watch the tutorial video to gain more insight into how to scale your AI!

From there finally we can build our model. To build the model, due to fMRI data's variable size nature, the model will load a single subjects data to figure out the dimension size and needed input shape. From there the tool will construct a convolutional neural network using your configuration as a framework. The model will summarize the structure once built succesfully...
```
net.built()
```

After building your network successfully, you can now load in some of your data and start training it with the respective ```net.load()``` and ```net.train()``` commands. Note you need to load at least 3 subjects per training batch. Here is an example of how to iterate through a large data sample...
```
window_size = 9
for window in range(0, len(net.config.subject_pool), window_size): 
    net.load(count = 3, session = '0')
    net.train()
```

You can also directly pass in portions of your config.subject_pool generated from the net.orient() function call for more custom batching...
```
subject_batch = net.config.subject_pool[12:21]
results = net.load(subject_batch, session = '0')
if results == None: # If wrangle failed with current batch, go to next subjects
    continue
net.train()
```
NOTE: The net.load() function will return a ```None``` result it it failed to load enough subjects from the batch of subject you passed in to load.

# Exploring what NeuroNet learned

NeuroNet was built with the hope to gain further insight into what the AI learns as useful patterns in predicting an outcome. Neuroscience has a massive body of research surrounding the different paradigms of the human experience, which makes AI an interesting avenue for exploring neuroimaging data given the known spatial patterns as a benchmark of sorts. To accomplish this NeuroNet has a seperate observer.lens() class object used to explore the NeuroNet you've trained. It can do anything from plotting your models accuracy to extracting the heatmaps and outputting them into MNI space for visualization as seen at the top of this library.

To do some basic analysis of the accuracy and/or loss of the model, you can call to the ```net.lens.plot_accuracy()```. This grabs the model history saved to your net.config object and plots it to your model directory using matplotlib.
```
net.lens.plot_accuracy(show = True)
```

You can also trace out a models predictions and the true labels for a given subject using the ```net.predict()``` call. You first will need to load the subject, here's an example script plotting the predictions for every subject in you subject pool that wasn't trained on...
```
for subject in net.config.subject_pool:
    if subject in net.config.previously_run: # Exclude trained
        continue
    net.predict(subject)
```

One of the most important functions apart of the `observer.lens()` class is the `lens.observe()` function, which takes an in depth dive into the feature maps generated apart of the AI. For every layer and every feature map, the tool deconvolves the maps back into their original shape and plots the features in MNI space to be visualized. For the final convolutional block in the NeuroNet model, a heatmap is generated to show the most important freature's used by the model. To generate all the feature and heat maps for a given outout, call the net.lens.observe() function and pass in the outcome you would like to observe like this...

```
net.lens.observe(-0.9) # To generate feature maps for a test sample that is close to -0.9
```

NOTE: The observe function will load a test subject that hasn't been seen by the model, and find a sample within the test subject set that is within a 0.1 range of your interest

# Transfer Learning with NeuroNet

(In development)

Transfer learning is completely achievable with NeuroNet by simply loading a pre-trained model built with NeuroNet and adjusting the config variable loaded before building. Below is an example of loading a pre-trained classification BOLDnet model, switching the activation to linear for a regression paradign, then building the model with the new configuration.

```
net.load_model('pre/trained/classification/model/directory/path/')

net.config.output_activation = 'linear' # Switch model to a regression output

net.build()

```



