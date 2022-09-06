#open noise2void environment
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
import urllib
import os
import zipfile

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# create a folder for our data
if not os.path.isdir('./data'):
    os.mkdir('data')

datagen = N2V_DataGenerator()

imgs = datagen.load_imgs_from_directory(directory="/home/betaglutamate/Documents/GitHub/01_local_ai/noise2void/data/S_NaCl_shock_2_220513/151140", filter='*.tif', dims='YX')

# Let's look at the shape of the image
# The function automatically added an extra dimension to the image.
# It is used to hold a potential stack of images, such as a movie.
# The image has four color channels (stored in the last dimension): RGB and Aplha.
# We are not interested in Alpha and will get rid of it.


#makes patches of images. Patches do not overlap
patch_shape=(64,64)
patches = datagen.generate_patches_from_list(imgs, shape=patch_shape)
X = patches[:5000]
X_val = patches[5000:]


# train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch 
# is shown once per epoch. 
config = N2VConfig(X, unet_kern_size=3, 
                   unet_n_first=64, unet_n_depth=3, train_steps_per_epoch=int(X.shape[0]/128), train_epochs=25, train_loss='mse', 
                   batch_norm=True, train_batch_size=128, n2v_perc_pix=0.198, n2v_patch_shape=(64, 64), 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, single_net_per_channel=False)

# Let's look at the parameters stored in the config-object.
vars(config)

# a name used to identify the model --> change this to something sensible!
model_name = 'n2v_chr_hansen'
# the base directory in which our model will live
basedir = 'models'
# We are now creating our network model.
model = N2V(config, model_name, basedir=basedir)

## train model
history = model.train(X, X_val)

print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss']);


model.export_TF(name='Noise2Void - 2D RGB Example', 
                description='This is the 2D Noise2Void example trained on RGB data in python.', 
                authors=["Tim-Oliver Buchholz", "Alexander Krull", "Florian Jug"],
                test_img=X_val[0], axes='YXC',
                patch_shape=patch_shape)