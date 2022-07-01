Notebook_version = '1.13.1'
Network = 'SplineDist (2D)'

root_dir = "/home/betaglutamate/Documents/GitHub/cellpose_local"
Data_folder = "/home/betaglutamate/Documents/GitHub/cellpose_local/content/one_hour_incubation/images"
Results_folder = "/home/betaglutamate/Documents/GitHub/cellpose_local/content/one_hour_incubation/result" 
Prediction_model_folder = "/home/betaglutamate/Documents/GitHub/cellpose_local/Chr_spline_model" 


import sys
before = [str(m) for m in sys.modules]
import tensorflow as tf
import os
import splinegenerator as sg
from spline_local.splinedist.utils import phi_generator, grid_generator, get_contoursize_max, export_imagej_rois
from spline_local.splinedist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from spline_local.splinedist.matching import matching, matching_dataset
from spline_local.splinedist.models import Config2D, SplineDist2D, SplineDistData2D

# ------- Variable specific to SplineDist -------
from csbdeep.utils import Path, normalize

# ------- Common variable to all ZeroCostDL4Mic notebooks -------
import numpy as np
from matplotlib import pyplot as plt
import os
from tifffile import imsave, imread
import sys
from pathlib import Path
import pandas as pd
import csv
from glob import glob
from tqdm import tqdm 
import cv2




#Prediction

Mask_images = True 
Tracking_file = False 


# model name and path
#@markdown ###Do you want to use the current trained model?
Use_the_current_trained_model = False 

#@markdown ###If not, please provide the path to the model folder:


#Here we find the loaded model name and parent path
Prediction_model_name = os.path.basename(Prediction_model_folder)
Prediction_model_path = os.path.dirname(Prediction_model_folder)


full_Prediction_model_path = Prediction_model_path+'/'+Prediction_model_name+'/'
if os.path.exists(full_Prediction_model_path):
  print("The "+Prediction_model_name+" network will be used.")
else:
  print('Please make sure you provide a valid model path and model name before proceeding further.')

#single images

Nuclei_number = []

for image in os.listdir(Data_folder):
    os.chdir(root_dir)

    print("Performing prediction on: "+image)

    X = imread(Data_folder+"/"+image)
    n_channel = 1 if X.ndim == 2 else X.shape[-1]  

    if n_channel == 1:
        axis_norm = (0,1)   # normalize channels independently
        print("Normalizing image channels independently")

    if n_channel > 1:
        axis_norm = (0,1,2) # normalize channels jointly
        print("Normalizing image channels jointly")  
        sys.stdout.flush()

    os.chdir('Chr_spline_model')
    model = SplineDist2D(None, name = Prediction_model_name, basedir = Prediction_model_path)
    names = [os.path.basename(f) for f in sorted(glob(Data_folder))]

    short_name = os.path.splitext(image)    

    # Save all ROIs and masks into results folder  

    img = normalize(X, 1,99.8, axis = axis_norm)    
    labels, details = model.predict_instances(img)    

    os.chdir(Results_folder)

    if Mask_images:
        imsave(str(short_name[0])+".tif", labels)
    Nuclei_array = details['coord']
    Nuclei_array2 = [str(short_name[0]), Nuclei_array.shape[0]]
    Nuclei_number.append(Nuclei_array2)

    my_df = pd.DataFrame(Nuclei_number)
    my_df.to_csv(Results_folder+'/object_count.csv', index=False, header=False)

         
print('---------------------')
print("Predictions completed.")   