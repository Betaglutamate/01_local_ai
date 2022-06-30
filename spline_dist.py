Notebook_version = '1.13.1'
Network = 'SplineDist (2D)'

Data_folder = "/home/betaglutamate/Documents/GitHub/cellpose_local/training_data/source"
Results_folder = "/home/betaglutamate/Documents/GitHub/cellpose_local/content" #@param {type:"string"}
Prediction_model_folder = "/home/betaglutamate/Documents/GitHub/cellpose_local/Chr_spline_model" #@param {type:"string"}


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
from zipfile import ZIP_DEFLATED



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


def get_contoursize_percentile_from_path(target_path, percentile = 99, show_histogram = False):
  # Percentile needs to be between 0 and 100
  contoursize = []
  Y_list = glob(target_path+"/*.tif") 
  for y in tqdm(Y_list):
    Y_im = imread(y)
    Y_im = fill_label_holes(Y_im)
    obj_list = np.unique(Y_im)
    obj_list = obj_list[1:]  
    
    for j in range(len(obj_list)):  
        mask_temp = Y_im.copy()     
        mask_temp[mask_temp != obj_list[j]] = 0
        mask_temp[mask_temp > 0] = 1
        
        mask_temp = mask_temp.astype(np.uint8)    
        contours,_ = cv2.findContours(mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        perimeter = cv2.arcLength(contours[0],True)
        contoursize = np.append(contoursize, perimeter)

  contoursize_max = np.amax(contoursize)     
  contoursize_percentile = np.percentile(contoursize, percentile)

  if show_histogram:
    # Histogram display
    n, bins, patches = plt.hist(x=contoursize, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Contour size')
    plt.ylabel('Frequency')
    plt.title('Contour size distribution')
    plt.text(200, 300, r'$Max='+str(round(contoursize_max,2))+'$')
    plt.text(200, 280, r'$'+str(percentile)+'th-per.='+str(round(contoursize_percentile,2))+'$')
    maxfreq = n.max();
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10);

  return contoursize_percentile


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
    os.chdir("/home/betaglutamate/Documents/GitHub/cellpose_local")

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

    os.chdir('/home/betaglutamate/Documents/GitHub/cellpose_local/content/test_results')

    if Mask_images:
        imsave(str(short_name[0])+".tif", labels)
    Nuclei_array = details['coord']
    Nuclei_array2 = [str(short_name[0]), Nuclei_array.shape[0]]
    Nuclei_number.append(Nuclei_array2)

    my_df = pd.DataFrame(Nuclei_number)
    my_df.to_csv(Results_folder+'/object_count.csv', index=False, header=False)

         
print('---------------------')
print("Predictions completed.")   