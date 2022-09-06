from n2v.models import N2V
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
from csbdeep.io import save_tiff_imagej_compatible

model_name = 'n2v_chr_hansen'
basedir = 'models'
model = N2V(config=None, name=model_name, basedir=basedir)

img = imread('')

# Here we process the image.
pred = model.predict(img, axes='YX')