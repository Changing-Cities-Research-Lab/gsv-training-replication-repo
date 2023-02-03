import warnings
import json
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import skimage #require scikit-image pkg
import sys
import torchvision.models as models
import pickle as pickle
import img_to_vec
import io

import urllib.request
import seaborn as sns

from PIL import Image
from torch.autograd import Variable
from torch import topk
from torch.nn import functional as F

def generate_CAM(image_name, model_name, output_class, url_path, num_classes = 2):
    """
    Inputs:
        - image_name: name of image to generate CAM
        - model_name: path to resnet model to use for feature extraction
        - output_class: desired prediction class for which to generate CAM
        - url_path: url for the S3 directory of images in which image is stored
        - num_classes: number of classes associated with resnet model output layer [default: 2]

    Outputs: Figure containing CAM heatmap overlayed on original image for desired class.
    """

    def get_activations(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]

    URL = url_path + image_name

    with urllib.request.urlopen(URL) as url:
        f = io.BytesIO(url.read())

    image = Image.open(f)
    print(image)

    width, height = image.size
    img_2_vec = img_to_vec.Img2Vec(model=model_name, num_output_labels=num_classes)
    prediction_var = img_2_vec.process_image(image)

    model = img_2_vec.model
    final_layer = model._modules.get('layer4')
    activated_features = img_to_vec.SaveFeatures(final_layer)

    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    weight_softmax_params
    class_idx = topk(pred_probabilities,1)[1].int()
    print("Predicted trash class for " + image_name + " is " + str(int(class_idx)))
    overlay = get_activations(activated_features.features, weight_softmax, output_class)
    fig = plt.figure(figsize=(5,4))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(image)
    plt.imshow(overlay[0], alpha=0.5, cmap='jet')
    plt.imshow(skimage.transform.resize(overlay[0], (height,width)), alpha=0.3, cmap='jet');
    im_title = image_name + " | prediction: " + str(int(class_idx)) + " | activation for: " + str(int(output_class))
    plt.title(im_title)
    #plt.show()
    print(prediction)
    fig.tight_layout(pad=0)
    return(fig)


def generate_CAM(args, image_name, num_classes, save_name=""):
    model_name, num_classes, label_column, input_csv, image_dir = args['--model_name'], int(args['--num_classes']), args['--true_label'], args['--input_csv'], args['--image_dir']

    def get_activations(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return [cam_img]
    image = Image.open(image_dir + "/" + image_name)
    width, height = image.size
    img_2_vec = img_to_vec.Img2Vec(model=model_name, num_output_labels=num_classes)
    prediction_var = img_2_vec.process_image(image)

    model = img_2_vec.model
    final_layer = model._modules.get('layer4')
    activated_features = img_to_vec.SaveFeatures(final_layer)

    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    print("")


    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    weight_softmax_params

    class_idx = topk(pred_probabilities,1)[1].int()

    overlay = get_activations(activated_features.features, weight_softmax, class_idx)
    print(overlay)
    fig = plt.figure(figsize=(5,4))
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.imshow(image)
    plt.imshow(overlay[0], alpha=0.5, cmap='jet')
    plt.imshow(skimage.transform.resize(overlay[0], (height,width)), alpha=0.3, cmap='jet');
    im_title = image_name + " | prediction: " + str(int(class_idx))
    plt.title(im_title)
    #plt.show()
    print(skimage.transform.resize(overlay[0], (height,width)))
    fig.tight_layout(pad=0)
    return(fig)

def make_CAM_images(args, data, dir_name="error_analysis/cams/", count=50):
    print(len(data), "data points")
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory " , dir_name ,  " created ")
    else:
        print("Directory " , dir_name ,  " exists ")
    
    for i, row in data[:min(len(data), count)].iterrows():
        img_cam = generate_CAM(args, row.image_name, args["--num_classes"])
        img_cam.savefig(dir_name + row.image_name)

        img_cam.canvas.draw() 

# Generate CAMS

discrepancies = pd.read_csv('PATH TO CSV WITH IMAGES WITH DISCREPANCIES')

dir_name = "DIRECTORY TO SAVE TO"
test_args = "PATH TO TRAINED RESNET MODEL"
print(dir_name)
make_CAM_images(test_args, discrepancies, str(dir_name), 30)