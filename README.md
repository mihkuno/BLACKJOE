> use python 3.10.12
> use RTX 4050 Nvidia GPU


## About The dataset

# Playing Cards > YOLOv8n
https://universe.roboflow.com/augmented-startups/playing-cards-ow27d

Provided by a Roboflow user
License: Public Domain

# Overview
The Playing Cards dataset is a collection of synthetically generated cards blended into various types of backgrounds. You will be able to perform object detection to detect both number and suit of the cards.

# Example Footage
![](https://i.imgur.com/eDtoiF3.gif)


# Training and Deployment

The playing cards model has been trained in Roboflow, available for inference on the Dataset tab.

One could also build a Card Counting model for either Black Jack or Poker using YOLOR. This is achieved using the Roboflow Platform which you can deploy the model for robust and real-time detections. You can learn more here:  https://augmentedstartups.info/YOLOR-Get-Started

Video Demo using YOLOR for training- https://youtu.be/2lGTZuaH4ec

# About Augmented Startups
We are at the forefront of Artificial Intelligence in computer vision. With over 90k subscribers on YouTube, we embark on fun and innovative projects in this field and create videos and courses so that everyone can be an expert in this field. Our vision is to create a world full of inventors that can turn their dreams into reality.



## About Roboflow


Playing Cards - v4 YOLOv8n
==============================

This dataset was exported via roboflow.com on March 18, 2023 at 3:59 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 24240 images.
Cards are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 15 percent of the image
* Random rotation of between -10 and +10 degrees
* Random shear of between -2° to +2° horizontally and -2° to +2° vertically
* Random brigthness adjustment of between -25 and +25 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 1.75 pixels
* Salt and pepper noise was applied to 2 percent of pixels


