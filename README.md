# dataset-format-converter-
This repository contains code for converting datasets in different formats. For training different types of image classification and segmentation models.
The scripts contained in this repo use computer vision techniques and simple Python libraries to perform their functions.

It is a lot of pain to convert dataset labels from one format to another as a student or professional. As we see YOLO has a different set of rules and format for its segmentation and detection models. Building a U-net will require labels in a different format and so do many other open-source models used for segmentation and detection. So I thought of creating this repo containing code to convert labels from one format to another.

The files and their function:-

1. coconew.py: Converts Jason to Yolo format for detection models.
2. convert.py: Converts Multi-Class Segmentation masks to polygon coordinates for training instance segmentation and detection models. Polygon coordinates are used for state-of-the-art segmentation models like YOLO SegV8.

Disclaimer:-
1. Remember to specify the correct path to the data.
2. In convert.py replace  the class names and the BGR channel with your own classes and BGR codes. (These can be found in the segmentation class directory of your data)
