# Overview

The ML-ready dataset of martian surface images used in this project was
taken from JPL's dataverse. It contains 29,686 labeled image tiles (png
and associated json) taken by the HiRISE camera. Each png file is
299x299 pixels in grayscale. The purpose of this project was not only to
determine if frost is present in an image, but also which indicators of
frost are present.\
I used the suggested training, validation, and test split information in
the source images txt files. In this project, I applied a convolutional
neural network (CNN) to train the model. Using hamming loss and a custom
metric: frost_presence_accuracy, I was able to evaluate my model's
performance. These metrics will be discussed later in this report.

# Contents
- **binary_fine_tune.ipynb**: The Jupyter Notebook for fine tuning InceptionV3 to make a purely binary predictor (frost/background)
- **train/val/text_source_images.txt** The suggested splits to prevent data leakage.
- **SUDS...Labeling_Guide.pdf** A provided PDF on how the images were labelled.

# Scope of this Dataset

This dataset was created to study Mars' seasonal frost cycle and its role in the planet's climate and surface evolution over the past ~2 billion years. Scientists are interested in identifying low-latitude frosted microclimates and their impact on climate, and future exploration of Mars will partially depend on finding potential habitable environments and resources for human explorers.

Previous studies of Mars' seasonal frost cycle were limited by a trade-off between coverage and spatial resolution, as scientists were forced to take either a global perspective using coarse observations or a local perspective focused on higher-resolution observations. To overcome this limitation, data science techniques could be useful to identify frost at a global scale with both high-resolution visible and coarse resolution thermal datasets. Toward that goal, models are needed to help identify frost in large-scale, high-resolution imagery. The dataset provided here is meant to address that need. It contains high resolution visible data (from HiRISE on MRO) with frost labels, which can be used to train machine learning models for frost identification in large datasets.

# Contents

## Data
To download the data, visit this link: https://dataverse.jpl.nasa.gov/file.xhtml?fileId=77435&version=3.0
Unzip the file, and place the folder titled 'data' in the project directory.
Images (png files) and labels (json files) are organized in the data directory by "subframes." Subframes are individual 5120x5120 pixel images which are crops of the original HiRISE images (often on the order of 50k x 10k pixels). Individual subframes were annotated by the contributors and then sliced into 299x299 "tiles." Each tile has an associated label for use in training ML algorithms

** Data Directory Tree**
```
data
├── ESP_017506_2450_0_5120_0_5120
│   ├── labels
│   │   └── background
│   │       ├── ESP_017506_2450_2990_3588_4186_4784.json
│   │       └── ...
│   │       └── ESP_017506_2450_4186_4784_4186_4784.json
│   └── tiles
│       └── background
│           ├── ESP_017506_2450_2990_3588_4186_4784.png
│           └── ...
│           └── ESP_017506_2450_4186_4784_4186_4784.png
...
└── ESP_069365_2440_10240_15360_10240_15360
│   ├── labels
│   │   └── frost
│   │       ├── ESP_069365_2440_10240_10838_10838_11436.json
│   │       └── ...
│   │       └── ESP_069365_2440_11436_12034_10838_11436.json
│   └── tiles
│       └── frost
│           ├── ESP_069365_2440_10240_10838_10838_11436.png
│           └── ...
│           └── ESP_069365_2440_11436_12034_10838_11436.png
```

# Methods

In order to determine if an image has frost, I decided to check if it has uniform albedo
and/or defrosting marks. These two classes are the majority classes, as others are rare
such as polygonal cracks. Thus, it prevents the model from confusion. I check if there's an annotation
containing either context at either medium or high confidence. Then, if a tile contains either of these
labels, it is to be treated as "frost." Otherwise, it is "background"

\
To preprocess the data, I used keras.applications.inception_v3.preprocess_input
This rescales the pixel values and performs z-score normalization using ImageNet's
mean and standard deviation. I also performed data augmentation with the
following parameters:\

- rotation_range=25
- width_shift_range=0.2
- height_shift_range=0.2
- horizontal_flip=True
- zoom_range=0.25

Finally, in the trainig dataset, I implemented a **tile_limit** which limits the amount of tiles loaded per image.
This way, there won't be images with several hundred tiles and others with only a dozen. This results
in a more even distribution and allows the model to generalize better.

# Neural Network

This model employs transfer learning using the InceptionV3 architecture, a Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset.

The pre-trained layers of InceptionV3 are used as feature extractors and are "frozen", meaning their weights are not updated during training.

Following these pre-trained layers, Global Average Pooling is performed to summarize the high-dimensional feature maps into lower dimensional ones, maintaining the most important information.

The model then includes a dense layer, a type of fully-connected layer often used in deep learning models to learn complex patterns. To avoid overfitting, we also include a dropout layer which randomly ignores a fraction of neurons during training.

Finally, a sigmoid activation function in the output layer is used to output a probability indicating whether the image belongs to the positive class or not.

# Analysis

The fine tuned InceptionV3 model merely uses the accuracy metric as it's
outputting values for one class (frost). Thus it is appropriate to use
this metric. So far we are able to achieve ~85% validation accuracy.
There is still a ~10% accuracy gap between validation and test, so further
improvements are needed.