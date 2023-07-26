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
- **context_confidence_cnn.ipynb**: The Jupyter Notebook for the multilabel CNN from scratch.
- **inception_fine_tune.ipynb**: The Jupyter Notebook for fine tuning InceptionV3 to make a purely binary predictor (frost/background)
- **vgg16.ipynb**: The Jupyter Notebook for fine tuning VGG16 to make a multilabel classification model to predict frost contexts.
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

I mapped each tile to a 6-element array, where each value is either a 0
or 1, corresponding to the presence of the following frost contexts
in-order:

* [defrosting marks, halos, polygonal cracks, slab ice cracks, uniform albedo, other]

Thus, the entry \[0,1,0,0,1,0\] corresponds to an image with halos and
uniform albedo.\
In order to determine the \"true\" label of each image, I created a
label aggregation method I call the **confidence-overlap weighted
aggregated label**. The process is as follows:\
In the JSON file of an image containing frost, the labelers provided
many useful metrics. Most importantly, they provided a frost_context
list, which are characteristics of the image that led them to classify
it as 'frost.'

-   Defrosting Marks

-   Halos

-   Polygonal Cracks

-   Slab ice cracks

-   Uniform Albedo

-   Other

They also provided a confidence value (low, medium, high) in their
labelling as well as the proportion of overlap. The way I determined the
\"true\" context of the images was to created a weighted average of the
labelers' decisions multiplied by overlap proportion, and finally a hard
0.5 cutoff. Firstly, I assigned the following weights to each confidence
value:

-   **low**: 0.4

-   **medium**: 0.7

-   **high**: 1.0

Then, iterate through all of the annotations. For each annotation, a
labelers' predictions (weighted with their confidence, multiplied by
overlap) was added to the final 6-element array. Then the array is
averaged, and is rounded to the nearest integer. To better demonstrate
this technique, I will provide a detailed example;\
\
Assume we have the following annotations for some tile:

1.  **labeler_1**

    -   **Confidence**: medium (0.7)

    -   **Context**: \[defrost marks, halos\]

    -   **Overlap**: 1.0

2.  **labeler_2**

    -   **Confidence**: high (1.0)

    -   **Context**: \[defrost marks, other\]

    -   **Overlap**: 0.95

3.  **labeler_3**

    -   **Confidence**: low (0.4)

    -   **Context**: \[halos, poly cracks, other\]

    -   **Overlap**: 0.7

Each labeler will, thus, make the following contributions:

1.  Labeler 1 has a weight of 0.7\*1.0, so they contribute
    [0.7, 0.7, 0, 0, 0, 0]

2.  Labeler 2 has a weight of 1.0\*0.95, so they contribute
    [0.95, 0, 0, 0, 0, 0.95]

3.  Labeler 3 has a weight of 0.4\*0.7, so they contribute
    [0, 0.28, 0.28, 0, 0, 0.28]

We now find the \"true\" label by summing up all the contributions and
taking an average

-   Sum each element columnwise:
    [0.7+0.95, 0.7+0.28, 0.28, 0, 0, 0.95+0.28]

-   Average each element by N (number of annotations):
    [1.65/3, 0.98/3, 0.28/3, 0, 0, 1.23/3]
    =[0.55, 0.32, 0.093, 0, 0, 0.41]

-   Perform the 0.5 cutoff =[1, 0, 0, 0, 0, 0]

\
To preprocess the data, I normalized the images by dividing by 255.0.
This rescales the pixel values to a range between 0 and 1. I then
shuffled the training input array, and performed data augmentation with
the following parameters:\

-   rotation_range=20,

-   width_shift_range=0.2,

-   height_shift_range=0.2,

-   shear_range=0.2,

-   zoom_range=0.2,

-   horizontal_flip=True,

-   vertical_flip=True

# Neural Network

I have 3 notebooks, each with a different implementation. The first is context_confidence_cnn, where
I developed a multi-label classifcation model using a CNN from scratch.
It consists of 3 2D convolutional layers, making use of
the ReLU activation function in each layer. Each of these convolutional layers consists of a
max-pooling layer and a dropout layer. After these 3 convolutional layers, there
is flattening to convert the matrix of features into a vector so that it
can be used in fully connected Dense layers. The flattened feature maps
are then passed through 3 dense layers, also with dropout regularization
to make predictions on each of the six target classes.\
\
By using a sigmoid activation function in the last layer, the model was
able to determine the presence or absence of each class independently.
Optimization was handled with the adam optimizer, and loss was computed
using binary_crossentropy. Binary crossentropy loss function is
well-suited for binary classification problems such as this, where it
measures the dissimilarity between the predicted probabilities and true
labels for each class independently.\

The inception_fine_tune.ipynb file fine tunes InceptionV3 to make a binary classification model.
This purely predicts if the image has frost or does not. It does not make predictions
on the frost context classes. \

The vgg16_fine_tune.ipynb file fine tunes VGG16 to make a multilabel classification model.
This is an improvement from the pure CNN model earlier. It makes predictions on each
individual frost context class. A way to improve this may be to implement a hierarchical
clustering algorithm as stipulated in these papers:
- https://www.cosmos.esa.int/documents/6109777/8766007/PSIDA2022_Abstracts+%281%29_Part49.pdf/6353daba-2e07-0f01-beca-a97d4d42ae5c?t=1652783921879
- https://www.cosmos.esa.int/documents/6109777/9316710/X13+-+Lu+-+22-PSIDA-DE-MERNet_final.pdf/3319e266-8071-3fb0-19c4-a3c1cb9a9dfe?t=1657278565405
- https://www.cs.waikato.ac.nz/~eibe/pubs/chains.pdf

# Analysis

To analyze the performance of the multilabel models, I implemented two custom
metrics:

-   **Hamming Loss**: For our purposes, this metric is superior to
    accuracy because it takes into account the similarity between
    predicted labels and true labels; it calculates the average fraction
    of labels that are incorrectly predicted, taking into account both
    false positives and false negatives. Pure accuracy would mark a
    prediction as invalid even if it was very similar. For instance, if
    the machine predicted frost is present with markers 'uniform albedo'
    and 'defrosting marks,' and the true labels were identical plus
    'polygonal cracks,' this would be marked as fully incorrect. With
    Hamming Loss, predictions like these are still rewarded.

-   **Frost Presence Accuracy**: Purely frost or no frost. Our model
    will output a 6-element array of 0's and 1's. If the array has any
    1's, then the model has detected a frost context, and it therefore
    thinks there is frost present. This metric simply checks the binary
    frost/background. If the prediction contains any 1's, treat it as
    \"frost.\" If all 0's, treat it as \"background.\"

The fine tuned InceptionV3 model merely uses the accuracy metric as it's
outputting values for one class (frost). Thus it is appropriate to use
this metric.