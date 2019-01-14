# CNN Spectral Image Classification - TensorFlow
Deep Learning (CNN) classification of spectral images on TensorFlow.

The DEMO of spectral Image classification service has been available at :

 https://lcrobert.pythonanywhere.com/pysa/recognition/

### Outline :

* Introduction
* DataSets
   - Image preprocess
   - Training set, Validation set and Test set 
   - Data input pipeline
* CNN model architecture
* Training process
* Result

Last update: 20190114

------

<br>

### Introduction 

* The spectral image mentioned here refers to image like this :  (!! NOT Hyperspectral images !!)
   ![](https://lcrobert.pythonanywhere.com/static/img/Sun_600_300.jpg "img demo")

   the more reference images can be find [here](https://spectralworkbench.org/) and [here](https://lcrobert.pythonanywhere.com/pysa/).

* The **7** spectral image types of current trained model includes : 

   | Label | Description |
   | :---: | :---------- |
   |   0   | LED |
   |   1   | CCFL, Fluorescent lamp, Hg lamp  ( A light source with element mercury ) |
   |   2   | Sun, Sky, Star, Incandescent lamp, Halogen lamp  ( A continuum light source ) |
   |   3   | Neon lamp |
   |   4   | Helium lamp |
   |   5   | Hydrogen lamp |
   |   6   | Nitrogen lamp |

 <br>

### DataSets :

The images in the datasets mainly taken from [SpectralWorkbench](https://spectralworkbench.org/) website and the other part is collected by myself. These images will be preprocessed and divided into training, validation and test sets with labels, then converted to TFRecord format files as input datasets of training process. The TFRecord files can be found in InputData folder. 

* **Image preprocess :**  ( see  *s1_LabelImage.py*  , *ImgPreprocessSingle.py* , *s2_ImgPreprocess.py*)

    1. After labeling  images,  manually check each image to ensure it's classified correctly and has good image quality because of unstable image quality and information of the source website . 
    2. The main feature of a spectral image is its variation of color and light intensity along the direction of spectral dispersion (x-axis), which means we could select a rectangular area having a small part in y-axis direction with the region only covering spectral signal in the x-axis direction. So each image will go through the following processing : (I) Ensure that the spectral dispersion direction is along the x-axis. (II) Select and crop the effective spectrum area.  (III) Scale to the same image size which is  (15, 480). 

* **Training, Validation and Test sets**   ( see  *s3_ImgToTFrecord.py* )

    1. Check the distribution of image types of the datasets : 

       |   Label   |  0   |  1   |  2   |  3   |  4   |  5   |  6   |
       | :-------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
       | Data size | 506  | 598  | 280  | 119  |  54  |  81  | 103  |

    2. Separate datasets into training set and test set with shuffle in-place. The ratio was 9 : 1 .

    3. Out of the 1741 images in the original datasets, there was an imbalance in the image types. So for the **training set**,  I oversampled the under-represented types and made the image count at least equal to 400. The method of oversampling in this case is a simple process of random replication of images.

    4. Separate **training set** into **training set and validation set** with shuffle in-place. The ratio  was 9 : 1 .  In order to increase the diversity of the training set,  randomly selected a half of the images and then flipped the original along horizontal axis.

    5. Create the TFRecords files :

       |           | Training set | Validation set | Test set |
       | --------- | ------------ | -------------- | -------- |
       | Data size | 2696         | 300            | 175      |

* **Data input pipeline**   ( see  *ReadTFRecords.py* )

<br>

### CNN model architecture: 

- To be continued...<br>

  



   
