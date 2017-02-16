## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview

This project trains a deep neural networks model to recognize the traffic signs using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

![demo_image](./additional_images/prob_result_0.png)

![demo_image](./additional_images/prob_result_4.png)

![demo_image](./additional_images/prob_result_9.png)


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.

2. Clone the project and start the notebook.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
3. Follow the instructions in the `Traffic_Sign_Recognition.ipynb` notebook.

### How to run the script

1. The script assumes that the downloaded image data is placed in ```./traffic-signs-data/```. Otherwise you have to change the value of ```image_dir``` in the two ipynb files.

2. Run ```Gereate_fake_data.ipynb``` first to generate the additional images.

3. Run ```Traffic_Sign_Classifier.ipynb```. This is the main file that trains and tests the model.
