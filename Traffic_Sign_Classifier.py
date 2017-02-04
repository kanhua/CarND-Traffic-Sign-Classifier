
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:

# Load pickled data
import pickle
import numpy as np


# TODO: Fill this in based on where you saved the training and testing data

training_file = "./traffic-signs-data/train.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# In[2]:

train['labels'].shape


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below.

# In[3]:

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = train["labels"].shape[0]

# TODO: Number of testing examples.
n_test = test["labels"].shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = train["features"][0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(train["labels"]).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[4]:

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')


# In[5]:

img_id=10
test_img=train["features"][img_id]
plt.imshow(test_img)
print("Label of this image: %s"%train["labels"][img_id])


# ### Visualize the distribution of the labels in train data

# In[6]:

unique_values,unique_counts=np.unique(train["labels"],return_counts=True)
plt.bar(unique_values,unique_counts)
plt.xlabel("label")
plt.ylabel("occurences")


# ### Visualize the distribution of the labels in test data

# In[7]:

unique_values,unique_counts=np.unique(test["labels"],return_counts=True)
plt.bar(unique_values,unique_counts)
plt.xlabel("label")
plt.ylabel("occurences")


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[8]:

### Preprocess the data here.
### Feel free to use as many code cells as needed.


# ### Proprocessing 1: convert the image from RGB space to YUV space

# In[9]:



# ### Proprocessing 2: normalize the image

# In[10]:

def reduce_dim(data):
    new_data=np.zeros((data.shape[0],data.shape[1],data.shape[2],1))
    
    new_data[:,:,:,0]=data[:,:,:,0]
    
    return new_data
def select_label(X,y,index_list):
    idx=np.in1d(y,index_list)
    new_X=X[idx,:,:,:]
    new_y=y[idx]
    
    return new_X,new_y
def substract_mean(X):
    mean_X=np.mean(X,axis=(1,2),keepdims=True)
    return (X-mean_X)/255-0.5

def normalize_image(X):
    X=X.astype('float32')
    return X/255.-0.5


# ### Question 1 
# 
# _Describe how you preprocessed the data. Why did you choose that technique?_

# In[ ]:




# **Answer:**

# In[11]:

### Generate additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.


# ### Question 2
# 
# _Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

# **Answer:**

# In[12]:

### Define your architecture here.
### Feel free to use as many code cells as needed.


# only select part of the data

# In[13]:

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# select only Y channel

t_train_y=train["features"]
t_test_y=test["features"]

select_id=np.unique(y_train)
sel_train_X,sel_train_y=select_label(X_train,y_train,select_id)
sel_test_X,sel_test_y=select_label(X_test,y_test,select_id)
n_classes=len(select_id)


# In[14]:

for i in range(10):
    plt.figure()
    plt.imshow(sel_train_X[i])


# In[15]:

X_train, X_validation, y_train, y_validation = train_test_split(sel_train_X, 
                                                                sel_train_y, test_size=0.2, random_state=2)


# In[19]:

X_train[0].dtype


# In[20]:

sel_train_X.dtype


# In[ ]:

import copy
X_train=normalize_image(copy.copy(X_train))
X_test=normalize_image(copy.copy(X_test))
X_validation=normalize_image(copy.copy(X_validation))


# In[ ]:

plt.hist(X_validation[20].ravel(),bins=100)


# In[ ]:

import tensorflow as tf
EPOCHS = 10
BATCH_SIZE = 128
#BATCH_SIZE = 1280


# In[ ]:

from tensorflow.contrib.layers import flatten
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    l1_filter_width=5
    l1_filter_height=5
    l1_k_output=6
    color_channels=3
    
    l1_filter=tf.Variable(tf.truncated_normal([l1_filter_width, 
                                               l1_filter_height, color_channels, l1_k_output],mean=mu,stddev=sigma))
    l1_bias=tf.Variable(tf.zeros(l1_k_output))
    
    l1_stride=1
    l1_x = tf.nn.conv2d(x, l1_filter, strides=[1, l1_stride, l1_stride, 1], padding='VALID')
    
    
    l1_x = l1_x+ l1_bias
    
    
    # TODO: Activation.
    l1_x=tf.nn.relu(l1_x)

   
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    
    l2_x=tf.nn.max_pool(
    l1_x,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    
    l2_k_output=16
    
    l2_filter=tf.Variable(tf.truncated_normal([l1_filter_width, 
                                               l1_filter_height, l1_k_output, l2_k_output],
                         mean=mu,stddev=sigma))
    
    l2_bias=tf.Variable(tf.zeros(l2_k_output))
    l2_x = tf.nn.conv2d(l2_x, l2_filter, strides=[1, l1_stride, l1_stride, 1], padding='VALID')
    l2_x = tf.nn.bias_add(l2_x, l2_bias)

    
    # TODO: Activation.
    
    l2_x=tf.nn.relu(l2_x)
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    
    l3_x=tf.nn.max_pool(l2_x,
                       ksize=[1,2,2,1],
                       strides=[1,2,2,1],
                       padding='VALID')
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    
    l4_x=flatten(l3_x) # use "-1" to reduce the tensor to 1-d
    
    #assert tf.shape(l4_x)[0]==400
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    
    features_n=400
    hidden_layer_1_n=120
    
    print(l4_x.get_shape())
    
    weights_0=tf.Variable(tf.truncated_normal([features_n,hidden_layer_1_n],mean=mu,stddev=sigma))
    biases_0=tf.Variable(tf.zeros(hidden_layer_1_n))
    
    l4_x=tf.matmul(l4_x,weights_0)+biases_0
    
    # TODO: Activation.
    
    l5_x=tf.nn.relu(l4_x)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    
    hidden_layer_2_n=84
    
    weights_1=tf.Variable(tf.truncated_normal([hidden_layer_1_n,hidden_layer_2_n],mean=mu,stddev=sigma))
    biases_1=tf.Variable(tf.zeros([hidden_layer_2_n]))
    
    l5_x=tf.add(tf.matmul(l5_x,weights_1),biases_1)
    
    # TODO: Activation.
    
    l6_x=tf.nn.relu(l5_x)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = n_classes.
    
    hidden_layer_3_n=n_classes
    
    weights_2=tf.Variable(tf.truncated_normal([hidden_layer_2_n,hidden_layer_3_n],mean=mu,stddev=sigma))
    biases_2=tf.Variable(tf.zeros([hidden_layer_3_n]))
    
    logits=tf.add(tf.matmul(l6_x,weights_2),biases_2)
    
    return logits


# Define the training and test set

# In[ ]:

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)


# training pipeline

# In[ ]:

rate = 1e-3

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# In[ ]:

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# In[ ]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("number of training samples:%s"%num_examples)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


# ### Question 3
# 
# _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)  For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow
# ](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._
# 

# **Answer:**

# In[ ]:

### Train your model here.
### Feel free to use as many code cells as needed.


# ### Question 4
# 
# _How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_
# 

# **Answer:**

# ### Question 5
# 
# 
# _What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

# **Answer:**

# ---
# 
# ## Step 3: Test a Model on New Images
# 
# Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Implementation
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Question 6
# 
# _Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._
# 
# 

# **Answer:**

# In[ ]:

### Run the predictions here.
### Feel free to use as many code cells as needed.


# ### Question 7
# 
# _Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._
# 
# _**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._
# 

# **Answer:**

# In[ ]:

### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.


# ### Question 8
# 
# *Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)*
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# **Answer:**

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



