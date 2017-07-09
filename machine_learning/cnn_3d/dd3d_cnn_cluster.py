# this notebook will be used to create a cnn for state indentification for a double dot data set with 3 dimensions.
# CNN for learning!

# learn the states of a double dot
import numpy as np
import tensorflow as tf
import glob
import os

# get the data
data_folder_path = "/wrk/ssk4/dd3d_data/"
files = glob.glob(data_folder_path + "*.npy")
# shuffling the files to avoid any single dot bias
import random
random.shuffle(files)
files = files[:]

n_samples = len(files)
train_sample_ratio = 0.5
n_train = int(train_sample_ratio * n_samples)

print("Total number of samples :",n_samples)
print("Training samples :",n_train)
print("Test samples :",n_samples - n_train)
n_train = 1
n_samples = 1

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# application logic will be added here
def cnn_model_fn(features,labels,mode):
    '''Model function for CNN'''
    #input layer
    input_layer = tf.cast(tf.reshape(features,[-1,50,50,50]),tf.float32)
    
    conv1 = tf.layers.conv2d(inputs=input_layer,
                            filters=32,
                            kernel_size=5,
                            padding="same",
                            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2,strides=2)
    
    conv2 = tf.layers.conv2d(inputs=pool1,
                            filters=64,
                            kernel_size=5,
                            padding="same",
                            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2,strides=2)
    
    flat = tf.contrib.layers.flatten(inputs=pool2)
    # dense output layer
    out1 = tf.layers.dense(inputs=flat,units=1024,activation=tf.nn.relu)  
    dropout1 = tf.layers.dropout(
      inputs=out1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    
    out = tf.layers.dense(inputs=dropout1, units=125000)
    
    loss = None
    train_op = None

    # Calculate loss( for both TRAIN AND EVAL modes)
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(out,labels)

    # Configure the training op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=1e-3,
            optimizer=tf.train.AdamOptimizer)

    # Generate predictions
    predictions= {
        "states" : tf.rint(out),
    }
    
    # Returna  ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode,predictions=predictions,loss=loss, train_op=train_op)
    
    
def get_train_inputs():
    n_batch = 1
    index = np.random.choice(n_train,n_batch,replace=False)
    inp = []
    oup = []
    for i in index:
        dat = np.load(files[i]).item()
        curr_map = np.array([x['current'] for x in dat['output']])
        def convert_to_state(dat_ele_output):
            state = dat_ele_output['state']
            if state == "QPC":
                return 0
            elif state == "ShortCircuit":
                return -1
            elif state == "Dot":
                return len(dat_ele_output['charge_state'])
            else:
                #invalid
                return -2

        state_map = np.array([convert_to_state(x) for x in dat['output']])
        inp += [curr_map]
        oup += [state_map]

    inp = np.array(inp,dtype=np.float32)
    oup = np.array(oup,dtype=np.float32)
    
    x = tf.constant(inp)
    y = tf.constant(oup)
    
    return x,y

def get_test_inputs():
    inp = []
    oup = []
    for file in files[n_train:]:
        dat = np.load(file).item()
        curr_map = np.array([x['current'] for x in dat['output']])
        def convert_to_state(dat_ele_output):
            state = dat_ele_output['state']
            if state == "QPC":
                return 0
            elif state == "ShortCircuit":
                return -1
            elif state == "Dot":
                return len(dat_ele_output['charge_state'])
            else:
                #invalid
                return -2

        state_map = np.array([convert_to_state(x) for x in dat['output']])
        inp += [curr_map]
        oup += [state_map]

    inp = np.array(inp,dtype=np.float32)
    oup = np.array(oup,dtype=np.float32)
    
    x = tf.constant(inp)
    y = tf.constant(oup)
    
    return x,y


# create the estimator
dd_classifier = learn.Estimator(model_fn=cnn_model_fn,model_dir = "/wrk/ssk4/tensorflow_models/dd3d/")

# set up logging for predictions
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

metrics = {
    "accuracy" : learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="states"),
}
for _ in range(1):
    dd_classifier.fit(
        input_fn=get_train_inputs,
        steps=100,
        monitors=[logging_hook])
    
    eval_results=dd_classifier.evaluate(input_fn=get_train_inputs,metrics=metrics,steps=1)
    print("Train accuracy",eval_results)
    eval_results=dd_classifier.evaluate(input_fn=get_test_inputs,metrics=metrics,steps=1)
    print("Validation accuracy",eval_results)

print("Total number of samples :",n_samples)
print("Training samples :",n_train)
print("Test samples :",n_samples - n_train)
eval_results=dd_classifier.evaluate(input_fn=get_test_inputs,metrics=metrics,steps=1)
print("Test accuracy",eval_results)
