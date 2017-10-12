# learn the states of a double dot
import numpy as np
import tensorflow as tf
import glob
import os

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# application logic will be added here
def cnn_model_fn(features,labels,mode):
    '''Model function for CNN'''
    #input layer
    input_layer = tf.cast(tf.reshape(features,[-1,100,100,1]),tf.float32)
    
    # Concolutional layer1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[5,5],strides=5)

    # Dense layer
    pool2_flat = tf.contrib.layers.flatten(pool1)
    dense0 = tf.layers.dense(inputs=pool2_flat,units=512,activation=tf.nn.relu)
    dropout0 = tf.layers.dropout(inputs=dense0,rate=0.5,training=mode == learn.ModeKeys.TRAIN)
    
    dense1 = tf.layers.dense(inputs=dropout0,units=256,activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1,rate=0.5,training=mode == learn.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1,units=128,activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2,rate=0.5,training=mode == learn.ModeKeys.TRAIN)

    # encode layer
    encode = tf.layers.dense(inputs=dropout2,units=4)
    
    # dense output layer
    out_layer = tf.layers.dense(inputs=encode,units=10000)

    loss = None
    train_op = None

    # Calculate loss( for both TRAIN AND EVAL modes)
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(labels=labels, predictions=out_layer)

    # Configure the training op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.01,
            optimizer="Adam")

    # Generate predictions
    predictions= {
        "states" : tf.rint(out_layer),
    }

    # Returna  ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode,predictions=predictions,loss=loss, train_op=train_op)
    
def get_train_inputs():
    n_batch = 25
    index = np.random.choice(np.arange(train_data.shape[0]),n_batch,replace=False)
    x = tf.constant(train_data[index])
    y = tf.constant(train_labels[index])
    return x,y

def get_test_inputs():
    x = tf.constant(test_data)
    y = tf.constant(test_labels)
    return x,y

# get the data
data_files = glob.glob(os.path.expanduser('/wrk/ssk4/dataproc/*100*.npy'))
inp = []
oup = []
for file in data_files:
    data_dict = np.load(file).item()
    inp += [data_dict['current_map']]
    oup += [data_dict['state_map'].flatten()]

inp = np.array(inp)
oup = np.array(oup)
n_samples = inp.shape[0]
train_sample_ratio = 0.8
n_train = int(train_sample_ratio * n_samples)
print("Total number of samples :",n_samples)
print("Training samples :",n_train)
print("Test samples :",n_samples - n_train)
train_data = inp[:n_train]
train_labels = oup[:n_train]

test_data = inp[n_train:]
test_labels = oup[n_train:]

# create the estimator
dd_classifier = learn.Estimator(model_fn=cnn_model_fn,model_dir="/wrk/ssk4/tensorflow_models/")

# set up logging for predictions
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

metrics = {
    "accuracy" : learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="states"),
}
for _ in range(100):
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

#def get_subimage():
#    subimage_input = np.zeros((100,100))
#    subimage_inputs[20:60,20:60] = test_data[0][20:60,20:60]
#    return subimage_input

#res = dd_classifier.predict(input_fn=get_subimage):
#print(res)
#print(test_labels[0][20:60,20:60])
#predictions = dd_classifier.predict(x=test_data)
#res_list = []
#for i,p in enumerate(predictions):
#    res_list += [p['states'].reshape((100,100))]


#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
#plt.subplot(2,1,1)
#plt.pcolor(res_list[3])
#plt.subplot(2,1,2)
#plt.pcolor(test_labels[3].reshape((100,100)))





