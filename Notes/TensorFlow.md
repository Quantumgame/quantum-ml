# TensorFlow Notes

*These are notes on TensorFlow, a machine learning library from Google which I am going to use. These notes have been compiled from the Getting Started section.*. 

### Installing TensorFlow
~~~
pip install tensorflow
~~~

## Getting Started

The central unit of data is a tensor. A tensor's rank is its number of dimensions.

~~~python
import tensorflow as tf
~~~
**The Computational Graph**
Think of tensorflow programs as consisting of two discrete sections:
1. Building the computational graph
2. Running the computational graph

- A computational graph is a series of operations arranged into a graph of nodes. 

- Each node takes zero or more tensors as inputs and produces a tensor as output.

- To actually evaluate the nodes, we must run the computational graph within a session. Create a session object and invoke its run method.

- Operations are also nodes.

- TensorFlow provides a utility called **TensorBoard** that can display a picture of the computational graph.

- A graph can be parameterized to accept external inputs, known as placeholders. A placeholder promises to provide a value later.

- To make a model trainable, we need to be able to modify the graph to get new outputs with the same input. **Variables** allow us to add trainable parameters to the graph. They are constructed with a type and a initial value.

- Constants are initialized when you call tf.constant, and their value can never change. By contrast, variables are not initialized when you call tf.Variable. To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:

~~~python
init = tf.global_variables_initializer()
sess.run(init)
~~~

- tf.train API :  TensorFlow provides optimizers that slowly change each variable to minimize the loss function. The simplest optimizer is gradient descent.

---

## MNIST

### Softmax Regresssions
If you want to assign probabilities to an object being one of several different things, softmax is the thing to do.

A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.


