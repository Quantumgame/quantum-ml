### 9th May 2017
- I figured out a way to plot all the loci enclosing the charge stability diagram. But I don't know how to convert it into a picture and which boundaries to choose.

### 12th May 2017
I realised that the charge stability diagram I was plotting was all wrong. CS diagrams are in terms of the two gate voltages and not the lead potentials.

### 16th May 2017
Developing code for the Markov chain model to produce I-V for single quantum dot.

### 25th May 2017
- Developed code for general thomas-fermi for N dots. The code is under ndot/lib
- dot_classifier is a module under ndot/lib which classifies the landscape into leads, barriers and dots, not implemented

### 29th May 2017
I am working on building a potential landscape classifier using Thomas-Fermi. Code under dot_classifier_tf/
Basic algorithm 
- Solve TF for a fixed mu
- Find the dots by looking by n

Created module potential.py : will be used to create potential profiles for testing
First stage of dot classfier with n < 0 being classified as a barrier seems to be working. Though it fails when the Coulomb interation is strong and long-ranged.

### 1st June 2017
**I am trying to build a finite size single quantum dot, ML on it's IV to learn the charge states.**

- Developed a model for a single dot potential, the dot being modelled as a parabolic potential well and the barriers being modelled as Gaussians

- Machine Learning for a single dot, single IV accomplished.

- Improving on the potential model, as potentials from a cylindrical wire

- Could not include tunneling

### 6th June 2017

Wrote a DNN network, works with 99% accuracy with under a minute runtime.
Network structure was [6,12,6]

### 12th June 2017 

- Moved all old code to a junk folder. nanowire_model will be the development folder.
- Created a machine_learning folder at root of repo, will be used for further machine learning code.
- Added screening to the K matrix to prevent $n <= 0$

### 13th June 2017
- iterative fixed mu solver is not very useful. It is only used for obtaining an estimate on number of electrons in the dot.
- added a new function to calculate N_d (number of electrons on each dot) from an input n and the mask stored in ThomasFermi object
- implemented startegy argument in fixed N solver with 'simple','simple\_iter','opt' and 'opt\_iter' values.
	- simple : one iteration of the fixed N solver using the linear system
	- simple_iter : iterations of the fixed N solver using the linear system
	- opt : solves the optimisation problem for energy with n >= 0 as a constraint
	- opt_inter : iteratively solves the opt problem until the mask converges
- fixed a bug in mask_info calculation from mask. The mask_info dictionary has to set to empty {} when calculating from a new mask, else the old values persist.
- added mu_d calculation in the opt solver using V + K n = mu, the dot potential used is the value of mu at the center of the dot
- In each iteration of fixed N solver, I have chosen to update the mask based on the turning points and not the n density. The n density is not reliable because: n <= 0 at some points and finding n ~ 0 points is numerically unstable. (needs a reliable eps such that n - eps ~ 0, such an eps is not available offhand)
- I am deciding that there is no such thing as a dot with 0 e-. It is simply a no dot state.
- changed the default strategy in fixed N solver to opt_iter
- tunneling included, it is making the low N peaks smaller than the larger N peaks
- trying to see if machine can learn the full (V,I) -> C map.

### 14th June 2017
- Can single layer NN learn scale invariance?
- I think the problem with single layer not learning is lack of data
- DNN seems to be working, though I don't know how to quantify the accuracy.
Here is my idea, 

> accur = avg over test samples [(count\_CS == predicted\_CS)/n_out ]


### 15th June 2017
- tf.matmul requires tensors of same rank. So if the rank does not match, use tf.reshape.
- Lesson learned: single layer classifier is not going to learn, the training time would be too much, I am not thinking of building a DNN
- Moving to pooling for reduing dimensionality, does pooling really reduce dimensionality??
- I am thinking of a hybrid arch, where a NN is used to learn E_C,V_b and so on.

### 16th June 2017
- Trying just white noise, this seems to be related to the Anderson model

### 19th June 2017
- It is a good idea to store data in the form of dictionaries in npy format.

### 20th June 2017
- Deep networks, random permutation of data and rint were crucial in learning scale invariance.