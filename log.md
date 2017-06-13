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

---


