```
A Support Tensor Train Machines (STTM) [1] for Matlab&copy;/Octave&copy;
--------------------------------------------------------------------------------------------------

This package contains Matlab/Octave code for the methods mentioned in Support Tensor Train Machines, namely STTM. The demo code demonstrates a CIFAR-10 bianry classification example. The CIFAR-10 samples and the required toolbox are provided and can be downloaded at https://drive.google.com/file/d/1zecyTldlvOowMz9QyzkIQ2FZO9n57Hck/view?usp=sharing.


1. Requirements
------------

* Matlab or Octave.

* CIFAR-10 dataset.

* TT toolbox [2].

* Tensorlab toolbox [3].

The CIFAR-10 dataset and the two toolboxes can be downloaded through the above link. Put the dataset on the same directory as this README file and add the two toolboxes into Matlab/Octave toolbox dependencies. 


2. Functions
------------

* STTM_demo

Demonstrates the usage of the STTM algorithm. 

* [e, labels, W, b] = f2_STTM(train_X, train_labels, W_r);

STTM training function given the training samples and the TT-ranks.


3. Reference
------------
[1]. "A Support Tensor Train Machine"

Authors: Cong Chen, Kim Batselier, Ching-Yun Ko, Ngai Wong

[2]. https://github.com/oseledets/TT-Toolbox

[3]. https://www.tensorlab.net/
```