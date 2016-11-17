# mnist with tfrecords pipeline, validation and multi-gpus training
This repo is a tutorial for demonstrating the tfrecords input, in-progress validation and multi-gpus
 trainig of TensorFlow without puzzling code like in official tutorial cifar10.
 
 To run this repo, you need at least two gpus in your workstation and do:
 ```python
 python mnist_multigpus_train.py
 ```