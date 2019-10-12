
# Supervised learning: curiosity by prioritized data


This repo contains the sample implementation for this article:

https://medium.com/p/c9528849760a

https://github.com/lorenzob/curiosity/blob/master/docs/README.md (GitHub version)


The basic idea is to give a higher priorty to the harder samples in the data set, by training on these more often. See the article for details.


*NOTE: I'm not yet sure if I did a big silly mistake somewhere, I double checked everything a few times but...you know. Please let me know.*


**Code**


Samples are provided for tensorflow and keras (eager mode) and pytorch.

Current examples:

 * MINIST
 * Fashion-MNIST
 * CIFAR-10
 * Linear regression

There is also one implementation of the "pool" idea (mnist_eager_pool.py).

This is the code used to make the sample charts in the article.

Please let me know what do you think, if it works, if there are mistakes, suggestions, etc.


