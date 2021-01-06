# DCGANs-to-Generate-Clothing-Images-Using-Fashion-MNIST
The goal of this project was to train and visualize the outputs of a simple Deep Convolutional GAN (DCGAN) to generate realistic-looking (but fake) images of clothing

The FashionMNIST training dataset to train the DCGAN. Images are grayscale and size 28 × 28

## Discriminator Architecture (kernel size = 5 × 5 with stride = 2 in both directions):
* 3 2D convolutions 
  * (1 × 28 × 28)
    * Leaky ReLU with slope 0.3
    * Dropout with parameter 0.3.
  * (64 × 14 × 14)
    * Leaky ReLU with slope 0.3
    * Dropout with parameter 0.3.
  * (128 × 7 × 7)
    * Leaky ReLU with slope 0.3
    * Dropout with parameter 0.3.
* a dense layer that takes the flattened output of the last convolution and maps it to a
scalar.

## Generator Architecture:
* there is a dense layer that takes a Gaussian noise vector of length 100 and maps it to a vector of sixe 7 x 7 x 256
* 4 convolutional Layers will follow
  * (256 × 7 × 7)
    * Batch Normalization (BN)
    * Leaky ReLU with slope 0.3
  * (128 × 7 × 7)
    * Batch Normalization (BN)
    * Leaky ReLU with slope 0.3
  * (64 × 14 × 14)
    * Batch Normalization (BN)
    * Leaky ReLU with slope 0.3
  * (1 × 28 × 28)
    * `tanh` activation

The cross-entropy loss was used for training both the generator and the discriminator. The
Adam optimizer was also used and the the learning rate was set to 10E−4

Train went for 50 epochs. Intermediate images generated were displayed after T = 10, T = 30, and
T = 50 epochs. 

Loss curves were reported for both the discriminator and the generator loss over all epochs,

