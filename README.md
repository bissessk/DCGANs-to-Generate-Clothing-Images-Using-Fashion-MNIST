# DCGANs-to-Generate-Clothing-Images-Using-Fashion-MNIST
The goal of this project was to train and visualize the outputs of a simple Deep Convolutional GAN (DCGAN) to generate realistic-looking (but fake) images of clothing

The FashionMNIST training dataset to train the DCGAN. Images are grayscale and size $$28 × 28$$

## Discriminator Architecture (kernel size = 5 × 5 with stride = 2 in both directions):
* 3 2D convolutions 
  * $$(1 × 28 × 28)$$
    * Leaky ReLU with slope 0.3
    * Dropout with parameter 0.3.
  * $$(64 × 14 × 14)$$
    * Leaky ReLU with slope 0.3
    * Dropout with parameter 0.3.
  * $$(128 × 7 × 7)$$
    * Leaky ReLU with slope 0.3
    * Dropout with parameter 0.3.
* a dense layer that takes the flattened output of the last convolution and maps it to a
scalar.

## Generator Architecture:
* a dense layer that takes a unit Gaussian noise vector of length 100 and maps it to a
vector of size 7 ∗ 7 ∗ 256. No bias terms.
* several transpose 2D convolutions (256 × 7 × 7 → 128 × 7 × 7 → 64 × 14 × 14 →
1 × 28 × 28). No bias terms.
* each convolutional layer (except the last one) is equipped with Batch Normalization
(BN), followed by Leaky ReLU with slope 0.3. The last (output) layer is equipped
with tanh activation (no BN).

The cross-entropy loss was used for training both the generator and the discriminator. The
Adam optimizer was also used and the the learning rate was set to $$10^{−4}$$

Train went for 50 epochs. Intermediate images generated were displayed after T = 10, T = 30, and
T = 50 epochs. 

Loss curves were reported for both the discriminator and the generator loss over all epochs,

