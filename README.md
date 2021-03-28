# Hadamard-Network
These code is a single layer of 1x1 convolution or FWHT.

Requirement: TensorFlow 2.4.1, Python 3

1. run conv1x1.py, fast_hadamard.py and hadamard.py to generate the model for speed test.
2. Run speed_test.py on the computer, or speed_test_nano on Jetson Nano 4GB.

On Lenovo R720 (Intel Core i7-7700HQ CPU, NVIDIA GTX-1060 GPU with Max-Q design and 16GB DDR4 RAM), it takes about 0.09~0.1 second with pb model for both cases. Hadamard runs the slowest, and conv1x1 and fast Hadamard are in close speed.

On Nvidia Jetson, it takes about 2.3204 seconds for conv1x1, 5.1580 seconds for Hadamard and 1.1973 second for fast Hadamard.
