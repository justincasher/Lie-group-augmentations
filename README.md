# Lie group augmentations
This is the code behind my joint work with Alessandro Selvitella on applying Lie groups to augment images. In particular, we apply PGL(C, 2), PGL(R, 2)^2, and PGL(R, 3) Lie group actions to augment images on the CIFAR-100 benchmark, then train neural networks on these images, achieving an increase in accuracy in each case. 

These augmentation methods are implemented in the file ```augmentor_module.c```. After compiling this file, you can import it into ```augmentor.py``` to transform tensors. The files ```training.py``` and ```main.py``` are examples of how these augmentations can be applied to training neural networks. I also included ```sliders.py``` to visualize how Lie group augmentations distort images.

## Compiling
In order to run the code, the file ```augmentor_module.c``` has to be compiled on your machine. 

### Step 1: Finding ```Python.h```
Before compiling it, you will need to locate the ```Python.h``` header file, and then type this file path at the beginning of ```augmentor_module.c```. On Ubuntu (e.g. when using a server), this can be done by using the following command:
```
find ~/ -type f -name "Python.h"
```
On my Windows machine, it was located at the following path:
```
C:\Users\FirstName LastName\AppData\Local\Programs\Python\Python312\include\Python.h
```
On my Mac, I had a harder time finding ```Python.h```, which had the following file path:
```
/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12/Python.h
```

### Step 2: Compiling ```augmentor_module.c```
To compile ```augmentor_module.c``` on Windows, you will need to download MinGW (to run gcc) and enter the following command: 
```
gcc -Wall -pedantic -shared -fPIC -o augmentor.dll augmentor_module.c
```
To compile ```augmentor_module.c``` on Ubuntu, you likewise use the following command:
```
gcc -Wall -pedantic -shared -fPIC -o augmentor.so augmentor_module.c
```

## License

This software is available under the MIT License.
