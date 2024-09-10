# Lie group augmentations
This is the code behind my joint work with Alessandro Selvitella "On image data augmentation via low-dimensional Lie groups". 

In order to run the code, the file ```augmentor_module.c``` has to be compiled on your machine. Before compiling it, you will need to locate the ```Python.h``` header file, and then enter this file path at the start of ```augmentor_module.c```. On Ubuntu, this can be done by using the following command:
```
find ~/ -type f -name "Python.h"
```
On my Mac, ```Python.h``` had the following file path:
```
C:\Users\FirstName LastName\AppData\Local\Programs\Python\Python312\include\Python.h
```
To compile ```augmentor_module.c``` on Windows, you will need to download MinGW (to run gcc) and enter the following command: 
```
gcc -Wall -pedantic -shared -fPIC -o augmentor.dll augmentor_module.c
```
To compile ```augmentor_module.c``` on Ubuntu, you likewise use the following command:
```
gcc -Wall -pedantic -shared -fPIC -o augmentor.so augmentor_module.c
```
