# includes: general
import numpy as np
import random

# includes: torch
import torch
import torchvision
import torchvision.transforms as transforms

# includes: C augmentator
import ctypes 

c_augmentor = ctypes.WinDLL(r"...\augmentor.dll")

ND_POINTER_1 = np.ctypeslib.ndpointer(
    dtype=np.float64, 
    ndim=1,
    flags="C",
)

c_augmentor.complexPGL2TransformImage.argtypes = [
    ND_POINTER_1, 
    ctypes.c_int, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
]

c_augmentor.complexPGL2TransformImage.restype = None

c_augmentor.realPGL2SqrTransformImage.argtypes = [
    ND_POINTER_1, 
    ctypes.c_int, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
]

c_augmentor.realPGL2SqrTransformImage.restype = None

c_augmentor.realPGL3TransformImage.argtypes = [
    ND_POINTER_1, 
    ctypes.c_int, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
    ctypes.c_double,
]

c_augmentor.realPGL3TransformImage.restype = None

# main augmentor 
def augment(images, image_size, deviation, complexPGL2Pct, realPGL2SquaredPct, realPGL3Pct) : 

    # rescale x2
    new_images = torchvision.transforms.Resize(
        size=[image_size * 2, image_size * 2], 
        antialias=False, 
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    )(images)
    
    # convert to np
    np_images = new_images.numpy().astype(dtype=np.float64)

    # randomly choose augmentations
    untransformed_images = set()
    for j in range(len(np_images)) : 
        sample = random.randint(0, 99) 

        # PGL(2, C) case
        if sample < complexPGL2Pct : 
            # generate coefficients
            a_r = random.uniform(1-deviation, 1+deviation)
            a_i = random.uniform(-deviation, deviation)
            b_r = random.uniform(-deviation, deviation)
            b_i = random.uniform(-deviation, deviation)
            c_r = random.uniform(-deviation, deviation)
            c_i = random.uniform(-deviation, deviation)
            d_r = random.uniform(1-deviation, 1+deviation)
            d_i = random.uniform(-deviation, deviation)
            
            # convert 3d tensor to 1d array
            reshaped_image = np_images[j].ravel()

            # augment
            c_augmentor.complexPGL2TransformImage(
                reshaped_image, 
                image_size, 
                deviation, 
                a_r, 
                a_i,
                b_r, 
                b_i, 
                c_r, 
                c_i, 
                d_r, 
                d_i,
            )
            
            # convert back to 3d tensor
            np_images[j] = reshaped_image.reshape((3, 64, 64))
        
        # PGL(2, R)^2 case
        elif sample < complexPGL2Pct + realPGL2SquaredPct : 
            # generate coefficients
            a_1 = random.uniform(1-deviation, 1+deviation)
            b_1 = random.uniform(-deviation, deviation)
            c_1 = random.uniform(-deviation, deviation)
            d_1 = random.uniform(1-deviation, 1+deviation)
            a_2 = random.uniform(1-deviation, 1+deviation)
            b_2 = random.uniform(-deviation, deviation)
            c_2 = random.uniform(-deviation, deviation)
            d_2 = random.uniform(1-deviation, 1+deviation)

            # convert 3d tensor to 1d array
            reshaped_image = np_images[j].ravel()

            # augment
            c_augmentor.realPGL2SqrTransformImage(
                reshaped_image, 
                image_size,
                deviation,
                a_1, 
                b_1, 
                c_1, 
                d_1, 
                a_2, 
                b_2, 
                c_2, 
                d_2,
            )

            # convert back to 3d tensor
            np_images[j] = reshaped_image.reshape((3, 64, 64))
        
        # PGL(3, R) case
        elif sample < complexPGL2Pct + realPGL2SquaredPct + realPGL3Pct : 
            # generate coefficients
            a_1 = random.uniform(1-deviation, 1+deviation)
            b_1 = random.uniform(-deviation, deviation)
            c_1 = random.uniform(-deviation, deviation)
            a_2 = random.uniform(-deviation, deviation)
            b_2 = random.uniform(1-deviation, 1+deviation)
            c_2 = random.uniform(-deviation, deviation)
            a_3 = random.uniform(-deviation, deviation)
            b_3 = random.uniform(-deviation, deviation)
            c_3 = random.uniform(1-deviation, 1+deviation)

            # convert 3d tensor to 1d array
            reshaped_image = np_images[j].ravel()

            # augment
            c_augmentor.realPGL3TransformImage(
                reshaped_image, 
                image_size,
                deviation,
                a_1,
                b_1,
                c_1,
                a_2,
                b_2,
                c_2,
                a_3, 
                b_3,
                c_3,
            )

            # convert back to 3d tensor
            np_images[j] = reshaped_image.reshape((3, 64, 64))

        # untransformed, i.e., RandomCrop, case
        else :
            untransformed_images.add(j)

    # convert back to tensor
    new_images = torch.from_numpy(np_images)

    # rescale to original
    new_images = torchvision.transforms.Resize(
        size=[image_size, image_size], 
        antialias=False, 
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    )(new_images)

    # crop untransformed images
    for j in range(len(new_images)) :
        if j in untransformed_images :
            new_images[j] = transforms.RandomCrop(32, padding=4)(new_images[j])
            new_images[j] = transforms.RandomHorizontalFlip()(new_images[j])

    # normalize
    new_images = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(new_images)
    
    # update pointers
    for i in range(len(images)) : images[i] = new_images[i]
