# GLOBAl VARIABLES
# width = range of position deviations for slider
width = 0.15
# step = fineness of possible choices
step = 0.005
# ----------------

# includes: plotting
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# includes: general
import numpy as np
import os
import concurrent.futures
import random

# includes: torch
import torch
import torchvision
import torchvision.transforms as transforms

# includes: C augmentator
import ctypes 

# import c_augmentor if using windows (see README)
c_augmentor = ctypes.WinDLL(r"...\augmentor.dll") 

# import c_augmentor if using Ubuntu/Mac (see README)
# c_augmentor = ctypes.CDLL(r"...\augmentor.so") 

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
]

c_augmentor.realPGL3TransformImage.restype = None

# PGL(2, C) slider
def C_PGL2(trainloader) : 
    """
    Starts a PGL(C, 2) slider, allowing you to view the 
    effects of these transformations on an image.
    """

    # get images to augment
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # create plotting figure
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("PGL(2, C) slider")
    axcolor = 'yellow'

    # plot images
    np_image = np.transpose(torchvision.utils.make_grid(images).numpy(), (1, 2, 0))
    img = ax.imshow(np_image)

    a_r_slider_ax = plt.axes([0.20, 0.95, 0.65, 0.03], facecolor=axcolor)
    a_r_slider = Slider(a_r_slider_ax, "a_r value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    a_i_slider_ax = plt.axes([0.20, 0.90, 0.65, 0.03], facecolor=axcolor)
    a_i_slider = Slider(a_i_slider_ax, "a_i value", valstep=step, valmin=-width, valmax=width, valinit=0)

    b_r_slider_ax = plt.axes([0.20, 0.85, 0.65, 0.03], facecolor=axcolor)
    b_r_slider = Slider(b_r_slider_ax, "b_r value", valstep=step, valmin=-width, valmax=width, valinit=0)

    b_i_slider_ax = plt.axes([0.20, 0.80, 0.65, 0.03], facecolor=axcolor)
    b_i_slider = Slider(b_i_slider_ax, "b_i value", valstep=step, valmin=-width, valmax=width, valinit=0)

    c_r_slider_ax = plt.axes([0.20, 0.15, 0.65, 0.03], facecolor=axcolor)
    c_r_slider = Slider(c_r_slider_ax, "c_r value", valstep=step, valmin=-width, valmax=width, valinit=0)
    
    c_i_slider_ax = plt.axes([0.20, 0.10, 0.65, 0.03], facecolor=axcolor)
    c_i_slider = Slider(c_i_slider_ax, "c_i value", valstep=step, valmin=-width, valmax=width, valinit=0)

    d_r_slider_ax = plt.axes([0.20, 0.05, 0.65, 0.03], facecolor=axcolor)
    d_r_slider = Slider(d_r_slider_ax, "d_r value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    d_i_slider_ax = plt.axes([0.20, 0, 0.65, 0.03], facecolor=axcolor)
    d_i_slider = Slider(d_i_slider_ax, "d_i value", valstep=step, valmin=-width, valmax=width, valinit=0)

    def update(val) :
        # clone and rescale x2
        new_images = torch.clone(images)
        
        new_images = torchvision.transforms.Resize(
            size=[32 * 2, 32 * 2], 
            antialias=False, 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )(images)
        
        # convert to np
        np_images = new_images.numpy().astype(dtype=np.float64)

        for j in range(len(np_images)) : 
            # convert 3d tensor to 1d array
            reshaped_image = np_images[j].ravel()

            # augment
            c_augmentor.complexPGL2TransformImage(
                reshaped_image, 
                64, 
                a_r_slider.val, 
                a_i_slider.val,
                b_r_slider.val, 
                b_i_slider.val, 
                c_r_slider.val, 
                c_i_slider.val, 
                d_r_slider.val, 
                d_i_slider.val,
            )

            # convert back to 3d tensor
            np_images[j] = reshaped_image.reshape((3, 64, 64))

        # convert back to tensor
        new_images = torch.from_numpy(np_images)

        # rescale to original 
        new_images = torchvision.transforms.Resize(
            size=[32, 32], 
            antialias=False, 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )(new_images)

        # plot new images
        np_new_image = np.transpose(torchvision.utils.make_grid(new_images).numpy(), (1, 2, 0))
        ax.imshow(np_new_image)
        fig.canvas.draw_idle()

    # on slider change augment images
    a_r_slider.on_changed(update)
    a_i_slider.on_changed(update)
    b_r_slider.on_changed(update)
    b_i_slider.on_changed(update)
    c_r_slider.on_changed(update)
    c_i_slider.on_changed(update)
    d_r_slider.on_changed(update)
    d_i_slider.on_changed(update)

    plt.show()

# PGL(2, R)^2 slider
def R_PGL2_sq(trainloader) : 
    """
    Starts a PGL(R, 2)^2 slider, allowing you to view the 
    effects of these transformations on an image.
    """

    # get images to augment
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # create plotting figure
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('PGL(2, R)^2 slider')
    axcolor = 'yellow'

    # plot images
    np_image = np.transpose(torchvision.utils.make_grid(images).numpy(), (1, 2, 0))
    img = ax.imshow(np_image)

    a_1_slider_ax = plt.axes([0.20, 0.95, 0.65, 0.03], facecolor=axcolor)
    a_1_slider = Slider(a_1_slider_ax, "a_1 value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    b_1_slider_ax = plt.axes([0.20, 0.90, 0.65, 0.03], facecolor=axcolor)
    b_1_slider = Slider(b_1_slider_ax, "b_1 value", valstep=step, valmin=-width, valmax=width, valinit=0)

    c_1_slider_ax = plt.axes([0.20, 0.85, 0.65, 0.03], facecolor=axcolor)
    c_1_slider = Slider(c_1_slider_ax, "c_1 value", valstep=step, valmin=-width, valmax=width, valinit=0)

    d_1_slider_ax = plt.axes([0.20, 0.80, 0.65, 0.03], facecolor=axcolor)
    d_1_slider = Slider(d_1_slider_ax, "d_1 value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    a_2_slider_ax = plt.axes([0.20, 0.15, 0.65, 0.03], facecolor=axcolor)
    a_2_slider = Slider(a_2_slider_ax, "a_2 value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)
    
    b_2_slider_ax = plt.axes([0.20, 0.10, 0.65, 0.03], facecolor=axcolor)
    b_2_slider = Slider(b_2_slider_ax, "b_2 value", valstep=step, valmin=-width, valmax=width, valinit=0)

    c_2_slider_ax = plt.axes([0.20, 0.05, 0.65, 0.03], facecolor=axcolor)
    c_2_slider = Slider(c_2_slider_ax, "c_2 value", valstep=step, valmin=-width, valmax=+width, valinit=0)

    d_2_slider_ax = plt.axes([0.20, 0, 0.65, 0.03], facecolor=axcolor)
    d_2_slider = Slider(d_2_slider_ax, "d_2 value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    def update(val) :
        # clone and rescale x2
        new_images = torch.clone(images)
        
        new_images = torchvision.transforms.Resize(
            size=[32 * 2, 32 * 2], 
            antialias=False, 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )(images)
        
        # convert to np
        np_images = new_images.numpy().astype(dtype=np.float64)

        for j in range(len(np_images)) : 
            # convert 3d tensor to 1d array
            reshaped_image = np_images[j].ravel()

            # augment
            c_augmentor.realPGL2SqrTransformImage(
                reshaped_image, 
                64,
                a_1_slider.val, 
                b_1_slider.val,
                c_1_slider.val,
                d_1_slider.val,
                a_2_slider.val,
                b_2_slider.val,
                c_2_slider.val,
                d_2_slider.val
            )

            # convert back to 3d tensor
            np_images[j] = reshaped_image.reshape((3, 64, 64))

        # convert back to tensor
        new_images = torch.from_numpy(np_images)

        # rescale to original 
        new_images = torchvision.transforms.Resize(
            size=[32, 32], 
            antialias=False, 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )(new_images)

        # plot new images
        np_new_image = np.transpose(torchvision.utils.make_grid(new_images).numpy(), (1, 2, 0))
        ax.imshow(np_new_image)
        fig.canvas.draw_idle()

    # on slider change augment images
    a_1_slider.on_changed(update)
    b_1_slider.on_changed(update)
    c_1_slider.on_changed(update)
    d_1_slider.on_changed(update)
    a_2_slider.on_changed(update)
    b_2_slider.on_changed(update)
    c_2_slider.on_changed(update)
    d_2_slider.on_changed(update)

    plt.show()

# PGL(3, R) slider
def R_PGL3(trainloader) :
    """
    Starts a PGL(R, 3) slider, allowing you to view the 
    effects of these transformations on an image.
    """

    # get images to augment
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # create plotting figure
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("PGL(3, R) slider")
    axcolor = 'yellow'

    # plot images
    np_image = np.transpose(torchvision.utils.make_grid(images).numpy(), (1, 2, 0))
    img = ax.imshow(np_image) 

    a_1_slider_ax = plt.axes([0.20, 0.95, 0.65, 0.03], facecolor=axcolor)
    a_1_slider = Slider(a_1_slider_ax, "a_1 value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    b_1_slider_ax = plt.axes([0.20, 0.90, 0.65, 0.03], facecolor=axcolor)
    b_1_slider = Slider(b_1_slider_ax, "b_1 value", valstep=step, valmin=-width, valmax=width, valinit=0)

    c_1_slider_ax = plt.axes([0.20, 0.85, 0.65, 0.03], facecolor=axcolor)
    c_1_slider = Slider(c_1_slider_ax, "c_1 value", valstep=step, valmin=-width, valmax=width, valinit=0)

    a_2_slider_ax = plt.axes([0.20, 0.80, 0.65, 0.03], facecolor=axcolor)
    a_2_slider = Slider(a_2_slider_ax, "a_2 value", valstep=step, valmin=-width, valmax=width, valinit=0)
    
    b_2_slider_ax = plt.axes([0.20, 0.20, 0.65, 0.03], facecolor=axcolor)
    b_2_slider = Slider(b_2_slider_ax, "b_2 value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    c_2_slider_ax = plt.axes([0.20, 0.15, 0.65, 0.03], facecolor=axcolor)
    c_2_slider = Slider(c_2_slider_ax, "c_2 value", valstep=step, valmin=-width, valmax=width, valinit=0)

    a_3_slider_ax = plt.axes([0.20, 0.10, 0.65, 0.03], facecolor=axcolor)
    a_3_slider = Slider(a_3_slider_ax, "a_3 value", valstep=step, valmin=-width, valmax=width, valinit=0)
    
    b_3_slider_ax = plt.axes([0.20, 0.05, 0.65, 0.03], facecolor=axcolor)
    b_3_slider = Slider(b_3_slider_ax, "b_3 value", valstep=step, valmin=-width, valmax=width, valinit=0)

    c_3_slider_ax = plt.axes([0.20, 0, 0.65, 0.03], facecolor=axcolor)
    c_3_slider = Slider(c_3_slider_ax, "c_3 value", valstep=step, valmin=1-width, valmax=1+width, valinit=1)

    # updates coefficients when sliders are moved
    def update(val) :
        # clone and rescale x2
        new_images = torch.clone(images)
        
        new_images = torchvision.transforms.Resize(
            size=[32 * 2, 32 * 2], 
            antialias=False, 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )(images)
        
        # convert to np
        np_images = new_images.numpy().astype(dtype=np.float64)

        for j in range(len(np_images)) : 
            # convert 3d tensor to 1d array
            reshaped_image = np_images[j].ravel()

            # augment
            c_augmentor.realPGL3TransformImage(
                reshaped_image, 
                64,
                a_1_slider.val, 
                b_1_slider.val,
                c_1_slider.val,
                a_2_slider.val,
                b_2_slider.val,
                c_2_slider.val,
                a_3_slider.val,
                b_3_slider.val,
                c_3_slider.val
            )

            # convert back to 3d tensor
            np_images[j] = reshaped_image.reshape((3, 64, 64))

        # convert back to tensor
        new_images = torch.from_numpy(np_images)

        # rescale to original 
        new_images = torchvision.transforms.Resize(
            size=[32, 32], 
            antialias=False, 
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        )(new_images)

        # plot new images
        np_new_image = np.transpose(torchvision.utils.make_grid(new_images).numpy(), (1, 2, 0))
        ax.imshow(np_new_image)
        fig.canvas.draw_idle()

    # on slider change augment images
    a_1_slider.on_changed(update)
    b_1_slider.on_changed(update)
    c_1_slider.on_changed(update)
    a_2_slider.on_changed(update)
    b_2_slider.on_changed(update)
    c_2_slider.on_changed(update)
    a_3_slider.on_changed(update)
    b_3_slider.on_changed(update)
    c_3_slider.on_changed(update)

    plt.show()

# main method imports data and asks which type of augmentation
# the user would like to visualize
if __name__ == '__main__' : 
    # import dataset
    batch_size = 4

    trainset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True,
        download=True, 
        transform=transforms.ToTensor()
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )    
    
    quit_running = False
    while quit_running == False : 
        invalid_digit = True
        while invalid_digit : 
            print("\nWhich augmentation slider would you like to start?")
            print("\t(1) PGL(2, C)")
            print("\t(2) PGL(2, R)^2")
            print("\t(3) PGL(3, R)")
            print("\t(4) Quit\n")

            chosen_aug = input("Slider: ")

            if not chosen_aug.isdigit() : 
                print("\nYou have entered an invalid input, try again")
                continue 
                
            chosen_aug = int(chosen_aug)
            if 1 <= chosen_aug and chosen_aug <= 4 : 
                invalid_digit = False
            else : 
                print("\nYou have entered an invalid input, try again")
                continue 
        
        if chosen_aug == 1 : 
            C_PGL2(trainloader)
        elif chosen_aug == 2 : 
            R_PGL2_sq(trainloader)
        elif chosen_aug == 3 :
            R_PGL3(trainloader)
        elif chosen_aug == 4 : 
            quit_running = True 

        

    