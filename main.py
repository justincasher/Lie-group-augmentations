# general
import numpy as np
import matplotlib.pyplot as plt
import os
import concurrent.futures
import multiprocessing as mp
import random

# torch
import torch
import torchvision 
import torchvision.transforms as transforms

# training
from training import training
    
if __name__ == '__main__' : 
    """
    Main method. Trains networks by randomly selecting deviations
    (or whatever is desired)
    """
    
    for i in range(10) :
        deviation = random.uniform(0, 0.015) 

        print(f"--- Running PGL(2, C) with deviation {round(deviation, 4)} ---")
        
        name = "PGL(2, C)_dev=" + str(round(deviation, 4)) + "_pct=20_" + str(i+1)
        training(deviation=deviation, complexPGL2Pct=20, realPGL2SquaredPct=0, realPGL3Pct=0, save_name=name)
