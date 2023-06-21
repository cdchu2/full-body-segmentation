import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import segmentation_models_pytorch as sm
import albumentations as A
from albumentations.pytorch import ToTensorV2 #np.array -> torch.tensor
from torchmetrics import Dice, JaccardIndex # dice vs iou
import os
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

# conda install --file requirements.txt