from lib import *
from dataset import *

img_folder_path = "C:/Users/Admin/ss-torch/data/images"
mask_folder_path = "C:/Users/Admin/ss-torch/data/masks"

color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
trainsize = 256

train_dataset = CarSegmentationDataset(img_folder_path, mask_folder_path, phase='train', transform=DataTransform(input_size=trainsize, color_mean=color_mean, color_std=color_std))
img, mask= train_dataset.__getitem__(10)

print(img.shape)
print(mask.shape)