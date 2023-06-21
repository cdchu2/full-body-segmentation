from lib import *
from dataset import *

img_folder_path = "C:/Users/Admin/ss-torch/data/images"
mask_folder_path = "C:/Users/Admin/ss-torch/data/masks"

train_dataset = CarSegmentationDataset(img_folder_path, mask_folder_path, phase='train')
img, mask= train_dataset.__getitem__(10)

train_images, test_images, train_masks, test_masks = train_test_split(
    train_dataset.images, train_dataset.masks, test_size=0.2, random_state=42
)
# train_dataset = CarSegmentationDataset(img_folder_path, mask_folder_path, phase='train', images=train_images, masks=train_masks)
# test_dataset = CarSegmentationDataset(img_folder_path, mask_folder_path, phase='test', images=test_images, masks=test_masks)

print(img.shape)