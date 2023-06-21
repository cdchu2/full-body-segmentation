import sys
sys.path.append("..")
from lib import *

trainsize = 256
class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": A.Compose([
                A.Resize(width=trainsize, height=trainsize),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.Blur(),
                A.RGBShift(),
                A.Cutout(num_holes=8, max_h_size=25, max_w_size=25, fill_value=0),
                A.Normalize(),
                ToTensorV2(),
            ]),
            "val": A.Compose([
                A.Resize(width=trainsize, height=trainsize),
                A.Normalize(),
                ToTensorV2(),
            ])
        }
    
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](image=img, mask=anno_class_img)


class CarSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, phase, transform=None, images=None, masks=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.phase = phase
        self.transform = transform

        if images is not None and masks is not None:
            self.images = images
            self.masks = masks
        else:
            self.images = sorted(os.listdir(image_dir))  
            self.masks = sorted(os.listdir(mask_dir))   

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = (mask > 0).astype(np.uint8)
        mask = np.expand_dims(mask, axis=2)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError("Height and Width of image and mask should be equal.")

        if self.transform:
            transformed = self.transform(self.phase, img=image, anno_class_img=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask
