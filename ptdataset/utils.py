from models.dataset import DatasetLabelMeModel
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from torchvision import transforms
from . import LABEL_TO_ID
from .settings import settings
import numpy as np
import torch
import json
import os
import cv2

def load_dataset_in_memory() -> list[DatasetLabelMeModel]:
    result = []
    for f in os.listdir(settings.DATASET_PATH):
        if f.endswith(".json"):
            full_path = os.path.join(settings.DATASET_PATH, f)
            with open(full_path, 'r') as f:
                data = json.load(f)
                result.append(DatasetLabelMeModel(**data))
    return result

def create_mask_from_dataset(data: DatasetLabelMeModel):
    mask = np.zeros((data.imageHeight, data.imageWidth), dtype=np.uint8)

    for shape in data.shapes:
        pts = np.array([list(p) for p in shape.points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], LABEL_TO_ID[shape.label])

    return mask

def store_masks_to_disk(data: list[DatasetLabelMeModel], include_scaled_masks: bool = False):
    for i, d in enumerate(data):
        out_path=f"{settings.MASKS_DIR_NAME}/{d.imagePath}_mask.png"
        # Actual data needed, will look as basically black
        mask = create_mask_from_dataset(d)
        cv2.imwrite(out_path, mask)

        if include_scaled_masks:
            # Human friendly masks, only for visualization
            out_scaled_path=f"{settings.MASKS_DIR_NAME}/scaled/{d.imagePath}_mask.png"
            scaled_mask = (mask * (255 // (mask.max() if mask.max() > 0 else 1))).astype(np.uint8)
            cv2.imwrite(out_scaled_path, scaled_mask)

        if i % 10 == 0:
            print(f'Processed image {i + 1}/{len(data)}')

class CustomPytorchDataset(Dataset):
    def __init__(self, dataset: list[DatasetLabelMeModel]):
        self.img_dir = settings.DATASET_PATH
        self.mask_dir = settings.MASKS_DIR_NAME
        self.files = [ v.imagePath for v in dataset ]
        self.transform_img = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = self.transform_img(img)

        mask = cv2.imread(os.path.join(self.mask_dir, fname + "_mask.png"), 0)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()

        return img, mask

def evaluate_model(model, loader, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(settings.DEVICE)
            masks = masks.to(settings.DEVICE)
            outputs = model(imgs)['out']
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(masks.cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, labels=list(range(num_classes)), average=None)
    return (accuracy, f1)