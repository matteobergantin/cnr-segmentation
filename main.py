from ptdataset.utils import CustomPytorchDataset, evaluate_model, load_dataset_in_memory
from ptdataset import ID_TO_COLOR, ID_TO_LABEL
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from ptdataset.settings import settings
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
import torch
import cv2
import os

DATASET_DATA = load_dataset_in_memory()

model = deeplabv3_resnet50(weights=settings.DEEPLAB_PRETRAINED_WEIGHTS)
model.classifier[4] = torch.nn.Conv2d(256, settings.NUM_CLASSES, kernel_size=(1,1))
model.to(settings.DEVICE)

if settings.RUN_MODE == "test":
    print("Loading model state dict from:", settings.TRAINED_MODEL_PATH)
    state_dict = torch.load(settings.TRAINED_MODEL_PATH)
    model.load_state_dict(state_dict)
else:
    # Shuffle and split the dataset in 80% train, 20% val
    # Commented for debugging
    #random.shuffle(DATASET_DATA)
    train_dataset = CustomPytorchDataset(DATASET_DATA[:int(len(DATASET_DATA) * 0.8)])
    val_dataset = CustomPytorchDataset(DATASET_DATA[int(len(DATASET_DATA) * 0.8):])

    train_loader = DataLoader(train_dataset, batch_size=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    print("Begin Training...")
    weights = torch.tensor(settings.CLASS_WEIGHTS).to(settings.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(train_loader):
            imgs, masks = imgs.to(settings.DEVICE), masks.to(settings.DEVICE)

            optimizer.zero_grad()
            output = model(imgs)["out"]
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader)}")

if settings.EVALUATE_MODEL:
    print("Evaluating model...")
    accuracy, f1_scores = evaluate_model(model, val_loader, num_classes=settings.NUM_CLASSES, device=settings.DEVICE)

    print(f"Pixel Accuracy: {(accuracy * 100):.2f}%")
    for i, score in enumerate(f1_scores):
        print(f"F1-score for class {ID_TO_LABEL.get(i, 'background')}: {score:.4f}")

if settings.RUN_MODE == "test":
    print("Running Model on random sample")
    # Pick random sample
    idx = random.randint(0, len(DATASET_DATA)-1)
    sample = DATASET_DATA[idx]

    img_path = os.path.join(settings.DATASET_PATH, sample.imagePath)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (512, 512))

    mask_path = os.path.join(settings.MASKS_DIR_NAME, sample.imagePath + "_mask.png")
    mask = cv2.imread(mask_path, 0)
    mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    mask_color = np.zeros_like(img_resized)
    for cls_id, color in ID_TO_COLOR.items():
        mask_color[mask_resized == cls_id] = color

    input_tensor = torch.from_numpy(img_resized.transpose(2,0,1)/255.).unsqueeze(0).float().to(settings.DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_mask = output.argmax(1).squeeze(0).cpu().numpy()

    pred_overlay = np.zeros_like(img_resized)
    for cls_id, color in ID_TO_COLOR.items():
        pred_overlay[pred_mask == cls_id] = color

    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(img_resized)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(mask_color)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_overlay)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()