from ptdataset.utils import CustomPytorchDataset, evaluate_model, load_dataset_in_memory, store_masks_to_disk
from ptdataset import ID_TO_COLOR, ID_TO_LABEL
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from ptdataset.settings import settings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import cv2
import os

if __name__ == '__main__':
    training_data = load_dataset_in_memory()

    model = deeplabv3_resnet50(weights=settings.DEEPLAB_PRETRAINED_WEIGHTS)
    model.classifier[4] = torch.nn.Conv2d(256, settings.NUM_CLASSES, kernel_size=(1,1))
    model.to(settings.DEVICE)

    if settings.RUN_MODE == "test":
        print("Loading model state dict from:", settings.TRAINED_MODEL_PATH)
        state_dict = torch.load(settings.TRAINED_MODEL_PATH, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        if settings.GENERATE_MASKS:
            print("Generating masks for dataset")
            store_masks_to_disk(training_data)
        # Shuffle and split the dataset in 80% train, 20% val
        # Commented for debugging
        #random.shuffle(DATASET_DATA)
        train_dataset = CustomPytorchDataset(training_data[:int(len(training_data) * 0.8)])
        val_dataset = CustomPytorchDataset(training_data[int(len(training_data) * 0.8):])

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
        image_files = os.listdir(settings.INPUT_IMAGES_DIR)
        print(f"Starting to process {len(image_files)} images...")
        model.eval()

        legend_patches = []
        for cls_id, color in ID_TO_COLOR.items():
            if cls_id == 0:
                continue
            legend_color = tuple(np.array(color) / 255.0)
            legend_patches.append(
                mpatches.Patch(color=legend_color, label=ID_TO_LABEL[cls_id])
            )

        for idx, filename in enumerate(image_files):
            print(f"[{idx + 1}/{len(image_files)}] Processing {filename}")

            img_path = os.path.join(settings.INPUT_IMAGES_DIR, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))

            input_tensor = (
                torch.from_numpy(img_resized.transpose(2, 0, 1))
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(settings.DEVICE)
            )

            with torch.no_grad():
                output = model(input_tensor)["out"]
                pred_mask = output.argmax(1).squeeze(0).cpu().numpy()

            pred_overlay = img_resized.copy()

            for cls_id, color in ID_TO_COLOR.items():
                if cls_id == 0:
                    continue
                pred_overlay[pred_mask == cls_id] = (
                    0.5 * img_resized[pred_mask == cls_id]
                    + 0.5 * np.array(color)
                ).astype(np.uint8)


            fig = plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(img_resized)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(pred_overlay)
            plt.title("Predicted Segmentation")
            plt.axis("off")

            fig.legend(
                handles=legend_patches,
                loc="lower center",
                ncol=len(legend_patches),
                bbox_to_anchor=(0.5, -0.05),
                fontsize=11,
            )

            output_path = os.path.join(
                settings.OUTPUT_IMAGES_DIR,
                os.path.splitext(filename)[0] + "_prediction.png",
            )

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
