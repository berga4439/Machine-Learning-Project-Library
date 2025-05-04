import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#data set for card dataset
class CardSet(Dataset):
    def __init__(self, csv_file, set_type, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["data set"] == set_type].reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.data["card type"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = "./dataset/" + self.data.iloc[item]["filepaths"]
        label = self.class_to_idx[self.data.iloc[item]["card type"]]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

#makes sure every image is 224x224 and convert it to tensor for CNN
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# My model is a simple CNN that reduces the image data to 14 classes
# 3 channel 224x224 to 16 channels
# max pool to reduce size
# 16 to 32 channels etc.
# Then the model is flattened to linear for classification
class CardCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(CardCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


def trainNN(epochs=5, batch_size=16, lr=0.001):
    train_set = CardSet("./dataset/cards.csv", "train", transform=transform)


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    card_cnn = CardCNN()

    cross_entropy = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(card_cnn.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, (image, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = card_cnn(image)
            loss = cross_entropy(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0

    return card_cnn



model = trainNN(epochs=5, batch_size=32)


print("Model training done")

test_set = CardSet("./dataset/cards.csv", "test", transform=transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())



cm = confusion_matrix(y_true, y_pred)
disp_cm = ConfusionMatrixDisplay(cm, display_labels=test_set.classes)
disp_cm.plot()
plt.show()

