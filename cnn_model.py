# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load USPS Dataset
train_dataset = torchvision.datasets.USPS(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.USPS(root='./data', train=False, transform=transform, download=True)

plt.imshow(train_dataset[0][0].squeeze())

train_dataset[0][0].shape

# Split dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 32
# Define dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
num_classes = 10

class CNN_model(nn.Module):
    def __init__(self, num_classes, configuration):
        super(CNN_model, self).__init__()
        if configuration == "configuration1":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
            self.fc = nn.Linear(32*4*4, num_classes)
        elif configuration == "configuration2":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(64*4*4, num_classes)
        elif configuration == "configuration3":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(16*4*4, num_classes)
        else:
            raise ValueError("Invalid configuration provided.")

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

cnn_model_conf1 = CNN_model(num_classes, configuration="configuration1")
cnn_model_conf2 = CNN_model(num_classes, configuration="configuration2")
cnn_model_conf3 = CNN_model(num_classes, configuration="configuration3")

# cnn_model = CNN_model(num_classes)

criterion = nn.CrossEntropyLoss()
cnn_optimizer_conf1 = optim.Adam(cnn_model_conf1.parameters(), lr=0.001)
cnn_optimizer_conf2 = optim.Adam(cnn_model_conf2.parameters(), lr=0.001)
cnn_optimizer_conf3 = optim.Adam(cnn_model_conf3.parameters(), lr=0.001)

# Train CNN model
def train_cnn(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    print("Training CNN : ")
    writer = SummaryWriter("runs/cnn_experiment")
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        running_train_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            running_train_loss += loss.item()
            optimizer.step()

            if (i+1) % 100 == 0:
                print (f'CNN Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

        avg_loss =  (running_train_loss / len(train_loader))
        writer.add_scalar('CNN/Train Loss', avg_loss, epoch)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f'CNN Validation Accuracy: {accuracy:.4f}')
            writer.add_scalar('CNN/Validation Accuracy', accuracy, epoch)
    writer.close()

    return model

# Train CNN model
print("Training Configuration 1")
cnn_model_conf1 = train_cnn(cnn_model_conf1, criterion, cnn_optimizer_conf1, train_loader, val_loader,num_epochs = 10)

print("Training Configuration 2")
cnn_model_conf2 = train_cnn(cnn_model_conf2, criterion, cnn_optimizer_conf2, train_loader, val_loader,num_epochs = 10)

print("Training Configuration 3")
cnn_model_conf3 = train_cnn(cnn_model_conf3, criterion, cnn_optimizer_conf3, train_loader, val_loader,num_epochs = 10)

def evaluate_cnn_model(cnn_model, test_loader):
    cnn_model.eval()
    cnn_predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs.data, 1)
            cnn_predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    cnn_accuracy = accuracy_score(true_labels, cnn_predictions)
    cnn_precision = precision_score(true_labels, cnn_predictions, average='weighted')
    cnn_recall = recall_score(true_labels, cnn_predictions, average='weighted')
    cnn_conf_matrix = confusion_matrix(true_labels, cnn_predictions)
    print("CNN Evaluation Metrics:")
    print("Accuracy:", cnn_accuracy)
    print("Precision:", cnn_precision)
    print("Recall:", cnn_recall)
    print("Confusion matrix:", cnn_conf_matrix)

    # Add PR Curve
    writer = SummaryWriter("runs/cnn_experiment")
    class_probs = []
    class_label = []
    with torch.no_grad():
        for features, labels in test_loader:
            output = cnn_model(features)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)
    classes = [str(i) for i in range(10)]

    # Helper function for adding PR curve to TensorBoard
    def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
        tensorboard_truth = test_label == class_index
        tensorboard_probs = test_probs[:, class_index]

        writer.add_pr_curve(classes[class_index],
                            tensorboard_truth,
                            tensorboard_probs,
                            global_step=global_step)
        writer.close()

    # Plot all the PR curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_label)

print()
print("Evaluating Configuration 1")
evaluate_cnn_model(cnn_model_conf1, test_loader)

print()
print("Evaluating Configuration 2")
evaluate_cnn_model(cnn_model_conf2, test_loader)

print()
print("Evaluating Configuration 3")
evaluate_cnn_model(cnn_model_conf3, test_loader)