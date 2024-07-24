import os
import sys
import random
import time
import copy
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from PIL import Image

warnings.filterwarnings("ignore")


def load_paths_from_folder(data_path):
    label_names = os.listdir(data_path)
    label_dict = {i: label for i, label in enumerate(label_names)}

    file_paths = []
    for label, folder_name in label_dict.items():
        label_folder_path = os.path.join(data_path, folder_name)
        for file in os.listdir(label_folder_path):
            file_paths.append((os.path.join(label_folder_path, file), label))

    random.shuffle(file_paths)
    train_num = int(0.6 * len(file_paths))
    val_num = int(0.25 * len(file_paths))
    return file_paths[:train_num], file_paths[train_num:train_num + val_num], file_paths[
                                                                              train_num + val_num:], label_dict


class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.file_list[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.file_list)


def run_validation(model, criterion, val_loader):
    model.eval()
    pred_labels, orig_labels = [], []
    running_loss, ntotal = 0.0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, targets)
            ntotal += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            orig_labels.extend(targets.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    val_loss = running_loss / ntotal
    val_acc = accuracy_score(orig_labels, pred_labels)
    val_f1 = f1_score(orig_labels, pred_labels, average='macro')
    return val_loss, val_acc, val_f1, orig_labels, pred_labels


def plot_train_val_loss(tra_val_loss):
    epochs = list(range(1, len(tra_val_loss) + 1))
    train_loss = [x[0] for x in tra_val_loss]
    val_loss = [x[1] for x in tra_val_loss]
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def run_training(model, criterion, num_epochs, lr, check_after):
    best_f1, best_epoch = 0, 0
    train_val_loss_epoch = []
    start_training = time.time()

    for epoch in range(num_epochs):
        # Adjust learning rate
        if epoch < num_epochs // 6:
            current_lr = lr
        elif epoch < num_epochs // 3:
            current_lr = lr / 2
        elif epoch < num_epochs // 2:
            current_lr = lr / 4
        elif epoch < 3 * (num_epochs // 4):
            current_lr = lr / 10
        else:
            current_lr = lr / 20

        optimizer = optim.Adam(model.parameters(), lr=current_lr) if epoch >= 2 else optim.SGD(model.parameters(),
                                                                                               lr=current_lr,
                                                                                               momentum=0.9,
                                                                                               weight_decay=1e-4)
        model.train()
        running_loss, running_corrects, ntotal = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.data, 1)
            ntotal += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == targets.data)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'lr: {current_lr:.6f}')
        print(f'Train Loss: {running_loss / ntotal:.4f} Accuracy: {running_corrects.item() / ntotal:.4f}')

        # Validation
        if (epoch + 1) % check_after == 0:
            val_loss, val_acc, val_f1, _, _ = run_validation(model, criterion, val_loader)
            train_val_loss_epoch.append([running_loss / ntotal, val_loss])
            print(
                f"Epoch: {epoch + 1}/{num_epochs}\tVal Loss: {val_loss:.4f}\tAccuracy: {val_acc:.4f}\tF1-score: {val_f1:.4f}")

            if val_f1 > best_f1:
                print('Saving model')
                best_f1, best_epoch = val_f1, epoch
                best_model = copy.deepcopy(model)
                torch.save({
                    'model': best_model.state_dict(),
                    'f1': best_f1,
                    'epoch': epoch,
                }, 'resnet18_best.pth')

    print(f'Training complete in {time.time() - start_training:.0f}s')
    print(f'Best validation F1 score: {best_f1:.4f} at epoch: {best_epoch + 1}')
    plot_train_val_loss(train_val_loss_epoch)
    return train_val_loss_epoch


if __name__ == '__main__':
    # GPU setup
    use_gpu = torch.cuda.is_available()
    print('Using GPU:', use_gpu)
    device = torch.device("cuda" if use_gpu else "cpu")

    # Data setup
    train_list, val_list, test_list, label_dict = load_paths_from_folder("Chicken_Duck_Dataset")
    print("Number of files in training set:", len(train_list))
    print("Number of files in validation set:", len(val_list))
    print("Number of files in test set:", len(test_list))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    batch_size = 16
    train_loader = DataLoader(CustomDataset(train_list, transform=data_transforms['train']),
                              batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(CustomDataset(val_list, transform=data_transforms['val']),
                            batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model setup
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    # Start training
    print('Start training ...')
    criterion = nn.CrossEntropyLoss()
    train_val_loss = run_training(model, criterion, num_epochs=35, lr=1e-4, check_after=1)
