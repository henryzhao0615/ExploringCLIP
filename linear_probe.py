import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import open_clip
from tqdm import tqdm
import numpy as np
import os
from utils import *
set_seed(42)

def sample_few_shot(dataset, shots, num_classes=10):
    """
    Given a dataset (or a Subset), sample 'shots' examples per class.
    Returns a Subset containing the few-shot samples.
    """
    
    if isinstance(dataset, Subset):
        underlying = dataset.dataset
        indices = dataset.indices
    else:
        underlying = dataset
        indices = list(range(len(dataset)))
        
    
    class_to_indices = {i: [] for i in range(num_classes)}
    for idx in indices:
        target = underlying.targets[idx]
        class_to_indices[target].append(idx)
    
    selected_indices = []
    for i in range(num_classes):
        
        selected = np.random.choice(class_to_indices[i], shots, replace=False)
        selected_indices.extend(selected)
    return Subset(underlying, selected_indices)

def train(model, classifier, train_loader, optimizer, device, num_epochs=50):
    """
    Train the linear classifier (on top of frozen CLIP features) using AdamW.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval() 
    classifier.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
           
            with torch.no_grad():
                features = model.encode_image(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        epoch_loss /= len(train_loader.dataset)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return classifier

def evaluate_classifier(model, classifier, loader, device):
    """
    Evaluate the classifier (accuracy) on a given dataset loader.
    """
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = model.encode_image(images)
            logits = classifier(features)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def find_peak(model, train_loader, val_loader, device, wd_list, num_epochs=20):
    """
    Performs hyperparameter tuning over a list of weight decay values.
    For each candidate, trains the linear classifier for a few epochs and 
    returns the best weight decay (and the classifier state_dict) according to validation accuracy.
    """
    best_acc = 0.0
    best_wd = None
    best_state = None
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        feature_dim = model.encode_image(dummy).shape[-1]

    for wd in wd_list:
        classifier = nn.Linear(feature_dim, 10).to(device)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=wd)
        train(model, classifier, train_loader, optimizer, device, num_epochs=num_epochs)
        acc = evaluate_classifier(model, classifier, val_loader, device)
        print(f"WD: {wd:.6f}, Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_wd = wd
            best_state = classifier.state_dict()
    print(f"Best WD: {best_wd}, Val Acc: {best_acc:.4f}")
    return best_wd, best_state

def infer(model, classifier, test_loader, device):

    classifier.eval()
    model.eval()
    all_preds = []
    all_labels = []
    all_top5 = [] 
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model.encode_image(images)
            logits = classifier(features)

     
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        
            top5_preds = logits.topk(5, dim=1).indices 
            all_top5.append(top5_preds.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    top5_all = torch.cat(all_top5)

    top5_correct = top5_all.eq(all_labels.unsqueeze(1)).any(dim=1)
    top5_accuracy = top5_correct.float().mean().item()

    accuracy = (all_preds == all_labels).float().mean().item()
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy (Top-1): {accuracy:.4f}")
    print(f"Test Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)])
    plt.yticks(tick_marks, [str(i) for i in range(10)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    # if not os.path.exists("fig"):
    #     os.makedirs("fig")
    # save_path = os.path.join("fig", "confusion_matrix.png")
    # plt.savefig(save_path)
    # print(f"Confusion matrix saved to {save_path}")
    
    # plt.show()

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


def evaluate(args):
    # Grid of weight decay values to search over.
    wd_list = np.logspace(-5, -1, num=10).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Open CLIP ViT-B-32 model and its preprocessing transform.
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.to(device)
    # Freeze CLIP parameters.
    for param in model.parameters():
        param.requires_grad = False

    # Load CIFAR-10 full training and test datasets.
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
    
    # Split full_train into training and validation sets (80/20 split).
    num_train = int(0.8 * len(full_train))
    num_val = len(full_train) - num_train
    train_set, val_set = random_split(full_train, [num_train, num_val])
    # breakpoint()

    # Define the few-shot settings.
    shots_list = [1, 2, 4, 8, 16, "full"]
    results = {}

    # For each shot experiment, perform hyperparameter tuning, training, and evaluation.
    for shot in shots_list:
        print(f"\n=== {shot}-shot experiment ===")
        # For few-shot, sample k examples per class; for full shot, use the entire training set.
        if shot == "full":
            train_subset = train_set
        else:
            train_subset = sample_few_shot(train_set, shot, num_classes=10)
            
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        full_train_loader = DataLoader(full_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Determine feature dimension using a dummy input.
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            feature_dim = model.encode_image(dummy).shape[-1]

        # Use the validation set to search for the best weight decay.
        best_wd, best_state = find_peak(model, train_loader, val_loader, device, wd_list, num_epochs=args.tune_epochs)
        
        # Initialize a new classifier and load the best state from tuning.
        classifier = nn.Linear(feature_dim, 10).to(device)
        classifier.load_state_dict(best_state)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=best_wd)
        
        # Train final classifier for more epochs.
        if shot == "full":
            classifier = train(model, classifier, full_train_loader, optimizer, device, num_epochs=args.train_epochs)
        else:
            classifier = train(model, classifier, train_loader, optimizer, device, num_epochs=args.train_epochs)
        
        print("Final performance on test set:")
        metrics = infer(model, classifier, test_loader, device)
        results[shot] = metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Probe Evaluation on CIFAR-10 using Open CLIP.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument('--tune_epochs', type=int, default=20, help="Epochs for weight decay tuning (find_peak).")
    parser.add_argument('--train_epochs', type=int, default=50, help="Epochs for final training with best WD.")
    args = parser.parse_args()
    evaluate(args)
