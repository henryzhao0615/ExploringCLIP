import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import open_clip
from functools import partial
from tqdm import tqdm
import numpy as np
import os
from utils import *
set_seed(42)

def zero_shot_classifier(model, tokenizer, classnames, templates, device):
    """
    Computes zero-shot text features for each class.
    For each class, multiple text prompts are generated using the provided templates.
    The features for each class are the (normalized) average of the encoded text prompts.
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            # Generate text prompts for this class
            texts = [template.format(cls=classname) for template in templates]
            # Tokenize the texts
            tokenized_text = tokenizer(texts).to(device)
            # Encode the texts
            text_features = model.encode_text(tokenized_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Average over the multiple prompts (if more than one)
            class_feature = text_features.mean(dim=0)
            class_feature = class_feature / class_feature.norm()
            zeroshot_weights.append(class_feature)
        # Stack into a matrix of shape [feature_dim, num_classes]
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def run_classification(args, model, preprocess, zeroshot_weights, device, template):
    """
    Loads the CIFAR-10 test dataset, computes image features using the CLIP image encoder,
    and then computes classification metrics using the dot product of image and text features.
    """
    # Load CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    all_top1_preds = []
    all_top5_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            # Obtain image features and normalize
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Scale the logits by the learned logit scale
            logit_scale = model.logit_scale.exp()
            # Compute logits (cosine similarity * scale)
            logits = logit_scale * (image_features @ zeroshot_weights)
            
            # Top-1 prediction
            preds_top1 = logits.argmax(dim=1)
            all_top1_preds.append(preds_top1.cpu())
            
            # Top-5 predictions
            preds_top5 = torch.topk(logits, k=5, dim=1).indices.cpu()
            all_top5_preds.append(preds_top5)
            
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_top1_preds = torch.cat(all_top1_preds)
    all_labels = torch.cat(all_labels)
    all_top5_preds = torch.cat(all_top5_preds)

    # Compute Top-1 accuracy
    top1_accuracy = (all_top1_preds == all_labels).float().mean().item()
    
    # Compute Top-5 accuracy: count sample if true label appears in top 5
    top5_correct = 0
    for i in range(all_labels.size(0)):
        if all_labels[i] in all_top5_preds[i]:
            top5_correct += 1
    top5_accuracy = top5_correct / all_labels.size(0)

    # Compute precision, recall, F1 using macro-average for multi-class
    precision = precision_score(all_labels, all_top1_preds, average='macro')
    recall = recall_score(all_labels, all_top1_preds, average='macro')
    f1 = f1_score(all_labels, all_top1_preds, average='macro')
    cm = confusion_matrix(all_labels, all_top1_preds)

    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 绘制混淆矩阵图像，并在每个单元格中显示值
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(test_dataset.classes))
    plt.xticks(tick_marks, test_dataset.classes, rotation=45)
    plt.yticks(tick_marks, test_dataset.classes)
    
    # 在图像上标注数值
    thresh = cm.max() / 2.0  # 用于确定文字颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if not os.path.exists("fig"):
        os.makedirs("fig")
    save_path = os.path.join("fig", f"confusion_matrix_{template}.png")
    plt.savefig(save_path)
    plt.show()

def evaluate(args):
    """
    Set up the model, tokenizer, and transforms.
    Generate zero-shot text features and then run evaluation on CIFAR-10.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Open CLIP ViT-B-32 model and its transforms
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device)
    model.eval()

    # CIFAR-10 class names
    classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

    # Use the template provided by the user. You can allow multiple templates by comma-separating.
    templates = [t.strip() for t in args.template.split(",")]

    # Build zero-shot text features for each class using the given templates
    zeroshot_weights = zero_shot_classifier(model, tokenizer, classnames, templates, device)

    # Run zero-shot classification and evaluation
    run_classification(args, model, preprocess, zeroshot_weights, device, args.template)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot evaluation using Open CLIP on CIFAR-10.")
    parser.add_argument('--template', type=str, default="a photo of a [{cls}]",
                        help="Template for constructing text prompts. Use '{cls}' as placeholder for the class name.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for evaluation.")
    args = parser.parse_args()
    evaluate(args)
