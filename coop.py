import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import copy
import os
import matplotlib.pyplot as plt
from utils import *
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ViT-B-32"
pretrained = "openai"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()
for param in model.parameters():
    param.requires_grad = False

###############################################################################
# load data 
###############################################################################
full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocess)
val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)
class_names = full_train_dataset.classes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

###############################################################################
# model 
###############################################################################
class PromptLearner(nn.Module):
    def __init__(self, clip_model, class_names, n_ctx=4):
        """
        Initializes the learnable prompt embeddings for each class.
        Params
          clip_model: CLIP model
          class_names: class name as a list 
          n_ctx: number of context tokens need to learn
        """
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.class_names = class_names
        self.n_classes = len(class_names)
        ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.context_embeddings = nn.Parameter(torch.randn(n_ctx, ctx_dim) * 0.02)
        tokenizer = open_clip.get_tokenizer(model_name)
        class_token_ids = tokenizer(class_names)  # [n_classes, context_length]
        eos_id = int(class_token_ids.max().item())
        pad_id = 0
        dummy_id = eos_id - 1
        context_length = class_token_ids.shape[1]
        prompts = []
        for ids in class_token_ids:
            ids = ids.tolist()
            eos_index = ids.index(eos_id)
            new_ids = [dummy_id] * self.n_ctx + ids[:eos_index+1]
            new_ids = new_ids[:context_length]
            if len(new_ids) < context_length:
                new_ids += [pad_id] * (context_length - len(new_ids))
            prompts.append(new_ids)
        self.register_buffer('prompt_token_ids', torch.tensor(prompts, dtype=torch.long))
        self.eos_id = eos_id

    def forward(self):
        # By replacing the dummy token with learnable context, 
        # the text encoder of CLIP is used to calculate the text feature.
        token_ids = self.prompt_token_ids.to(self.clip_model.token_embedding.weight.device)  # [n_classes, context_length]
        x = self.clip_model.token_embedding(token_ids)  # [n_classes, context_length, dim]
        ctx = self.context_embeddings.unsqueeze(0).expand(self.n_classes, -1, -1)  # [n_classes, n_ctx, dim]
        x[:, :self.n_ctx, :] = ctx
        x = x + self.clip_model.positional_embedding
        attn_mask = self.clip_model.attn_mask
        if attn_mask is not None:
            L = x.shape[1]
            attn_mask = attn_mask[:L, :L]
        x = self.clip_model.transformer(x, attn_mask=attn_mask)
        x = self.clip_model.ln_final(x)
        eos_mask = (self.prompt_token_ids == self.eos_id)
        eos_idx = eos_mask.int().argmax(dim=1)
        text_feats = x[torch.arange(self.n_classes), eos_idx, :]
        if self.clip_model.text_projection is not None:
            if isinstance(self.clip_model.text_projection, nn.Linear):
                text_feats = self.clip_model.text_projection(text_feats)
            else:
                text_feats = text_feats @ self.clip_model.text_projection

        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats

prompt_learner = PromptLearner(model, class_names, n_ctx=4).to(device)
optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=1e-3, weight_decay=0.001)

###############################################################################
# Training 
###############################################################################
epochs = 50
best_val_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    prompt_learner.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        image_feats = model.encode_image(images, normalize=True)
        text_feats = prompt_learner()
        logits = model.logit_scale.exp() * image_feats @ text_feats.T  # [batch, n_classes]
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    
    avg_loss = total_loss / len(train_loader.dataset)
    

    prompt_learner.eval()
    total_samples = 0
    total_top1_correct = 0
    top5_correct = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            image_feats = model.encode_image(images, normalize=True)
            text_feats = prompt_learner()
            logits = model.logit_scale.exp() * image_feats @ text_feats.T
            total_samples += labels.size(0)
            top1_preds = logits.argmax(dim=1)
            total_top1_correct += (top1_preds == labels).sum().item()
            
            _, top5_preds = logits.topk(5, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1
            
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    
    val_top1_acc = 100 * total_top1_correct / total_samples
    val_top5_acc = 100 * top5_correct / total_samples
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds_np = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()
    precision = precision_score(labels_np, preds_np, average='macro')
    recall = recall_score(labels_np, preds_np, average='macro')
    f1 = f1_score(labels_np, preds_np, average='macro')
    conf_matrix = confusion_matrix(labels_np, preds_np)
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    print(f"Val Top1 Accuracy: {val_top1_acc:.2f}% | Top5 Accuracy: {val_top5_acc:.2f}%")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
    
    if val_top1_acc > best_val_acc:
        best_val_acc = val_top1_acc
        best_model_state = copy.deepcopy(prompt_learner.state_dict())

###############################################################################
# Evaluation
###############################################################################
if best_model_state is not None:
    prompt_learner.load_state_dict(best_model_state)

prompt_learner.eval()
total_samples = 0
total_top1_correct = 0
top5_correct = 0
all_logits = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        image_feats = model.encode_image(images, normalize=True)
        text_feats = prompt_learner()
        logits = model.logit_scale.exp() * image_feats @ text_feats.T
        total_samples += labels.size(0)
        top1_preds = logits.argmax(dim=1)
        total_top1_correct += (top1_preds == labels).sum().item()
        _, top5_preds = logits.topk(5, dim=1)
        for i in range(labels.size(0)):
            if labels[i] in top5_preds[i]:
                top5_correct += 1
                
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

test_top1_acc = 100 * total_top1_correct / total_samples
test_top5_acc = 100 * top5_correct / total_samples
all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)
preds_np = all_logits.argmax(dim=1).numpy()
labels_np = all_labels.numpy()
precision = precision_score(labels_np, preds_np, average='macro')
recall = recall_score(labels_np, preds_np, average='macro')
f1 = f1_score(labels_np, preds_np, average='macro')
conf_matrix = confusion_matrix(labels_np, preds_np)

print("===== Test Set Evaluation =====")
print(f"Test Top1 Accuracy: {test_top1_acc:.2f}%")
print(f"Test Top5 Accuracy: {test_top5_acc:.2f}%")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# visulize
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(test_dataset.classes))
plt.xticks(tick_marks, test_dataset.classes, rotation=45)
plt.yticks(tick_marks, test_dataset.classes)
    
thresh = conf_matrix.max() / 2.0
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
    
if not os.path.exists("fig"):
    os.makedirs("fig")
save_path = os.path.join("fig", f"confusion_matrix_CoOp.png")
plt.savefig(save_path)
plt.show()
