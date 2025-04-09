# Vision-Language Learning Methods

This repository contains implementations for various vision-language learning methods. It includes scripts for zero-shot classification, linear probing, and Cooperative Prompting (CoOP) as introduced in the paper *[Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)*. Additionally, there is a `visualize` folder dedicated to scripts for generating report figures.

## Scripts

### Zero-shot Classification
The `zeroshot_classification.py` script allows you to perform zero-shot classification with a customizable template. Use the `{cls}` placeholder within your template to specify where the class name should appear. For example, you might set the template as:

```
a photo of a [{cls}]
```

**Usage Example:**
```bash
python zeroshot_classification.py --template "your template here!" --batch_size 64
```
*Note:* Replace `"your template here!"` with your actual template string.

### Linear Probe
The `linear_probe.py` script implements linear probing (1,2,4,8,16,full) for model fine-tuning. It allows you to adjust several training parameters such as batch size, tuning epochs, and training epochs.

**Usage Example:**
```bash
python linear_probe.py --batch_size 64 --tune_epochs 20 --train_epochs 50
```

Customize these parameters according to the requirements of your experiment.

### Cooperative Prompting (CoOP)
The `coop.py` script implements the CoOP method based on the paper *"Learning to Prompt for Vision-Language Models"*. This method adapts prompts to better leverage the synergy between vision and language models.

**Usage Example:**
```bash
python coop.py 
```

**Tip:**  
For above experiments, it is recommended to use `nohup` to preserve the results and logs even after you log out of your session. An example command with `nohup` on CUDA 1 is:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u coop.py  > res/coop_res.log &
```

## Visualization
The `visualize` folder contains scripts that generate the figures and charts used in the project report. Use these scripts to create visual representations of your experimental results.
