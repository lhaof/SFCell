import os
import sys
import random
import warnings
from torch.utils.data import Sampler
from PIL import Image
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import logging
import pandas as pd

import argparse

from SFCell_network import VSSM as SFCell # import model
from utils import GaussianPaddingOrCentralCrop, temporal_collate_fn, CSVDataset, ConfusionMatrix, seed_worker, ndarray_to_pil


TASK_CONFIGS = {
    "2cls": {
        "classes": ["CHC", "WBC"],
    },
    "3cls": {
        "classes": ["CD14", "CD3", "CD66b"],
    },
}


def get_SFCell_model(
    wave,
    J,
    num_classes=3,
    device="cuda",
    use_temporal=False,
    active_branches=("spatial", "frequency", "local"),
):
    model = SFCell(
        wave=wave,
        J=J,
        depths=[2, 2, 4, 2],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        use_temporal=use_temporal,
        active_branches=active_branches,
    ).to(device)

    return model

def set_random_seed(seed):
    """Set global random seeds for reproducible execution."""
    # Python random module.
    random.seed(seed)
    # NumPy random generator.
    np.random.seed(seed)
    # PyTorch CPU and GPU generators.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU execution.
        # Disable nondeterministic cuDNN algorithms.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to: {seed}")


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    """Plot training and validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss curves.
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy curves.
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure.
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training curves have been saved to: {plot_path}")

def save_training_history(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    """Save training history to a CSV file."""
    history = {
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }
    
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    logging.info(f"Training history has been saved to: {history_path}")
    
    # Log the final training summary.
    logging.info("\n=== Final Training Summary ===")
    logging.info(f"Final training loss: {train_losses[-1]:.4f}")
    logging.info(f"Final validation loss: {val_losses[-1]:.4f}")
    logging.info(f"Final training accuracy: {train_accuracies[-1]:.4f}")
    logging.info(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    logging.info(f"Best validation accuracy: {max(val_accuracies):.4f}")


def setup_logging(output_dir, log_file):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isabs(log_file):
        log_file = os.path.join(output_dir, log_file)
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True,
    )
    return log_file

def load_weight(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        new_state_dict[key] = value
    model.load_state_dict(new_state_dict, strict=True)


def save_class_indices(labels, output_dir):
    class_indices_path = os.path.join(output_dir, 'class_indices.json')
    cla_dict = {idx: label for idx, label in enumerate(labels)}
    with open(class_indices_path, 'w', encoding='utf-8') as json_file:
        json.dump(cla_dict, json_file, indent=4)
    logging.info(f"Class mapping has been saved to: {class_indices_path}")


def append_pred_rows(pred_rows, probs_tensor, labels_tensor, mask_tensor, labels):
    probs_tensor = torch.softmax(probs_tensor, dim=-1)
    B, T = mask_tensor.shape
    for b in range(B):
        for t in range(T):
            if mask_tensor[b, t]:
                gt = int(labels_tensor[b, t].detach().cpu().item())
                row = {'gt': gt}
                for class_idx in range(len(labels)):
                    row[str(class_idx)] = float(probs_tensor[b, t, class_idx].detach().cpu().item())
                pred_rows.append(row)


def append_single_frame_pred_rows(pred_rows, probs_tensor, labels_tensor, labels):
    probs_tensor = torch.softmax(probs_tensor, dim=-1)
    for i in range(labels_tensor.shape[0]):
        gt = int(labels_tensor[i].detach().cpu().item())
        row = {'gt': gt}
        for class_idx in range(len(labels)):
            row[str(class_idx)] = float(probs_tensor[i, class_idx].detach().cpu().item())
        pred_rows.append(row)



def main(opt):
    task_config = TASK_CONFIGS[opt.task]
    num_classes = len(task_config['classes'])

    all_branches = ["spatial", "frequency", "local"]
    if opt.disable_branch == "none":
        active_branches = all_branches
    else:
        active_branches = [branch for branch in all_branches if branch != opt.disable_branch]

    output_dir = os.path.abspath(opt.output_dir)
    log_file_path = setup_logging(output_dir, opt.log_file)
    set_random_seed(opt.seed)

    gpu_id = opt.gpu_id
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    config = {
        'task': opt.task,
        'wave': opt.wave,
        'J': opt.J,
        'csv_path': opt.csv_path,
        'target_size': opt.target_size,
        'batch_size': opt.batch_size,
        'output_dir': output_dir,
        'num_classes': num_classes,
        'random_seed': opt.seed,
        'disable_branch': opt.disable_branch,
        'active_branches': active_branches,
        'max_seq_length': opt.max_seq_length,
        'weight_path': opt.weight_path,
        'preds_test_csv': opt.preds_test_csv,
    }

    logging.info("Starting SFCell test-only evaluation")
    logging.info(f"Device: {device}")
    logging.info(f"GPU ID: {gpu_id}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Log file: {log_file_path}")
    logging.info(f"Configuration:\n{config}")

    padding = GaussianPaddingOrCentralCrop(target_size=config['target_size'], debug=False)
    test_transform = transforms.Compose([
        transforms.Lambda(ndarray_to_pil),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = CSVDataset(
        csv_path=config['csv_path'],
        task=config['task'],
        transform=test_transform,
        padding=padding,
        is_train=False,
        max_seq_length=config['max_seq_length']
    )

    labels = list(TASK_CONFIGS[opt.task]["classes"])
    if len(labels) != config['num_classes']:
        raise ValueError(
            f"Expected {config['num_classes']} class labels for task {config['task']}, "
            f"but received {len(labels)}."
        )
    save_class_indices(labels, output_dir)

    nw = min([os.cpu_count(), config['batch_size'] if config['batch_size'] > 1 else 0, 8])
    if opt.num_workers is not None:
        nw = opt.num_workers
    logging.info('Using {} dataloader workers every process'.format(nw))

    test_generator = torch.Generator()
    test_generator.manual_seed(opt.seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=nw,
        worker_init_fn=seed_worker,
        generator=test_generator,
        collate_fn=temporal_collate_fn,
    )

    model = get_SFCell_model(
        wave=config['wave'],
        J=config['J'],
        num_classes=config['num_classes'],
        device=device,
        use_temporal=True,
        active_branches=config['active_branches'],
    )
    logging.info(f"Ablation settings: disable_branch={config['disable_branch']}, active_branches={config['active_branches']}")
    load_weight(model, opt.weight_path, device)
    logging.info(f"Model weights loaded from: {opt.weight_path}")

    model.eval()
    confusion = ConfusionMatrix(num_classes=config['num_classes'], labels=labels, config=config)
    total_test = 0
    correct_test = 0
    pred_rows = []
    test_predictions = []
    test_targets = []

    np.random.seed(opt.seed)
    random.seed(opt.seed)
    test_loader.generator.manual_seed(opt.seed)

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="test_only", file=sys.stdout):
            if len(test_data) == 3:
                test_images, test_labels, test_mask = test_data
                B, T, C, H, W = test_images.shape

                test_images = test_images.permute(0, 1, 3, 4, 2).to(device)
                test_labels = test_labels.to(device)
                test_mask = test_mask.to(device)

                outputs = model(test_images, mask=test_mask)
                if outputs.dim() == 2 and outputs.shape[0] == B * T:
                    outputs = outputs.view(B, T, -1)
                elif outputs.dim() == 2 and outputs.shape[0] == B:
                    outputs = outputs.unsqueeze(1)

                append_pred_rows(pred_rows, outputs, test_labels, test_mask, labels)

                for b in range(B):
                    for t in range(T):
                        if test_mask[b, t]:
                            predicted_classes = torch.argmax(outputs[b, t], dim=0)
                            pred_val = predicted_classes.cpu().item()
                            label_val = test_labels[b, t].cpu().item()
                            confusion.update([pred_val], [label_val])
                            test_predictions.append(pred_val)
                            test_targets.append(label_val)
                            total_test += 1
                            if predicted_classes == test_labels[b, t]:
                                correct_test += 1
            else:
                test_images, test_labels = test_data
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                outputs = model(test_images)
                predicted_classes = torch.argmax(outputs, dim=1)
                confusion.update(predicted_classes.cpu().numpy(), test_labels.cpu().numpy())
                append_single_frame_pred_rows(pred_rows, outputs, test_labels, labels)
                test_predictions.extend(predicted_classes.detach().cpu().tolist())
                test_targets.extend(test_labels.detach().cpu().tolist())
                total_test += test_labels.size(0)
                correct_test += (predicted_classes == test_labels).sum().item()

    confusion.summary()
    epoch_test_accuracy = correct_test / total_test if total_test > 0 else 0.0
    epoch_test_macro_f1 = f1_score(
        test_targets,
        test_predictions,
        average='macro',
        labels=list(range(config['num_classes'])),
        zero_division=0,
    ) if total_test > 0 else 0.0
    logging.info(f'Test-only accuracy: {epoch_test_accuracy:.4f}')
    logging.info(f'Test-only macro-F1: {epoch_test_macro_f1:.4f}')

    preds_test_path = opt.preds_test_csv
    if not os.path.isabs(preds_test_path):
        preds_test_path = os.path.join(output_dir, preds_test_path)
    preds_test_df = pd.DataFrame(pred_rows, columns=['gt'] + [str(i) for i in range(config['num_classes'])])
    preds_test_df.to_csv(preds_test_path, index=False, encoding='utf-8-sig')
    logging.info(f"Test predictions have been saved to: {preds_test_path}")
    logging.info('Finished test_only')


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", type=str, required=True, choices=sorted(TASK_CONFIGS),
        help="Evaluation task: 2cls for CHC-WBC or 3cls for WBC three-class",
    )
    parser.add_argument(
        "--csv_dir", "--csv_path", dest="csv_path", type=str, required=True,
        help="Path to a single task-ready evaluation CSV file",
    )
    parser.add_argument("--weight_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for evaluation outputs")
    parser.add_argument("--log_file", type=str, default="test.log", help="Log file name or absolute path")
    parser.add_argument("--preds_test_csv", type=str, default="preds_test.csv", help="Prediction CSV file name or absolute path")

    parser.add_argument("--target_size", type=int, default=128, help="Input image size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of DataLoader workers; selected automatically by default")
    parser.add_argument('--wave', type=str, default='haar', help="Wavelet basis function")
    parser.add_argument('--J', type=int, default=2, help="Wavelet decomposition level")
    parser.add_argument("--disable_branch", type=str, default="none", choices=["none", "spatial", "frequency", "local"], help="Disable one branch for ablation; use none for the full model")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--max_seq_length", type=int, default=10, help="Maximum temporal sequence length")

    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())

'''
python test_SFCell.py \
  --task 2cls \
  --csv_dir data/splits/chc_wbc/test.csv \
  --weight_path weights/SFCell_2cls_best_test.pth \
  --output_dir test/chc-wbc \
  --wave haar \
  --J 2 \
  --target_size 128 \
  --disable_branch none \
  --batch_size 32 \
  --max_seq_length 10 \
  --gpu_id 0 \
  --seed 3

python test_SFCell.py \
  --task 2cls \
  --csv_dir data/splits/chc_wbc/external_test.csv \
  --weight_path weights/SFCell_2cls_best_test.pth \
  --output_dir test/chc-wbc \
  --wave haar \
  --J 2 \
  --target_size 128 \
  --disable_branch none \
  --batch_size 32 \
  --max_seq_length 10 \
  --gpu_id 0 \
  --seed 3

python test_SFCell.py \
  --task 3cls \
  --csv_dir data/splits/wbc_3class/external_test.csv \
  --weight_path weights/SFCell_3cls_best_test.pth \
  --output_dir test/wbc3-external \
  --wave haar \
  --J 2 \
  --target_size 128 \
  --disable_branch none \
  --batch_size 32 \
  --max_seq_length 10 \
  --gpu_id 0 \
  --seed 2

'''
