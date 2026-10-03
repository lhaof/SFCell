import os
import sys
import json
import random
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
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
    '2cls': {
        'classes': ['CHC', 'WBC'],
        'selection_metric': 'macro_f1',
        'model_prefix': 'SFCell_2cls',
    },
    '3cls': {
        # Preserve the class order used by the released three-class weights.
        'classes': ['CD14', 'CD3', 'CD66b'],
        'selection_metric': 'accuracy',
        'model_prefix': 'SFCell_3cls',
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

def plot_training_curves(history, output_dir):
    """Plot loss, accuracy, and Macro-F1 curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    curves = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy'),
        ('macro_f1', 'Macro-F1'),
    ]
    for axis, (metric, title) in zip(axes, curves):
        axis.plot(epochs, history[f'train_{metric}'], 'b-', label='Training', linewidth=2)
        axis.plot(epochs, history[f'val_{metric}'], 'r-', label='Validation', linewidth=2)
        axis.set_title(f'Training and Validation {title}')
        axis.set_xlabel('Epochs')
        axis.set_ylabel(title)
        axis.legend()
        axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved to: {plot_path}")


def save_training_history(history, output_dir):
    """Save all recorded training metrics to a CSV file."""
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        **history,
    })
    history_path = os.path.join(output_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    logging.info(f"Training history saved to: {history_path}")




def flatten_valid_steps(outputs, labels, mask):
    """Return logits and labels for valid temporal positions only."""
    batch_size, time_steps = labels.shape
    if outputs.dim() == 2 and outputs.shape[0] == batch_size * time_steps:
        outputs = outputs.view(batch_size, time_steps, -1)
    if outputs.dim() != 3:
        raise ValueError(
            f"Expected temporal output with 3 dimensions, got shape {tuple(outputs.shape)}"
        )
    valid_mask = mask.bool()
    return outputs[valid_mask], labels[valid_mask]


def run_epoch(model, loader, loss_function, device, num_classes, optimizer=None, description=""):
    """Run one training or evaluation epoch and return aggregate metrics."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []

    progress = tqdm(loader, desc=description, file=sys.stdout)
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for data in progress:
            if len(data) == 3:
                images, labels, mask = data
                images = images.permute(0, 1, 3, 4, 2).to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                outputs = model(images, mask=mask)
                logits, targets = flatten_valid_steps(outputs, labels, mask)
            else:
                images, targets = data
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)

            if targets.numel() == 0:
                continue

            loss = loss_function(logits, targets)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits, dim=-1)
            sample_count = targets.numel()
            total_loss += loss.item() * sample_count
            total_samples += sample_count
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

            running_accuracy = (
                np.mean(np.asarray(all_predictions) == np.asarray(all_targets))
                if all_targets else 0.0
            )
            progress.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{running_accuracy:.4f}")

    if total_samples == 0:
        raise RuntimeError("The data loader produced no valid samples.")

    accuracy = float(np.mean(np.asarray(all_predictions) == np.asarray(all_targets)))
    macro_f1 = float(
        f1_score(
            all_targets,
            all_predictions,
            labels=list(range(num_classes)),
            average='macro',
            zero_division=0,
        )
    )
    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": all_predictions,
        "targets": all_targets,
    }


def save_class_indices(class_to_idx, output_dir):
    class_indices_path = os.path.join(output_dir, 'class_indices.json')
    index_to_class = {index: name for name, index in class_to_idx.items()}
    with open(class_indices_path, 'w', encoding='utf-8') as json_file:
        json.dump(index_to_class, json_file, indent=4)
    logging.info(f"Class mapping saved to: {class_indices_path}")


def create_data_loader(dataset, batch_size, shuffle, num_workers, seed, pin_memory):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=temporal_collate_fn,
    )


def main(opt):
    task_config = TASK_CONFIGS[opt.task]
    class_names = task_config['classes']
    num_classes = len(class_names)
    selection_metric = task_config['selection_metric']

    all_branches = ['spatial', 'frequency', 'local']
    active_branches = (
        all_branches
        if opt.disable_branch == 'none'
        else [branch for branch in all_branches if branch != opt.disable_branch]
    )
    ablation_tag = 'full' if opt.disable_branch == 'none' else f"no-{opt.disable_branch}"

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            'training_results',
            f"{opt.task}-{ablation_tag}-seed{opt.seed}",
        )
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, 'training.log'),
                mode='a',
                encoding='utf-8',
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

    set_random_seed(opt.seed)
    device = torch.device(
        f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else 'cpu'
    )
    pin_memory = device.type == 'cuda'

    checkpoint_path = os.path.join(output_dir, 'training_checkpoint.pth')
    resume_training = os.path.exists(checkpoint_path) and not opt.no_resume

    config = {
        'task': opt.task,
        'classes': class_names,
        'selection_metric': selection_metric,
        'train_csv': os.path.abspath(opt.train_csv),
        'val_csv': os.path.abspath(opt.val_csv),
        'test_csv': os.path.abspath(opt.test_csv) if opt.test_csv else None,
        'target_size': opt.target_size,
        'batch_size': opt.batch_size,
        'num_epochs': opt.num_epochs,
        'learning_rate': opt.lr,
        'weight_decay': opt.weight_decay,
        'output_dir': output_dir,
        'num_classes': num_classes,
        'patience': opt.patience,
        'random_seed': opt.seed,
        'active_branches': active_branches,
        'max_seq_length': opt.max_seq_length,
        'wave': opt.wave,
        'J': opt.J,
    }
    logging.info(f"Configuration:\n{json.dumps(config, indent=2)}")
    logging.info(f"Using device: {device}")
    logging.info(f"Model selection metric for {opt.task}: {selection_metric}")

    padding = GaussianPaddingOrCentralCrop(
        target_size=opt.target_size,
        debug=False,
    )
    train_transform = transforms.Compose([
        transforms.Lambda(ndarray_to_pil),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.Lambda(ndarray_to_pil),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = CSVDataset(
        csv_path=opt.train_csv,
        task=opt.task,
        transform=train_transform,
        padding=padding,
        is_train=True,
        max_seq_length=opt.max_seq_length,
    )
    val_dataset = CSVDataset(
        csv_path=opt.val_csv,
        task=opt.task,
        transform=eval_transform,
        padding=padding,
        is_train=False,
        max_seq_length=opt.max_seq_length,
    )
    test_dataset = None
    if opt.test_csv:
        test_dataset = CSVDataset(
            csv_path=opt.test_csv,
            task=opt.task,
            transform=eval_transform,
            padding=padding,
            is_train=False,
            max_seq_length=opt.max_seq_length,
        )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation class mappings do not match.")
    if test_dataset and train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Training and test class mappings do not match.")
    save_class_indices(train_dataset.class_to_idx, output_dir)

    if opt.num_workers < 0:
        num_workers = min(
            os.cpu_count() or 1,
            opt.batch_size if opt.batch_size > 1 else 0,
            8,
        )
    else:
        num_workers = opt.num_workers

    train_loader = create_data_loader(
        train_dataset,
        opt.batch_size,
        True,
        num_workers,
        opt.seed,
        pin_memory,
    )
    val_loader = create_data_loader(
        val_dataset,
        opt.batch_size,
        False,
        num_workers,
        opt.seed,
        pin_memory,
    )
    test_loader = (
        create_data_loader(
            test_dataset,
            opt.batch_size,
            False,
            num_workers,
            opt.seed,
            pin_memory,
        )
        if test_dataset is not None else None
    )

    model = get_SFCell_model(
        wave=opt.wave,
        J=opt.J,
        num_classes=num_classes,
        device=device,
        use_temporal=True,
        active_branches=active_branches,
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_macro_f1': [],
        'val_macro_f1': [],
    }
    start_epoch = 0
    patience_counter = 0
    best_val_score = float('-inf')
    best_val_epoch = 0
    best_test_score = float('-inf')
    best_test_epoch = 0

    if resume_training:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if checkpoint.get('task', opt.task) != opt.task:
                raise ValueError(
                    f"Checkpoint task {checkpoint.get('task')} does not match {opt.task}."
                )
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            patience_counter = checkpoint['patience_counter']
            best_val_score = checkpoint['best_val_score']
            best_val_epoch = checkpoint['best_val_epoch']
            best_test_score = checkpoint.get('best_test_score', float('-inf'))
            best_test_epoch = checkpoint.get('best_test_epoch', 0)
            history = checkpoint['history']
            logging.info(
                f"Resuming from epoch {start_epoch + 1}; "
                f"best validation {selection_metric}={best_val_score:.4f} "
                f"at epoch {best_val_epoch}."
            )
        except Exception as error:
            logging.exception(f"Unable to resume from checkpoint: {error}")
            logging.info("Starting training from scratch.")
            start_epoch = 0
            patience_counter = 0
            best_val_score = float('-inf')
            best_val_epoch = 0
            best_test_score = float('-inf')
            best_test_epoch = 0
            history = {key: [] for key in history}

    model_prefix = task_config['model_prefix']
    best_val_path = os.path.join(output_dir, f"{model_prefix}_best_val.pth")
    best_test_path = os.path.join(output_dir, f"{model_prefix}_best_test.pth")

    for epoch in range(start_epoch, opt.num_epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            num_classes,
            optimizer=optimizer,
            description=f"train {epoch + 1}/{opt.num_epochs}",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            loss_function,
            device,
            num_classes,
            optimizer=None,
            description=f"val {epoch + 1}/{opt.num_epochs}",
        )

        for metric in ('loss', 'accuracy', 'macro_f1'):
            history[f'train_{metric}'].append(train_metrics[metric])
            history[f'val_{metric}'].append(val_metrics[metric])

        logging.info(
            "[epoch %d] train_loss=%.4f train_acc=%.4f train_macro_f1=%.4f "
            "val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch + 1,
            train_metrics['loss'],
            train_metrics['accuracy'],
            train_metrics['macro_f1'],
            val_metrics['loss'],
            val_metrics['accuracy'],
            val_metrics['macro_f1'],
        )


        val_score = val_metrics[selection_metric]
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_val_path)
            logging.info(
                f"Best validation model saved to {best_val_path}; "
                f"{selection_metric}={best_val_score:.4f}."
            )
        else:
            patience_counter += 1
            logging.info(
                f"Validation {selection_metric} did not improve; "
                f"early-stopping counter: {patience_counter}/{opt.patience}."
            )

        if test_loader is not None:

            np.random.seed(opt.seed)
            random.seed(opt.seed)
            test_loader.generator.manual_seed(opt.seed)

            test_metrics = run_epoch(
                model,
                test_loader,
                loss_function,
                device,
                num_classes,
                optimizer=None,
                description=f"test {epoch + 1}/{opt.num_epochs}",
            )
            test_score = test_metrics[selection_metric]

            logging.info(
                "[epoch %d] test_loss=%.4f test_acc=%.4f test_macro_f1=%.4f",
                epoch + 1,
                test_metrics['loss'],
                test_metrics['accuracy'],
                test_metrics['macro_f1'],
            )

            confusion = ConfusionMatrix(
                num_classes=num_classes,
                labels=class_names,
                config=config,
            )

            confusion.update(
                test_metrics["predictions"],
                test_metrics["targets"],
            )

            confusion.summary()

            logging.info(
                "\nTest confusion matrix:\n%s",
                confusion.matrix.astype(int),
            )

            if test_score > best_test_score:
                best_test_score = test_score
                best_test_epoch = epoch + 1
                torch.save(model.state_dict(), best_test_path)
                logging.info(
                    f"Best test model saved to {best_test_path}; "
                    f"{selection_metric}={best_test_score:.4f}."
                )

        checkpoint = {
            'task': opt.task,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'patience_counter': patience_counter,
            'best_val_score': best_val_score,
            'best_val_epoch': best_val_epoch,
            'best_test_score': best_test_score,
            'best_test_epoch': best_test_epoch,
            'selection_metric': selection_metric,
            'history': history,
            'config': config,
        }
        torch.save(checkpoint, checkpoint_path)
        save_training_history(history, output_dir)

        if patience_counter >= opt.patience:
            logging.info(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best validation {selection_metric}={best_val_score:.4f} "
                f"at epoch {best_val_epoch}."
            )
            break

    plot_training_curves(history, output_dir)
    save_training_history(history, output_dir)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logging.info("Training complete; temporary checkpoint deleted.")

    if test_loader is not None:
        logging.info(
            f"Best test {selection_metric}={best_test_score:.4f} "
            f"at epoch {best_test_epoch}."
        )
    logging.info("Finished training.")


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train SFCell for binary CHC-WBC or three-class WBC classification.'
    )
    parser.add_argument(
        '--task',
        required=True,
        choices=sorted(TASK_CONFIGS),
        help='Classification task: 2cls or 3cls.',
    )
    parser.add_argument('--train_csv', required=True, help='Training CSV file.')
    parser.add_argument('--val_csv', required=True, help='Validation CSV file.')
    parser.add_argument(
        '--test_csv',
        default=None,
        help='Optional test CSV evaluated after each epoch.',
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Output directory. A task-specific directory is used by default.',
    )
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--target_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wave', default='haar', help='Wavelet basis.')
    parser.add_argument('--J', type=int, default=2, help='Wavelet decomposition level.')
    parser.add_argument(
        '--disable_branch',
        default='none',
        choices=['none', 'spatial', 'frequency', 'local'],
        help="Disable one branch for an ablation; 'none' uses the full model.",
    )
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=10,
        help='Maximum temporal sequence length.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=-1,
        help='Data-loader workers; -1 selects up to eight automatically.',
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Ignore an existing training checkpoint.',
    )
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())


'''
usage:
python train_SFCell.py \
  --task 2cls \
  --train_csv data/splits/chc_wbc/train.csv \
  --val_csv data/splits/chc_wbc/val.csv \
  --test_csv data/splits/chc_wbc/test.csv

python train_SFCell.py \
  --task 3cls \
  --train_csv data/splits/wbc_3class/train.csv \
  --val_csv data/splits/wbc_3class/val.csv
  --test_csv data/splits/wbc_3class/external_test.csv

'''