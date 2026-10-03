import os
import sys
import json
import random
import warnings
from PIL import Image
from prettytable import PrettyTable

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

TASK_CONFIGS = {
    "2cls": {
        "classes": ["CHC", "WBC"],
    },
    "3cls": {
        "classes": ["CD14", "CD3", "CD66b"],
    },
}
def seed_worker(worker_id):
    """Set random seeds for each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ndarray_to_pil(image):
    """Convert an ndarray loaded by Matplotlib into a PIL image."""
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


class ConfusionMatrix(object):
    """
    If the rendered figure is incomplete, check the Matplotlib version.
    This implementation was verified with Matplotlib 3.2.1 on Windows and Ubuntu.
    The prettytable package is also required.
    """
    def __init__(self, num_classes: int, labels: list, config):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.output_dir = config['output_dir']

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        logging.info(f"the model accuracy is {acc}")

        # precision, recall, specificity, f1
        table = PrettyTable()
        table.field_names = ["Class", "Precision", "Recall", "Specificity", "F1-Score", "Support"]

        # Store per-class metrics for aggregate calculations.
        precisions = []
        recalls = []
        f1_scores = []
        supports = []

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            support = np.sum(self.matrix[:, i])  # Number of ground-truth samples for this class.
            
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if (Precision + Recall) > 0 else 0.
            
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1, int(support)])
            
            # Store values for aggregate calculations.
            precisions.append(Precision)
            recalls.append(Recall)
            f1_scores.append(F1)
            supports.append(support)

        # Calculate macro and weighted averages.
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)

        weighted_precision = np.average(precisions, weights=supports)
        weighted_recall = np.average(recalls, weights=supports)
        weighted_f1 = np.average(f1_scores, weights=supports)

        # Append aggregate rows.
        table.add_row(["-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10])
        table.add_row(["Macro Avg", round(macro_precision, 3), round(macro_recall, 3), "-", round(macro_f1, 3), "-"])
        table.add_row(["Weighted Avg", round(weighted_precision, 3), round(weighted_recall, 3), "-", round(weighted_f1, 3), "-"])

        logging.info(f"\n{table}")

    def plot(self):
        matrix = self.matrix
        logging.info(f"\n{matrix}")
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # Set x-axis labels.
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # Set y-axis labels.
        plt.yticks(range(self.num_classes), self.labels)
        # Display the color bar.
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # Annotate each matrix cell.
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # Matrix indexing is [y, x], not [x, y].
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        logging.info(f"The confusion matrix has been saved to: {save_path}")
    
    def return_acc(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)

        return acc



class GaussianPaddingOrCentralCrop:
    def __init__(self, target_size, debug=False):
        self.target_size = target_size
        self.debug = debug  # Debug flag

    def __call__(self, img_ndarray):
        """
        Apply Gaussian padding and center cropping to an ndarray image
        Args:
            img_ndarray: Input ndarray with shape (H, W) for grayscale or (H, W, C) for RGB/RGBA
        Returns:
            Processed ndarray with shape (target_size, target_size) or (target_size, target_size, C)
        """
        # ------------ Validate input format ------------
        if self.debug:
            self._check_input_size(img_ndarray)
        
        # Normalize the input representation for grayscale, RGB, and RGBA images
        original_shape = img_ndarray.shape
        is_gray = len(original_shape) == 2  # Grayscale-image flag
        if not is_gray:
            C = original_shape[2]
            # RGBA→RGB
            if C == 4:
                img_ndarray = img_ndarray[:, :, :3]
                C = 3
        else:
            C = 1
            # Temporarily reshape grayscale input to (H, W, 1) for uniform processing
            img_ndarray = img_ndarray[:, :, np.newaxis]
        
        h, w = img_ndarray.shape[:2]  # Read H and W consistently

        # ========== Step 1: center crop images larger than the target size ==========
        crop_applied = False
        crop_h = min(h, self.target_size)
        crop_w = min(w, self.target_size)
        if crop_h < h or crop_w < w:
            # Compute ndarray center-crop coordinates
            center_h, center_w = h // 2, w // 2
            crop_h_half = crop_h // 2
            crop_w_half = crop_w // 2
            # Compute crop bounds without exceeding image limits
            start_h = max(0, center_h - crop_h_half)
            end_h = min(h, center_h + crop_h_half)
            start_w = max(0, center_w - crop_w_half)
            end_w = min(w, center_w + crop_w_half)
            # Apply the crop
            img_ndarray = img_ndarray[start_h:end_h, start_w:end_w, :]
            h, w = img_ndarray.shape[:2]
            crop_applied = True
        
        # ------------ Validate the cropped size ------------
        if self.debug:
            self._check_crop_size(img_ndarray, crop_applied, is_gray)

        # ========== Step 2: calculate symmetric padding ==========
        pad_h = self.target_size - h
        pad_w = self.target_size - w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # ========== Step 3: apply Gaussian-noise padding to the ndarray ==========
        pad_applied = False
        if pad_h > 0 or pad_w > 0:
            # Step 1: extract border pixels from each channel
            edge_pixels = []
            for c in range(C):
                channel = img_ndarray[:, :, c]  # Current channel: (H, W)
                # Extract the four image borders
                top_edge = channel[0, :]
                bottom_edge = channel[-1, :]
                left_edge = channel[1:-1, 0]
                right_edge = channel[1:-1, -1]
                # Concatenate border pixels
                channel_edges = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
                edge_pixels.append(channel_edges)

            # Step 2: calculate per-channel Gaussian parameters (mean and standard deviation)
            gauss_params = []
            for edges in edge_pixels:
                mean = np.mean(edges)
                std = np.std(edges)
                std = max(std, 1e-5)  # Avoid a zero standard deviation
                gauss_params.append((mean, std))

            # Step 3: generate Gaussian noise for each padding region
            dtype = img_ndarray.dtype
            # Generate top padding: (pad_top, w, C)
            top_noise = self._generate_gauss_noise((pad_top, w, C), gauss_params, dtype)
            # Generate bottom padding: (pad_bottom, w, C)
            bottom_noise = self._generate_gauss_noise((pad_bottom, w, C), gauss_params, dtype)
            # Generate left padding: (h, pad_left, C)
            left_noise = self._generate_gauss_noise((h, pad_left, C), gauss_params, dtype)
            # Generate right padding: (h, pad_right, C)
            right_noise = self._generate_gauss_noise((h, pad_right, C), gauss_params, dtype)
            # Generate padding for the four corners
            tl_noise = self._generate_gauss_noise((pad_top, pad_left, C), gauss_params, dtype)
            tr_noise = self._generate_gauss_noise((pad_top, pad_right, C), gauss_params, dtype)
            bl_noise = self._generate_gauss_noise((pad_bottom, pad_left, C), gauss_params, dtype)
            br_noise = self._generate_gauss_noise((pad_bottom, pad_right, C), gauss_params, dtype)

            # Step 4: concatenate all ndarray regions along the appropriate axes
            # 1. Concatenate the middle row: left + image + right (axis 1, width)
            middle = np.concatenate([left_noise, img_ndarray, right_noise], axis=1)
            # 2. Concatenate the top row: top-left + top + top-right (axis 1)
            top = np.concatenate([tl_noise, top_noise, tr_noise], axis=1)
            # 3. Concatenate the bottom row: bottom-left + bottom + bottom-right (axis 1)
            bottom = np.concatenate([bl_noise, bottom_noise, br_noise], axis=1)
            # 4. Concatenate all rows: top + middle + bottom (axis 0, height)
            padded_ndarray = np.concatenate([top, middle, bottom], axis=0)
            pad_applied = True
        else:
            padded_ndarray = img_ndarray

        # ------------ Validate the padded size ------------
        if self.debug:
            self._check_pad_size(padded_ndarray, pad_applied, is_gray)

        # ------------ Validate padding-region values ------------
        if self.debug and pad_applied:
            self._check_padding_values(padded_ndarray, pad_top, pad_bottom, pad_left, pad_right, is_gray)

        # ========== Final step: restore grayscale shape and validate output size ==========
        # Remove the final dimension from grayscale output to restore (H, W)
        if is_gray:
            padded_ndarray = padded_ndarray.squeeze(axis=-1)
        
        # Validate dimensions
        out_h, out_w = padded_ndarray.shape[:2]
        assert out_h == self.target_size and out_w == self.target_size, \
            f"Incorrect padded size: expected ({self.target_size},{self.target_size}); got ({out_h},{out_w})"

        return padded_ndarray

    def _generate_gauss_noise(self, shape, gauss_params, dtype):
        """
        Generate Gaussian noise as an ndarray
        :param shape: Noise shape (H, W, C)
        :param gauss_params: Per-channel Gaussian parameters [(mean, std), ...]
        :param dtype: Data type, matching the input image
        :return: Noise ndarray with shape (H, W, C)
        """
        H, W, C = shape
        noise = np.zeros(shape, dtype=dtype)
        for c in range(C):
            mean, std = gauss_params[c]
            # Generate Gaussian noise for the current channel
            channel_noise = np.random.normal(loc=mean, scale=std, size=(H, W))
            # Clip values to the valid image range, such as 0-255
            channel_noise = np.clip(channel_noise, 0, 255) if np.issubdtype(dtype, np.integer) else channel_noise
            noise[:, :, c] = channel_noise
        return noise

    def _check_input_size(self, img_ndarray):
        """Validate ndarray input dimensions and size"""
        if img_ndarray.ndim not in [2, 3]:
            warnings.warn(f"⚠️ Input dimension warning: expected 2D grayscale or 3D RGB, got {img_ndarray.ndim}D")
        else:
            if img_ndarray.ndim == 2:
                h, w = img_ndarray.shape
                print(f"Input size: grayscale image, height={h}, width={w}")
            else:
                h, w, C = img_ndarray.shape
                print(f"Input size: RGB image, channels={C}, height={h}, width={w}")
        print("-" * 50)

    def _check_crop_size(self, img_ndarray, crop_applied, is_gray):
        """Validate ndarray dimensions after cropping"""
        h, w = img_ndarray.shape[:2]
        C = 1 if is_gray else img_ndarray.shape[2]
        if crop_applied:
            print(f"Crop applied: cropped size=(height={h}, width={w}), channels={C}")
            if h > self.target_size or w > self.target_size:
                warnings.warn(f"⚠️ Crop warning: cropped dimensions still exceed target size {self.target_size}")
        else:
            print(f"Crop not required: original size=(height={h}, width={w}) (both are <= target size {self.target_size})")
        print("-" * 50)

    def _check_pad_size(self, padded_ndarray, pad_applied, is_gray):
        """Validate ndarray dimensions after padding"""
        h, w = padded_ndarray.shape[:2]
        C = 1 if is_gray else padded_ndarray.shape[2]
        if pad_applied:
            print(f"Padding applied: padded size=(height={h}, width={w}), channels={C}")
        else:
            print(f"Padding not required: image already matches target size {self.target_size}")
        if h != self.target_size or w != self.target_size:
            warnings.warn(f"⚠️ Padding warning: padded dimensions do not match target size {self.target_size}")
        print("-" * 50)

    def _check_padding_values(self, tensor, pad_top, pad_bottom, pad_left, pad_right, is_gray):
        """Inspect ndarray pixel values in padding regions"""
        H, W = tensor.shape[:2]
        # Use channel 0 for both grayscale and RGB images
        channel = tensor[:, :, 0] if not is_gray else tensor

        # Extract each padding region
        if pad_top > 0:
            top_pad = channel[:pad_top, :]
            print(f"Top padding (Gaussian noise) - mean: {top_pad.mean():.4f}, std: {top_pad.std():.4f}, range: [{top_pad.min():.4f}, {top_pad.max():.4f}]")
        
        if pad_bottom > 0:
            bottom_pad = channel[-pad_bottom:, :]
            print(f"Bottom padding (Gaussian noise) - mean: {bottom_pad.mean():.4f}, std: {bottom_pad.std():.4f}, range: [{bottom_pad.min():.4f}, {bottom_pad.max():.4f}]")
        
        if pad_left > 0:
            left_pad = channel[:, :pad_left]
            print(f"Left padding (Gaussian noise) - mean: {left_pad.mean():.4f}, std: {left_pad.std():.4f}, range: [{left_pad.min():.4f}, {left_pad.max():.4f}]")
        
        if pad_right > 0:
            right_pad = channel[:, -pad_right:]
            print(f"Right padding (Gaussian noise) - mean: {right_pad.mean():.4f}, std: {right_pad.std():.4f}, range: [{right_pad.min():.4f}, {right_pad.max():.4f}]")
        
        # Inspect the four corners
        if pad_top > 0 and pad_left > 0:
            top_left_corner = channel[0, 0]
            print(f"Top-left padding value (Gaussian noise): {top_left_corner:.4f}")
        
        if pad_top > 0 and pad_right > 0:
            top_right_corner = channel[0, -1]
            print(f"Top-right padding value (Gaussian noise): {top_right_corner:.4f}")
        
        if pad_bottom > 0 and pad_left > 0:
            bottom_left_corner = channel[-1, 0]
            print(f"Bottom-left padding value (Gaussian noise): {bottom_left_corner:.4f}")
        
        if pad_bottom > 0 and pad_right > 0:
            bottom_right_corner = channel[-1, -1]
            print(f"Bottom-right padding value (Gaussian noise): {bottom_right_corner:.4f}")
        
        print("-" * 50)

def temporal_collate_fn(batch):
    """
    Collate variable-length temporal sequences.
    batch: list of (images[T,C,H,W], labels[T], seq_length)
    Returns: (images[B,T_max,C,H,W], labels[B,T_max], mask[B,T_max])
    """
    images_list = []
    labels_list = []
    lengths = []
    
    for images, labels, seq_length in batch:
        images_list.append(images)
        labels_list.append(labels)
        lengths.append(seq_length)
    
    # Find the maximum sequence length in the batch.
    max_len = max(lengths)
    
    # padding
    B = len(batch)
    C, H, W = images_list[0].shape[1:]
    
    images_padded = torch.zeros(B, max_len, C, H, W)
    labels_padded = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    
    for i, (imgs, lbls, length) in enumerate(zip(images_list, labels_list, lengths)):
        images_padded[i, :length] = imgs
        labels_padded[i, :length] = lbls
        mask[i, :length] = True
    
    return images_padded, labels_padded, mask


class CSVDataset(Dataset):
    def __init__(self, csv_path, task, transform=None, padding=None, is_train=True,
                 max_seq_length=5):
        """Load a task-ready SFCell evaluation CSV file."""
        self.max_seq_length = max_seq_length

        if task not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task

        csv_path = os.path.abspath(csv_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

        self.df = pd.read_csv(csv_path)
        required_columns = {'image_file_path', 'track_id', 'frame_idx', 'cell_type'}
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise ValueError(
                f"CSV file {csv_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )
        if self.df.empty:
            raise ValueError(f"CSV file contains no samples: {csv_path}")

        if task == '3cls':
            self.df['cell_type'] = self.df['cell_type'].replace({'CD16': 'CD66b'})

        expected_class_order = TASK_CONFIGS[task]['classes']
        expected_classes = set(expected_class_order)
        actual_classes = set(self.df['cell_type'].dropna().unique())
        if actual_classes != expected_classes:
            raise ValueError(
                f"Expected cell_type labels {sorted(expected_classes)}, but found "
                f"{sorted(actual_classes)} in {csv_path}."
            )

        repo_root = os.path.dirname(os.path.abspath(__file__))
        self.df['image_file_path'] = self.df['image_file_path'].map(
            lambda path: path if os.path.isabs(path)
            else os.path.normpath(os.path.join(repo_root, path))
        )
        logging.info(f"Loaded {len(self.df)} samples from: {csv_path}")
        
        self.transform = transform
        self.padding = padding
        self.is_train = is_train
        
        # Build the class mapping.
        self.classes = expected_class_order
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        logging.info(f"Final dataset: {len(self.df)} samples, classes: {self.classes}")
        
        # Organize temporal sequences by track_id.
        if 'track_id' not in self.df.columns or 'frame_idx' not in self.df.columns:
            logging.warning(
                "The CSV has no track_id or frame_idx column; using single-frame mode."
            )
            self.use_temporal = False
            self.sequences = None
        else:
            self.use_temporal = True
            self._build_sequences()
        
        logging.info("Class distribution:")
        for cls in self.classes:
            count = len(self.df[self.df['cell_type'] == cls])
            logging.info(f"  {cls}: {count} samples")

    def _build_sequences(self):
        """
        Group frames by track_id and construct temporal subsequences.
        Each track is split into chunks of at most max_seq_length frames.
        Padding is handled by the collate function.
        """
        self.sequences = []
        
        # Group samples by track_id.
        grouped = self.df.groupby('track_id')
        
        for track_id, group_df in grouped:
            # Sort frames chronologically.
            group_df = group_df.sort_values('frame_idx')
            
            # Collect frame paths and labels for this track.
            img_paths = group_df['image_file_path'].tolist()
            cell_types = group_df['cell_type'].tolist()
            frame_ids = group_df['frame_idx'].tolist()
            
            # Split the track into fixed-length subsequences.
            num_sequences = (len(img_paths) + self.max_seq_length - 1) // self.max_seq_length
            
            for seq_idx in range(num_sequences):
                start_idx = seq_idx * self.max_seq_length
                end_idx = min(start_idx + self.max_seq_length, len(img_paths))
                
                sub_img_paths = img_paths[start_idx:end_idx]
                sub_cell_types = cell_types[start_idx:end_idx]
                sub_frame_ids = frame_ids[start_idx:end_idx]
                
                # Record the unpadded sequence length.
                actual_length = len(sub_img_paths)
                
                # Map class names to integer labels.
                labels = [self.class_to_idx[cell_type] for cell_type in sub_cell_types]
                
                self.sequences.append({
                    'img_paths': sub_img_paths,
                    'labels': labels,
                    'track_id': track_id,
                    'frame_ids': sub_frame_ids,
                    'length': actual_length
                })
        
        logging.info(f"Constructed {len(self.sequences)} temporal sequences.")

    def __len__(self):
        if self.use_temporal and self.sequences:
            return len(self.sequences)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if self.use_temporal and self.sequences:
            # Temporal mode: return one sequence.
            seq = self.sequences[idx]
            img_paths = seq['img_paths']
            labels = seq['labels']
            seq_length = seq['length']
            
            # Load only real frames, excluding collate-time padding.
            actual_img_paths = img_paths[:seq_length]
            actual_labels = labels[:seq_length]
            
            # Load every frame in the subsequence.
            images = []
            for img_path in actual_img_paths:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file does not exist: {img_path}")
                
                try:
                    image = plt.imread(img_path)
                    # print(image.shape)
                except Exception as e:
                    raise ValueError(f"Unable to load image {img_path}: {e}")
                
                # Convert RGBA input to RGB.
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = image[:, :, :3]
                
                # padding
                if self.padding is not None:
                    image = self.padding(image)
                
                # transform
                if self.transform is not None:
                    image = self.transform(image)
                
                images.append(image)
            
            # Stack frames into [T_actual, C, H, W].
            images = torch.stack(images, dim=0)
            labels = torch.tensor(actual_labels, dtype=torch.long)
            
            return images, labels, seq_length
        else:
            # Single-frame mode.
            row = self.df.iloc[idx]
            img_path = row['image_file_path']
            label = self.class_to_idx[row['cell_type']]
            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            
            try:
                image = plt.imread(img_path)
            except Exception as e:
                raise ValueError(f"Unable to load image {img_path}: {e}")
            
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]
            
            if self.padding is not None:
                image = self.padding(image)

            if self.transform is not None:
                image = self.transform(image)
                
            return image, label