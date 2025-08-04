import torch
import os
from os.path import join
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import platform
import torch.distributed as dist
import logging
import sys, socket
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import itertools
from sklearn.metrics import classification_report, confusion_matrix

from artifact_classifier import ArtifactClassifier 


# Configuration

CLASS_NAMES = ["hair", "gel_bubble", "ruler", "ink", "MEL", "NV"]

def parse_args():
    parser = argparse.ArgumentParser(description='Artifact Classifier Training/Testing Script')
    parser.add_argument('--data_path', type=str, help='Path to the dataset directory')
    parser.add_argument('--num_classes', type=int, default=4, choices=[4,6],
                        help='Number of classes for classification')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                      help='Mode to run the model in (train or test, default: train)')
    parser.add_argument('--checkpoint_dir', type=str, default="classifier",
                      help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='classifier/artifact_classifier.pth',
                      help='Path to model checkpoint for testing')
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='Number of epochs for training')
    parser.add_argument('--train_csv', type=str, default="splits/isic2019_train.csv")
    parser.add_argument('--val_csv', type=str, default="splits/isic2019_val.csv")
    parser.add_argument('--test_csv', type=str, default="splits/isic2019_test.csv")
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training/testing')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for training/testing')
    
    args = parser.parse_args()
    return args

def setup_logger(rank, mode):
    if rank != 0:
        return None
        
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    logger = logging.getLogger('isic_classifier')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    log_file = f'logs/isic_{mode}_artifact_{timestamp}.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info('='*50)
    logger.info(f'Starting new ISIC {mode} run')
    logger.info(f'Log file: {log_file}')
    logger.info('='*50)
    
    return logger

def setup(rank, world_size, mode):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    logger = setup_logger(rank, mode)
    
    if rank == 0:
        logger.info(f'Initializing process group with {world_size} GPUs')
        logger.info(f'System: {platform.system()}')
        logger.info(f'Python version: {sys.version}')
        logger.info(f'PyTorch version: {torch.__version__}')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            logger.info(f'CUDA version: {torch.version.cuda}')
            logger.info(f'Number of GPUs: {torch.cuda.device_count()}')
            for i in range(torch.cuda.device_count()):
                logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    
    return logger

def cleanup(rank, logger):
    if rank == 0 and logger:
        logger.info('Training completed')
    dist.destroy_process_group()

class ArtifactDataset(Dataset):
    def __init__(self, csv_file, data_path, class_names, transform=None):
        self.data = pd.read_csv(csv_file)
        # Assuming the CSV has columns for each class: 'hair', 'gel_bubble', 'ruler', 'ink'
        self.class_columns = class_names
        self.transform = transform
        self.base_path = data_path

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.base_path, self.data.iloc[idx]['image'] + '.jpg')
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            img_name = os.path.join(self.base_path, self.data.iloc[idx]['image'] + '_downsampled.jpg')
            image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Create multi-class label vector
        label = torch.tensor([self.data.iloc[idx][col] for col in self.class_columns], dtype=torch.float32)
        return image, label
    
def convert_prompt_to_metadata_isic(prompt: str):
    """
    Convert a single caption/prompt back into the metadata labels: 
    [hair, gel_bubble, ruler, ink, MEL, NV].
    """
    # Check disease
    MEL = 1 if "melanoma (MEL)" in prompt else 0
    NV = 1 if "melanocytic nevus (NV)" in prompt else 0
    
    # Check artifacts
    hair = 1 if "hair" in prompt else 0
    gel_bubble = 1 if "gel bubble" in prompt else 0
    ruler = 1 if "ruler" in prompt else 0
    ink = 1 if "ink" in prompt else 0
    
    return [hair, gel_bubble, ruler, ink, MEL, NV]

def train_model(args, rank, world_size, model, train_loader, val_loader, criterion, optimizer, scheduler, logger, num_epochs=20):
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.numel()
            
            if batch_idx % 50 == 0 and rank == 0 and logger:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                          f'Acc: {100 * train_correct/train_total:.2f}%, '
                          f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader.dataset)
        dist.all_reduce(torch.tensor(train_loss).to(rank))
        train_loss = train_loss / world_size
        
        if rank == 0 and logger:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(rank), labels.to(rank)
                    
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.numel()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100 * val_correct / val_total
            
            # Calculate per-class metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            logger.info(
                f'Epoch {epoch+1}/{num_epochs} - '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Acc: {val_accuracy:.2f}%, '
                f'LR: {scheduler.get_last_lr()[0]:.6f}'
            )
            
            # Print per-class metrics
            for i, class_name in enumerate(CLASS_NAMES[:args.num_classes]):
                class_acc = 100 * ((all_preds[:, i] == all_labels[:, i]).sum() / len(all_labels))
                logger.info(f'{class_name} Accuracy: {class_acc:.2f}%')
            
            if val_loss < best_val_loss:
                logger.info("Saving checkpoint...")
                best_val_loss = val_loss
                patience_counter = 0
                save_path = join(args.checkpoint_dir, 
                               f'artifact_classifier.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after epoch {epoch+1}')
                    break
        
        dist.barrier()

def test_model(args, rank, model, test_loader, criterion, logger):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics for each class
    if logger:
        logger.info(f'Test Results:')
        logger.info(f'Loss: {test_loss:.4f}')

        result_dict = {}
        
        for i, class_name in enumerate(CLASS_NAMES[:args.num_classes]):
            class_acc = 100 * ((all_preds[:, i] == all_labels[:, i]).sum() / len(all_labels))
            logger.info(f'{class_name} Accuracy: {class_acc:.2f}%')

            
            # Calculate precision, recall, and F1 score for each class
            true_pos = ((all_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum()
            false_pos = ((all_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
            false_neg = ((all_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f'{class_name} Precision: {precision:.4f}')
            logger.info(f'{class_name} Recall: {recall:.4f}')
            logger.info(f'{class_name} F1 Score: {f1:.4f}')

            result_dict[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': class_acc
            }

        # calculate on combinations of classes as well
        for combo in itertools.product([0,1], repeat=args.num_classes):
            combo_text = ', '.join([CLASS_NAMES[i] for i in range(args.num_classes) if combo[i] == 1])
            pred_combo = np.all(all_preds == np.array(combo), axis=1)
            true_combo = np.all(all_labels == np.array(combo), axis=1)

            tp_combo = np.sum(np.logical_and(pred_combo == 1, true_combo == 1))
            fp_combo = np.sum(np.logical_and(pred_combo == 1, true_combo == 0))
            fn_combo = np.sum(np.logical_and(pred_combo == 0, true_combo == 1))
            tn_combo = np.sum(np.logical_and(pred_combo == 0, true_combo == 0))
            
            logger.info(f'{combo_text} Precision: {tp_combo / (tp_combo + fp_combo):.4f}')
            logger.info(f'{combo_text} Recall: {tp_combo / (tp_combo + fn_combo):.4f}')
            logger.info(f'{combo_text} F1 Score: {2 * (tp_combo / (tp_combo + fp_combo)) * (tp_combo / (tp_combo + fn_combo)) / ((tp_combo / (tp_combo + fp_combo)) + (tp_combo / (tp_combo + fn_combo))):.4f}')
            logger.info(f'{combo_text} Accuracy: {(tp_combo + tn_combo) / (tp_combo + tn_combo + fp_combo + fn_combo):.4f}, count_total = {tp_combo + tn_combo + fp_combo + fn_combo}')

            result_dict[combo_text + "prompt"] = {
                'precision': tp_combo / (tp_combo + fp_combo),
                'recall': tp_combo / (tp_combo + fn_combo),
                'f1': 2 * (tp_combo / (tp_combo + fp_combo)) * (tp_combo / (tp_combo + fn_combo)) / ((tp_combo / (tp_combo + fp_combo)) + (tp_combo / (tp_combo + fn_combo))),
                'accuracy': (tp_combo + tn_combo) / (tp_combo + tn_combo + fp_combo + fn_combo)
            }
        # save to csv
        result_df = pd.DataFrame(result_dict).T
        result_df.to_csv('results.csv')
    
    return test_loss, all_preds, all_labels

def main_worker(rank, world_size, args):
    logger = setup_logger(rank, args.mode)
    
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.mode == 'train':
        setup(rank, world_size, args.mode)

        train_dataset = ArtifactDataset(args.train_csv, args.data_path, CLASS_NAMES[:args.num_classes], transform=train_transform)
        val_dataset = ArtifactDataset(args.val_csv, args.data_path, CLASS_NAMES[:args.num_classes], transform=test_transform)
        train_sampler = DistributedSampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            num_workers=4,
            pin_memory=True
        ) if rank == 0 else None
        
        model = ArtifactClassifier(args.num_classes).to(rank)
        model = DDP(model, device_ids=[rank])
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
        
        try:
            train_model(args, rank, world_size, model, train_loader, val_loader, criterion, 
                       optimizer, scheduler, logger, num_epochs=args.num_epochs)
        finally:
            cleanup(rank, logger)
            
    elif args.mode == 'test':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for test mode")
        
        # For testing, we only use one GPU
        if rank == 0:
            test_dataset = ArtifactDataset(args.test_csv, transform=test_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True
            )
            
            model = ArtifactClassifier().to(rank)
            checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{rank}')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            criterion = nn.BCEWithLogitsLoss()
            
            logger.info(f'Loading checkpoint from {args.checkpoint}')
            logger.info(f'Checkpoint epoch: {checkpoint["epoch"]}')
            logger.info(f'Validation loss: {checkpoint["val_loss"]:.4f}')
            logger.info(f'Validation accuracy: {checkpoint["val_accuracy"]:.2f}%')
            
            test_loss, all_preds, all_labels = test_model(args, rank, model, test_loader, criterion, logger)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #note the code doesn't work with multiple GPUS
    world_size = 1
    torch.multiprocessing.spawn(main_worker,
                              args=(world_size, args),
                              nprocs=world_size,
                              join=True)