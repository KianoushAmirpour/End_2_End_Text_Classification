import torch.nn as nn
import torch
from .logger import setup_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logger = setup_logger(__name__)


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1)) # (batch_size,1) , (batch_size,1)


def train_one_epoch(dataloader, model, optimizer, device, epoch):
    model.train()
    train_metrics = {'accuracy': []}
    num_batches = len(dataloader)
    train_epoch_loss = 0
    for batch_idx, data in enumerate(dataloader):
        input_ids = data['input_ids'].to(device, dtype=torch.long) # batch_size, max_len
        attention_mask = data['attention_mask'].to(device, dtype=torch.long) # batch_size, max_len
        targets = data['label'].to(device, dtype=torch.float) # batch_size        
        outputs = model(input_ids, attention_mask) # (batch_size, 1), this is logits
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets) # loss per batch
        if batch_idx % 100 == 0:
            logger.info(f'Training | Epoch: {epoch} | Step:{batch_idx}/{num_batches} | Loss: {loss.item():.4f}')
            y_preds = torch.round(torch.sigmoid(outputs.view(-1))).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            train_metrics['accuracy'].append(accuracy_score(targets_np, y_preds))
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()
    return train_epoch_loss / num_batches, train_metrics
        
def evaluate_one_epoch(dataloader, model, device, epoch):
    model.eval()
    valid_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    num_batches = len(dataloader)
    valid_epoch_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            targets = data['label'].to(device, dtype=torch.float)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)
            if batch_idx % 100 == 0:
                logger.info(f'Validation | Epoch: {epoch} | Step:{batch_idx}/{num_batches} | Loss: {loss.item():.4f}')
                y_preds = torch.round(torch.sigmoid(outputs.view(-1))).detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()
                accuracy = accuracy_score(targets_np, y_preds)
                precision = precision_score(targets_np, y_preds)
                recall = recall_score(targets_np, y_preds)
                f1 = f1_score(targets_np, y_preds)
                valid_metrics['accuracy'].append(accuracy)
                valid_metrics['precision'].append(precision)
                valid_metrics['recall'].append(recall)
                valid_metrics['f1'].append(f1)
            valid_epoch_loss += loss.item()
    return valid_epoch_loss / num_batches, valid_metrics
