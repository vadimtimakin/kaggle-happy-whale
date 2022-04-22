import numpy as np
import random
import torch
import os


def set_seed(seed: int):
    """Set a random seed for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_model(config, model, epoch, train_loss, metric, optimizer,
                 epochs_since_improvement, scheduler, scaler, is_best):
    '''Save PyTorch model.'''
    filename = 'best.ckpt' if is_best else 'last.ckpt'

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'metric': metric,
        'optimizer': optimizer.state_dict(),
        'epochs_since_improvement': epochs_since_improvement,
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
    }, os.path.join(config.paths.save_dir, filename))


def print_report(t, train_loss, metric, best_metric, lr):
    '''Print report of one epoch.'''
    print(f'Time: {t} s')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Current MAP@5: {metric:.3f}')
    print(f'Best MAP@5: {best_metric:.3f}')
    print(f'Learning Rate: {lr}')


def save_log(path, epoch, train_loss, metric, best_metric):
    '''Save log of one epoch.'''
    with open(path, 'a') as file:
        file.write(f'Epoch: {epoch} ')
        file.write(f'TrainLoss: {train_loss:.4f} ')
        file.write(f'MAP@5: {metric:.3f} ')
        file.write(f'Best: {best_metric:.3f} ')
        file.write('\n')


def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def map_per_image(label: np.array, predictions: np.array) -> float:
    """Auxiliary function for calculating metric."""
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def count_metric(labels: np.array, predictions: np.array) -> float:
    """MAP@5 metric."""
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])