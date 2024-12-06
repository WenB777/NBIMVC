import logging
import torch
import numpy as np
import math


def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def next_batch(X1, X2, X3, X4, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    indices = np.arange(tot)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        if end_idx - start_idx < 2:
            continue
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_indices = indices[start_idx:end_idx]

        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_indices)

def next_batch1(x1, x2, x3, x4, batch_size, complete_ratio=0.5):
    """Return data for next batch"""
    tot = x1.shape[0]
    total = math.ceil(tot / batch_size)

    mask_x3 = x3 == 1
    mask_x4 = x4 == 1
    mask_complete = mask_x3 & mask_x4
    x1_complete = x1[mask_complete]
    x2_complete = x2[mask_complete]

    x1_incomplete = x1[~mask_complete]
    x1_mask_incomplete = x3[~mask_complete]

    x2_incomplete = x2[~mask_complete]
    x2_mask_incomplete = x4[~mask_complete]

    select_size = int(batch_size * complete_ratio)
    incomplete_size = batch_size - select_size
    
    start_idx = 0
    end_idx = 0

    for i in range(total):
        complete_1_idx = torch.randint(0, x1_complete.size(0), (select_size,))
        complete_2_idx = torch.randint(0, x2_complete.size(0), (select_size,))
        complete_1 = x1_complete[complete_1_idx]
        complete_2 = x2_complete[complete_2_idx]
        
        end_idx = min(tot, start_idx + incomplete_size)
        if end_idx - start_idx < 2:
            continue
        
        batch_x1 = x1_incomplete[start_idx:end_idx]
        batch_x2 = x2_incomplete[start_idx:end_idx]
        batch_x3 = x1_mask_incomplete[start_idx:end_idx]
        batch_x4 = x2_mask_incomplete[start_idx:end_idx]

        batch_x1 = torch.cat([batch_x1, complete_1], dim=0)
        batch_x2 = torch.cat([batch_x2, complete_2], dim=0)
        batch_x3 = torch.cat([batch_x3, torch.ones(select_size, dtype=torch.int).cuda()], dim=0)
        batch_x4 = torch.cat([batch_x4, torch.ones(select_size, dtype=torch.int).cuda()], dim=0)
        batch_x1, batch_x2, batch_x3, batch_x4 = shuffle_batch(batch_x1, batch_x2, batch_x3, batch_x4)

        yield batch_x1, batch_x2, batch_x3, batch_x4, start_idx
        
        start_idx = end_idx


def shuffle_batch(batch_x1, batch_x2, batch_x3, batch_x4):
    """Shuffle the batch data"""
    indices = torch.randperm(batch_x1.size(0))
    return batch_x1[indices], batch_x2[indices], batch_x3[indices], batch_x4[indices]

def next_batch_local(X1, X2, X3, X4, mask, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]
        batch_x4 = X4[start_idx: end_idx, ...]
        batch_mask =mask[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2, batch_x3, batch_x4, batch_mask, start_idx, end_idx)

def next_batch2(X1, X2, batch_size):
    # generate next batch, just two views
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size) - 1  # fix the last batch
    if tot % batch_size == 0:
        total += 1

    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        yield batch_x1, batch_x2, (i + 1)


def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        print('ACC:'+ str(arg[0]))
        print('NMI:'+ str(arg[1]))
        print('ARI:'+ str(arg[2]))
        output = "ACC {:.3f} std {:.3f} NMI {:.3f} std {:.3f} ARI {:.3f} std {:.3f}".format( np.mean(arg[0]),
                                                                                             np.std(arg[0]),
                                                                                             np.mean(arg[1]),
                                                                                             np.std(arg[1]),
                                                                                             np.mean(arg[2]),
                                                                                             np.std(arg[2]))
    elif len(arg) == 1:
        print(arg)
        output = "ACC {:.3f} std {:.3f}".format(np.mean(arg), np.std(arg))

    print(output)
    return


def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
