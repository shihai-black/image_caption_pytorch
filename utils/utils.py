# -*- coding: utf-8 -*-
# @project：2021_MAXP
# @author:caojinlei
# @file: utils.py
# @time: 2021/11/01
import os.path

import torch
import torch.distributed as dist


def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)


def cleanup():
    dist.destroy_process_group()


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoints(output_folder, data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_opt,
                     decoder_opt, bleu4,
                     is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_opt': encoder_opt,
             'decoder_opt': decoder_opt}
    filename = 'checkpoint_' + data_name + '.pth'
    torch.save(state, os.path.join(output_folder, filename))
    if is_best:
        torch.save(state, os.path.join(output_folder, 'BEST_' + filename))


def load_checkpoints(output_folder, data_name):
    filename = 'checkpoint_' + data_name + '.pth'
    checkpoints = torch.load(os.path.join(output_folder, 'BEST_' + filename))
    return checkpoints
