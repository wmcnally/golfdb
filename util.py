import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def correct_preds(probs, labels, tol=-1):
    """
    Gets correct events in full-length sequence using tolerance based on number of frames from address to impact.
    Used during validation only.
    :param probs: (sequence_length, 9)
    :param labels: (sequence_length,)
    :return: array indicating correct events in predicted sequence (8,)
    """

    events = np.where(labels < 8)[0]
    preds = np.zeros(len(events))
    if tol == -1:
        tol = int(max(np.round((events[5] - events[0])/30), 1))
    for i in range(len(events)):
        preds[i] = np.argsort(probs[:, i])[-1]
    deltas = np.abs(events-preds)
    correct = (deltas <= tol).astype(np.uint8)
    return events, preds, deltas, tol, correct


def freeze_layers(num_freeze, net):
    # print("Freezing {:2d} layers".format(num_freeze))
    i = 1
    for child in net.children():
        if i ==1:
            j = 1
            for child_child in child.children():
                if j <= num_freeze:
                    for param in child_child.parameters():
                        param.requires_grad = False
                j += 1
        i += 1