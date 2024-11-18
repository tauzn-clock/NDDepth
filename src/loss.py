import torch
import torch.nn as nn


def silog_loss(depth_gt, depth_est):
    
    assert depth_gt.shape == depth_est.shape
    
    variance_focus = 0.85
    scale = 10.0
    d = torch.log(depth_est) - torch.log(depth_gt)
    batch_error = torch.sqrt((d ** 2).mean(dim=(2,3)) - variance_focus * (d.mean(dim=(2,3)) ** 2)) * scale
    return batch_error

def histogram_intersection_loss(a, b, num_bins=100):
    assert a.shape == b.shape, 'Input tensors must have the same shape'
    
    loss = torch.zeros(a.shape[0]).to(a.device)
    
    for batch in range(a.shape[0]):
        
        a[batch] -= a[batch].min()
        a[batch] /= a[batch].max()
        b[batch] -= b[batch].min()
        b[batch] /= b[batch].max()
        
        # Compute histograms (normalized over the range [0, 1])
        hist_a = torch.histc(a[batch], bins=num_bins, min=0, max=1)
        hist_b = torch.histc(b[batch], bins=num_bins, min=0, max=1)
        
        # Normalize histograms (so they sum to 1)
        hist_a = hist_a / hist_a.sum()
        hist_b = hist_b / hist_b.sum()

        # Compute histogram intersection (sum of minimums of the bins)
        intersection = torch.sum(torch.min(hist_a, hist_b))

        # The loss is the inverse of intersection, i.e., the smaller the intersection, the larger the loss
        loss[batch] = 1 - intersection
    return loss

def get_metrics(gt, pred):
    
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).int().float().mean()
    d2 = (thresh < 1.25 ** 2).int().float().mean()
    d3 = (thresh < 1.25 ** 3).int().float().mean()

    rms = (gt - pred) ** 2
    rms = torch.sqrt(rms.mean())

    log_rms = (torch.log(gt) - torch.log(pred)) ** 2
    log_rms = torch.sqrt(log_rms.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    err = torch.abs(torch.log10(pred) - torch.log10(gt))
    log10 = torch.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]
