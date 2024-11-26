import torch
import torch.nn as nn


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log((depth_est*mask + 1e-7)) - torch.log((depth_gt*mask + 1e-7))
        return (torch.sqrt((d ** 2).mean(dim=(1,2,3)) - self.variance_focus * (d.mean(dim=(1,2,3)) ** 2))).mean()
        #Mathmetically incorrect but necessary engineering hack
        #batch_error = torch.sqrt((d ** 2).mean(dim=(1,2,3)) - variance_focus * (d.mean(dim=(1,2,3)) ** 2)) * scale
        #return batch_error

def rms_loss(gt, pred):
    assert gt.shape == pred.shape, 'Input tensors must have the same shape'
    scale = 10.0
    
    rms = (gt - pred) ** 2
    rms = torch.sqrt(rms.mean(dim=(1,2,3)))

    return rms * scale

def get_metrics(gt, pred, mask):
    gt = gt * mask + 1e-7
    pred = pred*mask + 1e-7

    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).int().float().mean(dim=(1,2,3)).mean()
    d2 = (thresh < 1.25 ** 2).int().float().mean(dim=(1,2,3)).mean()
    d3 = (thresh < 1.25 ** 3).int().float().mean(dim=(1,2,3)).mean()

    rms = (gt - pred) ** 2
    rms = torch.sqrt(rms.mean(dim=(1,2,3))).mean()

    log_rms = (torch.log(gt) - torch.log(pred)) ** 2
    log_rms = torch.sqrt(log_rms.mean(dim=(1,2,3))).mean()

    abs_rel = (torch.abs(gt - pred) / gt).mean(dim=(1,2,3)).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean(dim=(1,2,3)).mean()

    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt((err ** 2).mean(dim=(1,2,3)) - (err).mean(dim=(1,2,3)) ** 2).mean() * 100

    err = torch.abs(torch.log10(pred) - torch.log10(gt))
    log10 = err.mean(dim=(1,2,3)).mean()

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]
