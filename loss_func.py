import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, focal=False):
        super(MultiCrossEntropyLoss, self).__init__()
        self.focal = focal

    def forward(self, input, target):
        #IN: input: unregularized logits [B, C] target: multi-hot representaiton [B, C]
        target_sum = torch.sum(target, dim=1)
        target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
        target = target/target_div
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        if not self.focal:
            output = torch.sum(-target * logsoftmax(input), 1)
        else:
            softmax = nn.Softmax(dim=1).to(input.device)
            p = softmax(input)
            output = torch.sum(-target * (1 - p)**2 * logsoftmax(input), 1)
        return torch.mean(output)
  
#all nets use the same loss func, but split function for extension.
def MHD_loss_function(y,MHD_output,opt, use_focal=False):
    y = y.float().cuda()
    
    loss_func = MultiCrossEntropyLoss(focal=use_focal)
    
    y=y.reshape(-1,y.size(-1))
    MHD_output=MHD_output.reshape(-1,MHD_output.size(-1))
    loss = loss_func(MHD_output,y)
    
    loss_dict={}
    loss_dict["cost"] = loss
    return loss_dict

def EDR_loss_function(y,EDR_output,opt, use_focal=False):
    y = y.float().cuda()
    
    loss_func = MultiCrossEntropyLoss(focal=use_focal)
    
    y=y.reshape(-1,y.size(-1))
    EDR_output=EDR_output.reshape(-1,EDR_output.size(-1))
    loss = loss_func(EDR_output,y)
    
    EDR_output_fore = EDR_output[:,:-1]*y[:,:-1]
    y_fore=y[:,:-1]
    loss2 = torch.nn.L1Loss()(EDR_output_fore,y_fore)
    
    loss_dict={}
    loss_dict["cost"] = loss + opt['edr_loss_foredist']*loss2
    return loss_dict
        
def ASD_loss_function(y,ASD_output,opt, use_focal=False):
    y = y.float().cuda()
    
    loss_func = MultiCrossEntropyLoss(focal=use_focal)
    
    y=y.reshape(-1,y.size(-1))
    ASD_output=ASD_output.reshape(-1,ASD_output.size(-1))
    loss = loss_func(ASD_output,y)
    
    loss_dict={}
    loss_dict["cost"] = loss
    return loss_dict
