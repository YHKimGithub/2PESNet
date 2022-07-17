import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

#multi head detector
class MHD(torch.nn.Module):
    def __init__(self, opt):
        super(MHD, self).__init__()
        
        self.feat_dim = opt["mhd_feat_dim"]
        self.c_hidden = opt["mhd_hidden_dim"]
        self.mhd_best_loss = 10000000
        self.output_dim = opt["mhd_out_dim"]  
        
        self.rnn_cell = nn.LSTMCell(self.feat_dim,self.c_hidden, bias=True)
        self.rnn_cell_ed = nn.LSTMCell(self.feat_dim,self.c_hidden, bias=True)
        self.fc = nn.Linear(self.c_hidden,self.output_dim, bias=True)
        self.fc_ed = nn.Linear(self.c_hidden,self.output_dim, bias=True)
        
        init.kaiming_normal_(self.rnn_cell.weight_ih)
        init.kaiming_normal_(self.rnn_cell.weight_hh)
        init.kaiming_normal_(self.rnn_cell_ed.weight_ih)
        init.kaiming_normal_(self.rnn_cell_ed.weight_hh)
        

    def forward(self, x):
        input_size = x.size()
        
        #action detection head
        h=torch.zeros(input_size[0], self.c_hidden, requires_grad = True).cuda()
        c=torch.zeros(input_size[0], self.c_hidden, requires_grad = True).cuda()
        score_tensor=[]
        for i in range(0, input_size[1]):
            h,c = self.rnn_cell(x[:,i,:], (h,c))
            score = self.fc(h)
            score_tensor.append(score)            
        score_tensor=torch.stack(score_tensor,dim=1)
         
        #action end head
        h=torch.zeros(input_size[0], self.c_hidden, requires_grad = True).cuda()
        c=torch.zeros(input_size[0], self.c_hidden, requires_grad = True).cuda()
        ed_tensor=[]
        for i in range(0, input_size[1]):
            h,c = self.rnn_cell_ed(x[:,i,:], (h,c))
            ed = self.fc_ed(h)
            ed_tensor.append(ed)
        ed_tensor=torch.stack(ed_tensor,dim=1)
            
        return score_tensor, ed_tensor
        
    def forward_infer(self, x, h, c):
        input_size = x.size()
        
        h[0],c[0] = self.rnn_cell(x, (h[0],c[0]))
        score = self.fc(h[0])
        h[1],c[1] = self.rnn_cell_ed(x, (h[1],c[1]))
        ed = self.fc_ed(h[1])
        
        return score, ed, h, c

#end detection refinement
class EDR(torch.nn.Module):
    def __init__(self, opt):
        super(EDR, self).__init__()
        
        self.input_dim = opt["edr_input_dim"]
        self.hidden = opt["edr_hidden_dim"]
        self.edr_best_loss = 10000000
        self.output_dim = opt["edr_out_dim"]  
        
        self.fc1 = nn.Linear(self.input_dim,self.hidden, bias=False)
        self.fc2 = nn.Linear(self.hidden,self.output_dim, bias=False)
        self.dropout = nn.Dropout(p=0.6)
                
    def forward(self, x):
        input_size = x.size()
        
        x= self.fc1(x)
        x= self.dropout(x)
        x= torch.relu(x)
        x= self.fc2(x)
        return x
  
#action start detection    
class ASD(torch.nn.Module):
    def __init__(self, opt):
        super(ASD, self).__init__()
        
        self.feat_dim = opt["asd_feat_dim"]
        self.c_hidden = opt["asd_hidden_dim"]
        self.mhd_best_loss = 10000000
        self.output_dim = opt["asd_out_dim"]  
        
        self.rnn_cell_st = nn.LSTMCell(self.feat_dim,self.c_hidden, bias=True)
        self.fc_st = nn.Linear(self.c_hidden,self.output_dim, bias=True)
        
        init.kaiming_normal_(self.rnn_cell_st.weight_ih)
        init.kaiming_normal_(self.rnn_cell_st.weight_hh)
        
    def forward(self, x):
        input_size = x.size()
        h=torch.zeros(input_size[0], self.c_hidden, requires_grad = True).cuda()
        c=torch.zeros(input_size[0], self.c_hidden, requires_grad = True).cuda()
        st_tensor=[]
        for i in range(0, input_size[1]):
            h,c = self.rnn_cell_st(x[:,i,:], (h,c))
            st = self.fc_st(h)
            st_tensor.append(st)
        st_tensor=torch.stack(st_tensor,dim=1)
            
        return st_tensor
        
    def forward_infer(self, x, h, c):
        input_size = x.size()
        
        h,c = self.rnn_cell_st(x, (h,c))
        st = self.fc_st(h)
        return st, h, c
