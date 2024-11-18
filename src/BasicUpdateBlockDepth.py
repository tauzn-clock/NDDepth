import torch
from torch import nn
import torch.nn.functional as F

class BasicUpdateBlockDepth(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=3, context_dim=192):
        super(BasicUpdateBlockDepth, self).__init__()
                
        self.encoder = ProjectionInputDepth(cost_dim=cost_dim, hidden_dim=hidden_dim, out_chs=hidden_dim)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs+context_dim)
        self.d_head = DHead(hidden_dim, hidden_dim=hidden_dim)
        self.project = nn.Conv2d(256, hidden_dim, 1, padding=0)

    def forward(self, depth1, uncer1, depth2, uncer2, context, gru_hidden, seq_len=3):

        depth1_list = []
        depth1_list.append(depth1)
        depth2_list = []
        depth2_list.append(depth2)

        gru_hidden = torch.tanh(self.project(gru_hidden))
        diff = (depth1.detach() - depth2.detach()).abs()

        for i in range(seq_len):

            input_features = self.encoder(torch.cat([diff, uncer1.detach(), uncer2.detach()], 1), depth1.detach(),  depth2.detach())
            input_c = torch.cat([input_features, context], dim=1)

            gru_hidden = self.gru(gru_hidden, input_c)
            delta_d = self.d_head(gru_hidden)

            delta_d1 = delta_d[:, :1]
            delta_d2 = delta_d[:, 1:]
            
            depth1 = (depth1.detach() + delta_d1).clamp(1e-3, 1)
            depth2 = (depth2.detach() + delta_d2).clamp(1e-3, 1)
         
            depth1_list.append(depth1)
            depth2_list.append(depth2)
            
        return depth1_list, depth2_list

class ProjectionInputDepth(nn.Module):
    def __init__(self, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.convd1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, 64, 3, padding=1)

        self.convd3 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convd4 = nn.Conv2d(hidden_dim, 64, 3, padding=1)
        
        self.convd = nn.Conv2d(64*2+hidden_dim, out_chs - 2, 3, padding=1)
        
    def forward(self, cost, depth1, depth2):

        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))

        d1 = F.relu(self.convd1(depth1))
        d1 = F.relu(self.convd2(d1))

        d2 = F.relu(self.convd3(depth2))
        d2 = F.relu(self.convd4(d2))

        cor_d = torch.cat([cor, d1, d2], dim=1)
        
        out_d = F.relu(self.convd(cor_d))
                
        return torch.cat([out_d, depth1, depth2], dim=1)
    
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+192):
        super(SepConvGRU, self).__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1))) 
        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
    
class DHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(DHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_du, act_fn=F.tanh):
        out = self.conv2(self.relu(self.conv1(x_du)))
        return act_fn(out)