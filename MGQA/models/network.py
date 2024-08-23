import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from models.modules import DomainLevelGragh


# DualGragh Regression
class Reg_Domain(nn.Module):
    def __init__(self, do_emb_size, eg_emb_size, pretrain):
        super(Reg_Domain, self).__init__()


        # for key, p in self.named_parameters():
        #     # print(key)
        #     p.requires_grad = False

        self.domainlevelgraph = DomainLevelGragh(8000, do_emb_size, eg_emb_size, pretrain)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.predictor = nn.Sequential(
        #     nn.Linear(5120, 2000),
        #     nn.ReLU(True),
        #     nn.Linear(2000, 1)
        #     )
        self.hyperpredmos = HyperPred(8000 + eg_emb_size)

        # self.classifier = nn.Linear(256, 25)


        for m in self.modules():
            self.weights_init(m)


    def forward(self, input, length):
        # input = F.pad(input, pad=(1, 1, 1, 1), mode='constant', value=1)
        # input = input.view(input.shape[0],input.shape[1], -1)
        # input = input[:,:,:,0,0]
        B, l ,_= input.shape
        score = torch.zeros((B))
        '''
        x: (N, C, H, W); In Kadid-10k, (N, 3, 224, 224).
        N: batch size (i.e. number of domain graph nodes)
        '''
          # (N, 2048, 7, 7)
        
        for i in range(B):   
            # print(length[i][0])
            x = input[i, :int(length[i][0])]
            # x = x[:,0,:]
            ins_emb, eg_emb_eg, level_pred, do_emb = self.domainlevelgraph(x)

            # regression
            mean, scale = self.hyperpredmos(torch.cat([ins_emb, eg_emb_eg], -1))
            x = self._mos_vae(mean, scale)

            # # node only
            # mean, scale = self.hyperpredmos(do_emb)
            # x = self._mos_vae(mean, scale)

            # # edge only
            # mean, scale = self.hyperpredmos(eg_emb_eg)
            # x = self._mos_vae(mean, scale)

            
            # x = self.global_pool(x).view(x.size(0), -1)
            # x = self.predictor(x)   # (N, M) -> (N, 1)

            # do_code = do_emb.mean(0)
            # type_pred = self.classifier(do_emb)
            score[i] = torch.mean(x)
            
            
            # score [i] =   torch.mean(self.predictor(x))

        return score


    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def _mos_vae(self,mean,scale):
        # (N, P)
        noise = torch.randn(mean.size()).cuda()
        mos_pred = mean + noise * scale
        return mos_pred

class HyperPred(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super(HyperPred, self).__init__()

        self.fc = nn.Sequential(nn.Linear(in_dim, in_dim // 2, bias=True),
                                  nn.ReLU(),
                                  nn.Linear(in_dim // 2, out_dim, bias=True))
        # for m in self.modules():
        #     self.weights_init(m)

    def forward(self, x):
        # input: (N, K)
        # output: (N, 2)
        x = self.fc(x)
        mean, scale = x.split(1, dim=1)  # (N, 1) * 2
        return mean, scale

    # def weights_init(self, m):
    #     if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)


if __name__ == '__main__':



    net =Reg_Domain(256,16, False)

    pretained_model = torch.load('/home/test/10t/wl/VQA_Model/Backbone/best.pth')
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in pretained_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    net.load_state_dict(model_dict, strict=False)

    device = torch.device('cuda')
    net = net.to(device)
    input = torch.ones((2,1,320,3,3)).to(device)
    x = net(input)
    print(x)
