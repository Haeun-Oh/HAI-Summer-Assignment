import torch
import torch.nn as nn

from .hparams import HyperParameters

class Model(nn.Module):
    def __init__(self, hparam: HyperParameters): #:HyperParameters로 hparam의 type을 보여줘서 가독성이 좋아지게 한다.
        super(Model, self).__init__()
        self.hparam = hparam
        self.lstm = nn.LSTM(input_size=hparam.input_size,
                            hidden_size=hparam.hidden_size,
                            num_layers=hparam.n_layers,
                            bidirectional=hparam.bidirectional,
                            batch_first=True)

    self.fc = nn.Linear(hparam.n_layers*self.n_directions*hparam.hidden_size, hparam.out_size)

    def forward(self, image):
        """ My Deep Learning Model's forward method 
        :param: image: [batch_size, seq_length(28), input_dim(28)]
        :return: label [batch_size, out_size(10)]
        """    
        n_directions = 2 if self.hparam.bidirectional else 1
        h_0 = torch.zeros(self.hparam.n_layers*n_directions, image.size(0),self.hparam.hidden_size).to(self.hparam.device)
        c_0 = torch.zeros(self.hparam.n_layers*n_directions, image.size(0),self.hparam.hidden_size).to(self.hparam.device)

        #h_n: [num_layers * num_directions, batch, hidden_size]
        _, (h_n, _) = self.lstm(image, (h_0, c_0))

        #[batch, num_layers * num_directions, hidden_size]로 구성 위치를 바꿔준다.
        #h_n: [batch, num_layers * num_directions* hidden_size]
        h_n = torch.reshape(h_n.permute(1, 0, 2), (image.size(0), -1))#batch_size만큼 앞에 잡아주고 나머지는 다 staking?

        #out: [batch, out_size(10)]
        out = self.fc(h_n)

        return out