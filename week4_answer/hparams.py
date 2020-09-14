#hyper parameters정리

from importify import Serializable

class HyperParameters(Serializable):
    def __init__(self):
        super(HyperParameters, self).__init__()
        self.device = 'cpud'

        self.batch_size = 128
        self.shuffle = True
        self.num_workers = 0

        self.input_size = 28
        self.hidden_size = 64
        self.n_layers = 2
        self.bidirectional = True

        self.out_size = 10

        #training param
        self.epoch = 100
        self.logging_interval = 10
        self.ckpt_interval = 10
        self.lr = 0.0001
        