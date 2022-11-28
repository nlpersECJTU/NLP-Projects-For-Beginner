import json
import time
import numpy as np


class Config:
    def __init__(self, args):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.dataset = config['dataset']
        self.ckpdir = config['ckpdir']
        self.logdir = config['logdir']

        self.batch_size = config['batch_size']
        self.max_len = config['max_len']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.ffn_hidden = config['ffn_hidden']
        self.drop_prob = config['drop_prob']
        
        self.device = config['device']
        self.init_lr = config['init_lr']
        self.factor = config['factor']
        self.adam_eps = config['adam_eps']
        self.patience = config['patience']
        self.warmup = config['warmup']
        self.clip = config['clip']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']

        self.use_beam = config['use_beam']
        self.decode_max_len = config['decode_max_len']
        self.beam_size = config['beam_size']
        self.length_penalty = config['length_penalty']
        
        # random seed
        self.seed = np.random.randint(1e8)
        # self.seed = 57760604
        
        # logg path
        self.logdir = "{}/{}.log".format(self.logdir, self.dataset)
        # self.logdir = "{}/{}_{}.log".format(self.logdir, self.dataset, time.strftime("%m-%d_%H-%M-%S"))
        
        # ckpoint path
        self.ckpdir = "{}/{}.ckp".format(self.ckpdir, self.dataset)
        # self.ckpoint = "{}/{}.ckp_{}".format(self.ckpdir, self.dataset, time.strftime("%m-%d_%H-%M-%S"))
        

        # params from CMD have the highest priority
        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

        # bool
        self.use_beam = bool(self.use_beam)

        
    def __repr__(self):
        return '{}'.format(self.__dict__.items())
    