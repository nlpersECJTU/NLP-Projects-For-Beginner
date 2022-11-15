import json
import time
import numpy as np

class Config:
    def __init__(self, args):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.dataset = config['dataset']
        self.dtrain = config['dtrain']
        self.ddev = config['ddev']
        self.dtest = config['dtest']

        self.bert_dir = config['bert_dir']
        self.ckpdir = config['ckpdir']
        self.logdir = config['logdir']
        
        self.train_max_seq_length = config['train_max_seq_length']
        self.eval_max_seq_length = config['eval_max_seq_length']

        self.device = config['device']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.bert_lr = config['bert_lr']
        self.other_lr = config['other_lr']
        self.clip_grad_norm = config['clip_grad_norm']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.save_steps = config['save_steps']
        self.warmup_proportion = config['warmup_proportion']


        # random seed
        self.seed = np.random.randint(1e8)
        # self.seed = 57854639
        
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

        
    def __repr__(self):
        return '{}'.format(self.__dict__.items())
    