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
        self.word_embed = config['word_embed']
        self.num_classes = config['num_classes']

        self.outdir = config['outdir']
        self.logdir = config['logdir']

        self.max_word_num = config['max_word_num']
        self.max_sent_num = config['max_sent_num']
        
        self.user_dim = config['user_dim']        
        self.prod_dim = config['prod_dim']
                
        self.word_dim = config['word_dim']
        self.finetune = config['finetune']
        self.word_hidden_size = config['word_hidden_size']
        self.sent_hidden_size = config['sent_hidden_size']

        self.device = config['device']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.dropout = config['dropout']
        self.stop_at = config['stop_at']
        self.eval_at = config['eval_at']


        # random seed
        # self.seed = 123
        self.seed = np.random.randint(1e8)
        
        # logg path
        # self.logdir = "{}/{}.txt".format(self.logdir, self.dataset)
        self.logdir = "{}/{}_{}.txt".format(self.logdir, self.dataset, time.strftime("%m-%d_%H-%M-%S"))
        
        # model path
        # self.model = "{}/{}.model".format(self.outdir, self.dataset)
        self.model = "{}/{}.model_{}".format(self.outdir, self.dataset, time.strftime("%m-%d_%H-%M-%S"))
        

        # params from CMD have the highest priority
        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

        
        # boolean params
        self.finetune = bool(self.finetune)


    def __repr__(self):
        return '{}'.format(self.__dict__.items())
    
