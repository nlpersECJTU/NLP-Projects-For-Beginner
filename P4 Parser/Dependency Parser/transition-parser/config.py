import json


class Config:
    def __init__(self, args):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.dataset = config['dataset']
        self.dtrain = config['dtrain']
        self.ddev = config['ddev']
        self.dtest = config['dtest']

        self.outdir = config['outdir']
        self.logdir = config['logdir']
        self.vocab = config['vocab']
        self.model = config['model']

        self.wdims = config['wdims']
        self.pdims = config['pdims']
        self.rdims = config['rdims']

        self.activation = config['activation']
        self.lstm_layers = config['lstm_layers']
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.hidden_size = config['hidden_size']
        self.window = config['window']

        self.oracle = config['oracle']

        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.optim = config['optim']
        self.lr = config['lr']
        self.dropout = config['dropout']
        self.seed = config['seed']
        self.debug = config['debug']

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return '{}'.format(self.__dict__.items())
    
