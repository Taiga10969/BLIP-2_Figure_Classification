class Config(object):
    def __init__(self):
        
        self.dataset_path = '/taiga/Datasets/moonshot-dataset'

        self.model_name = 'Salesforce/blip2-opt-2.7b'

        self.epoch = 10
        self.batch_size = 32

        self.max_length = 16

        self.category = ['chart', 'network architecture', 'qualitative evaluation']
        
        self.prompt_1 = 'What type of figure in this image?'
        self.ans_template_1 = 'this figure type is {}.'  # 'What type of figure in this image?'に対する回答


        # argument setting of optimizer (AdamW)
        ''' ===AdamW argument===
        >> params       : 最適化するパラメータ (model.parameters())
        >> lr           : 学習率 (default=1e-3)
        >> betas        : 勾配とその2乗の移動平均を計算するために使用される係数 (default=(0.9, 0.999))
        >> eps          : 数値の安定性を向上させるために分母に項を追加する (default=1e-8)
        >> weight_decay : 重み減衰係数 (default=1e-2)
        []
        ========================'''
        self.lr = 1e-5              #0.0005
        self.betas = (0.9, 0.999)
        self.eps = 1e-6             #0.000001
        self.weight_decay = 0.1
        
        # lr_schefuler
        '''===lr_scheduler.CosineAnnealingWarmRestarts argument===
        >> optimizer  : wrapped optimizer
        >> T_0        : 最初の restart の反復回数
        >> T_mult     : 再起動後にT_iを増加させる数 (default=1)
        >> eta_min    : 最小学習率 (default=0)
        >> last_epoch : 最後の epoch 値 (defailt=-1)
        >> verbose    : 更新ごとに stdout にメッセージを出力するか (default=False)
        []
        '''
        self.t_0 = 15
        self.t_mult = 1
