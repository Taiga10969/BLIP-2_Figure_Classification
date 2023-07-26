import configs
import utils
import datasets

import torch.optim as optim


config = configs.Config()

model = utils.SimpleModel()


optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=config.t_0, 
                                                              T_mult=config.t_mult)

utils.plot_learning_rate(optimizer, lr_scheduler, config.epoch)
