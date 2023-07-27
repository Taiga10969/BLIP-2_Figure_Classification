import configs
import utils
import datasets

from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim


config = configs.Config()

model = utils.SimpleModel()

train_dataset = datasets.ms_FigureClassification(dataset_path=config.dataset_path, 
                                                 category=config.category, 
                                                 ans_template=config.ans_template_1, 
                                                 prompt=config.prompt_1, 
                                                 max_length=config.max_length, 
                                                 train=True,  
                                                 change_id=False, 
                                                 transform=None, 
                                                 processor=None, 
                                                 )

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)


optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=config.t_0, 
                                                              T_mult=config.t_mult)

#utils.plot_learning_rate(optimizer, lr_scheduler, config.epoch)

lr_values = []

for epoch in range(1, config.epoch+1, 1):
    with tqdm(train_loader) as pbar:
        pbar.set_description(f'[train epoch : {epoch}]')

        for img_paths, prompts, labels in pbar:
            lr = optimizer.param_groups[0]['lr']
            lr_values.append(lr)
            lr_scheduler.step()

# 学習率の曲線をプロット
plt.plot(lr_values)
plt.xlabel('iterator')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.savefig('learning_rate_scheduler.svg')
plt.savefig('learning_rate_scheduler.png')
plt.close()
