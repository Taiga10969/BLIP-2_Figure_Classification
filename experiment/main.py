import configs
import utils
import datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from transformers import AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration
from PIL import Image


utils.setup_torch_seed()

config = configs.Config()

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')


# model definition
processor = AutoProcessor.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = Blip2ForConditionalGeneration.from_pretrained(config.model_name, 
                                                      torch_dtype=torch.float16)

## 'qformer.encoder'を含むサブモジュール以外のパラメータをフリーズする

# Vision encoder をフリーズする
for name, param in model.named_parameters():
    if 'vision_model.encoder' in name:
        param.requires_grad = False

#for name, param in model.named_parameters():
#    if 'qformer.encoder' not in name:
#        param.requires_grad = False

model = nn.DataParallel(model)
model.to(device)




train_dataset = datasets.ms_FigureClassification(dataset_path=config.dataset_path, 
                                                 category=config.category, 
                                                 ans_template=config.ans_template_1, 
                                                 prompt=config.prompt_1, 
                                                 max_length=config.max_length, 
                                                 train=True,  
                                                 change_id=False, 
                                                 transform=None, 
                                                 processor=None, 
                                                 device=device
                                                 )

test_dataset = datasets.ms_FigureClassification(dataset_path=config.dataset_path, 
                                                 category=config.category, 
                                                 ans_template=config.ans_template_1, 
                                                 prompt=config.prompt_1, 
                                                 max_length=config.max_length, 
                                                 train=False,  
                                                 change_id=False, 
                                                 transform=None, 
                                                 processor=None, 
                                                 device=device
                                                 )


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=config.t_0, 
                                                              T_mult=config.t_mult)

best_val_loss=10.0
best_train_loss=10.0
iteration=0

for epoch in range(1, config.epoch+1, 1):
    with tqdm(train_loader) as pbar:
        pbar.set_description(f'[train epoch : {epoch}]')

        model.train()
        sum_train_loss = 0.0
        train_loss = 0.0
        train_count = 0

        for img_paths, prompts, labels in pbar:
            train_count+=1
            iteration+=1
            optimizer.zero_grad()

            image_list = []
            for path in img_paths:
                image = Image.open(path)
                image_list.append(image)


            inputs = processor(images=image_list, 
                               text=prompts, 
                               padding='max_length', 
                               truncation=True, 
                               max_length=config.max_length, 
                               return_tensors='pt').to(device, torch.float16)
            
            labels = processor(text=labels, 
                               padding='max_length', 
                               truncation=True, 
                               max_length=config.max_length, 
                               return_tensors='pt')["input_ids"]

            outputs = model(pixel_values=inputs["pixel_values"], 
                            input_ids=inputs["input_ids"], 
                            labels = labels)

            
            loss = outputs.loss.mean()

            sum_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            ave_loss=sum_train_loss/train_count,

            pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=ave_loss[0], lr = optimizer.param_groups[0]['lr']))
            with open(config.record_dir+'/loss_record.csv', 'a') as f:
                print(f'{iteration}, {loss.item()}', file=f)
            
            if train_count%100 ==0 and best_train_loss > ave_loss[0]:
                best_train_loss = ave_loss[0]
                torch.save(model.module.state_dict(), config.record_dir+"/train_best.pth")
            
        train_loss=sum_train_loss/train_count

        
        

    with tqdm(test_loader) as pbar:
        pbar.set_description(f'[valid epoch : {epoch}]')

        model.eval()
        sum_valid_loss = 0.0
        valid_loss = 0.0
        valid_count = 0

        for img_paths, prompts, labels in pbar:
            valid_count+=1

            image_list = []
            for path in img_paths:
                image = Image.open(path)
                image_list.append(image)


            inputs = processor(images=image_list, 
                               text=prompts, 
                               padding='max_length', 
                               truncation=True, 
                               max_length=config.max_length, 
                               return_tensors='pt').to(device, torch.float16)
            
            labels = processor(text=labels, 
                               padding='max_length', 
                               truncation=True, 
                               max_length=config.max_length, 
                               return_tensors='pt')["input_ids"]

            outputs = model(pixel_values=inputs["pixel_values"], 
                            input_ids=inputs["input_ids"], 
                            labels = labels)

            
            loss = outputs.loss.mean()
            sum_valid_loss += loss.item()

            pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=sum_valid_loss/valid_count))

        valid_loss=sum_valid_loss/valid_count
        
    torch.save(model.module.state_dict(), config.record_dir+"/epoch_"+str(epoch)+".pth")

    if best_val_loss > valid_loss:
        best_val_loss = valid_loss
        torch.save(model.module.state_dict(), config.record_dir+"/best_val_loss.pth")

    with open(config.record_dir+'/epoch_loss_record.csv', 'a') as f:
        print(f'{epoch}, {train_loss}, {valid_loss}', file=f)
