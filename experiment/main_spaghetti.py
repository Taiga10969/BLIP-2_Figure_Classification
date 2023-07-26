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

model = nn.DataParallel(model)
model.to(device)



# 画像をリサイズするための変換
#transform = transforms.Compose([transforms.Resize((224, 224)),
#                                transforms.ToTensor()])


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


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=config.t_0, 
                                                              T_mult=config.t_mult)


for epoch in range(1, config.epoch+1, 1):
    with tqdm(train_loader) as pbar:
        pbar.set_description(f'[train epoch : {epoch}]')

        model.train()
        sum_train_loss = 0.0
        train_loss = 0.0
        train_count = 0

        for img_paths, prompts, labels in pbar:
            train_count+=1
            optimizer.zero_grad()

            #print('img_paths : ', img_paths)

            image_list = []
            for path in img_paths:
                image = Image.open(path)
                image_list.append(image)

            #print('image_list : ', image_list)
            #print('prompts : ', prompts)
            #print('labels : ', labels)


            inputs = processor(images=image_list, text=prompts, padding='max_length', truncation=True, max_length=config.max_length ,return_tensors='pt').to(device, torch.float16)
            #input = {k: v.to(self.device) for k, v in input.items()}
            labels = processor(text=labels, padding='max_length', truncation=True, max_length=config.max_length, return_tensors='pt')["input_ids"]
            #labels = labels.to(device)

            #pixel_values = pixel_values.to(device, torch.float16)
            #input_ids = input_ids.to(device)
            #labels = labels.to(device)
            #labels[labels == tokenizer.pad_token_id] = -100

            #print('pixel_values : ', pixel_values.shape)
            #print('input_ids : ', input_ids.shape)
            #print('labels : ', labels.shape)

            #inputs = processor(images=images, text=config.prompt_1, return_tensors='pt').to(device, torch.float16)
            #inputs = {k: v.to(device) for k, v in inputs.items()} #データをGPUデバイスに移動させる

            #labels = processor(text=labels, return_tensors='pt')['input_ids'].to(device)

            outputs = model(pixel_values=inputs["pixel_values"], 
                            input_ids=inputs["input_ids"], 
                            labels = labels)
            #print(outputs.loss)
            #print('loss : ', outputs.loss.mean())
            
            loss = outputs.loss.mean()

            #print(loss)
            
            loss.backward()
            optimizer.step()

            pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=sum_train_loss/train_count, lr = optimizer.param_groups[0]['lr']))
