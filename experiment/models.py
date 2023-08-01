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

from torchinfo import summary

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



if __name__ == '__main__':

    summary(model)

    for name, param in model.named_parameters():
        #print('name : ', name, end='\t\t')
        print('name : ', name)
        #print('param.requires_grad = ', param.requires_grad)


    ## 'qformer.encoder'を含むサブモジュール以外のパラメータをフリーズする
    #for name, param in model.named_parameters():
    #    if 'qformer.encoder' not in name:
    #        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if any(f'language_model.model.decoder.layers.{layer_num}.' in name for layer_num in range(1, 16)):
            param.requires_grad = False


    for name, param in model.named_parameters():
        print('name : ', name, end='\t\t')
        print('param.requires_grad = ', param.requires_grad)


    summary(model)
