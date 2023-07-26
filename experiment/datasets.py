import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from sklearn.model_selection import train_test_split

# moonshot dataset figure classification Dataset
'''====argument====
>> dataset_path : moonshot-dataset までのパス
>> category     : 含まれる論文図のカテゴリ分類 (idと紐付け)
>> ans_template : 期待する回答のテンプレート文 (文中の{}部分にcategoryのテキストが入る)
>> train        : trainデータの場合はTrueを指定 (default=True)
>> transform    : 出力する画像への変換の定義 (default=None)
==============='''

class ms_FigureClassification(Dataset):
    def __init__(self, 
                 dataset_path, 
                 category, 
                 ans_template, 
                 prompt, 
                 max_length, 
                 train=True, 
                 change_id=False, 
                 transform=None, 
                 processor=None, 
                 device='cpu'):

        self.dataset_path = dataset_path
        self.category = category
        self.ans_template = ans_template
        self.prompt = prompt
        self.max_length = max_length
        self.train = train
        self.transform = transform
        self.change_id = change_id
        self.processor = processor
        self.device = device

        self.annotation_file = os.path.join(dataset_path, 'annotation_files.json')    
        self.data_list = []
        self.labels = []
        
        with open(self.annotation_file) as f:
            annotation = json.load(f)
            for entry in annotation["annotations"]:
                img_path = entry["img_path"]
                category = entry["category"]
                self.data_list.append(img_path)
                self.labels.append(category)
        

        # データをトレーニングデータとテストデータに分割
        train_size = 0.9  # トレーニングデータの割合
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.data_list, 
                                                                                                self.labels, 
                                                                                                train_size=train_size, 
                                                                                                stratify=self.labels)

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, idx):

        if self.train:
            img_path = self.train_data[idx]
            label_id = self.train_labels[idx]
        else:
            img_path = self.test_data[idx]
            label_id = self.test_labels[idx]
        
        label = self.category[label_id]
        label  =self.ans_template.format(label)
        
        img_path = os.path.join(self.dataset_path, img_path)
        
        if self.change_id == True and self.processor != None and self.transform != None:

            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)

            input = self.processor(images=image, text=self.prompt, padding='max_length', truncation=True, max_length=self.max_length ,return_tensors='pt')
            input = {k: v.to(self.device) for k, v in input.items()}
            label = self.processor(text=label, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

            return input['pixel_values'].squeeze() , input['input_ids'].squeeze(), label['input_ids'].squeeze()
        
        else:
            return img_path, self.prompt, label



if __name__=='__main__':
    import configs
    from transformers import AutoProcessor

    config = configs.Config()

    # check GPU usage
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count()==0: print('Use 1 GPU')
    else: print(f'Use {torch.cuda.device_count()} GPUs')

    processor = AutoProcessor.from_pretrained(config.model_name)


    # 画像をリサイズするための変換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = ms_FigureClassification(dataset_path=config.dataset_path, 
                                            category=config.category, 
                                            ans_template=config.ans_template_1, 
                                            prompt=config.prompt_1, 
                                            max_length=config.max_length, 
                                            train=True, 
                                            transform=transform, 
                                            change_id=False, 
                                            processor=processor, 
                                            device=device
                                            )
    
    img_path, prompt, label = train_dataset[0]
    print('img_path : ', img_path)
    print('prompt : ', prompt)
    print('label : ', label)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    img_paths, prompts, labels = next(iter(train_loader))

    print('img_paths : ', img_paths)

    image_list = []
    for path in img_paths:
        image = Image.open(path)
        image_list.append(image)
    
    print('image_list : ', image_list)
    print('prompts : ', prompts)
    print('labels : ', labels)



    
    #pixel_value, input_id, label = train_dataset[0]
    #print('pixel_value : ', pixel_value.shape)
    #print('input_id : ', input_id.shape)
    #print('label : ', label.shape)
    #
    #train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    #
    ##one_batch = next(iter(train_loader))
    #pixel_values, input_ids, labels = next(iter(train_loader))
    #print('pixel_values : ', pixel_values.shape)
    #print('input_ids : ', input_ids.shape)
    #print('labels : ', labels.shape)


    '''
    test_dataset = ms_FigureClassification(dataset_path=config.dataset_path, 
                                            category=config.category, 
                                            ans_template=config.ans_template_1, 
                                            train=False, 
                                            transform=transform)

    image, label, label_id = train_dataset[0]
    print('image.shape : ', image.shape)
    print('label : ', label)
    print('label_id : ', label_id)

    print('len(train_dataset) : ', len(train_dataset))
    print('len(test_dataset) : ', len(test_dataset))

    ## DataLoader exsample
    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)'''
