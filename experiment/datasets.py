import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from sklearn.model_selection import train_test_split

# moonshot dataset figure classification Dataset

class ms_FigureClassification(Dataset):
    def __init__(self, dataset_path, category, template, train=True, transform=None):

        self.dataset_path = dataset_path
        self.annotation_file = os.path.join(dataset_path, 'annotation_files.json')
        self.train = train
        self.transform = transform
        self.category = category
        self.template = template
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
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data_list, self.labels, train_size=train_size, stratify=self.labels
        )

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
        label  =self.template.format(label)
        
        image = Image.open(os.path.join(self.dataset_path, img_path))
        if self.transform:
            image = self.transform(image)
        
        return image, label, label_id



## データセットのインスタンスを作成
#data_path = "data.json"  # JSONファイルのパスを適宜修正
#train_dataset = CustomDataset(data_path, train=True, transform=transform)
#test_dataset = CustomDataset(data_path, train=False, transform=transform)
#
## DataLoaderを作成
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



if __name__=='__main__':
    import configs
    config = configs.Config()


    # 画像をリサイズするための変換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    train_dataset = ms_FigureClassification(dataset_path=config.dataset_path, 
                                            category=config.category, 
                                            template=config.template, 
                                            train=True, 
                                            transform=transform)

    image, label, _ = train_dataset[0]
    print('image.shape : ', image.shape)
    print('label : ', label)

    print('len(train_dataset) : ', len(train_dataset))
    #print('len(test_dataset) : ', len(test_dataset))
