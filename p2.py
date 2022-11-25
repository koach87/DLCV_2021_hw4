import torch
import torch.optim as optim
from torchvision import models
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import glob
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import sys
args = sys.argv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = {'Couch': 0,
 'Helmet': 1,
 'Refrigerator': 2,
 'Alarm_Clock': 3,
 'Bike': 4,
 'Bottle': 5,
 'Calculator': 6,
 'Chair': 7,
 'Mouse': 8,
 'Monitor': 9,
 'Table': 10,
 'Pen': 11,
 'Pencil': 12,
 'Flowers': 13,
 'Shelf': 14,
 'Laptop': 15,
 'Speaker': 16,
 'Sneakers': 17,
 'Printer': 18,
 'Calendar': 19,
 'Bed': 20,
 'Knives': 21,
 'Backpack': 22,
 'Paper_Clip': 23,
 'Candles': 24,
 'Soda': 25,
 'Clipboards': 26,
 'Fork': 27,
 'Exit_Sign': 28,
 'Lamp_Shade': 29,
 'Trash_Can': 30,
 'Computer': 31,
 'Scissors': 32,
 'Webcam': 33,
 'Sink': 34,
 'Postit_Notes': 35,
 'Glasses': 36,
 'File_Cabinet': 37,
 'Radio': 38,
 'Bucket': 39,
 'Drill': 40,
 'Desk_Lamp': 41,
 'Toys': 42,
 'Keyboard': 43,
 'Notebook': 44,
 'Ruler': 45,
 'ToothBrush': 46,
 'Mop': 47,
 'Flipflops': 48,
 'Oven': 49,
 'TV': 50,
 'Eraser': 51,
 'Telephone': 52,
 'Kettle': 53,
 'Curtains': 54,
 'Mug': 55,
 'Fan': 56,
 'Push_Pin': 57,
 'Batteries': 58,
 'Pan': 59,
 'Marker': 60,
 'Spoon': 61,
 'Screwdriver': 62,
 'Hammer': 63,
 'Folder': 64}
decoder = {0: 'Couch',
 1: 'Helmet',
 2: 'Refrigerator',
 3: 'Alarm_Clock',
 4: 'Bike',
 5: 'Bottle',
 6: 'Calculator',
 7: 'Chair',
 8: 'Mouse',
 9: 'Monitor',
 10: 'Table',
 11: 'Pen',
 12: 'Pencil',
 13: 'Flowers',
 14: 'Shelf',
 15: 'Laptop',
 16: 'Speaker',
 17: 'Sneakers',
 18: 'Printer',
 19: 'Calendar',
 20: 'Bed',
 21: 'Knives',
 22: 'Backpack',
 23: 'Paper_Clip',
 24: 'Candles',
 25: 'Soda',
 26: 'Clipboards',
 27: 'Fork',
 28: 'Exit_Sign',
 29: 'Lamp_Shade',
 30: 'Trash_Can',
 31: 'Computer',
 32: 'Scissors',
 33: 'Webcam',
 34: 'Sink',
 35: 'Postit_Notes',
 36: 'Glasses',
 37: 'File_Cabinet',
 38: 'Radio',
 39: 'Bucket',
 40: 'Drill',
 41: 'Desk_Lamp',
 42: 'Toys',
 43: 'Keyboard',
 44: 'Notebook',
 45: 'Ruler',
 46: 'ToothBrush',
 47: 'Mop',
 48: 'Flipflops',
 49: 'Oven',
 50: 'TV',
 51: 'Eraser',
 52: 'Telephone',
 53: 'Kettle',
 54: 'Curtains',
 55: 'Mug',
 56: 'Fan',
 57: 'Push_Pin',
 58: 'Batteries',
 59: 'Pan',
 60: 'Marker',
 61: 'Spoon',
 62: 'Screwdriver',
 63: 'Hammer',
 64: 'Folder'}

filenameToPILImage = lambda x: Image.open(x)
class OfficeDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Resize((128,128)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        # label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, path, index
        # return image, encoder[label]

    def __len__(self):
        return len(self.data_df)


def output(model,test_dataloader, csv_path):
    model.eval()
    print('start output')
    ids, fns, labels = [], [], []
    for data,fn,id in tqdm(test_dataloader):
        out = model(data.to(device))
        pred = out.argmax(dim=1)

        ids += [i.item() for i in id]
        fns += [i for i in fn]
        labels += [i.item() for i in pred]
    
    pd.DataFrame(
        {
            "id":ids,
            "filename":fns,
            "label":[ decoder[i] for i in labels ]
        }
    ).to_csv(csv_path,index = False)

test_dataset = OfficeDataset(args[1], args[2])
test_loader = DataLoader(test_dataset, batch_size=4)
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(512, 65)
)
model.load_state_dict(torch.load('p2.pt', map_location=device))
model.to(device)

output(model, test_loader, args[3])
