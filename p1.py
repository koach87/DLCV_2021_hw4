import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import random
import torch.nn.functional as F
import sys
args = sys.argv

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

val_N_way = 5
val_K_shot = 1
val_N_query = 15

filenameToPILImage = lambda x: Image.open(x)
# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

class Convnet(nn.Module):
    def __init__(self, in_chaneels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_chaneels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_chaneels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_chaneels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def predict(model):
    with torch.no_grad():
        ans = None
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(tqdm(val_loader)):
            model.eval()
            # split data into support and query data
            support_input = data[:val_N_way * val_K_shot,:,:,:] 
            query_input   = data[val_N_way * val_K_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * val_K_shot] : i for i in range(val_N_way)}
            
            # TODO: extract the feature of support and query data
            # shape:torch.Size([N-way*K-shot, 1600]) torch.Size([N-way*N-query, 1600])
            support_feature = model(support_input.to(device))
            query_feature = model(query_input.to(device))
            support_feature = support_feature.view(val_N_way,val_K_shot,1600).mean(dim=1)

            # TODO: calculate the prototype for each class according to its support data
            # ref:https://discuss.pytorch.org/t/calculating-eucledian-distance-between-two-tensors/85141
            query_to_clses = None
            for i in query_feature:
                i = i.view(1,-1)
                reg = ((support_feature - i)**2).sum(1).view(1,-1)
                if query_to_clses is not None:
                    query_to_clses = torch.cat((query_to_clses, reg), 0)
                else:
                    query_to_clses = reg

            a = query_to_clses.argmin(dim=1).view(1,-1)
            # TODO: classify the query data depending on the its distense with each prototype
            if ans is not None:
                ans = torch.cat((ans, a), 0)
            else:
                ans = a
        pd.DataFrame(
            ans.tolist(),
            columns=[f'query{i}'for i in range(val_N_query*val_N_way)],
        ).to_csv(args[4], index_label='episode_id')

val_csv, val_data_dir = args[1],args[2]
val_dataset = MiniDataset(val_csv, val_data_dir)

val_loader = DataLoader(
        val_dataset, batch_size=val_N_way * (val_N_query + val_K_shot),
        pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args[3]))
print(device)
model = torch.load('p1.pth',map_location=device)
model.to(device)
predict(model)

