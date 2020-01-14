#%%
import numpy as np
import matplotlib.pyplot as plt
import os 
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import transforms


#%%
class MyDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file+' does not exists!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir + self.names_list[idx].split(' ')[0]
        if not os.path.isfile(image_path):
            print(image_path+' does not exists!')
            return None
        image = cv2.imread(image_path, True)
        label = int(self.names_list[idx].split(' ')[1])

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample



class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image_new = np.transpose(image)
        return {'image': torch.from_numpy(image_new),
                'label': sample['label']}



def show_images_batch(sample_batched):
    images_batch, labels_batch = \
        sample_batched['image'], sample_batched['label']
    grid = make_grid(images_batch)
    plt.imshow(grid.numpy().transpose(1,2,0))

# %%
transformed_trainset = MyDataset(root_dir='data1/train',
                          names_file='data1/train/train.txt',
                          transform=transforms.Compose(
                              [ToTensor()]
                          ))

trainset_dataloader = DataLoader(dataset=transformed_trainset,
                                 batch_size=7,
                                 shuffle=True,
                                 num_workers=1)
# %%

# sample_batch:  Tensor , NxCxHxW
plt.figure()
for i_batch, sample_batch in enumerate(trainset_dataloader):
    show_images_batch(sample_batch)
    plt.axis('off')
    plt.ioff()
    plt.show()


plt.show()

# %%
