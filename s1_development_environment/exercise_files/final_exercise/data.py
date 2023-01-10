import torch
import numpy as np
from torch.utils.data import Dataset 


class CorruptMnist(Dataset):
    def __init__(self,train):
        if train:

            # exchange with the corrupted mnist dataset
            path = "/Users/hanluhe/Documents/MLops/dtu_mlops/data/corruptmnist/"

            ## load all training datasets
            images = []
            labels = []
            test = []

            for i in range(0,5):
                with np.load(path+'train_' +str(i)+'.npz') as f:
                    images.append(f['images'])
                    labels.append(f['labels'])

            images = torch.tensor(np.concatenate([c for c in images])).reshape(-1, 1, 28, 28)
            labels = torch.tensor(np.concatenate([c for c in labels]))
        else:
        # Load the validation data 
            path = "/Users/hanluhe/Documents/MLops/dtu_mlops/data/corruptmnist/"
            with np.load(path+'test.npz') as f:
                images_test, labels_test = f['images'], f['labels']
                images = torch.from_numpy(images_test).reshape(-1, 1, 28, 28)
                labels = torch.from_numpy(labels_test)

        self.images = images
        self.labels = labels

    def __len__(self):
            return self.labels.numel()
        
    def __getitem__(self, idx):
        return self.images[idx].float(), self.labels[idx]


if __name__ == "__main__":
    dataset_train = CorruptMnist(train=True)
    dataset_test = CorruptMnist(train=False)
    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)