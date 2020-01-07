"""
Pytorch 를 이용해 Data loader 를 만든다.
"""
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data.sampler import SubsetRandomSampler


class NkDataSet(Dataset):

    # 데이터 초기화 시켜주는 작업
    def __init__(self, file_path):

        self.trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(128, 4),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(file_path, header=None)
        # asarray is convert the input to an array
        self.image_arr = np.asarray(self.data_info.iloc[:, 0][1:])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1][1:])
        self.label_arr = self.label_arr.astype(np.long) - 1

        self.label_arr = torch.from_numpy(self.label_arr)

        self.data_len = len(self.data_info.index)
        self.img_path = "/tmp/pycharm_project_917/img/faces_images/"

    # 경로를 통해서 실제 데이터의 접근을 해서 데이터를 돌려주는 함수
    def __getitem__(self, index):

        img_name = self.img_path + self.image_arr[index-1]
        img_as_img = Image.open(img_name)
        img_as_tensor = self.trans(img_as_img)

        img_label = self.label_arr[index-1]

        return img_as_tensor, img_label, img_name

    # 데이터의 전체 길이를 구하는 함수
    def __len__(self):
        return self.data_len

'''

shuffle_dataset = True

# Creating data indices for training and validation splits:

if shuffle_dataset :


# Creating PT data samplers and loaders:


'''


def get_data_loader(args):

    validation_split = .2
    random_seed = 42

    csv_path = '/tmp/pycharm_project_917/files/train_vision.csv'
    custom_dataset = NkDataSet(csv_path)

    test_csv_path = "/tmp/pycharm_project_917/files/test_vision.csv"
    test_custom_dataset = NkDataSet(test_csv_path)

    dataset_size = len(custom_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=64,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=64,
                                                    sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_custom_dataset, batch_size=1, shuffle=False)

    print("dasta set len i s", len(test_loader))
    return train_loader, val_loader, test_loader

'''
# csv 의 경로를 설정해 줘야 한다.
csv_path = '/tmp/pycharm_project_exam_vision/files/train_vision.csv'
custom_dataset = NkDataSet(csv_path)
my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=1,
                                                shuffle=False)


# enumerate 는 list 의 있는 내용을 순서를 매기면서 프린트를 한다.
for i, (images, labels, img_name) in enumerate(my_dataset_loader):
    print(labels, img_name)
'''
