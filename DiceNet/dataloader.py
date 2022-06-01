import os
#import scipy.io as io
from torch.utils import data
from skimage import io
import numpy as np

IMG_EXTENSIONS = ['.mat']

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
   # print(classes)
    #class_low = classes[:3] + ['108']
    #print(class_low)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    #class_to_idx_low = {class_low[i]: i for i in range(len(class_low))}
    #print(class_to_idx)
    #print(class_to_idx_low)
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images

def make_dataset_low(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        #print(target)
        if target in ['101','102','103']:
            #print(target)
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


def mat_loader(path):
    #data = io.loadmat(path)
    img = io.imread(path)
    
    
    delta= []
    for i in range(1,60,1):
        image = np.abs(img[i-1] - img[i])
        delta.append(image)
    delta_array = np.array(delta)
    if delta_array.shape[0] == 59:
            delta_array = np.swapaxes(delta_array, 0, 1)
            delta_array = np.swapaxes(delta_array, 1, 2)
    
    #print(delta_array.shape)

    data = {}
    data['data'] = delta_array
    
    '''
    if img.shape[0] == 60:
            img = np.swapaxes(img, 0, 1)
            img = np.swapaxes(img, 1, 2)
    
    data = {}
    data['data'] = img
    '''

    return data

def default_loader(path):
    return mat_loader(path)


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        #imgs_low = make_dataset_low(root, class_to_idx_low)
        #print(len(imgs_low))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        #self.imgs_low = imgs_low
        self.classes = classes
        self.class_to_idx = class_to_idx
        #self.class_to_idx_low = class_to_idx_low
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        #print(index)
        #path_low, target_low = self.imgs_low[index]
        img = self.loader(path)
        if self.transform is not None:
            for i in self.transform:
                img = i(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            #target_low = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
'''
def to_one_hot_vector(num_class, label):
    b = np.zeros((label.shape[0], num_class))
    b[np.arange(label.shape[0]), label] = 1

    return b


from math import log10

class ImageFolder(data.Dataset):
    def __init__(self, df, transform=None, target_transform=None,
                 loader=default_loader):
        
        self.df = df
        self.image_files_list = self.df['file_name'].tolist()
        self.labels = self.df['label'].tolist()
    
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        
        img_name = self.image_files_list[idx]
        img = mat_loader(img_name)
        
        if self.transform is not None:
            for i in self.transform:
                img = i(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            #target_low = self.target_transform(target)

        label = log10(self.labels[idx]) 
        
        return img, label


'''