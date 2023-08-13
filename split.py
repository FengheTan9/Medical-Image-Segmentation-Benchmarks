import os
import random
import argparse

from glob import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="busi", help='dataset_name')
parser.add_argument('--dataset_root', type=str, default="./data", help='dir')
args = parser.parse_args()

if __name__ == '__main__':

    name = args.dataset_name
    root = os.path.join(args.dataset_root, args.dataset_name)

    img_ids = glob(os.path.join(root, 'images', '*.png'))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.3, random_state=random.randint(0, 1024))

    with open(os.path.join(root, '{}_train.txt'.format(name)), 'w') as file:
        for i in train_img_ids:
            file.write(i + '\n')
    print("build train file successfully, path is: {}".format(os.path.join(root, '{}_train.txt'.format(name))))

    with open(os.path.join(root, '{}_val.txt'.format(name)), 'w') as file:
        for i in val_img_ids:
            file.writelines(i + '\n')
    print("build validate file successfully, path is: {}".format(os.path.join(root, '{}_val.txt'.format(name))))

