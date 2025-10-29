import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import tarfile

def unpack_cifar10(path_to_cifar10, path_to_destination):

    if os.path.isdir(path_to_destination)==False:
         os.makedirs(path_to_destination)
         file = tarfile.open(path_to_cifar10)
         file.extractall(path_to_destination)
         file.close()

    return None

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict

def construct_image(data,label,storage_dir,batch):

    for i in range(data.shape[0]):
        row=data[i,:]
        red=row[0:1024].reshape((32,32))
        green=row[1024:1024*2].reshape((32,32))
        blue=row[1024*2:1024*3].reshape((32,32))

        im=np.array([red,green,blue])
        im=np.swapaxes(im,0,-1)
        im=np.swapaxes(im,0,1)

        im = Image.fromarray(im)
        im.save(os.path.join(storage_dir,'%s_%s_%s.png'%(batch,i,label[i])))

    return None

# CIFAR data is downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
Path_to_data_storage='.//Desktop//CIFAR//'    #Path to directory where the .tar.gz file is stored

path_to_cifar10=os.path.join(Path_to_data_storage,'cifar-10-python.tar.gz')
Path_to_compressed_data=os.path.join(Path_to_data_storage,'DATA')
unpack_cifar10(path_to_cifar10, Path_to_compressed_data)
Path_to_images=os.path.join(Path_to_data_storage,'Images')
Path_to_batch_data=os.path.join(Path_to_compressed_data,'cifar-10-batches-py')

if os.path.isdir(Path_to_images):
    shutil.rmtree(Path_to_images)
    os.mkdir(Path_to_images)
else:
    os.mkdir(Path_to_images)

all_files=os.listdir(Path_to_batch_data)
all_files=[i for i in all_files if 'data_batch' in i]

for i in range(len(all_files)):
    path=os.path.join(Path_to_batch_data,all_files[i])
    batch_nr=all_files[i].split('_')[-1]
    dictionary=unpickle(path)
    data=dictionary[b'data']
    labels=dictionary[b'labels']
    filenames=dictionary[b'filenames']
    Data=construct_image(data,labels,Path_to_images,batch_nr)



