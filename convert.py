from pathlib import Path
import argparse
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import bcolz
import pickle
import cv2
import numpy as np
from torchvision import transforms as trans
import os
import numbers

def save_rec_to_img_dir(rec_path, save_correct_channel_order=False, save_as_png=False):

    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        if not isinstance(header.label, numbers.Number):
            label = int(header.label[0])
        else:
            label = int(header.label)

        if save_correct_channel_order:
            # this option saves the image in the right color.
            # but the training code uses PIL (RGB)
            # and validation code uses Cv2 (BGR)
            # so we want to turn this off to deliberately swap the color channel order.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()

        if save_as_png:
            img_save_path = label_path/'{}.png'.format(idx)
            img.save(img_save_path)
        else:
            img_save_path = label_path/'{}.jpg'.format(idx)
            img.save(img_save_path, quality=95)

def load_bin(path, rootdir, image_size=[112,112]):
    test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = test_transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='./faces_emore', type=str)

    args = parser.parse_args()
    rec_path = Path(args.rec_path)
    # save_rec_to_img_dir(rec_path)
    
    # bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    # bin_files = ['calfw', 'vgg2_fp', 'cfp_ff']
    bin_files = list(filter(lambda x: os.path.splitext(x)[1] in ['.bin'], os.listdir(args.rec_path)))
    bin_files = [i.split('.')[0] for i in bin_files]
    # import pdb; pdb.set_trace()
    for i in range(len(bin_files)):
        load_bin(rec_path/(bin_files[i]+'.bin'), Path("/home/thucth/thucth/data/face-regconition/faces_emore/" + bin_files[i]))

