# -*- coding: utf-8 -*-
# @project：image_caption_base
# @author:caojinlei
# @file: preprocess.py
# @time: 2022/02/09
import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from cv2 import imread, resize
from random import seed, sample, choice


def create_input_file(dataset, info_path, image_folder, captions_per_image, min_word_freq, output_folder, max_len,
                      split_ratio=[0.6, 0.1, 0.3]):
    # assert sum(split_ratio) == 1  切割率要求为1
    image_names = []
    image_captions = []
    word_freq = {}
    with open(info_path, 'r') as f:
        for lines in f.readlines():
            result = lines.strip().split(',')  # 读每一行数据
            name = result[0]  # 第一个是jpg名称
            caption_list = result[1:]  # 后面是caption集合
            captions = []
            for i in range(len(caption_list)):
                caption = caption_list[i].split(' ')
                if len(caption) <= max_len:  # 判断是否符合最大长度
                    captions.append(caption)
                    for word in caption:
                        if word_freq.get(word):
                            word_freq[word] += 1
                        else:
                            word_freq[word] = 1
            if len(captions) == 0:
                continue
            else:
                image_names.append(name)
                image_captions.append(captions)

    assert len(image_names) == len(image_captions)  # 判断读取数据是否正确

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    image_length = len(image_names)
    train_index_end = int(split_ratio[0] * image_length)
    val_index_end = int(sum(split_ratio[:2]) * image_length)

    # 训练集合
    train_image_names = image_names[:train_index_end]
    train_image_captions = image_captions[:train_index_end]
    # 验证集合
    val_image_names = image_names[train_index_end:val_index_end]
    val_image_captions = image_captions[train_index_end:val_index_end]
    # 测试集合
    test_image_names = image_names[val_index_end:]
    test_image_captions = image_captions[val_index_end:]

    # 输出映射结果
    with open(os.path.join(output_folder, 'word_map' '.json'), 'w') as j:
        json.dump(word_map, j)

    #
    seed(123456)
    for names, captions, split_name in [(train_image_names, train_image_captions, 'TRAIN'),
                                        (val_image_names, val_image_captions, 'VAL'),
                                        (test_image_names, test_image_captions, 'TEST')]:
        with h5py.File(os.path.join(output_folder, split_name + '_IMAGES' + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image
            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('image', (len(names), 3, 256, 256), dtype='uint8')
            print("\nReading %s images and captions, storing to file...\n" % split_name)

            enc_captions = []
            cap_lengths = []
            for i, name in enumerate(tqdm(names)):
                # Sample captions
                if len(captions[i]) < captions_per_image:
                    caption = captions[i] + [choice(captions[i]) for _ in range(captions_per_image - len(captions[i]))]
                else:
                    caption = sample(captions[i], k=captions_per_image)
                img = imread(os.path.join(image_folder, name))
                # 图片标准化
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)

                img = resize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                # 标题预处理
                for j, c in enumerate(caption):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    # Find caption lengths

                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    cap_lengths.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(cap_lengths)
            print(f'captions length:{len(cap_lengths)}')

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split_name + '_CAPTIONS_' + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split_name + '_CAPLENS_' + '.json'), 'w') as j:
                json.dump(cap_lengths, j)


def flicker8k_process(info_path, process_path):
    re_dict = {}
    with open(info_path, 'r') as f:
        for lines in f.readlines():
            result = lines.strip().split(',')
            name = result[0]
            caption = ' '.join(result[1:])
            if re_dict.get(name):
                re_dict[name] = re_dict[name] + ',' + caption
            else:
                re_dict[name] = caption
    with open(process_path, 'w') as f:
        for k, v in re_dict.items():
            f.writelines(k + ',' + v + '\n')


if __name__ == '__main__':
    dataset = 'Flicker8k'
    info_path = '/data/cjl/Flicker8k/archive/captions.txt'
    process_path = '/data/cjl/Flicker8k/archive/pre_captions.txt'
    image_folder = '/data/cjl/Flicker8k/archive/Images'
    output = f'inputs/{dataset}'
    captions_per_image = 5
    min_word_freq = 5
    max_length = 50
    flicker8k_process(info_path, process_path)
    create_input_file(dataset, process_path, image_folder, captions_per_image, min_word_freq, output, max_length)
