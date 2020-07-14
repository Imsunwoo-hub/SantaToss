# -*- coding: utf-8 -*-

import os
import sys
import csv
import pickle
import gzip
from datetime import datetime
from config import config
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


# Sub1 Req. 3-1	이미지 경로 및 캡션 불러오기
def get_path_caption(caption_file_path):
    f = open(caption_file_path, 'r')
    caption_file = csv.reader(f)
    next(caption_file) # 맨 첫줄은 데이터 컬럼명이라 건너뜀

    img_path_list = []  # 모든 이미지 파일명(.jpg)을 담을 리스트(1*n개=31782) [a.jpg, b.jpg, c.jpg, ... ]
    caption_list = []   # 이미지에 대한 모든 캡션을 담을 리스트(이미지 1개당 캡션 5개, n*5개=158910) [[a is a, a is b, ...], [b is b, b is c, ...]]
    for i, line in enumerate(caption_file):
        data = '|'.join(line).split('|') # csv의 구분자인 ,(comma)로 분리된 data를 |로 이어붙인 뒤, |를 기준으로 재분할. ['1000092795.jpg', '0', 'Two young', 'White males are...']

        # 이미지 파일명 저장
        img_path_list.append(data[0])

        # data[1]은 캡션 번호라서 필요 X
        caption = data[2] # 이미지의 캡션([2]~)
        # ,를 포함하던 문장 이어붙이기 ex) 'Two young, White males are...'
        for cap in data[3:]:
            if len(cap) <= 0: break
            else: caption += ',' + cap
        caption_list.append('<start> ' + caption + ' <end> ')

    f.close()
    print("데이터셋 로딩 완료!")

    img_path_list, caption_list = shuffle(img_path_list, caption_list)

    with open(config.description_path, 'w') as f:
        for caption in caption_list:
            f.write("%s\t" % caption)

    return img_path_list[:config.dataset_size], caption_list[:config.dataset_size]


# Sub1 Req. 3-2	전체 데이터셋을 분리해 저장하기
def dataset_split_save(img_paths, captions):
    train_dataset_path = config.train_dataset_path
    val_dataset_path = config.validation_dataset_path
    te_dataset_path = config.test_dataset_path

    # train data와 val data로 분리
    train_img_paths, te_img_paths, train_captions, te_captions = train_test_split(img_paths, captions, test_size=0.2, random_state=0)
    train_img_paths, val_img_paths, train_captions, val_captions = train_test_split(train_img_paths, train_captions, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2

    # 분리한 data 저장
    with open(train_dataset_path, 'w') as f:
        for img_path, caption in zip(train_img_paths, train_captions):
            f.write("%s\t%s\n" % (img_path, caption))

    with open(te_dataset_path, 'w') as f:
        for img_path, caption in zip(te_img_paths, te_captions):
            f.write("%s\t%s\n" % (img_path, caption))

    with open(val_dataset_path, 'w') as f:
        for img_path, caption in zip(val_img_paths, val_captions):
            f.write("%s\t%s\n" % (img_path, caption))

    print("데이터셋 분리 및 .txt 저장 완료!")
    return train_img_paths, val_img_paths, train_captions, val_captions


# Sub1 Req. 3-3	저장된 데이터셋 불러오기
def get_data_file(dataset_path):
    img_paths = []
    captions = []

    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split("\t")
            img_paths.append(data[0]) # 이미지 파일명
            captions.append(data[1].rstrip("\n")) # 이미지의 토큰들

    print("%s 데이터셋 로딩 완료!" % dataset_path)
    return img_paths, captions


# Sub2 Req. 1-1. batch_size 개수만큼 이미지 데이터 로딩
def loading_imgs(img_paths):
    img = tf.io.read_file(img_paths)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.image_size, config.image_size))
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    return img, img_paths


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


# Sub2 Req. 1-2. 이미지 정규화
def image_normalization(imgs):
    mean = np.mean(imgs, axis=(0, 1)) # axis: (x, y) 기준으로 평균 계산
    std = np.var(imgs, axis=(0, 1))

    imgs = (imgs-mean)/std

    return imgs


# Sub2 Req. 3-1. tf.data.Dataset 생성
def convert_to_dataset(imgs, tokenizer):

    dataset = tf.data.Dataset.from_tensor_slices((imgs, tokenizer))

    return dataset


# Sub2 Req. 3-2. Image Data Augmentation by Daseul
def image_augmentation(dataset):
    augmented_dataset = (
        dataset
        .map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return augmented_dataset


def augment(image, captions):
    # degree = random.random

    image_size = config.image_size
    crop_size = int(image_size * 0.85)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.image.random_brightness(image, max_delta=0.5)
    # image = tfa.image.rotate(image, angles=0.1)
    # image = tf.image.adjust_saturation(image, 3)
    # image = tf.image.resize_with_crop_or_pad(image, 500, 500)  # Add 6 pixels of padding

    return image, captions


# Image Data Augmentation visualize
def visualize(raw ,original, augmented):
    fig = plt.figure()
    plt.subplot(1,3,1)
    plt.title('Original image')
    plt.imshow(raw)
    plt.subplot(1,3,2)
    plt.title('Normalization image')
    plt.imshow(original)
    plt.subplot(1,3,3)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()


# Sub2 Req. 2-1 텍스트 데이터 토큰화
def text_tokenizer(tr_captions, val_captions):
    captions = tr_captions + val_captions
    max_length = calc_max_length_caption(captions)

    tokenizer = Tokenizer(num_words=config.top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[1] = '<start>'
    tokenizer.word_index['<start>'] = 1
    tokenizer.index_word[2] = '<end>'
    tokenizer.word_index['<end>'] = 2
    tokenizer.fit_on_texts(captions)

    tr_tokens = tokenizer.texts_to_sequences(tr_captions)
    tr_tokens = sequence.pad_sequences(tr_tokens, maxlen=max_length, padding='post')

    val_tokens = tokenizer.texts_to_sequences(val_captions)
    val_tokens = sequence.pad_sequences(val_tokens, maxlen=max_length, padding='post')

    # path = "./datasets/textTokenizers/"
    # now = datetime.today()
    # name = str(now)+"captionToken.txt"
    # name = "".join(i for i in name if i not in "\/:*?<>|")
    # name = path + name
    #
    # with open(name, 'w') as f:
    #     for i in range(len(captions)) :
    #         for j in range(len(captions[i])) :
    #             f.write("\n"+str(captions[i][j]) + " = " +str(tokens[i][j]))
    return tokenizer, tr_tokens, val_tokens


def calc_max_length_caption(captions):
    max = 0
    for caption in captions:
        word = caption.split(" ")
        word = [n for n in word if n != "." and n != "," and n != " "]
        length = len(word)
        if max < length:
            max = length

    return max

# Sub2 Req. 2-2 피클 저장 및 불러오기
def save_load(tokenizer):
    path = ".\\datasets\\textTokenizers\\"
    ## tokens를 저장할 폴더가 없으면 생성 by daseul
    if not os.path.isdir(path):
        os.mkdir(path)

    name = "Tokenizer.pickle"
    name = "".join(i for i in name if i not in "\/:*?<>|")
    name = path + name

    with open(name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return
