from utility.config import config

import tensorflow as tf
from sklearn.utils import shuffle

import os
import numpy as np
import json
import pickle


def download_coco_dataset():
    annotation_folder = 'annotations/'
    if not os.path.exists('./datasets/' + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath('../train/datasets'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        # os.remove(annotation_zip)

    # Download image files
    image_folder = 'train2014/'
    if not os.path.exists('./datasets/' + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('../train/datasets'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        # os.remove(image_zip)


def load_dataset():
    with open("./datasets/annotations/captions_train2014.json", 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        img_id = annot['image_id']
        full_coco_image_path = config.dataset_path + 'COCO_train2014_' + '%012d.jpg' % (img_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    train_captions, image_name_vector = shuffle(all_captions, all_img_name_vector)
    train_captions = train_captions[:config.dataset_size]
    image_name_vector = image_name_vector[:config.dataset_size]

    return image_name_vector, train_captions


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.img_size, config.img_size))
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    return img, image_path


def extract_image_features(img_names, image_features_extract_model):
    encode_train = sorted(set(img_names))
    
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for i, (img, path) in enumerate(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def tokenizer_to_captions(train_captions):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=config.top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    max_length = calc_max_length(train_seqs)

    # save word index
    word_index = tokenizer.word_index

    tmp_dict = {}
    for word, index in word_index.items():
        tmp_dict[index] = word

    name = "./tokenizer.pickle"
    with open(name, 'wb') as f:
        pickle.dump(tmp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, cap_vector, max_length


def map_func(image_name, cap):
    img_tensor = np.load(image_name.decode('utf-8') + '.npy')

    return img_tensor, cap