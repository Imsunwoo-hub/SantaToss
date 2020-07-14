from config import config
from utils import utils, train_utils, test_utils

from data import preprocess
from models import encoder, decoder
from models.encoder import CNN_Encoder
from models.decoder import RNN_Decoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

cnn_encoder = CNN_Encoder(config.embedding_dim)
rnn_decoder = RNN_Decoder(config.embedding_dim, config.units, config.top_k+1)

## check point ##
checkpoint_path = config.checkpoint_path
ckpt = tf.train.Checkpoint(encoder=cnn_encoder,
                           decoder=rnn_decoder,
                           optimizer=train_utils.optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
ckpt.restore(ckpt_manager.latest_checkpoint)
## END: check point ##

#tf.keras.models.save_model(rnn_decoder, config.rnn_model_path)

input_img = config.user_image
if input_img == '':
    209274114
    input_img = ".\\datasets\\images\\20200129_143652_871.jpg";
    #input_img = ".\\datasets\\images\\2784746.jpg";

image_dataset = tf.data.Dataset.from_tensor_slices([input_img])
image_dataset = image_dataset.map(preprocess.loading_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)

for img, path in tqdm(image_dataset):
    batch_features = encoder.extract_feature_from_image(img)

    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

image_dataset = tf.data.Dataset.from_tensor_slices([input_img])
image_dataset = image_dataset.map(lambda item1: tf.numpy_function(
          test_utils.map_func, [item1], [tf.float32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)


tokenizerPath = '.\\datasets\\textTokenizers\\Tokenizer.pickle'
with open(tokenizerPath, 'rb') as handle:
    tokenizer = pickle.load(handle)



for img_tensor in image_dataset:
    attention_plot = np.zeros((50, 64))
    features = cnn_encoder(img_tensor[0])

    hidden = rnn_decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    res_caption = []
    for i in range(50):
        predictions, hidden, attention_weigth = rnn_decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weigth, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        if tokenizer.index_word[predicted_id] == '<unk>':
            continue
        if tokenizer.index_word[predicted_id] in '<end>':
            break
        res_caption.append(tokenizer.index_word[predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)

    final_res = ""
    for cap in res_caption:
        final_res += cap + ' '

    plt.title(final_res)
    image = plt.imread(input_img)
    plt.imshow(image)
    plt.show()