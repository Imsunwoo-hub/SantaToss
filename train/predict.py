from config import config
import utils
from models.cnn import CNN_Encoder
from models.rnn import RNN_Decoder, BahdanauAttention

import tensorflow as tf

import pickle

def predict(imgae_name):
    max_length = 49
    name = "tokenizer.pickle"
    with open(name, 'rb') as f:
        tokenizer = pickle.load(f)

    image_features_extract_model = tf.keras.models.load_model(config.extractor_path)

    encoder = CNN_Encoder(config.embedding_dim)
    decoder = RNN_Decoder(config.embedding_dim, config.units, config.top_k+1)
    attention = BahdanauAttention(config.units)

    optimizer = tf.keras.optimizers.Adam()

    ckpt_path = config.checkpoints_path
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    final_res = []
    for i in range(3):
        result = utils.evaluate(imgae_name,
                          image_features_extract_model,
                          encoder,
                          attention,
                          decoder,
                          max_length,
                          tokenizer)

        result = ' '.join(result)
        final_res.append(result)

    return '\n'.join(final_res)

predict("./datasets/train2014/COCO_train2014_%012d.jpg" % (9))