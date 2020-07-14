import tensorflow as tf
import numpy as np

from data import preprocess
from models import encoder
from utils import train_utils

def evaluate(dataset, max_length, cnn_encoder, rnn_decoder, tokenizer):
    attention_plot = np.zeros((max_length, 64))

    val_loss = 0
    results = []
    for (batch, (img_tensor, target)) in enumerate(dataset):
        hidden = rnn_decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        features = cnn_encoder(img_tensor)

        total_loss = 0
        for i in range(1, target.shape[1]):
            predictions, hidden, attention_weigth = rnn_decoder(dec_input, features, hidden)

            tmp_result = []
            for i in range(max_length):
                # attention_plot[i] = tf.reshape(attention_weigth, (-1,)).numpy()

                predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
                tmp_result.append(tokenizer.index_word[predicted_id])

                if tokenizer.index_word[predicted_id] == '<end>':
                    break;

                # dec_input = tf.expand_dims([predicted_id], 0)
            results.append(tmp_result)
            total_loss += train_utils.loss_function(target[:, i], predictions)

            dec_input = tf.expand_dims(target[:, i], 1)

        loss = (total_loss / int(target.shape[1]))
        val_loss += loss

    # attention_plot = attention_plot[:len(result), :]
    return results, val_loss, attention_plot


def map_func(img_name):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')

    return img_tensor


def validation_step(tokenizer, cnn_encoder, rnn_decoder, img_tensor, target, max_length):
    loss = 0

    hidden = rnn_decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    features = cnn_encoder(img_tensor)

    for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, hidden, _ = rnn_decoder(dec_input, features, hidden)

        loss += train_utils.loss_function(target[:, i], predictions)

        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    return loss, total_loss