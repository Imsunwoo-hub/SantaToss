from utility.config import config

import tensorflow as tf


def create_image_features_extract_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    image_features_extract_model.save(config.extractor_path) # HDF5
    tf.saved_model.save(image_features_extract_model, "./models/extractor/1/") # SavedModel

    return image_features_extract_model


def load_image_features_extract_model():
    image_features_extract_model = tf.keras.models.load_model(config.extractor_path)

    return image_features_extract_model


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    @tf.function(input_signature=[tf.TensorSpec([None, 64, 2048], tf.float32)])
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x