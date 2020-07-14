import tensorflow as tf
from config import config


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        return x


# Req. 4-1. Pre-trained 모델로 이미지 특성 추출 by Daseul
def create_cnn():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_feature_extract_model = tf.keras.Model(new_input, hidden_layer)

    # save pre-trained model
    image_feature_extract_model.save(config.cnn_model_path)


def load_cnn():
    model = tf.keras.models.load_model(config.cnn_model_path)
    # model.summary()

    return model


def load_image(image, tokens):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, tokens


def extract_feature_from_image(img):
    # 이미지의 feature를 추출할 pre-trained model 생성 및 저장
    if config.load_cnn_model:
        create_cnn()

    # 저장된 pre-trained CNN 모델 로딩
    model = load_cnn()

    features = model(img)
    features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))

    return features