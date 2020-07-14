import os
from config import config
from utils import utils, train_utils, test_utils

from data import preprocess
from models import encoder, decoder
from models.encoder import CNN_Encoder
from models.decoder import RNN_Decoder

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime


for tc in range(10):
    # GPU 자동 할당
    tf.debugging.set_log_device_placement(True)

    # config 저장
    utils.save_config("sunwoo_test", config)

    # 이미지 경로 및 캡션 불러오기
    # img_paths, captions = preprocess.get_path_caption(config.dataset_file_path)
    # img_paths = [config.img_file_path + path for path in img_paths]

    # 전체 데이터셋을 분리해 저장하기
    # img_name_train, img_name_val, token_train, token_val = preprocess.dataset_split_save(img_paths, captions)

    # 저장된 데이터셋 불러오기
    tr_img_paths, tr_captions = preprocess.get_data_file(config.train_dataset_path)
    val_img_paths, val_captions = preprocess.get_data_file(config.validation_dataset_path)


    shuffle(tr_img_paths, tr_captions)
    tr_img_paths, tr_captions = tr_img_paths[:10000], tr_captions[:10000]
    shuffle(val_img_paths, val_captions)
    val_img_paths, val_captions = val_img_paths[:2000], val_captions[:2000]

    # 이미지 데이터 로딩
    # image_dataset = tf.data.Dataset.from_tensor_slices(sorted(set(img_paths)))
    # image_dataset = image_dataset.map(preprocess.loading_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(config.batch_size)

    # Req. 3-2. Image Data Augmentation by Daseul
    # augmented_dataset = preprocess.image_augmentation(dataset).batch(config.batch_size)
    # utils.visualize_dataset(augmented_dataset)

    # Req. 3-2. Image Data Augmentation 생성 for visualization by Minsu
    # img, label = preprocess.augment(origin_imgs, imgs, captions)

    # # Req. 4-1. Pre-trained 모델로 이미지 특성 추출 by Daseul
    # for img, path in tqdm(image_dataset):
    #     batch_features = encoder.extract_feature_from_image(img)
    #
    #     for bf, p in zip(batch_features, path):
    #         path_of_feature = p.numpy().decode("utf-8")
    #         np.save(path_of_feature, bf.numpy())

    # Req. 2-1. 텍스트 토큰화
    tokenizer, tr_tokens, val_tokens = preprocess.text_tokenizer(tr_captions, val_captions)

    # Req. 2-2. tokenizer 저장 및 불러오기
    preprocess.save_load(tokenizer)

    # # Req. 3-1. tf.data.Dataset 생성
    tr_dataset = preprocess.convert_to_dataset(tr_img_paths, tr_tokens)
    val_dataset = preprocess.convert_to_dataset(val_img_paths, val_tokens)

    # Use map to load the numpy files in parallel
    tr_dataset = tr_dataset.map(lambda item1, item2: tf.numpy_function(
              preprocess.map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(
              preprocess.map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    tr_dataset = tr_dataset.shuffle(config.buffer_size).batch(config.batch_size)
    tr_dataset = tr_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.shuffle(config.buffer_size).batch(config.batch_size)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # # Req. 5 word_embedding
    # # preprocess.word_embedding(tokenizer, captions)
    #
    # encoder & decoder 생성
    cnn_encoder = CNN_Encoder(config.embedding_dim)
    rnn_decoder = RNN_Decoder(config.embedding_dim, config.units, config.top_k+1)

    # checkpoint
    ckpt_path = config.checkpoint_path
    ckpt = tf.train.Checkpoint(encoder=cnn_encoder,
                               decoder=rnn_decoder,
                               optimizer=train_utils.optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=10)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)
    # END: checkpoint


    EPOCHS = config.epochs
    loss_plot = []

    # for tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tr_log_dir = "./logs/" + current_time + "/train"
    val_log_dir = "./logs/" + current_time + "/val"

    tr_summary_writer = tf.summary.create_file_writer(tr_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # tr_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tr_summary_writer, histogram_freq=1)
    # val_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=val_summary_writer, histogram_freq=1)
    vloss = 5
    count = 0
    for epoch in range(0, EPOCHS):
        # train
        start = time.time()

        total_tr_loss = 0
        for (batch, (img_tensor, target)) in enumerate(tr_dataset):
            tr_batch_loss, tr_loss = train_utils.train_step(tokenizer, cnn_encoder, rnn_decoder, img_tensor, target)
            total_tr_loss += tr_loss

            if batch % 100 == 0:
                print('train: Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, batch, tr_batch_loss.numpy() / int(target.shape[1])))

        # save log
        total_tr_loss /= config.train_dataset_size
        with tr_summary_writer.as_default():
            tf.summary.scalar('loss', total_tr_loss, step=epoch)

        print('train: Epoch {} Total Loss {} Time {}'.format(epoch+1, total_tr_loss, (time.time() - start)))
        # END: train

        start = time.time()

        max_length = 0
        for val_token in val_tokens:
            max_length = max(max_length, len(val_token))

        total_val_loss = 0
        for (batch, (img_tensor, target)) in enumerate(val_dataset):
            val_batch_loss, val_loss = test_utils.validation_step(tokenizer, cnn_encoder, rnn_decoder, img_tensor, target,
                                                                  max_length)
            total_val_loss += val_loss

            if batch % 10 == 0:
                print('validation: Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch,
                                                                         val_batch_loss.numpy() / int(target.shape[1])))

        if (total_val_loss / config.validation_dataset_size) > vloss:
            count += 1
            if count == 3:
                break
        elif (total_val_loss / config.validation_dataset_size) < vloss:
            vloss = total_val_loss / config.validation_dataset_size
            count = 0
            ckpt_manager.save()
            ckpt.restore(ckpt_manager.latest_checkpoint)

        # save log
        total_val_loss /= config.validation_dataset_size
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', total_val_loss, step=epoch)

        print('validation: Epoch {} Total Loss {} Time {}'.format(epoch + 1, total_val_loss, (time.time() - start)))
        # END: evaluate

        if epoch % 5 == 0:
            ckpt_manager.save()
            ckpt.restore(ckpt_manager.latest_checkpoint)
            # evaluate


if ckpt_manager.latest_checkpoint:
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

#tf.saved_model.save(rnn_decoder, config.rnn_model_path)

#여기서 한번
input_img = config.user_image
if input_img == '':
    input_img = ".\\datasets\\images\\209274114.jpg";


ml = len(tr_tokens[0])

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

for img_tensor in image_dataset:
    attention_plot = np.zeros((50, 64))
    features = cnn_encoder(img_tensor[0])

    hidden = rnn_decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    res_caption = []
    for i in range(ml):
        predictions, hidden, attention_weigth = rnn_decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weigth, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        if tokenizer.index_word[predicted_id] == '<end>':
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

