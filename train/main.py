from utility.config import config
from utility import train_utils, preprocessing
from models import cnn
from models.cnn import CNN_Encoder
from models.rnn import RNN_Decoder, BahdanauAttention

import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import time
import datetime


if config.download_dataset:
    preprocessing.download_coco_dataset()

img_names, captions = preprocessing.load_dataset()

if config.create_extractor:
    image_features_extract_model = cnn.create_image_features_extract_model()
else:
    image_features_extract_model = cnn.load_image_features_extract_model()

preprocessing.extract_image_features(img_names, image_features_extract_model)

tokenizer, cap_vector, max_length = preprocessing.tokenizer_to_captions(captions)

img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_names, cap_vector, test_size=0.2)

tr_dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
tr_dataset = tr_dataset.map(lambda item1, item2: tf.numpy_function(
    preprocessing.map_func, [item1, item2], [tf.float32, tf.int32]),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
val_dataset = val_dataset.map(lambda item1, item2: tf.numpy_function(
    preprocessing.map_func, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

tr_dataset = tr_dataset.shuffle(config.buffer_size).batch(config.batch_size)
tr_dataset = tr_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.shuffle(config.buffer_size).batch(config.batch_size)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

encoder = CNN_Encoder(config.embedding_dim)
decoder = RNN_Decoder(config.embedding_dim, config.units, config.top_k+1)
attention = BahdanauAttention(config.units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, config.checkpoints_path, max_to_keep=None)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)

tr_num_steps = len(img_name_train) // config.batch_size
val_num_steps = len(img_name_val) // config.batch_size

tr_loss_plot = []
val_loss_plot = []

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tr_log_dir = "./logs/" + current_time + "/train"
val_log_dir = "./logs/" + current_time + "/val"

tr_summary_writer = tf.summary.create_file_writer(tr_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

for epoch in range(start_epoch, config.epoch):
    start = time.time()
    tr_total_loss = 0

    for(tr_batch, (img_tensor, target)) in enumerate(tr_dataset):
        tr_batch_loss, t_loss = train_utils.train_step(img_tensor, target, decoder, encoder, tokenizer, optimizer, loss_object, attention)
        tr_total_loss += t_loss

        if tr_batch % 100 == 0:
            print('train: Epoch {} Batch {} Loss {:.4f}'.format(epoch+1, tr_batch, tr_batch_loss.numpy()/int(target.shape[1])))

    with tr_summary_writer.as_default():
        tf.summary.scalar('loss', tr_total_loss/tr_num_steps, step=epoch)
    tr_loss_plot.append(tr_total_loss/tr_num_steps)

    ckpt_manager.save()
    tf.saved_model.save(decoder, "./models/decoder/" + str(epoch) + "/")

    print('train: Epoch {} Loss {:.6f}'.format(epoch+1, tr_total_loss/tr_num_steps))
    print('train: Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # validation
    start = time.time()
    val_total_loss = 0

    for (val_batch, (img_tensor, target)) in enumerate(val_dataset):
        val_batch_loss, val_loss = train_utils.validation_step(img_tensor, target, decoder, encoder, tokenizer, optimizer, loss_object, attention)
        val_total_loss += val_loss

        if val_batch % 10 == 0:
            print('validation: Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, val_batch, val_batch_loss.numpy() / int(target.shape[1])))

    with val_summary_writer.as_default():
        tf.summary.scalar('loss', val_total_loss / val_num_steps, step=epoch)
    val_loss_plot.append(val_total_loss/val_num_steps)

    print('validation: Epoch {} Loss {:.6f}'.format(epoch + 1, val_total_loss / val_num_steps))
    print('validation: Time taken for 1 epoch {} sec\n'.format(time.time() - start))


plt.plot(tr_loss_plot)
plt.plot(val_loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train-Validation Loss')
plt.savefig('loss.png')