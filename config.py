import argparse

# Req. 2-1	Config.py 파일 생성
parser = argparse.ArgumentParser(description="Description: for image captioning model trainging")

# 캡션 데이터가 있는 파일 경로 (예시)
# path(dataset, image, check point, description, pre-trained CNN)
parser.add_argument("-d_path", '--dataset_file_path', type=str, metavar='', default='./datasets/captions.csv', help="path to caption&image file list(.csv)")
parser.add_argument("-i_path", '--img_file_path', type=str, metavar='', default='./datasets/images/', help="path to image file")
parser.add_argument("-tr_path", '--train_dataset_path', type=str, metavar='', default="./datasets/train.txt", help=".txt path of train data list")
parser.add_argument("-val_path", '--validation_dataset_path', type=str, metavar='', default="./datasets/val.txt", help=".txt path of validation data list")
parser.add_argument("-te_path", '--test_dataset_path', type=str, metavar='', default="./datasets/test.txt", help=".txt path of test data list")
parser.add_argument("-ckpt_path", '--checkpoint_path', type=str, metavar='', default="./checkpoints/ckpt", help="checkpoint path")
parser.add_argument("-desc_path", '--description_path', type=str, metavar='', default="./datasets/description.txt", help="description path")
parser.add_argument("-u_img", '--user_image', type=str, metavar='', default='', help="Set user image for test")
parser.add_argument("-cnn_path", '--cnn_model_path', type=str, metavar='', default="./checkpoints/CNN/pretrained_cnn.h5", help="cnn model(.h5) path")
parser.add_argument("-rnn_path", '--rnn_model_path', type=str, metavar='', default="./checkpoints/RNN/", help="rnn decoder model(.pb) path")

# size(dataset, mini-batch, image, buffer)
parser.add_argument("-tr_dsize", '--train_dataset_size', type=int, metavar='', default=6400, help="Set train dataset size")
parser.add_argument("-val_dsize", '--validation_dataset_size', type=int, metavar='', default=1600, help="Set validation dataset size")
parser.add_argument("-bs", '--batch_size', type=int, metavar='', default=64, help="batch size")
parser.add_argument("-i_size", '--image_size', type=int, metavar='', default=299, help="Set image size(resize)")
parser.add_argument("-buf_size", '--buffer_size', type=int, metavar='', default=1000, help="Buffer Size")

# train
parser.add_argument("-cnn", '--load_cnn_model', type=bool, metavar='', default=False, help="Load pre-trained CNN model with weights of ImageNet")
parser.add_argument("-emb_dim", '--embedding_dim', type=int, metavar='', default=256, help="Set embedding dimension")
parser.add_argument("-u", '--units', type=int, metavar='', default=512, help="units")
parser.add_argument("-top", '--top_k', type=int, metavar='', default=5000, help="top-k")
parser.add_argument("-e", '--epochs', type=int, metavar='', default=20, help="epochs")

config = parser.parse_args()