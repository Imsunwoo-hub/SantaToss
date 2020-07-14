import argparse

parser = argparse.ArgumentParser(description="Description: for image captioning model trainging")

parser.add_argument('--download_dataset', type=bool, metavar='', default=True, help="")
parser.add_argument('--dataset_path', type=str, metavar='', default='./datasets/train2014/', help="")
parser.add_argument('--dataset_size', type=int, metavar='', default=100, help="")
parser.add_argument('--create_extractor', type=bool, metavar='', default=True, help="")
parser.add_argument('--extractor_path', type=str, metavar='', default='./models/extractor.h5', help="")
parser.add_argument('--img_size', type=int, metavar='', default=299, help="")
parser.add_argument('--batch_size', type=int, metavar='', default=64, help="")
parser.add_argument('--buffer_size', type=int, metavar='', default=1000, help="")
parser.add_argument('--embedding_dim', type=int, metavar='', default=256, help="")
parser.add_argument('--units', type=int, metavar='', default=512, help="")
parser.add_argument('--features_shape', type=int, metavar='', default=2048, help="")
parser.add_argument('--attention_features_shape', type=int, metavar='', default=64, help="")
parser.add_argument('--checkpoints_path', type=str, metavar='', default='./checkpoints/train', help="")
parser.add_argument('--top_k', type=int, metavar='', default=5000, help="")
parser.add_argument('--user_image', type=str, metavar='', default='', help="")
parser.add_argument('--epoch', type=int, metavar='', default=1, help="")

config = parser.parse_args()