import os
from enum import Enum

DATA_FOLDER = os.path.abspath('../../data')
DATASET_DIRECTORY = os.path.join(DATA_FOLDER, 'dataset')
TRAINING_DIRECTORY = os.path.join(DATA_FOLDER, 'training')
IMAGES_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'images')
FEATURES_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'features')
FILTERED_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'filtered_dataset')
VALIDATED_DATASET_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'validated_dataset')
DATASET_PATH = os.path.join(DATASET_DIRECTORY, 'dataset.tsv')
FILTER_RULES_PATH = os.path.join(DATASET_DIRECTORY, 'filter_rules.json')
SPLIT_DIRECTORY = os.path.join(DATASET_DIRECTORY, 'split')

CHUNK_SIZE = 100_000
IMAGE_SHAPE = (224, 244, 3)

VOCABULARY_SIZE = 7500
START_TOKEN = '<start>'
END_TOKEN = '<end>'
PADDING_TOKEN = '<pad>'

TRAIN_SIZE = 0.8
TEST_SIZE = 0.1

BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 128
EPOCHS = 100


class ModelType(Enum):
    INCEPTION_RES_NET_V2 = 'InceptionResNetV2'
    EFFICIENT_NET_B7 = 'EfficientNetB7'
