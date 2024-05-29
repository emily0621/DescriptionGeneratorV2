import tensorflow as tf
from keras import Model
from keras.layers import Dense, Embedding, LSTM
from config import EMBEDDING_DIM, BATCH_SIZE, UNITS, VOCABULARY_SIZE, IMAGE_SHAPE, END_TOKEN, START_TOKEN, ModelType
from keras.applications import InceptionResNetV2, EfficientNetB7
from keras.applications import inception_resnet_v2, efficientnet_v2
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice


class BahdanauAttention(Model):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        score = self.V(attention_hidden_layer)

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Encoder(Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.fc = Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class Decoder(Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = Dense(self.units)
        self.fc2 = Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, memory_state, carry_state = self.lstm(x)

        x = self.fc1(output)

        x = tf.reshape(x, (-1, x.shape[2]))

        x = self.fc2(x)

        return x, memory_state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class FeatureExtractor:

    def __init__(self):
        super().__init__()
        self.model_type = ModelType.INCEPTION_RES_NET_V2
        if self.model_type == ModelType.INCEPTION_RES_NET_V2:
            self.model = InceptionResNetV2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
        if self.model_type == ModelType.EFFICIENT_NET_B7:
            self.model = EfficientNetB7(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')

    def preprocess_input(self, image):
        if self.model_type == ModelType.INCEPTION_RES_NET_V2:
            return inception_resnet_v2.preprocess_input(image)
        if self.model_type == ModelType.EFFICIENT_NET_B7:
            return efficientnet_v2.preprocess_input(image)

    def preprocess_image(self, image):
        img = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        img = tf.image.resize(img, (224, 244))
        img = self.preprocess_input(img)
        return tf.expand_dims(img, axis=0)

    def get_features(self, image):
        f = self.model.predict(self.preprocess_image(image), verbose=False)
        return tf.reshape(f, (f.shape[0], -1, f.shape[3]))[0]


class AppModel:

    def __init__(self):
        super().__init__()

        self.encoder = Encoder(EMBEDDING_DIM)
        self.encoder(tf.zeros((BATCH_SIZE, 25, 1536)))
        self.encoder.load_weights('Encoder.weights.h5')

        self.decoder = Decoder(EMBEDDING_DIM, UNITS, VOCABULARY_SIZE)
        self.decoder(tf.zeros((BATCH_SIZE, 1)), tf.zeros((BATCH_SIZE, 25, EMBEDDING_DIM)), tf.zeros((BATCH_SIZE, UNITS)))
        self.decoder.load_weights('Decoder.weights.h5')

        with open('Tokenizer.pickle', 'rb') as handle:
            self.tokenizer: Tokenizer = pickle.load(handle)

        self.END_TOKEN_INDEX = self.tokenizer.word_index.get(END_TOKEN)
        self.START_TOKEN_INDEX = self.tokenizer.word_index.get(START_TOKEN)

        self.feature_extractor = FeatureExtractor()

    def decode_caption(self, caption):
        sentence = ''
        for i in caption:
            if i == self.END_TOKEN_INDEX:
                return sentence
            sentence += self.tokenizer.index_word.get(i) + ' '
        return sentence

    def predict(self, image):
        features = self.feature_extractor.get_features(image).numpy()

        img = np.expand_dims(features, axis=0)

        hidden = self.decoder.reset_state(batch_size=BATCH_SIZE)
        dec_input = tf.expand_dims([self.START_TOKEN_INDEX] * BATCH_SIZE, 1)

        features = self.encoder(img)
        decoded_predictions = tf.zeros((BATCH_SIZE, 0), dtype=tf.int64)
        for i in range(1, 100):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)

            indexes = tf.argmax(predictions, axis=1).numpy()

            if indexes[0] == self.END_TOKEN_INDEX:
                break

            dec_input = tf.expand_dims(indexes, 1)
            decoded_predictions = tf.concat([decoded_predictions, dec_input], axis=1)

        return self.decode_caption(decoded_predictions.numpy()[0])

    @staticmethod
    def evaluate(predicted: str, expected: list[str]):
        result = dict()
        result['Bleu'] = Bleu(min(4, len(predicted.split(' ')))).compute_score({1: expected}, {1: [predicted]})[0]
        result['Rouge'] = Rouge().compute_score({1: expected}, {1: [predicted]})[0]
        result['Meteor'] = Meteor().compute_score({1: expected}, {1: [predicted]})[0]
        result['Spice'] = Spice().compute_score({1: expected}, {1: [predicted]})[0]
        return result
