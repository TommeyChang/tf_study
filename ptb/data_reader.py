import collections
import os
import urllib.request
import tarfile

import tensorflow as tf


class DataReader(object):

    def __init__(self, batch_size=0, num_step=0, data_path='./data_set',
                 data_name='simple-examples.tgz',
                 data_url='http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'):
        self.filename = data_name
        self.batch_size = batch_size
        self.num_step = num_step
        self.data_path = data_path
        self.data_url = data_url

    def _read_words(self, file_name):
        with tf.gfile.GFile(file_name, 'r') as f:
            return f.read().replace('\n', 'eos').split()

    def _build_vocab(self, file_name):
        data = self._read_words(file_name)

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        return word_to_id

    def _file_to_word_ids(self, file_name, word_to_id):
        data = self._read_words(file_name)
        return [word_to_id[word] for word in data if word in word_to_id]

    def ptb_raw_data(self):
        data_name = self.data_path + '/' + self.filename

        try:
            os.mkdir(self.data_path)
        except IOError:
            pass

        # if the data is not in the local host, fetch it from remote host
        if not os.path.exists(data_name):
            print('Fetch the data from remote host...')
            _ = urllib.request.urlretrieve(self.data_url, data_name)
            print('Success fetch the data, the data is %s' % data_name)

        with tarfile.open(data_name) as tar_f:
            tar_f.extractall(self.data_path)
            for root, _, files in os.walk(self.data_path):
                for file in files:
                    if 'ptb.train.txt' in file:
                        train_path = os.path.join(root, file)
                    if 'ptb.valid.txt' in file:
                        valid_path = os.path.join(root, file)
                    if 'ptb.test.txt' in file:
                        test_path = os.path.join(root, file)

            word_to_id = self._build_vocab(train_path)
            train_data = self._file_to_word_ids(train_path, word_to_id)
            valid_data = self._file_to_word_ids(valid_path, word_to_id)
            test_data = self._file_to_word_ids(test_path, word_to_id)

            return train_data, valid_data, test_data

    def ptb_producer(self, raw_data, name=None):
        with tf.name_scope(name, 'PTBProducer', [raw_data, self.batch_size, self.num_step]):
            tensor_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.int32)

            data_len = tf.size(tensor_data)
            batch_len = data_len // self.batch_size
            data = tf.reshape(tensor=tensor_data[0:self.batch_size * batch_len],
                              shape=[self.batch_size, batch_len])
            epoch_size = (batch_len - 1) // self.num_step
            assertion = tf.assert_positive(
                epoch_size,
                message='Epoch_size == 0, decrease batch_size or num_steps.'
            )
            with tf.control_dependencies([assertion]):
                epoch_size = tf.identity(epoch_size, name='epoch_size')

            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            x = tf.strided_slice(data, [0, i * self.num_step],
                                 [self.batch_size, (i + 1) * self.num_step])
            x.set_shape([self.batch_size, self.num_step])
            y = tf.strided_slice(data, [0, i * self.num_step],
                                 [self.batch_size, (i + 1) * self.num_step])
            y.set_shape([self.batch_size, self.num_step])
            return x, y


if __name__ == '__main__':
    data_reader = DataReader()
    data_reader.ptb_raw_data()








