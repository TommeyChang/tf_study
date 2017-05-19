import collections
import math
import os
import random
import zipfile
import urllib
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Word_To_Vec(object):

    def __init__(self):

        self.data_url = 'http://mattmahoney.net/dc/'
        self.file_name = 'text8.zip'
        self.data_expected_bytes = 31344016
        self.vocabulary_size = 50000
        self.batch_size = 128
        self.embedding_size = 128
        self.skip_window = 1
        self.num_skip = 1
        self.valid_size = 16
        self.valid_window = 100
        self.num_sampled = 64
        self.device_name = '/cpu:0'

        self.plot_data_num = 500

    def data_download(self):
        filename = self.file_name
        data_url = self.data_url
        expected_bytes = self.data_expected_bytes
        if not os.path.exists(filename):
            filename, _ = urllib.urlretrieve(data_url + filename, filename)
        data_stat_info = os.stat(filename)
        if data_stat_info.st_size == expected_bytes:
            print 'Found and verified %s.' % filename
        else:
            raise Exception('Failed to verify %s. You may try again or get it with a browser from website %s' %
                            (filename, data_url))
        return filename

    def read_data(self, filename='text8.zip'):
        with zipfile.ZipFile(filename) as zf:
            data = tf.compat.as_str(zf.read(zf.namelist()[0])).split()
        return data

    def build_data_set(self, words, vocabulary_size=None):
        if vocabulary_size is None:
            vocabulary_size = self.vocabulary_size
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
        word_dic = {}
        for word, _ in count:
            word_dic[word] = len(word_dic)
        data_list = []
        unk_count = 0
        for word in words:
            index = word_dic.get(word, 0)
            if index == 0:
                unk_count += 1
            data_list.append(index)
        count[0][1] = unk_count
        reverse_word_dic = dict(zip(word_dic.values(), word_dic.keys()))
        print 'Most common words (+UNK): ', count[:5]
        print 'Sample data: ', data_list[:10], [reverse_word_dic[i] for i in data_list[:10]]
        return data_list, count, word_dic, reverse_word_dic

    def generate_data_batch(self, data, batch_size=None, num_skips=None, skip_window=None):
        batch_size = self.batch_size
        num_skips = self.num_skip
        skip_window = self.skip_window
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        data_batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        label_batch = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        data_index = 0
        while True:
            data_buffer = collections.deque(maxlen=span)
            for _ in xrange(span):
                data_buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
            for i in range(batch_size // num_skips):
                target = skip_window
                target_to_avoid = [skip_window]
                for j in xrange(num_skips):
                    while target in target_to_avoid:
                        target = random.randint(0, span - 1)
                    target_to_avoid.append(target)
                    data_batch[i * num_skips + j] = data_buffer[skip_window]
                    label_batch[i * num_skips + j, 0] = data_buffer[target]
                data_buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
            yield data_batch, label_batch

    def compute_graph(self, device_name='/cpu:0'):

        embedding_size = self.embedding_size
        num_sampled = self.num_sampled

        batch_size = self.batch_size

        valid_size = self.valid_size
        valid_window = self.valid_window
        valid_sample = np.random.choice(valid_window, valid_size, replace=True)

        main_graph = tf.Graph()
        with main_graph.as_default():

            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_data_set = tf.constant(valid_sample)

            with tf.device(device_name):
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, embedding_size], -1., 1.)
                )
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, embedding_size],
                                        stddev=1. / math.sqrt(embedding_size))
                )
                nce_bias = tf.Variable(tf.zeros([self.vocabulary_size]))

            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                 biases=nce_bias,
                                                 labels=train_labels,
                                                 inputs=embed,
                                                 num_sampled=num_sampled,
                                                 num_classes=self.vocabulary_size))
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_data_set)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True
            )

        model = (train_inputs, train_labels, optimizer, main_graph, loss, normalized_embeddings)
        metric = (valid_sample, similarity)

        return model, metric

    def model_train(self, data, model, metric, r_dic, epoches=100000):

        batch_size = self.batch_size
        valid_size = self.valid_size

        train_input = model[0]
        train_label = model[1]
        train_op = model[2]
        train_graph = model[3]
        loss = model[4]
        final_embdeeing = model[5]

        valid_sample = metric[0]
        similarity = metric[1]

        data_iter = self.generate_data_batch(data)

        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            print 'Initialized'

            average_loss = 0

            for epoch in xrange(epoches):
                data_batch, label_batch = data_iter.next()

                _, loss_val = sess.run([train_op, loss],
                                       feed_dict={train_input: data_batch, train_label: label_batch})

                average_loss += loss_val

                if epoch % 2000 == 0:
                    if epoch > 0:
                        average_loss /= 2000
                    print 'Average at epoch %d, average loss: %f.' % (epoch, average_loss)
                    average_loss = 0

                if epoch % 10000 == 9999:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = r_dic[valid_sample[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        close_word = ','.join(r_dic[nearest[:top_k]])
                        print '%s %s' % (log_str, close_word)
            final_result = final_embdeeing.eval()

        return final_result

    def visual(self, data, r_dic, image_name='tsne.png'):
        plot_only = self.plot_data_num
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(data[:plot_only, :])
        labels = [r_dic[i] for i in xrange(plot_only)]
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings.'
        plt.figure(figsize=(18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                         ha='right', va='bottom')
        plt.savefig(image_name)





def main_function():
    word2vec = Word_To_Vec()
    file_name = word2vec.data_download()
    words = word2vec.read_data(filename=file_name)
    data, data_count, data_dic, data_reverse_dic = word2vec.build_data_set(words)
    model, metric = word2vec.compute_graph()
    word_embeddings = word2vec.model_train(data, model, metric, data_reverse_dic)
    word2vec.visual(word_embeddings, data_reverse_dic)


if __name__ == '__main__':
    main_function()

