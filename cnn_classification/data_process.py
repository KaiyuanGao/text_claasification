import numpy as np
import re


def clean_str(string):
    """
    Clean the text data using the same code as the original paper
    from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string: input string to process
    :return: a sentence with lower representation and delete spaces
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Load positive and negative sentences from the raw data files
    :param positive_data_file: pos
    :param negative_data_file: neg
    :return: split sentences and labels
    """
    # load file
    positive_examples = list(open(positive_data_file, 'r', encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, 'r', encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # clean sents
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # generate labels
    positive_labels = [[1, 0] for _ in positive_examples]
    negative_labels = [[0, 1] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """

    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1) / batch_size) + 1
    for epoch in range(num_epochs):
        # shuffle data
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[start_index:end_index]