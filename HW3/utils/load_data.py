import pandas as pd
import numpy as np
import torch


def construct_iter(dataset, batch_size):
    dataset = torch.utils.data.TensorDataset(*dataset)
    return torch.utils.data.DataLoader(dataset, batch_size, True)


def get_lexical_data(
    e1_start_index, e1_end_index, e2_start_index, e2_end_index, words, vocal_size
):
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for i in range(e1_start_index.shape[0]):
        x = e1_start_index[i][0]
        y = e1_end_index[i][0]
        l1.append([words[i][x]])
        l3_tmp = [words[i][x - 1]]
        if words[i][y + 1] == -1:
            l3_tmp.append(vocal_size)
        else:
            l3_tmp.append(words[i][y + 1])
        l3.append(l3_tmp)
    for i in range(e2_start_index.shape[0]):
        x = e2_start_index[i][0]
        y = e2_end_index[i][0]
        l2.append([words[i][x]])
        l4_tmp = [words[i][x - 1]]
        if words[i][y + 1] == -1:
            l4_tmp.append(vocal_size)
        else:
            l4_tmp.append(words[i][y + 1])
        l4.append(l4_tmp)
    l1 = torch.tensor(l1)
    l2 = torch.tensor(l2)
    l3 = torch.tensor(l3)
    l4 = torch.tensor(l4)
    lexical_data = torch.concat([l1, l2, l3, l4], dim=1)
    return lexical_data


def get_word_feat(words, max_len):
    words_feats = []
    for i in range(words.shape[0]):
        words_feat = []
        for j in range(1, max_len + 1):
            words_feat.append([words[i][j - 1], words[i][j], words[i][j + 1]])
        words_feats.append(words_feat)
    return torch.tensor(words_feats)


def get_dis_feat(index, max_len):
    feats = []
    for i in range(index.shape[0]):
        feat = []
        pos = index[i][0]
        for j in range(max_len):
            feat.append(j - pos + max_len)
        feats.append(feat)
    feats = torch.tensor(feats)
    return feats


def get_iter(data_csv, batch_size, vocal_size, max_len):
    labels = torch.tensor(data_csv.iloc[:, [1]].values.reshape(-1, 1)).squeeze()
    len = data_csv.iloc[:, [2]].values.reshape(-1, 1)
    e1_start_index = data_csv.iloc[:, [3]].values.reshape(-1, 1) + 1
    e1_end_index = data_csv.iloc[:, [4]].values.reshape(-1, 1) + 1
    e2_start_index = data_csv.iloc[:, [5]].values.reshape(-1, 1) + 1
    e2_end_index = data_csv.iloc[:, [6]].values.reshape(-1, 1) + 1
    words = data_csv.iloc[:, 7:].values.reshape(-1, max_len)
    # add padding at the begin and the end of a sentence
    words_padding = []
    for i in range(words.shape[0]):
        words_padding.append([vocal_size])
    words_padding = np.array(words_padding)
    words = np.concatenate((words_padding, words, words_padding), axis=1)
    mask = torch.tensor(data_csv.iloc[:, 7:].values.reshape(-1, max_len))
    mask = mask == vocal_size
    mask = mask.unsqueeze(2)
    lexical_data = get_lexical_data(
        e1_start_index, e1_end_index, e2_start_index, e2_end_index, words, vocal_size
    )
    word_feat = get_word_feat(words, max_len)
    dis1 = get_dis_feat(e1_start_index, max_len)
    dis2 = get_dis_feat(e2_start_index, max_len)
    return construct_iter(
        (lexical_data, word_feat, dis1, dis2, mask, labels), batch_size
    )


# construct data iter and load pretrained word embedding
def init(batch_size, vocal_size, max_len):
    train_csv = pd.read_csv("data/train.csv", header=None)
    test_csv = pd.read_csv("data/test.csv", header=None)
    word_vec_csv = pd.read_csv("data/word_vec.csv", header=None)
    train_iter = get_iter(train_csv, batch_size, vocal_size, max_len)
    test_iter = get_iter(test_csv, batch_size, vocal_size, max_len)
    word_embedding_weight = torch.tensor(word_vec_csv.values)
    word_embedding_weight_padding = torch.zeros(1, 50)
    word_embedding_weight = torch.concat(
        [word_embedding_weight, word_embedding_weight_padding]
    )
    return train_iter, test_iter, word_embedding_weight
