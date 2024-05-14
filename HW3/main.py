import utils.load_data as load
import model.cnn as exec
import torch
from model.cnn import CNN
from torch import nn
from torch import optim

# from model.cnn import *

batch_size = 128
num_labels = 19
word_vec_dim = 50
pos_vec_dim = 50
vocal_size = 24490
pos_size = 200
hidden_dim = 20
vocal_size = 24490
epochs = 40
max_len = 93

if __name__ == "__main__":
    train_iter, test_iter, word_embedding_weight = load.init(
        batch_size, vocal_size, max_len
    )
    model = CNN(
        word_vec_dim,
        word_embedding_weight,
        pos_vec_dim,
        pos_size,
        hidden_dim,
        num_labels,
        max_len,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exec.train(
        model, optimizer, loss_fn, train_iter, test_iter, epochs, torch.device("cuda")
    )
