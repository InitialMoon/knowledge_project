import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        word_vec_dim,
        word_embedding_weight,
        pos_vec_dim,
        pos_size,
        hidden_dim,
        out_dim,
        max_len,
    ):
        super().__init__()
        self.word_vec_dim = word_vec_dim
        self.pos_vec_dim = pos_vec_dim
        self.max_len = max_len
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_weight)
        self.dis1_embedding = nn.Embedding(pos_size, pos_vec_dim)
        self.dis2_embedding = nn.Embedding(pos_size, pos_vec_dim)
        self.conv = nn.Conv2d(
            in_channels=max_len,
            out_channels=max_len,
            kernel_size=(5, 1),
            dtype=torch.double,
        )
        self.pool = nn.MaxPool1d(word_vec_dim)
        self.linear_sentence = nn.Linear(self.max_len, hidden_dim, dtype=torch.float64)
        self.linear_full = nn.Linear(
            hidden_dim + 6 * self.word_vec_dim, out_dim, dtype=torch.float64
        )

    def get_lexical_feat(self, lexical_data):
        lexical_feat = self.word_embedding(lexical_data).view(-1, 6 * self.word_vec_dim)
        return lexical_feat

    def get_sentence_feat(self, word_data, dis1, dis2, mask):
        word_feat = self.word_embedding(word_data)
        dis1_feat = self.dis1_embedding(dis1).unsqueeze(2)
        dis2_feat = self.dis2_embedding(dis2).unsqueeze(2)
        inter_res = self.conv(
            torch.cat([word_feat, dis1_feat, dis2_feat], dim=2)
        ).squeeze(2)
        inter_res.masked_fill_(mask, -1e50)
        inter_res = self.pool(inter_res).squeeze(2)
        sentence_feat = torch.sigmoid(self.linear_sentence(inter_res))
        return sentence_feat

    def forward(self, lexical_data, word_data, dis1, dis2, mask):
        lexical_feat = self.get_lexical_feat(lexical_data)
        sentence_feat = self.get_sentence_feat(word_data, dis1, dis2, mask)
        full_feat = torch.cat([lexical_feat, sentence_feat], dim=-1)
        y = self.linear_full(full_feat)
        return y


@torch.no_grad()
def evaluation(model, loss_fn, data_iter, device):
    model.to(device)
    model.eval()
    loss = 0
    tp = 0
    fn = 0
    fp = 0
    total = 0
    for lexical_data, word_feat, dis1, dis2, mask, y in data_iter:
        lexical_data = lexical_data.to(device)
        word_feat = word_feat.to(device)
        dis1 = dis1.to(device)
        dis2 = dis2.to(device)
        mask = mask.to(device)
        y = y.to(device)
        y_hat = model(lexical_data, word_feat, dis1, dis2, mask)
        loss += loss_fn(y_hat, y)
        y //= 2
        y_hat = torch.argmax(y_hat, dim=1).squeeze()
        y_hat //= 2
        tp += (y == y_hat).sum()
        total += y.shape[0]
    acc = tp / total
    loss /= total
    return acc, loss


def train(model, optimizer, loss_fn, train_iter, test_iter, epochs, device):
    model.to(device)
    train_f1_arr = []
    train_loss_arr = []
    val_f1_arr = []
    val_loss_arr = []
    for i in range(epochs):
        model.train()
        for lexical_data, word_feat, dis1, dis2, mask, y in train_iter:
            optimizer.zero_grad()
            lexical_data = lexical_data.to(device)
            word_feat = word_feat.to(device)
            dis1 = dis1.to(device)
            dis2 = dis2.to(device)
            mask = mask.to(device)
            y = y.to(device)
            y_hat = model(lexical_data, word_feat, dis1, dis2, mask)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

        train_f1, train_loss = evaluation(model, loss_fn, train_iter, device)
        val_f1, val_loss = evaluation(model, loss_fn, test_iter, device)
        print(
            f"epoch {i + 1}/ {epochs}, train_acc: {train_f1:.4f}, train_loss: {train_loss:.4f}, val_acc: {val_f1:.4f}, val_loss: {val_loss:.4f}"
        )
        train_f1_arr.append(train_f1)
        train_loss_arr.append(train_loss)
        val_f1_arr.append(val_f1)
        val_loss_arr.append(val_loss)

    return (train_f1_arr, train_loss_arr, val_f1_arr, val_loss_arr)
