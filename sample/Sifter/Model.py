import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(window_size * embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view(inputs.shape[0], -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob


def load_batch():
    i = 0
    while i < 100:
        i += 1
        yield torch.randint(0, 100, size=(64, 4)), torch.randint(0, 100, size=(64,))


def forward_backward(model, context, target):
    context = torch.tensor(context, dtype=torch.long)
    target = torch.tensor(target, dtype=torch.long)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    """
        负对数似然损失函数，用于处理多分类问题，输入是对数化的概率值。
        对于包含N NN个样本的batch数据 D ( x , y ) D(x, y)D(x,y)，x xx 是神经网络的输出，
        进行了归一化和对数化处理。y yy是样本对应的类别标签，每个样本可能是C种类别中的一个。
    """
    loss_function = nn.NLLLoss()
    # 梯度清零
    model.zero_grad()
    # 开始前向传播
    train_predict = model(context)
    loss = loss_function(train_predict, target)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    return loss.item()


'''
TEST
'''


def test():
    epochs = 50
    model = CBOW(100, 32, 4)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 存储损失的集合
    losses = []
    """
        负对数似然损失函数，用于处理多分类问题，输入是对数化的概率值。
        对于包含N NN个样本的batch数据 D ( x , y ) D(x, y)D(x,y)，x xx 是神经网络的输出，
        进行了归一化和对数化处理。y yy是样本对应的类别标签，每个样本可能是C种类别中的一个。
    """
    loss_function = nn.NLLLoss()

    for epoch in trange(epochs):
        total_loss = 0
        count = 0
        for context, target in tqdm(load_batch()):
            count += 1
            # 梯度清零
            model.zero_grad()
            # 开始前向传播
            train_predict = model(context)
            loss = loss_function(train_predict, target)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print("============{}=========".format(total_loss / count))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), 'model._iter%d' % epoch)
