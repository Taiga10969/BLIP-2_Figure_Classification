import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def setup_torch_seed(seed=1):
    # pytorchに関連する乱数シードの固定を行う．
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 乱数シードを固定
setup_torch_seed()


def plot_learning_rate(optimizer, lr_scheduler, num_epochs):
    # 学習率を記録するためのリスト
    lr_values = []

    # モデルのトレーニングループの中で
    for epoch in range(num_epochs):
        # 学習率を取得
        lr = optimizer.param_groups[0]['lr']
        # 学習率を記録
        lr_values.append(lr)

        # 以下、トレーニングステップなどのコード

        # 学習率の更新
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 学習率の曲線をプロット
    plt.plot(lr_values)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig('learning_rate_scheduler.svg')
    plt.savefig('learning_rate_scheduler.png')
    plt.close()


# シンプルな線形モデルを定義します
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # パラメータとして1つの重みと1つのバイアスを持つ線形層を定義します
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # 入力を線形層に渡して2倍にします
        return 2 * self.linear(x)
