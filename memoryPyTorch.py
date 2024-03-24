#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- February 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  ###################################################################

from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

print(torch.__version__)


# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )

    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        # np.hstack((h, m))将输入h和输出m在水平方向上进行堆叠，创建一个新的记忆条目。
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        # 首先，调用self.remember(h, m)，方法将输入h和输出m存储到记忆中，即调用了remember方法，将记忆条目添加到内存中。
        self.remember(h, m)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        # 只要记忆计数器是训练间隔的倍数，就会进行训练。
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        # 这段代码首先根据内存计数器和内存大小的比较，选择要从内存中采样的记忆批次。
        # 如果内存中的记忆条目数量self.memory_counter超过了内存大小self.memory_size，
        # 则从整个内存中随机选择大小为self.batch_size的样本索引，存储在sample_index中。
        # 否则，从已存储的记忆条目中随机选择大小为self.batch_size的样本索引，也存储在sample_index中。
        # 然后，通过使用sample_index对内存数组self.memory进行索引操作，获取了批次记忆数据batch_memory。
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 这段代码根据神经网络的结构，将批次记忆数据batch_memory划分为输入h_train和输出m_train。
        # batch_memory[:, 0: self.net[0]]表示取批次记忆数据中的前self.net[0]列作为输入h_train，
        # batch_memory[:, self.net[0]:]表示取批次记忆数据中的剩余部分作为输出m_train。
        # 这里使用torch.Tensor将数据转换为PyTorch张量。
        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        # train the DNN
        # 定义了优化器optimizer，使用Adam算法进行参数优化。self.model.parameters()表示将模型中的所有可学习参数传递给优化器。
        # r是学习率，betas是Adam算法中的动量参数，weight_decay是L2正则化的权重衰减项。
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001)
        # 接着，定义了损失函数criterion，这里使用了二分类交叉熵损失（BCELoss）。
        criterion = nn.BCELoss()
        # 然后，将模型设置为训练模式self.model.train()，以启用训练相关的功能，例如Dropout和Batch Normalization。
        self.model.train()
        # 使用optimizer.zero_grad()，将优化器的梯度缓存清零，避免梯度累积。
        optimizer.zero_grad()
        # 通过模型self.model对输入h_train进行前向传播，得到预测值predict。
        predict = self.model(h_train)
        # 计算预测值predict和真实值m_train之间的损失loss
        loss = criterion(predict, m_train)
        # 调用loss.backward()进行反向传播，计算梯度。
        loss.backward()
        # 调用optimizer.step() 更新模型的参数，即执行一步优化器的参数更新。
        optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k = 1, mode = 'OP'):
        # to have batch dimension when feed into Tensor
        # 首先，将输入h转换为torch.Tensor，并添加一个新的维度作为批次维度，即h[np.newaxis, :]。
        # 这样做是为了符合模型的输入要求，因为模型通常期望输入具有批次维度。
        h = torch.Tensor(h[np.newaxis, :])
        # 接下来，将模型设置为评估模式self.model.eval()，以禁用训练过程中的一些功能，如Dropout。
        # 然后，通过模型对输入h进行前向传播，得到预测值m_pred。使用detach().numpy()将m_pred转换为NumPy数组，以便后续处理。
        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()
        # 接下来，根据mode参数的值进行不同的操作。如果mode是'OP'，则调用self.knm(m_pred[0], k)方法，
        # 传递预测结果m_pred[0]和k作为参数，返回使用某种操作（OP）选择的结果。
        # 如果mode是'KNN'，则调用self.knn(m_pred[0], k)方法，传递预测结果m_pred[0]和k作为参数，返回使用K最近邻（KNN）选择的结果。
        # 如果mode既不是'OP'也不是'KNN'，则打印错误消息，指示操作选择必须是'OP''KNN'。
        if mode is 'OP':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k = 1):
        # 这段代码是MemoryDNN类中的knm方法，用于根据给定的预测结果m生成k个有序保留的二进制动作。
        # return k order-preserving binary actions
        # 首先，创建一个空列表m_list来存储生成的二进制动作。
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        # 接下来，根据方程（8），生成第一个二进制的离线决策。
        # 这里使用了条件表达式1 * (m > 0.5)，将大于 0.5 的预测结果转换为1，小于等于0.5 的预测结果转换为0，并将结果添加到m_list中。
        m_list.append(1*(m>0.5))
        # 如果k大于1，则进一步生成剩余的 K - 1 个二进制离线决策。
        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            # 首先，计算预测结果与 0.5 的绝对值差值m_abs = abs(m - 0.5)。
            m_abs = abs(m-0.5)
            # 然后，使用 np.argsort 对 m_abs 进行排序，并获取前 k - 1 个最小值的索引，存储在 idx_list 中。
            idx_list = np.argsort(m_abs)[:k-1]
            # 接下来，使用循环遍历这k-1个索引，并根据方程（9）生成剩余的K-1个二进制离线决策。
            # 如果m中的某个元素大于0.5，则将该元素减去索引对应的值，并通过条件表达式1*(m - m[idx_list[i]] > 0)将大于0的结果转换为1，
            # 否则为0。如果m中的某个元素小于等于0.5，则将该元素减去索引对应的值，
            # 并通过条件表达式1*(m - m[idx_list[i]] >= 0)将大于等于0的结果转换为1，否则为0。
            # 然后将生成的二进制动作添加到m_list中。
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            # 如果 self.enumerate_actions 为空，代码导入了 itertools 模块，
            # 并使用 itertools.product 生成了所有长度为 self.net[0] 的二进制 offloading（转移）动作的排列组合。
            # 这些动作以列表的形式存储在 self.enumerate_actions 中。
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # 接下来，计算预测结果m与所有可能的二进制离线动作之间的2范数（欧氏距离的平方），
        # 使用((self.enumerate_actions - m)**2).sum(1)计算。这里使用了广播机制，
        # 将m与self.enumerate_actions的每个元素进行逐元素的差值计算，并对差值的平方进行求和。
        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

