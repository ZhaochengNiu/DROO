#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains the main code of DROO. It loads the training samples saved in ./data/data_#.mat, splits the samples into two parts (training and testing data constitutes 80% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. There are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, early access, 2019, DOI:10.1109/TMC.2019.2928811.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# Implementated based on the PyTorch 
from memoryPyTorch import MemoryDNN
from optimization import bisection

import time


def plot_rate(rate_his, rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    # 这两行代码将 rate_his 转换为 numpy 数组 rate_array 和 pandas 数据帧df，以便后续的数据处理和绘图操作。
    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
#    rolling_intv = 20
    # 这行代码绘制了计算速率的曲线。np.arange(len(rate_array)) + 1生成了一个从1到rate_array长度的数组作为x轴，
    # np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values)计算了滚动窗口内的平均值，
    # 并将结果作为y轴数据。'b'表示蓝色线条。
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    # 这行代码使用fill_between函数填充了计算速率曲线上下的区域。
    # np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values)计算了滚动窗口内的最小值，
    # np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values)计算了滚动窗口内的最大值。
    # color='b'表示填充颜色为蓝色，alpha=0.2表示填充区域的透明度为0.2。
    plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N = 10                       # number of users
    n = 30000                    # number of time frames
    K = N                        # initialize K = N
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure
    Delta = 32                   # Update interval for adaptive K

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    channel = sio.loadmat('./data/data_%d' %N)['input_h']
    rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    # 这行代码计算了按照 80: 20 比例划分数据集时的分割索引 split_idx。具体来说，它首先获取了数据集 channel 的长度，然后将其乘以
    # 0.8，即 80 %。然后，使用 int() 函数将结果转换为整数，以确保 split_idx 是一个整数值，因为索引通常是整数。这样就得到了分割索引
    # split_idx，该索引将数据集划分为训练集和测试集，其中前 split_idx 个元素用于训练，后面的元素用于测试。
    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # training data size


    mem = MemoryDNN(net = [N, 120, 80, N],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    for i in range(n):
        # 这个条件判断语句检查i是否是n的十分之一的倍数。如果是，它会打印一个格式化的字符串，表示当前迭代的百分比。
        # 例如，当i等于n的十分之一、二分之一、三分之一等时，它会打印出 "10.0%"、"20.0%"、"30.0%" 等。
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        # 这个条件判断语句检查i是否大于0且是Delta的倍数。如果满足条件，它会执行一系列操作。
        # 根据Delta的不同取值，它会计算max_k和K的值。具体来说：
        # 如果Delta > 1，它会获取k_idx_his列表中倒数第Delta个元素到最后一个元素的子列表，并找到其中的最大值，
        # 然后加1赋值给max_k。
        # 如果Delta = 1，它会获取k_idx_his列表中的最后一个元素，并加1赋值给 max_k。
        # 最后，它会将max_k + 1与N中的较小值赋值给K。
        if i > 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1;
            else:
                max_k = k_idx_his[-1] +1;
            K = min(max_k +1, N)
        # 这个条件判断语句根据i的值来确定i_idx的取值，可能是为了在训练和测试过程中使用不同的索引
        # 这个条件判断语句根据i的值将i_idx赋值为不同的结果。具体来说：
        # 如果i小于n - num_test，它将i对split_idx取模，并将结果赋值给i_idx。
        # 如果i大于等于n - num_test，它将计算i - n + num_test + split_idx的值，并将结果赋值给i_idx。
        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx
        # 这行代码根据索引 i_idx 从 channel 数组中获取一行数据，并将其赋值给变量 h。
        h = channel[i_idx,:]
        # m_list = mem.decode(h, K, decoder_mode)：使用mem对象的decode方法，根据h、K和decoder_mode解码出一个模式列表m_list。
        # the action selection must be either 'OP' or 'KNN'
        m_list = mem.decode(h, K, decoder_mode)
        # r_list = []：创建一个空列表r_list，用于存储计算出的奖励值。
        r_list = []
        # for m in m_list:：对于m_list中的每个模式m，执行以下操作：
        # r_list.append(bisection(h/1000000, m)[0])：使用bisection函数计算h/1000000和m作为参数，
        # 得到一个奖励值，并将其添加到r_list列表中。
        for m in m_list:
            r_list.append(bisection(h/1000000, m)[0])
        # mem.encode(h, m_list[np.argmax(r_list)])：
        # 使用mem对象的encode方法，将h和具有最大奖励值的模式m_list[np.argmax(r_list)]进行编码。
        # encode the mode with largest reward
        mem.encode(h, m_list[np.argmax(r_list)])
        # the main code for DROO training ends here




        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])


    total_time=time.time()-start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio)

    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
