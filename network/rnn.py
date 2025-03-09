import torch
from torch import nn
import numpy as np
'''
 pytorch RNN 模型
'''
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,  batch_first=True):
        super(RNN, self).__init__()
        self.rnn =  torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=batch_first,bias=False)
    def forward(self,x):
       return self.rnn(x)

input_x = [
    [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]],
]
input_size = len(input_x[0])
hidden_size = 4

rnn = RNN(input_size=len(input_x[0]), hidden_size=hidden_size, batch_first=True)

r , hidden_state = rnn(torch.FloatTensor(input_x))
print("=======RNN输出结果=========")
print(r[0][-1])
print("=======RNN权重============")
state_weight = rnn.state_dict()
# print(state_weight)
ih_w = rnn.state_dict()["rnn.weight_ih_l0"].numpy()
hh_w = rnn.state_dict()["rnn.weight_hh_l0"].numpy()

print("=======自定义输出结果=========")
'''
 自定义 RNN 前向计算过程
'''
def customize_rnn(input,ih_w,hh_w):
    h = np.zeros(hidden_size)
    hidden = []
    for x in input:
        h = np.tanh(np.dot(x,ih_w.T) + np.dot(h,hh_w.T))
        hidden.append(h)
    return h,hidden
r,hidden_state = customize_rnn(input_x[0],ih_w,hh_w)
print(r)
