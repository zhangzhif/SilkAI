import numpy as np
import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size=10, num_layers=1, *args, **kwargs):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,bias=False)

    def forward(self,x):
        return self.lstm(x)


input_x = [
    [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]],
]
input_size = len(input_x[0])
hidden_size = 4
lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
output,hidden =  lstm(torch.FloatTensor(input_x))

print("=========LSTM输出结果==========")
print(output[0][-1])

# print("=======LSTM权重============")
state_weight = lstm.state_dict()
# print(state_weight)

weight_ih_l0 = state_weight["lstm.weight_ih_l0"].numpy()



weight_hh_l0 = state_weight["lstm.weight_hh_l0"].numpy()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
'''
 自定义 LSTM 前向计算过程
'''
def customize_lstm(input,weight_ih_l0,weight_hh_l0):
    w_ii, w_if, w_ig, w_io = np.split(weight_ih_l0, 4)
    w_hi, w_hf, w_hg, w_ho = np.split(weight_hh_l0, 4)
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)
    hidden = []
    for x in input:
        i_t = sigmoid(np.dot(x, w_ii.T) +np.dot(h, w_hi.T))
        f_t = sigmoid(np.dot(x, w_if.T) +np.dot(h, w_hf.T))
        c_t = np.tanh(np.dot(x, w_ig.T) +np.dot(h, w_hg.T))
        o_t = sigmoid(np.dot(x, w_io.T) +np.dot(h, w_ho.T))
        c = i_t * c_t + f_t * c
        h = o_t * np.tanh(c)
        hidden.append(h)
    return h,hidden



r,hidden_state = customize_lstm(input_x[0],weight_ih_l0,weight_hh_l0)
print("=======自定义输出结果=========")
print(r)



