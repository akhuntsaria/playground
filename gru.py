import torch
import torch.nn as nn

torch.manual_seed(1337)

device = torch.device("mps")
#device = torch.device("cpu")
print(device)

rnn = nn.GRU(10, 20, 2).to(device=device)
#rnn = nn.LSTM(10, 20, 2).to(device=device)
input = torch.randn(5, 3, 10).to(device=device)
h0 = torch.randn(2, 3, 20).to(device=device)
output, hn = rnn(input, h0)
#c0 = torch.randn(2, 3, 20).to(device)
#output, hn = rnn(input, (h0, c0))

print(output[0])