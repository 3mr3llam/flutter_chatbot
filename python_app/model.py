import torch
import torch.nn as nn


# this is a feedforwrd neural network
class ChatNeuralNet(nn.Module):
    
    # input_size and num_tags must be fixed but hidden_size and be changed
    def __init__(self, input_size, hidden_size, num_tags):
        super(ChatNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_tags)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        #no activation and no softmax
        return out
