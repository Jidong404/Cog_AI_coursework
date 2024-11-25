import torch
import torch.nn as nn
import math
from torch.nn import init


class LeakyRNN(nn.Module):
    """Leaky RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms.
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        
        
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""


        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)


        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden


class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, e_prop=0.8,**kwargs):
        super().__init__()
        self.e_size = int(hidden_size * e_prop)
 
        self.rnn = LeakyRNN(input_size, hidden_size, **kwargs)

        self.fc = nn.Linear(self.e_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        pooling_layer = nn.AdaptiveAvgPool1d(self.e_size)
        pool_hidden = pooling_layer(rnn_output)
        out = self.fc(pool_hidden)
        return out, rnn_output, pool_hidden
    
if __name__ == "__main__":
    input_size = 3
    hidden_size = 50
    output_size = 3
    sequence_length = 100
    batch_size = 16

    model = RNNNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    
    inputs = torch.randn(sequence_length, batch_size, input_size)

    #outputs, hidden_state = model(inputs)
    outputs, hidden_state,pool_hidden = model(inputs)
    
    print("Output shape:", outputs.shape)  
    print("Total hidden state shape for all sequence is:", hidden_state.shape) 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print(f"Pooling hidden layer: {pool_hidden.shape}")
    print(model)