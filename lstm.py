import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pickle


# class LSTMModel(nn.Module):
#     def __init__(self):
#         super(LSTMModel, self).__init__()
        
#         #num_nodes = 19
#         rnn_units = 64
#         num_rnn_layers = 3        
#         input_dim = 512
        
#         self._input_dim = input_dim
#         #self._num_nodes = num_nodes
#         self._num_rnn_layers = num_rnn_layers
#         self._rnn_units = rnn_units
#         self._num_classes = 1
#         self.up_linear = nn.Linear(1,512)

                
#         self.linear_up = nn.Linear(1, 512)
#         # self.lstm = nn.LSTM(input_dim * num_nodes, 
#         #                   rnn_units, 
#         #                   num_rnn_layers,
#         #                   batch_first=True)
#         self.lstm = nn.LSTM(input_dim, 
#                           rnn_units, 
#                           num_rnn_layers,
#                           batch_first=True
#                           )                  
#         self.dropout = nn.Dropout(p=0.0) # dropout layer before final FC
#         self.fc = nn.Linear(rnn_units, 1) # final FC layer
#         self.relu = nn.ReLU()  
    
#     def forward(self, inputs):
#         """
#         Args:
#             inputs: (batch_size, max_seq_len, num_nodes, input_dim)
#             seq_lengths: (batch_size, )
#         """
        
#         inputs = self.up_linear(inputs.permute(0, 2, 1))
#         batch_size, max_seq_len, _ = inputs.shape
#         inputs = torch.reshape(inputs, (batch_size, max_seq_len, -1))  # (batch_size, max_seq_len, num_nodes*input_dim)
        
#         # initialize hidden states
#         initial_hidden_state, initial_cell_state = self.init_hidden(batch_size)

#         # LSTM
        
#         output, _ = self.lstm(inputs, (initial_hidden_state, initial_cell_state)) # (batch_size, max_seq_len, rnn_units)

#         print(output.shape)
#         last_out = self.last_relevant_pytorch(output, torch.ones(32)*178) # (batch_size, rnn_units)
        
#         # Dropout -> ReLU -> FC
#         logits = self.fc(self.relu(self.dropout(last_out))) # (batch_size, num_classes)       
        
#         return logits.squeeze()

#     def last_relevant_pytorch(self, output, lengths):

#         # masks of the true seq lengths
#         lengths = lengths.to(output.device)
#         masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2)).long()
#         time_dimension = 1 
#         #if batch_first else 0
#         masks = masks.unsqueeze(time_dimension)
#         last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
#         return last_output


#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_()
#         cell = weight.new(self._num_rnn_layers, batch_size, self._rnn_units).zero_()
#         return hidden, cell


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        
            #num_nodes = 19
        rnn_units = 64
        num_rnn_layers = 3        
        input_dim = 512
        
        self._input_dim = input_dim
        #self._num_nodes = num_nodes
        self._num_rnn_layers = num_rnn_layers
        self._rnn_units = rnn_units
        self._num_classes = 1
        self.up_linear = nn.Linear(1,512)

                
        self.linear_up = nn.Linear(1, 512)
        # self.lstm = nn.LSTM(input_dim * num_nodes, 
        #                   rnn_units, 
        #                   num_rnn_layers,
        #                   batch_first=True)
        self.lstm = nn.LSTM(input_dim, 
                          rnn_units, 
                          num_rnn_layers,
                          batch_first=True
                          )                  
        self.dropout = nn.Dropout(p=0.0) # dropout layer before final FC
        self.fc = nn.Linear(rnn_units, 1) # final FC layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, max_seq_len, num_nodes, input_dim)
            seq_lengths: (batch_size, )
        """
        
        inputs = self.up_linear(inputs.permute(0, 2, 1))
        batch_size, max_seq_len, _ = inputs.shape
        inputs = torch.reshape(inputs, (batch_size, max_seq_len, -1))  # (batch_size, max_seq_len, num_nodes*input_dim)
        
        # initialize hidden states
        initial_hidden_state, initial_cell_state = self.init_hidden(batch_size)

        # LSTM
        
        output, _ = self.lstm(inputs, (initial_hidden_state, initial_cell_state)) # (batch_size, max_seq_len, rnn_units)

        #print(output.shape)
        last_out = self.last_relevant_pytorch(output, torch.ones(output.shape[0])*117) # (batch_size, rnn_units)
        
        # Dropout -> ReLU -> FC
        logits = self.fc(self.relu(self.dropout(last_out))) # (batch_size, num_classes)

        # Sigmoid       
        logits = self.sigmoid(logits)

        return logits.squeeze()

    def last_relevant_pytorch(self, output, lengths):

        # masks of the true seq lengths
        lengths = lengths.to(output.device)
        masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2)).long()
        time_dimension = 1 
        #if batch_first else 0
        masks = masks.unsqueeze(time_dimension)
        last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
        return last_output


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new_zeros((self._num_rnn_layers, batch_size, self._rnn_units))
        cell = weight.new_zeros((self._num_rnn_layers, batch_size, self._rnn_units))
        return hidden, cell
