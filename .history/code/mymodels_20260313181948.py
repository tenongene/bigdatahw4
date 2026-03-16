import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
  ## =================================##
		self.fc1 = nn.Linear(178, 256)
		self.sigmoid = nn.Sigmoid()
		self.fc2 = nn.Linear(16, 5)



	def forward(self, x):
  ## =======+==========================##
		x = self.sigmoid(self.fc1(x))
		x = self.fc2(x)
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
  ## =================================##
  	
		self.conv1 = nn.Conv1d(in_channels=1,  out_channels=6,  kernel_size=5, stride=1)
		self.conv2 = nn.Conv1d(in_channels=6,  out_channels=16, kernel_size=5, stride=1)
		self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)
		self.relu  = nn.ReLU()
		self.fc1   = nn.Linear(16 * 41, 128)
		self.fc2   = nn.Linear(128, 5)

	def forward(self, x):
   
   ## =======+==========================##
		
		x = self.pool(self.relu(self.conv1(x)))  # (N, 6, 87)
		x = self.pool(self.relu(self.conv2(x)))  # (N, 16, 41)
		x = x.view(x.size(0), -1)               # (N, 656)
		x = self.relu(self.fc1(x))               # (N, 128)
		x = self.fc2(x)     
  # (N, 5)
		return x


class MyRNN(nn.Module):
	def __init__(self):
			super(MyRNN, self).__init__()
   ## =================================##
			self.gru = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
			self.fc  = nn.Linear(16, 5)

   
	def forward(self, x):
   ## =======+==========================##
   
		out, _ = self.gru(x)    # out: (N, 178, 16)
		out = out[:, -1, :]     # take last time step → (N, 16)
		out = self.fc(out)      # (N, 5)
  
		return out
  
		# return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
  
  ## =================================##
		self.fc1  = nn.Linear(dim_input, 32)
		self.tanh = nn.Tanh()
		self.gru  = nn.GRU(input_size=32, hidden_size=16, batch_first=True)
		self.fc2  = nn.Linear(16, 2)

	def forward(self, input_tuple):
		# HINT: Following two methods might be useful
		# 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn

		seqs, lengths = input_tuple
  
  ## =======+==========================##
  # Project to dense embedding
		x = self.tanh(self.fc1(seqs))           # (N, max_visits, 32)
 
		# Pack to skip padded time steps in GRU
		x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
		out, _ = self.gru(x)
		out, _ = pad_packed_sequence(out, batch_first=True)  # (N, max_visits, 16)
 
		# Gather the last valid hidden state for each patient using their true length
		idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
		out = out.gather(1, idx).squeeze(1)     # (N, 16)
 
		out = self.fc2(out)   
  
		return out

		# return seqs