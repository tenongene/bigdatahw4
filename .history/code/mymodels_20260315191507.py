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
		self.network = nn.Sequential(
			nn.Linear(178, 256),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 5)
	)


	def forward(self, x):
  ## =======+==========================##
		x = self.network(x)
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
  ## =================================##

		self.conv_layers = nn.Sequential(
				nn.Conv1d(1, 6, kernel_size=5, stride=1),
				nn.ReLU(),
				nn.MaxPool1d(2),
				nn.Conv1d(6, 16, kernel_size=5, stride=1),
				nn.ReLU(),
				nn.MaxPool1d(2),
				nn.Conv1d(16, 32, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.MaxPool1d(2)
		)
		
		self.fc_layers = nn.Sequential(
				nn.Linear(32 * 19, 256),
				nn.ReLU(),
				nn.Dropout(0.3),
				nn.Linear(256, 128),
				nn.ReLU(),
				nn.Dropout(0.3),
				nn.Linear(128, 5)
		)

	def forward(self, x):

  ## =======+==========================##
		
		x = self.conv_layers(x)
		x = x.view(x.size(0), -1)
		x = self.fc_layers(x)

		return x


class MyRNN(nn.Module):
	def __init__(self):
			super(MyRNN, self).__init__()

			self.gru = nn.GRU(
					input_size=1,
					hidden_size=64,
					num_layers=2,
					batch_first=True,
					dropout=0.3,
					bidirectional=True
			)
		
			self.fc = nn.Sequential(
					nn.Linear(64 * 2, 64),
					nn.ReLU(),
					nn.Dropout(0.3),
					nn.Linear(64, 5)
			)

	def forward(self, x):
			out, _ = self.gru(x)       
			out = out[:, -1, :]        
			x = self.fc(out)         

			return x


class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		# You may use the input argument 'dim_input', which is basically the number of features
  
  ## =================================##
		self.fc1  = nn.Linear(dim_input, 64)
		self.tanh = nn.Tanh()
		self.drop = nn.Dropout(0.3)
		# 2-layer bidirectional GRU
		self.gru  = nn.GRU(
			input_size=64,
			hidden_size=32,
			num_layers=2,
			batch_first=True,
			dropout=0.3,
			bidirectional=True
		)
		# bidirectional doubles output size: 32*2=64
		self.fc2  = nn.Linear(64, 2)

	def forward(self, input_tuple):
		seqs, lengths = input_tuple

		# Project to dense embedding
		x = self.drop(self.tanh(self.fc1(seqs)))     # (N, max_visits, 64)

		# Pack → GRU → unpack
		x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
		out, _ = self.gru(x)
		out, _ = pad_packed_sequence(out, batch_first=True)  # (N, max_visits, 64)

		# Gather last valid hidden state per patient
		idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
		out = out.gather(1, idx).squeeze(1)          # (N, 64)

		seqs = self.fc2(out)                          # (N, 2)

		return seqs