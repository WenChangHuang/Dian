import torch
class Lstm(torch.nn.Module):
	def __init__(self,opt):
		super(Lstm,self).__init__()
		self.embedding = torch.nn.Embedding(opt.VOCAB_SIZE, opt.EMBEDDING_DIM)
		self.rnn = torch.nn.LSTM(
			input_size = opt.SENT_LEN,
			hidden_size = 64,
			num_layers = 3,
			batch_first = True,
			)	
		self.fc1 = torch.nn.Linear(64,opt.NUM_CLASS_1)
		self.fc2 = torch.nn.Linear(64,opt.NUM_CLASS_2)
		self.fc3 = torch.nn.Sequential(
				torch.nn.Linear(64,128),
				torch.nn.BatchNorm1d(128),
				torch.nn.ReLU(),
				torch.nn.Linear(128,opt.NUM_CLASS_3)
				)
	def forward(self,x):
		emb = self.embedding(x)
		output,(h_n,c_n) = self.rnn(emb)
		output = h_n[-1,:,:]
		output1 = self.fc1(output)
		output2 = self.fc2(output)
		output3 = self.fc3(output)
		return output1,output2,output3	
