import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical


class Controller(nn.Module):
	
	@staticmethod
	def parse_Ctrl_args(parser):
		"""
		data loader related command line arguments parser
		:param parser:
		:return:
		"""
		parser.add_argument('--search_for', default='macro', choices=['macro'])
		parser.add_argument('--controller_lstm_size', type=int, default=64)
		parser.add_argument('--controller_lstm_num_layers', type=int, default=1)
		parser.add_argument('--controller_tanh_constant', type=float, default=1.5)
		parser.add_argument('--controller_skip_target', type=float, default=0.4)
		parser.add_argument('--controller_skip_weight', type=float, default=0.8)
		parser.add_argument('--controller_lr', type=float, default=0.001)
		parser.add_argument('--controller_entropy_weight', type=float, default=0.0001)
		parser.add_argument('--controller_bl_dec', type=float, default=0.99)
		parser.add_argument('--controller_num_aggregate', type=int, default=20)
		parser.add_argument('--controller_train_steps', type=int, default=50)
		parser.add_argument('--controller_l2', type=float, default=1e-4, help='Weight of l2_regularize for controller.')
		return parser
	
	'''
	https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py
	'''
	def __init__(self,
				 model_path,
				 search_for="macro",
				 search_whole_channels=True,
				 num_layers=12,
				 num_branches=6,
				 out_filters=36,
				 lstm_size=32,
				 lstm_num_layers=2,
				 tanh_constant=1.5,
				 temperature=None,
				 skip_target=0.4,
				 skip_weight=0.8,
				 entropy_weight=0.0001,
				 bl_dec=0.99,
				 num_aggregate=20,
				 sample_branch_id=True,
				 sample_skip_id=True):
		super(Controller, self).__init__()

		self.search_for = search_for
		self.search_whole_channels = search_whole_channels
		self.num_layers = num_layers
		self.num_branches = num_branches
		self.out_filters = out_filters

		self.lstm_size = lstm_size
		self.lstm_num_layers = lstm_num_layers
		self.tanh_constant = tanh_constant
		self.temperature = temperature

		self.skip_target = skip_target
		self.skip_weight = skip_weight
		
		self.entropy_weight = entropy_weight
		self.bl_dec = bl_dec
		self.num_aggregate = num_aggregate
		self.model_path = model_path
		self.sample_branch_id = sample_branch_id
		self.sample_skip_id = sample_skip_id

		self._create_params()

	def _create_params(self):
		'''
		https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L83
		'''
		self.w_lstm = nn.LSTM(input_size=self.lstm_size,
							  hidden_size=self.lstm_size,
							  num_layers=self.lstm_num_layers)

		self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the starting input

		if self.search_whole_channels:
			self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
			self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)
		else:
			assert False, "Not implemented error: search_whole_channels = False"

		self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
		self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
		self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

		self._reset_params()

	def _reset_params(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
				nn.init.uniform_(m.weight, -0.1, 0.1)

		nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
		nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

	def forward(self, sampling=False, test_mode=None):
		'''
		https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L126
		'''
		h0 = None  # setting h0 to None will initialize LSTM state with 0s

		anchors = []
		anchors_w_1 = []

		arc_seq = {}
		entropys = []
		log_probs = []
		skip_count = []
		skip_penaltys = []
		_ids = []

		inputs = self.g_emb.weight
		skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_target]).cuda()
		test_log_prob = torch.tensor([0.0]).cuda()
		for layer_id in range(self.num_layers):
			if self.search_whole_channels:
				inputs = inputs.unsqueeze(0)
				output, hn = self.w_lstm(inputs, h0)
				output = output.squeeze(0)
				h0 = hn

				logit = self.w_soft(output)
				if self.temperature is not None:
					logit /= self.temperature
				if self.tanh_constant is not None:
					logit = self.tanh_constant * torch.tanh(logit)
				branch_id_dist = Categorical(logits=logit)
				
				if layer_id > 2 and test_mode is not None:
					branch_id = torch.tensor(test_mode[layer_id][0]).cuda()
					test_log_prob += (branch_id_dist.log_prob(branch_id)).view(-1)
				else:
					if self.sample_branch_id or sampling:
						branch_id = branch_id_dist.sample()
					else:
						branch_id = torch.argmax(logit).reshape((-1))
				arc_seq[str(layer_id)] = [branch_id]

				log_prob = branch_id_dist.log_prob(branch_id)
				log_probs.append(log_prob.view(-1))
				entropy = branch_id_dist.entropy()
				entropys.append(entropy.view(-1))

				inputs = self.w_emb(branch_id)
				inputs = inputs.unsqueeze(0)
			else:
				# https://github.com/melodyguan/enas/blob/master/src/cifar10/general_controller.py#L171
				assert False, "Not implemented error: search_whole_channels = False"

			output, hn = self.w_lstm(inputs, h0)
			output = output.squeeze(0)
			if layer_id > 2:
				query = torch.cat(anchors_w_1, dim=0)
				query = torch.tanh(query + self.w_attn_2(output))
				query = self.v_attn(query)
				logit = torch.cat([-query, query], dim=1)
				if self.temperature is not None:
					logit /= self.temperature
				if self.tanh_constant is not None:
					logit = self.tanh_constant * torch.tanh(logit)

				skip_dist = Categorical(logits=logit)
				new_skip_dist = Categorical(logits=logit[:, 0])
				if test_mode is not None:
					skip = torch.tensor(test_mode[layer_id][1]).cuda()
					for id, skip_status in enumerate(skip):
						if skip_status == 0:
							test_log_prob += new_skip_dist.log_prob(torch.tensor([id]).cuda()).view(-1)
				else:
					if branch_id < 4:  # Revise here! Dividing line between unary and binocular operators
						if self.sample_skip_id or sampling:
							skip = torch.ones_like(query.view(-1))
							
							first = new_skip_dist.sample()
							second = new_skip_dist.sample()
							while first == second:
								second = new_skip_dist.sample()
							skip[first] = skip[second] = 0
							log_prob = new_skip_dist.log_prob(first) + new_skip_dist.log_prob(second)
						else:
							skip = torch.ones_like(query.view(-1))
							skip[torch.argsort(logit[:, 0], descending=True)[:2]] = 0
							log_prob = skip_dist.log_prob(skip)
							log_prob = torch.sum(log_prob)
					else:
						skip = torch.ones_like(query.view(-1))
						new_skip_dist = Categorical(logits=logit[:, 0])
						if self.sample_skip_id or sampling:
							rank = new_skip_dist.sample()
						else:
							rank = torch.argmax(logit[:, 0])
						skip[rank] = 0
						log_prob = new_skip_dist.log_prob(rank)
					
				_ids.append((branch_id.tolist()[0], skip.tolist()))
				
				arc_seq[str(layer_id)].append(skip)

				skip_prob = torch.sigmoid(logit)
				kl = skip_prob * torch.log(skip_prob / skip_targets)
				kl = torch.sum(kl)
				skip_penaltys.append(kl)

				log_probs.append(log_prob.view(-1))

				# entropy = skip_dist.entropy()
				entropy = new_skip_dist.entropy()
				entropy = torch.sum(entropy)
				entropys.append(entropy.view(-1))

				# Calculate average hidden state of all nodes that got skips
				# and use it as input for next step
				skip = skip.type(torch.float)
				skip = skip.view(1, layer_id)
				skip_count.append(torch.sum(skip))
				inputs = torch.matmul(skip, torch.cat(anchors, dim=0))
				inputs /= (1.0 + torch.sum(skip))

			else:
				inputs = self.g_emb.weight
				# inputs = inputs.squeeze(0)

			anchors.append(output)
			anchors_w_1.append(self.w_attn_1(output))
		if test_mode is not None:
			print(test_log_prob)
		self.ids = _ids
		self.sample_arc = arc_seq

		entropys = torch.cat(entropys)
		self.sample_entropy = torch.sum(entropys)

		log_probs = torch.cat(log_probs)
		self.sample_log_prob = torch.sum(log_probs)

		skip_count = torch.stack(skip_count)
		self.skip_count = torch.sum(skip_count)

		skip_penaltys = torch.stack(skip_penaltys)
		self.skip_penaltys = torch.mean(skip_penaltys)
	
	def save_model(self, model_path=None):
		"""
		save model
		"""
		if model_path is None:
			model_path = self.model_path
		dir_path = os.path.dirname(model_path)
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)
		torch.save(self.state_dict(), model_path)
	
	def load_model(self, model_path=None):
		"""
		load model
		"""
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		self.eval()
	'''
	def l2(self):
		l2 = 0
		for p in self.parameters():
			l2 += (p ** 2).sum()
		return l2
	'''