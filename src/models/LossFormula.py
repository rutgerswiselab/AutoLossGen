import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class LossLayer(nn.Module):
	'''
	https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L245
	'''
	def __init__(self, layer_id, in_planes, out_planes, epsilon=1e-6):
		super(LossLayer, self).__init__()

		self.layer_id = layer_id
		self.in_planes = in_planes
		self.out_planes = out_planes

		self.branch_0 = nn.Identity()
		self.branch_1 = nn.Sigmoid()
		self.branch_2 = nn.Tanh()
		self.branch_3 = nn.ReLU()
		self.epsilon = epsilon

	def forward(self, prev_layers, sample_arc, small_epsilon=False):
		layer_type = sample_arc[0]
		if self.layer_id > 0:
			skip_indices = sample_arc[1]
		else:
			skip_indices = []
		out = []
		for i, skip in enumerate(skip_indices):
			if skip != 1:
				out.append(prev_layers[i].reshape(-1, 1))
		out = torch.cat(out, 1).cuda()
		epsilon = 1e-6 if small_epsilon else self.epsilon
		if layer_type == 0:
			out = torch.sum(out, dim=1)
		elif layer_type == 1:
			out = torch.prod(out, dim=1)
		elif layer_type == 2:
			out = torch.max(out, dim=1)[0]
		elif layer_type == 3:
			out = torch.min(out, dim=1)[0]
		elif layer_type == 4:
			out = - out
		elif layer_type == 5:
			out = self.branch_0(out)
		elif layer_type == 6:
			out = torch.sign(out) * torch.log(torch.abs(out) + epsilon)
		elif layer_type == 7:
			out = (out) ** 2
		elif layer_type == 8:
			out = torch.sign(out) / (torch.abs(out) + epsilon)
		# The following operators are defined but not used
		elif layer_type == 9:
			out = self.branch_1(out)
		elif layer_type == 10:
			out = self.branch_2(out)
		elif layer_type == 11:
			out = self.branch_3(out)
		elif layer_type == 12:
			out = torch.abs(out)
		elif layer_type == 13:
			out = torch.sign(out) * torch.sqrt(torch.abs(out) + epsilon)
		elif layer_type == 14:
			out = torch.exp(out)
		else:
			raise ValueError("Unknown layer_type {}".format(layer_type))
		out = torch.clamp(out, min=1e-5, max=1e5)

		return out


class LossFormula(nn.Module):
	@staticmethod
	def parse_Formula_args(parser):
		parser.add_argument('--child_keep_prob', type=float, default=0.9)
		parser.add_argument('--child_lr_max', type=float, default=0.05)
		parser.add_argument('--child_lr_min', type=float, default=0.0005)
		parser.add_argument('--child_lr_T', type=float, default=10)
		parser.add_argument('--child_l2_reg', type=float, default=0.00025)
		parser.add_argument('--epsilon', type=float, default=1e-6)
		return parser
	
	def __init__(self,
				 model_path,
				 num_layers=12,
				 num_branches=6,
				 out_filters=24,
				 keep_prob=1.0,
				 fixed_arc=None,
				 epsilon=1e-6
				 ):
		super(LossFormula, self).__init__()

		self.num_layers = num_layers
		self.num_branches = num_branches
		self.out_filters = out_filters
		self.keep_prob = keep_prob
		self.fixed_arc = fixed_arc
		self.model_path = model_path

		self.layers = nn.ModuleList([])
		
		for layer_id in range(self.num_layers):
			layer = LossLayer(layer_id, self.num_layers + 3, 1, epsilon)
			self.layers.append(layer)
		
	def forward(self, x, y, sample_arc, small_epsilon=False):

		prev_layers = []
		for i in range(self.num_layers):
			prev_layers.append(torch.zeros_like(x))
		prev_layers[0] = x
		prev_layers[1] = y
		prev_layers[2] = torch.ones_like(x)
		for layer_id in range(3, self.num_layers):
			out = self.layers[layer_id](prev_layers, sample_arc[str(layer_id)], small_epsilon)
			prev_layers[layer_id] = out

		return torch.mean(prev_layers[-1])
	
	def log_formula(self, sample_arc, id):
		if id == 0:
			return 'pred'
		if id == 1:
			return 'label'
		if id == 2:
			return '1'
		skip_indices = sample_arc[str(id)][1]
		layer_type = int(sample_arc[str(id)][0][0])
		return_str = '('
		for i in range(id):
			if skip_indices[i] != 1:
				if layer_type == 0:
					return_str += self.log_formula(sample_arc, i) + ' + '
				elif layer_type == 1:
					return_str += self.log_formula(sample_arc, i) + ' * '
				elif layer_type == 2:
					return_str += self.log_formula(sample_arc, i) + ' , '
				elif layer_type == 3:
					return_str += self.log_formula(sample_arc, i) + ' , '
				elif layer_type == 4:
					return '- (' + self.log_formula(sample_arc, i) + ')'
				elif layer_type == 5:
					return self.log_formula(sample_arc, i)
				elif layer_type == 6:
					return 'Log (' + self.log_formula(sample_arc, i) + ')'
				elif layer_type == 7:
					return '(' + self.log_formula(sample_arc, i) + ') ^ 2'
				elif layer_type == 8:
					return '1 / (' + self.log_formula(sample_arc, i) + ')'
				elif layer_type == 9:
					return 'Sigmoid (' + self.log_formula(sample_arc, i) + ')'
				elif layer_type == 10:
					return 'Tanh (' + self.log_formula(sample_arc, i) + ')'
				elif layer_type == 11:
					return 'ReLU (' + self.log_formula(sample_arc, i) + ')'
				elif layer_type == 12:
					return '|' + self.log_formula(sample_arc, i) + '|'
				elif layer_type == 13:
					return 'Sqrt (' + self.log_formula(sample_arc, i) + ')'
				elif layer_type == 14:
					return 'e ^ (' + self.log_formula(sample_arc, i) + ')'
		if layer_type == 2:
			return_str = 'max' + return_str[:-3] + ')'
		elif layer_type == 3:
			return_str = 'min' + return_str[:-3] + ')'
		else:
			return_str = return_str[:-3] + ')'
		return return_str
	
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
