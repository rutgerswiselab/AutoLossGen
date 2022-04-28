# coding=utf-8

import torch
import logging
from time import time
from utils import utils, global_p
from tqdm import tqdm
import numpy as np
import os
import copy
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, accuracy_score

class BaseRunner(object):
	@staticmethod
	def parse_runner_args(parser):
		parser.add_argument('--load', type=int, default=0,
							help='Whether load model and continue to train')
		parser.add_argument('--epoch', type=int, default=100,
							help='Number of epochs.')
		parser.add_argument('--check_epoch', type=int, default=1,
							help='Check every epochs.')
		parser.add_argument('--early_stop', type=int, default=1,
							help='whether to early-stop.')
		parser.add_argument('--lr', type=float, default=0.01,
							help='Learning rate.')
		parser.add_argument('--batch_size', type=int, default=128,
							help='Batch size during training.')
		parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
							help='Batch size during testing.')
		parser.add_argument('--dropout', type=float, default=0.2,
							help='Dropout probability for each deep layer')
		parser.add_argument('--l2', type=float, default=1e-4,
							help='Weight of l2_regularize in loss.')
		parser.add_argument('--optimizer', type=str, default='GD',
							help='optimizer: GD, Adam, Adagrad')
		parser.add_argument('--metric', type=str, default="AUC",
							help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
		parser.add_argument('--skip_eval', type=int, default=0,
							help='number of epochs without evaluation')
		parser.add_argument('--skip_rate', type=float, default=1.005, help='bad loss skip rate')
		parser.add_argument('--rej_rate', type=float, default=1.005, help='bad training reject rate')
		parser.add_argument('--skip_lim', type=float, default=1e-5, help='bad loss skip limit')
		parser.add_argument('--rej_lim', type=float, default=1e-5, help='bad training reject limit')
		parser.add_argument('--lower_bound_zero_gradient', type=float, default=1e-4, help='bound to check zero gradient')
		parser.add_argument('--search_train_epoch', type=int, default=1, help='epoch num for training when searching loss')
		parser.add_argument('--step_train_epoch', type=int, default=1, help='epoch num for training each step')
		
		return parser

	def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
				 dropout=0.2, l2=1e-5, metrics='AUC,RMSE', check_epoch=10, early_stop=1, controller=None, loss_formula=None,
				 controller_optimizer=None, args=None):
		self.optimizer_name = optimizer
		self.learning_rate = learning_rate
		self.epoch = epoch
		self.batch_size = batch_size
		self.eval_batch_size = eval_batch_size
		self.dropout = dropout
		self.no_dropout = 0.0
		self.l2_weight = l2

		self.metrics = metrics.lower().split(',')
		self.check_epoch = check_epoch
		self.early_stop = early_stop
		self.time = None

		self.train_results, self.valid_results, self.test_results = [], [], []
		
		self.controller = controller
		self.loss_formula = loss_formula
		self.controller_optimizer = controller_optimizer
		self.args = args
		self.print_prediction = {}

	def _build_optimizer(self, model):
		optimizer_name = self.optimizer_name.lower()
		if optimizer_name == 'gd':
			logging.info("Optimizer: GD")
			optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
			# optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
		elif optimizer_name == 'adagrad':
			logging.info("Optimizer: Adagrad")
			optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
			# optimizer = torch.optim.Adagrad(model.parameters(), lr=self.learning_rate)
		elif optimizer_name == 'adam':
			logging.info("Optimizer: Adam")
			optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
			# optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
		else:
			logging.error("Unknown Optimizer: " + self.optimizer_name)
			assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
			optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
			# optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
		return optimizer

	def _check_time(self, start=False):
		if self.time is None or start:
			self.time = [time()] * 2
			return self.time[0]
		tmp_time = self.time[1]
		self.time[1] = time()
		return self.time[1] - tmp_time

	def batches_add_control(self, batches, train):
		for batch in batches:
			batch['train'] = train
			batch['dropout'] = self.dropout if train else self.no_dropout
		return batches

	def predict(self, model, data, data_processor, train=False):
		batches = data_processor.prepare_batches(data, self.eval_batch_size, train=train)
		batches = self.batches_add_control(batches, train=train)

		model.eval()
		predictions = []
		for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
			prediction = model.predict(batch)['prediction']
			predictions.append(prediction.detach().cpu())

		predictions = np.concatenate(predictions)
		sample_ids = np.concatenate([b[global_p.K_SAMPLE_ID] for b in batches])

		reorder_dict = dict(zip(sample_ids, predictions))
		predictions = np.array([reorder_dict[i] for i in data[global_p.K_SAMPLE_ID]])
		return predictions

	def fit(self, model, data, data_processor, epoch=-1, loss_fun=None, sample_arc=None, regularizer=True):  # fit the results for an input set
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		batches = data_processor.prepare_batches(data, self.batch_size, train=True)
		batches = self.batches_add_control(batches, train=True)
		batch_size = self.batch_size if data_processor.rank == 0 else self.batch_size * 2
		model.train()
		accumulate_size = 0
		to_show = batches if self.args.search_loss else tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1)
		for batch in to_show:
			accumulate_size += len(batch['Y'])
			model.optimizer.zero_grad()
			output_dict = model(batch)
			loss = output_dict['loss'] + model.l2() * self.l2_weight
			if loss_fun is not None and sample_arc is not None:
				loss = loss_fun(output_dict['prediction'], batch['Y'], sample_arc)
				if regularizer:
					loss += model.l2() * self.l2_weight
			loss.backward()
			torch.nn.utils.clip_grad_value_(model.parameters(), 50)
			if accumulate_size >= batch_size or batch is batches[-1]:
				model.optimizer.step()
				accumulate_size = 0
		model.eval()
		return output_dict

	def eva_termination(self, model):
		"""
		检查是否终止训练，基于验证集
		:param model: 模型
		:return: 是否终止训练
		"""
		metric = self.metrics[0]
		valid = self.valid_results
		# 如果已经训练超过100轮，且评价指标越小越好，且评价已经连续十轮非减
		if len(valid) > 100 and metric in utils.LOWER_METRIC_LIST and utils.strictly_increasing(valid[-10:]):
			return True
		# 如果已经训练超过100轮，且评价指标越大越好，且评价已经连续十轮非增
		elif len(valid) > 100 and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(valid[-10:]):
			return True
		# 训练好结果离当前已经100轮以上了
		elif len(valid) - valid.index(utils.best_result(metric, valid)) > 100:
			return True
		return False
	
	def predict_with_grad(self, model, data, data_processor, train=False):
		"""
		预测，不训练
		:param model: 模型
		:param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
		:param data_processor: DataProcessor实例
		:return: prediction 拼接好的 np.array
		"""
		batches = data_processor.prepare_batches(data, self.eval_batch_size, train=train)
		batches = self.batches_add_control(batches, train=train)

		model.eval()
		predictions = []
		for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
			prediction = model.predict(batch)['prediction']
			predictions.append(prediction)

		predictions = torch.cat(predictions)
		sample_ids = np.concatenate([b[global_p.K_SAMPLE_ID] for b in batches])

		reorder_dict = dict(zip(sample_ids, predictions))
		predictions = torch.tensor([reorder_dict[i] for i in data[global_p.K_SAMPLE_ID]], requires_grad=True).cuda()
		return predictions

	def train(self, model, data_processor, skip_eval=0):

		# 获得训练、验证、测试数据，epoch=-1不shuffle
		train_data = data_processor.get_train_data(epoch=-1)
		validation_data = data_processor.get_validation_data()
		test_data = data_processor.get_test_data()
		self._check_time(start=True)  # 记录初始时间

		# 训练之前的模型效果
		init_train = self.evaluate(model, train_data, data_processor, metrics=self.metrics[0:1]) \
			if train_data is not None else [-1.0] * len(self.metrics)
		init_valid = self.evaluate(model, validation_data, data_processor, metrics=self.metrics[0:1]) \
			if validation_data is not None else [-1.0] * len(self.metrics)
		init_test = self.evaluate(model, test_data, data_processor) \
			if test_data is not None else [-1.0] * len(self.metrics)
		
		logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
			utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
			self._check_time()) + ','.join(self.metrics))
		min_reward = torch.tensor(-1.0).cuda()
		if model.optimizer is None:
			model.optimizer = self._build_optimizer(model)
		last_search_cnt = self.controller.num_aggregate * self.args.controller_train_steps
		try:
			for epoch in range(self.epoch):
				self._check_time()
				epoch_train_data = data_processor.get_train_data(epoch=epoch)
				self.loss_formula.eval()
				self.controller.zero_grad()
				epoch_val_for_train_data = data_processor.get_val_data_for_train(epoch=epoch)
				if self.args.search_loss:
					start_auc = self.evaluate(model, validation_data, data_processor)[0]
					baseline = torch.tensor(start_auc).cuda()
					cur_model = copy.deepcopy(model)
					grad_dict = dict()
					test_pred = torch.rand(20).cuda() * 0.8 + 0.1 # change range here
					test_label = torch.rand(20).cuda()
					test_pred.requires_grad = True
					max_reward = min_reward.clone().detach()
					best_arc = None
					for i in tqdm(range(last_search_cnt), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
						while True:
							reward = None
							self.controller()  # perform forward pass to generate a new architecture
							sample_arc = self.controller.sample_arc
							if test_pred.grad is not None:
								test_pred.grad.data.zero_()
							test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
							try:
								test_loss.backward()
							except RuntimeError:
								pass
							if test_pred.grad is None or torch.norm(test_pred.grad, float('inf')) < self.args.lower_bound_zero_gradient:
								reward = min_reward.clone().detach()
							if reward is None:
								for key, value in grad_dict.items():
									if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
										reward = value.clone().detach()
										break
							if reward is None:
								model.zero_grad()
								for j in range(self.args.search_train_epoch):
									last_batch = self.fit(model, epoch_train_data, data_processor, epoch=epoch, loss_fun=self.loss_formula, sample_arc=sample_arc, regularizer=False)
								reward = torch.tensor(self.evaluate(model, validation_data, data_processor)[0]).cuda()
								grad_dict[test_pred.grad.clone().detach()] = reward.clone().detach()
								model = copy.deepcopy(cur_model)
							if reward < baseline - self.args.skip_lim:
								reward = min_reward.clone().detach()
								reward += self.args.controller_entropy_weight * self.controller.sample_entropy
							else:
								if reward > max_reward:
									max_reward = reward.clone().detach()
									if self.args.train_with_optim:
										best_arc = copy.deepcopy(sample_arc)
								reward += self.args.controller_entropy_weight * self.controller.sample_entropy
								baseline -= (1 - self.args.controller_bl_dec) * (baseline - reward)
							baseline = baseline.detach()
							
							ctrl_loss = -1 * self.controller.sample_log_prob * (reward - baseline)
							ctrl_loss /= self.controller.num_aggregate
							if (i + 1) % self.controller.num_aggregate == 0:
								ctrl_loss.backward()
								grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.parameters(),
																	   self.args.child_grad_bound)
								self.controller_optimizer.step()
								self.controller.zero_grad()
							else:
								ctrl_loss.backward(retain_graph=True)
							break
					self.controller.eval()
					
					logging.info('Best auc during controller train: %.3f; Starting auc: %.3f' % (max_reward.item(), start_auc))
					last_search_cnt = 0
					if self.args.train_with_optim and best_arc is not None and max_reward > start_auc - self.args.rej_lim:
						sample_arc = copy.deepcopy(best_arc)
						for j in range(self.args.step_train_epoch):
							last_batch = self.fit(model, epoch_train_data, data_processor, epoch=epoch, loss_fun=self.loss_formula, sample_arc=sample_arc)
						new_auc = torch.tensor(self.evaluate(model, validation_data, data_processor)[0]).cuda()
						print('Optimal: ', self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
					else:
						grad_dict = dict()
						self.controller.zero_grad()
						while True:
							with torch.no_grad():
								self.controller(sampling=True)
								last_search_cnt += 1
							sample_arc = self.controller.sample_arc
							if test_pred.grad is not None:
								test_pred.grad.data.zero_()
							test_loss = self.loss_formula(test_pred, test_label, sample_arc, small_epsilon=True)
							try:
								test_loss.backward()
							except RuntimeError:
								pass
							if test_pred.grad is None or torch.norm(test_pred.grad, float('inf')) < self.args.lower_bound_zero_gradient:
								continue
							dup_flag = False
							for key in grad_dict.keys():
								if torch.norm(test_pred.grad - key, float('inf')) < self.args.lower_bound_zero_gradient:
									dup_flag = True
									break
							if dup_flag:
								continue
							print(self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
							grad_dict[test_pred.grad.clone().detach()] = True
							model = copy.deepcopy(cur_model)
							model.zero_grad()
							for j in range(self.args.step_train_epoch):
								last_batch = self.fit(model, epoch_train_data, data_processor, epoch=epoch, loss_fun=self.loss_formula, sample_arc=sample_arc)
							new_auc = torch.tensor(self.evaluate(model, validation_data, data_processor)[0]).cuda()
							if new_auc > start_auc - self.args.rej_lim:
								break
							print('Epoch %d: Reject!' % (epoch + 1))
					
					last_search_cnt = max(last_search_cnt // 10, self.controller.num_aggregate * self.args.controller_train_steps)
					if last_search_cnt % self.controller.num_aggregate != 0:
						last_search_cnt = (last_search_cnt // self.controller.num_aggregate + 1) * self.controller.num_aggregate
					logging.info(self.loss_formula.log_formula(sample_arc=sample_arc, id=self.loss_formula.num_layers - 1))
					self.controller.train()
				else:
					last_batch = self.fit(model, epoch_train_data, data_processor, epoch=epoch, loss_fun=None, sample_arc=None)
				training_time = self._check_time()

				if epoch >= skip_eval:
					metrics = self.metrics[0:1]
					train_result = self.evaluate(model, train_data, data_processor, metrics=metrics) \
						if train_data is not None else [-1.0] * len(self.metrics)
					valid_result = self.evaluate(model, validation_data, data_processor, metrics=metrics) \
						if validation_data is not None else [-1.0] * len(self.metrics)
					test_result = self.evaluate(model, test_data, data_processor) \
						if test_data is not None else [-1.0] * len(self.metrics)
					testing_time = self._check_time()

					self.train_results.append(train_result)
					self.valid_results.append(valid_result)
					self.test_results.append(test_result)

					# 输出当前模型效果
					logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
								 % (epoch + 1, training_time, utils.format_metric(train_result),
									utils.format_metric(valid_result), utils.format_metric(test_result),
									testing_time) + ','.join(self.metrics))
					
					if not self.args.search_loss:
						print("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
								 % (epoch + 1, training_time, utils.format_metric(train_result),
									utils.format_metric(valid_result), utils.format_metric(test_result),
									testing_time) + ','.join(self.metrics))
					# 如果当前效果是最优的，保存模型，基于验证集
					if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
						model.save_model()
						self.controller.save_model()
						self.loss_formula.save_model()
					# 检查是否终止训练，基于验证集
					if self.args.search_loss == False and self.eva_termination(model) and self.early_stop == 1:
						logging.info("Early stop at %d based on validation result." % (epoch + 1))
						break
				if epoch < skip_eval:
					logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))
		except KeyboardInterrupt:
			logging.info("Early stop manually")
			save_here = input("Save here? (1/0) (default 0):")
			if str(save_here).lower().startswith('1'):
				model.save_model()
				self.controller.save_model()
				self.loss_formula.save_model()

		# Find the best validation result across iterations
		best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
		best_epoch = self.valid_results.index(best_valid_score)
		logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
					 % (best_epoch + 1,
						utils.format_metric(self.train_results[best_epoch]),
						utils.format_metric(self.valid_results[best_epoch]),
						utils.format_metric(self.test_results[best_epoch]),
						self.time[1] - self.time[0]) + ','.join(self.metrics))
		best_test_score = utils.best_result(self.metrics[0], self.test_results)
		best_epoch = self.test_results.index(best_test_score)
		logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
					 % (best_epoch + 1,
						utils.format_metric(self.train_results[best_epoch]),
						utils.format_metric(self.valid_results[best_epoch]),
						utils.format_metric(self.test_results[best_epoch]),
						self.time[1] - self.time[0]) + ','.join(self.metrics))
		model.load_model()
		self.controller.load_model()
		self.loss_formula.load_model()

	def evaluate(self, model, data, data_processor, metrics=None):  # evaluate the results for an input set
		"""
		evaluate模型效果
		:param model: 模型
		:param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
		:param data_processor: DataProcessor
		:param metrics: list of str
		:return: list of float 每个对应一个 metric
		"""
		if metrics is None:
			metrics = self.metrics
		predictions = self.predict(model, data, data_processor)
		return model.evaluate_method(predictions, data, metrics=metrics)
