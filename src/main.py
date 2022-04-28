# coding=utf-8

import argparse
import torch
import os
import sys
import numpy as np
from data_loader.DataLoader import DataLoader
from data_processor.DataProcessor import DataProcessor
from runner.BaseRunner import BaseRunner
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.global_p import *
from models.BiasedMF import BiasedMF
from models.DeepModel import DeepModel
from models.controller import Controller
from models.LossFormula import LossFormula


def main():
	parser = argparse.ArgumentParser(description='Model')
	parser.add_argument('--gpu', type=str, default='0', help='Set CUDA_VISIBLE_DEVICES')
	parser.add_argument('--verbose', type=int, default=logging.INFO, help='Logging Level, 0, 10, ..., 50')
	parser.add_argument('--log_file', type=str, default='../log/log_0.txt', help='Logging file path')
	parser.add_argument('--result_file', type=str, default='../result/result.npy', help='Result file path')
	parser.add_argument('--random_seed', type=int, default=42, help='Random seed of numpy and pytorch')
	parser.add_argument('--model_name', type=str, default='BiasedMF', help='Choose model to run.')
	parser.add_argument('--model_path', type=str, help='Model save path.',
						default=os.path.join(MODEL_DIR, 'biasedMF.pt'))  # '%s/%s.pt' % (model_name, model_name)))
	parser.add_argument('--controller_model_path', type=str, help='Controller Model save path.',
						default=os.path.join(MODEL_DIR, 'controller.pt'))
	parser.add_argument('--shared_cnn_model_path', type=str, help='Shared CNN Model save path.',
						default=os.path.join(MODEL_DIR, 'loss_formula.pt'))
	parser.add_argument('--formula_path', type=str, help='Loss Formula save path.',
						default=os.path.join(MODEL_DIR, 'Formula.txt'))
	parser.add_argument('--u_vector_size', type=int, default=64, help='Size of user vectors.')
	parser.add_argument('--i_vector_size', type=int, default=64, help='Size of item vectors.')
	
	parser.add_argument('--child_num_layers', type=int, default=12)
	parser.add_argument('--child_num_branches', type=int, default=8)  # different layers
	parser.add_argument('--child_out_filters', type=int, default=36)
	parser.add_argument('--sample_branch_id', action='store_true')
	parser.add_argument('--sample_skip_id', action='store_true')
	parser.add_argument('--search_loss', action='store_true', help="To search a loss or verify a loss")
	parser.add_argument('--train_with_optim', action='store_true')
	parser.add_argument('--child_grad_bound', type=float, default=5.0)
	parser.add_argument('--smooth_coef', type=float, default=1e-6)
	parser.add_argument('--layers', type=str, default='[64, 16]',
						help="Size of each layer. (For Deep RS Model.)")
	parser.add_argument('--loss_func', type=str, default='BCE',
						help='Loss Function. Choose from ["BCE", "MSE", "Hinge", "Focal", "MaxR", "SumR", "LogMin"]')
	
	parser = DataLoader.parse_data_args(parser)
	parser = DataProcessor.parse_dp_args(parser)
	parser = BaseRunner.parse_runner_args(parser)
	parser = Controller.parse_Ctrl_args(parser)
	parser = LossFormula.parse_Formula_args(parser)
	args, extras = parser.parse_known_args()
	
	# random seed & gpu
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	
	controller = Controller(search_for=args.search_for,
							search_whole_channels=True,
							num_layers=args.child_num_layers + 3,
							num_branches=args.child_num_branches,
							out_filters=args.child_out_filters,
							lstm_size=args.controller_lstm_size,
							lstm_num_layers=args.controller_lstm_num_layers,
							tanh_constant=args.controller_tanh_constant,
							temperature=None,
							skip_target=args.controller_skip_target,
							skip_weight=args.controller_skip_weight,
							entropy_weight=args.controller_entropy_weight,
							bl_dec=args.controller_bl_dec,
							num_aggregate=args.controller_num_aggregate,
							model_path=args.controller_model_path,
							sample_branch_id=args.sample_branch_id,
							sample_skip_id=args.sample_skip_id)
	controller = controller.cuda()

	loss_formula = LossFormula(num_layers=args.child_num_layers + 3,
							 num_branches=args.child_num_branches,
							 out_filters=args.child_out_filters,
							 keep_prob=args.child_keep_prob,
							 model_path=args.shared_cnn_model_path,
							 epsilon=args.epsilon)
	loss_formula = loss_formula.cuda()

	# https://github.com/melodyguan/enas/blob/master/src/utils.py#L218
	controller_optimizer = torch.optim.Adam(params=controller.parameters(),
											lr=args.controller_lr,
											betas=(0.0, 0.999),
											eps=1e-3)
	# logging
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(filename=args.log_file, level=args.verbose)
	logging.info(vars(args))
	
	model_name = eval(args.model_name)
	data_loader = DataLoader(path=args.path, dataset=args.dataset, label=args.label, sep=args.sep)
	
	model = model_name(user_num=data_loader.user_num, item_num=data_loader.item_num,
					   u_vector_size=args.u_vector_size, i_vector_size=args.i_vector_size, model_path=args.model_path,
					   smooth_coef=args.smooth_coef, layers=args.layers, loss_func=args.loss_func
					   )
	
	# use gpu
	if torch.cuda.device_count() > 0:
		# model = model.to('cuda:0')
		model = model.cuda()
	data_processor = DataProcessor(data_loader, model, rank=False, test_neg_n=args.test_neg_n)
	runner = BaseRunner(optimizer=args.optimizer, learning_rate=args.lr, epoch=args.epoch, batch_size=args.batch_size,
						eval_batch_size=args.eval_batch_size, dropout=args.dropout, l2=args.l2, metrics=args.metric,
						check_epoch=args.check_epoch, early_stop=args.early_stop,
						loss_formula=loss_formula,
						controller=controller, controller_optimizer=controller_optimizer, args=args)
	runner.train(model, data_processor, skip_eval=args.skip_eval)
	runner.evaluate(model, data_processor.get_test_data(), data_processor)

if __name__ == '__main__':
	main()
