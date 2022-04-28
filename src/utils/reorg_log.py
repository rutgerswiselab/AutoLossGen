import numpy as np
filename = '../../log/log_0.txt'
epochs, trains, valids, tests = [], [], [], []
fin = open(filename, 'r')
fout = open('../../log/log_0.csv', 'w')
interval = 1
logs = fin.readlines()
fout.write('Epoch, train, validation, test, formula\n')
for i, log in enumerate(logs):
	if 'Epoch' in log:
		strs = log.strip().split(' ')
		index = 0
		while 'train' not in strs[index]:
			index += 1
		epoch = int(strs[index - 3])
		trains.append(float(strs[index + 1].split(',')[0]))
		valids.append(float(strs[index + 3]))
		tests.append(float(strs[index + 5]))
		pre_log = logs[i - 1]
		formula = '"' + pre_log.strip().split(':')[-1] + '"'
		if epoch % interval == 0:
			fout.write(','.join([str(epoch), str(np.mean(trains)), str(np.mean(valids)), str(np.mean(tests)), formula]) + '\n')
			trains, valids, tests = [], [], []
fout.close()
