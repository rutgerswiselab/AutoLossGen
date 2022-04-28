from scipy.misc import derivative
import numpy as np
import random

s = ''
pos_s = ''
eps = '1e-6'


def subs(ss):
	i = 0
	return_s = ''
	prev = 0
	while i < len(ss):
		if ss[i] == '/':
			j = i + 1
			count = 0
			while j < len(ss):
				if ss[j] == '(':
					count += 1
				elif ss[j] == ')':
					count -= 1
					if count == 0:
						mid_s = subs(ss[(i+2):j])
						mid_s = '(-1 if (' + mid_s + ')<0 else 1) * (abs(' + mid_s + ')+' + eps + ')'
						# mid_s = 'np.sign(' + mid_s + ') * (abs(' + mid_s + ')+' + eps + ')'
						return_s += ss[prev:(i+2)] + mid_s + ss[j]
						i = j
						prev = i + 1
						break
				j += 1
		elif ss[i] == 'l' and ss[i:(i+3)] == 'log':
			j = i + 3
			count = 0
			while j < len(ss):
				if ss[j] == '(':
					count += 1
				elif ss[j] == ')':
					count -= 1
					if count == 0:
						mid_s = subs(ss[(i+4):j])
						mid_s = '(' + mid_s + '+' + eps + ') if (' + mid_s + ')>=0 else -(' + mid_s + ')'
						return_s += ss[prev:i] + 'np.log(' + mid_s + ss[j]
						i = j
						prev = i + 1
						break
				j += 1
		i += 1
	return_s += ss[prev:]
	return return_s
	

def pos_f(input_x):
	x = input_x
	return eval(pos_s)


def neg_f(input_x):
	x = input_x
	return eval(neg_s)


if __name__ == '__main__':
	total_test, interval = 1000, 1000
	record_dict = {}
	random.seed(0)
	X = [random.random() for _ in range(total_test)]
	X = np.clip(X, float(eps), 1 - float(eps))
	fin = open('../../log/log_0.csv', 'r')
	fout = open('../../log/check_log_0.csv', 'w')
	line = fin.readline()
	fout.write(','.join(line.split(',')[:-1] + ['valid_check'] + line.split(',')[-1:]))
	for i, line in enumerate(fin):
		origin_s = line.split('"')[1]
		fout.write(line.split('"')[0])
		if 'label' not in origin_s:
			fout.write('0,"' + origin_s + '"\n')
			continue
		s = origin_s.lower()
		s = s.replace(' ', '').replace('pred', 'x')
		s = s.replace('^2', '**2').replace('e^', 'math.e**')
		pos_s, neg_s = subs(s.replace('label', '1')), subs(s.replace('label', '0'))
		count = 0
		for x in X:
			der = derivative(pos_f, x, float(eps))
			if der < -float(eps):
				count += 1
			der = derivative(neg_f, x, float(eps))
			if der > float(eps):
				count += 1
		fout.write(str(count)+',"' + origin_s + '"\n')
		if count > int(1.2*total_test):
			if count not in record_dict:
				record_dict[count] = 0
			record_dict[count] += 1
			print(i + 1, ': ', count)
			print(origin_s)
		if (i + 1) % interval == 0:
			print(i + 1)
	fin.close()
	fout.close()
