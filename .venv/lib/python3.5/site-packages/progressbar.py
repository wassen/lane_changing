# -*- coding: utf-8 -*-
import sys
import time

class ProgressBar:

	MAX_LENGTH_OF_BAR = 30

	@classmethod
	def single_generator(cls,iterable):
		pb = cls(iterable)
		for item in iterable:
			yield item
			pb.display_progressbar()

	def generator(self,index):
		for item in self.iterables[index]:
			yield item
			if index == len(self.iterables) - 1:
				self.display_progressbar()

	def display_progressbar(self, *currentNums): #, color = None


		if currentNums == ():
			currentNums = self.defaultCurrentNums;
			self.update_defaultCurrentNums();

		if not len(self.maxNums) == len(currentNums):
			print("引数の数が不正です。")
			raise

		percentage = self.get_percentage(currentNums)

		if percentage > 1:
			print("100%を超えました。引数が予期しない値であった可能性があります。")
			raise
		elif percentage == 1:
			lf = '\n'
		else :
			lf = ''

		self.time = time.time() - self.start;
		MAX_LENGTH_OF_BAR = self.__class__.MAX_LENGTH_OF_BAR
		lengthOfBar = int(MAX_LENGTH_OF_BAR * percentage)
		outputStr = "".join([
			'[',
			'='*lengthOfBar,('>' if lengthOfBar < MAX_LENGTH_OF_BAR else ''),
			' ' * (MAX_LENGTH_OF_BAR - lengthOfBar - 1),
			']',
			'{0:.1f}%'.format(percentage * 100),
			' 経過時間 ',
			self.__class__.get_h_m_s(self.time),
			lf
		])
		#\033[K カーソル位置〜行末迄をクリア
		sys.stderr.write('\r\033[K' + '\x1b[1;38;5;150m' + outputStr + '\x1b[0m')
		sys.stderr.flush()

	@classmethod
	def get_h_m_s(cls,second):
		second = int(second)
		return '{0:0>2.0f}:{1:0>2.0f}:{2:0>2.0f}'.format(second//60//60,second//60%60,second%60)

	def get_percentage(self,currentNums):
		currentNums = list(currentNums)
		#終わりを100%にするための措置
		currentNums[len(currentNums) - 1] += 1
		percentage = 0
		for i,(currentNum, maxNum) in enumerate(zip(currentNums,self.maxNums)):
			if currentNum > maxNum:
				print("最大値を超えています。")
				raise
			percentage += currentNum/self.__class__.prod(self.maxNums[:i+1])
		return percentage

	@classmethod
	def prod(cls,args):
		num = 1
		for arg in args:
			num*=arg
		return num

	def update_defaultCurrentNums(self):
		dcns = list(reversed(self.defaultCurrentNums))
		maxNums = list(reversed(self.maxNums))
		dcns[0] += 1
		for i, (maxNum, dcn) in enumerate(zip(maxNums, dcns)):
			if maxNum == dcn:
				dcns[i] = 0
				if len(dcns) > i + 1:
					dcns[i + 1] += 1
		self.defaultCurrentNums = list(reversed(dcns))

	def add(self,*args):
		if all(isinstance(arg,int) for arg in args):
			self.maxNums.extend(args)
			self.defaultCurrentNums.extend([0]*len(args))
		elif all(self.__class__.isiterable(arg) for arg in args):
			self.maxNums.extend([len(iterable) for iterable in args])
			self.defaultCurrentNums.extend([0]*len(args))
			self.iterables.extend(args)
		else:
			print("引数が不正です。")
			raise

	@classmethod
	def isiterable(cls, obj):
		return isinstance(obj, str) or hasattr(obj, '__iter__')

	def __init__(self,*args):
		self.start = time.time()
		self.time = 0
		if all(isinstance(arg,int) for arg in args):
			self.maxNums = list(args)
			self.defaultCurrentNums = [0]*len(args)
		elif all(self.__class__.isiterable(arg) for arg in args):
			self.maxNums = [len(iterable) for iterable in args]
			self.defaultCurrentNums = [0]*len(args)
			self.iterables = list(args)
		else:
			print("引数が不正です。")
			raise
		#TODO 0%の表示

#複数のプログレスバーを一度に表示することはある？
#class Color:
	#色の指定出来るようにしたい
	#じかんが流れているのに更新されていないように見えるし、別スレッドに投げたほうがいいのか？
