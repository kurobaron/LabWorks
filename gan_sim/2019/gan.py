# coding: utf-8
import numpy as np
from gen_sequential import SequentialGen
from dis_sequential import SequentialDis

class GAN:
	def __init__(self):
		self.inpGen = 200
		self.inpDis = 200
		self.generator = self.buildGenerator()
		self.discriminator = self.buildDiscriminator()

	def buildGenerator(self):
		model = SequentialGen()
		model.layersPre = [
			('Dense', self.inpGen, 100, 0.001, 0.01),('Dense', 100, 200, 0.001, 0.01)
			]
		"""('Dense', self.inpGen, 50, 0.001, 0.01),
			('LeakyRelu', 0.2), 
			('BatchNormalization', np.ones(50), np.zeros(50), 0.8, 0.001), 
			('Dense', 50, 100, 0.001, 0.01), 
			('LeakyRelu', 0.2), 
			('BatchNormalization', np.ones(100), np.zeros(100), 0.8, 0.001), 
			('Dense', 100, self.inpDis, 0.001, 0.01), 
			('Tanh', None)""" #変更点
		model.compile()
		return model

	def buildDiscriminator(self):
		model = SequentialDis()
		model.layersPre = [
			('Dense', self.inpDis, 100, 0.001, 0.01),
			('Dense', 100, 1, 0.001, 0.01), 
			('Sigmoid', None), 
			('MSELoss', None)
			]
		"""('Dense', self.inpDis, 50, 0.001, 0.01),
			('LeakyRelu', 0.2), 
			('BatchNormalization', np.ones(50), np.zeros(50), 0.8, 0.001), 
			('Dense', 50, 100, 0.001, 0.01), 
			('LeakyRelu', 0.2), 
			('BatchNormalization', np.ones(100), np.zeros(100), 0.8, 0.001), 
			('Dense', 100, 1, 0.001, 0.01), 
			('Sigmoid', None), 
			('MSELoss', None)""" #変更点
		model.compile()
		return model

	def fit(self, itersNum = 100000, batchSize = 100, validationSplit = 0.2):
		nameTrain = input('Input the name of train file : ')
		nameTest = input('Input the name of test file : ')
		fnameTrain = nameTrain + '.csv'
		fnameTest = nameTest + '.csv'
		dataTrain = np.loadtxt(fnameTrain, delimiter = ',')
		dataTest = np.loadtxt(fnameTest, delimiter = ',')
		indices = np.arange(dataTrain.shape[0])
		np.random.shuffle(indices)
		validationSize = int(dataTrain.shape[0]*validationSplit)
		xTrain, xTest = dataTrain[indices[:-validationSize], :], dataTrain[indices[-validationSize:], :]
		tTrain, tTest = dataTest[indices[:-validationSize], :], dataTest[indices[-validationSize:], :]
		itersNum = itersNum
		batchSize = batchSize
		trainSize = xTrain.shape[0]
		testSize = xTest.shape[0]
		genTrainLossList = []
		disTrainLossList = []
		genTestLossList = []
		disTestLossList = []
		iterPerEpoch = max(trainSize/batchSize, 1)
		for i in range(itersNum):
			#Discriminatorの学習
			self.discriminator.traineble = True
			batchMask = np.random.choice(trainSize, int(batchSize/2))
			xBatch = xTrain[batchMask]
			tBatch = tTrain[batchMask]
			vecVal = np.add(xBatch, tBatch)
			vecFal = np.add(xBatch, self.generator.predict(xBatch, True))
			lossVal = self.discriminator.train(vecVal, np.ones((int(batchSize/2), 1)), True)[0]
			lossFal = self.discriminator.train(vecFal, np.zeros((int(batchSize/2), 1)), True)[0]
			lossMean = 0.5*np.add(lossVal, lossFal)

			#Generatorの学習
			self.discriminator.trainable = False
			batchMask = np.random.choice(trainSize, batchSize)
			xBatch = xTrain[batchMask]
			vecFal = np.add(xBatch, self.generator.predict(xBatch, True))
			values = self.discriminator.train(vecFal, np.ones((batchSize, 1)), True)
			self.generator.backProp(values[1])

			disTrainLossList.append(lossMean)
			genTrainLossList.append(values[0])
			if i%iterPerEpoch == 0:
				#Discriminatorのテスト
				batchMask = np.random.choice(testSize, int(batchSize/2))
				xBatch = xTest[batchMask]
				tBatch = tTest[batchMask]
				vecVal = np.add(xBatch, tBatch)
				vecFal = np.add(xBatch, self.generator.predict(xBatch, False))
				lossVal = self.discriminator.train(vecVal, np.ones((int(batchSize/2), 1)), False)[0]
				lossFal = self.discriminator.train(vecFal, np.zeros((int(batchSize/2), 1)), False)[0]
				lossMean = 0.5*np.add(lossVal, lossFal)

				#Generatorのテスト
				batchMask = np.random.choice(testSize, batchSize)
				xBatch = xTest[batchMask]
				vecFal = np.add(xBatch, self.generator.predict(xBatch, False))
				values = self.discriminator.train(vecFal, np.ones((batchSize, 1)), False)
				disTestLossList.append(lossMean)
				genTestLossList.append(values[0])
				print("iter number : {}/{}, lossDisTest = {}, lossGenTest = {}".format(i, itersNum, lossMean, values[0]))

	def predict(self):
		nameSim = input('Input the name of file to simulate : ')
		fnameSim = nameSim + '.csv'
		dataSim = np.loadtxt(fnameSim, delimiter = ',')
		dataSim = np.reshape(dataSim, (1,200))
		fnameSimResult = nameSim + '_pred.csv'
		pred = self.generator.predict(dataSim, False)
		np.savetxt('./'+fnameSimResult, pred, delimiter = ',')