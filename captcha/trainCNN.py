import os
import json

import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,datasets

from PIL import Image

from captcha.configs.shared import GUID
from captcha.configs.cleanup import CleanUpManager
from captcha.configs.logging import get_logger
from captcha.classes.cnnModel import CharClassifier

tdata='./captcha/trainingdata/singleChar_Augment/'

log=get_logger(__name__)

def cnnTransform():
	transform=transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((10,8)),
		transforms.ToTensor(),
		transforms.Normalize((0.5,),(0.5,))
	])
	return transform

def cnnSetup(trainingData):
	transform=cnnTransform()

	idata=datasets.ImageFolder(
		root=trainingData,
		transform=transform,
	)

	trainSize=int(0.8*len(idata))
	valSize=len(idata)-trainSize

	trainData,valData=torch.utils.data.random_split(idata,[trainSize,valSize])
	batchSize=32

	trainLoader=DataLoader(trainData,batch_size=batchSize,shuffle=True)
	valLoader=DataLoader(valData,batch_size=batchSize,shuffle=True)

	return trainLoader,valLoader,idata

def cnnTrain(numClasses,trainLoader,valLoader):
	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model=CharClassifier(numClasses)
	model=model.to(device)

	criterion=torch.nn.CrossEntropyLoss()
	optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

	num_epochs=10
	for e in range(num_epochs):
		model.train()
		train_loss=0.0

		for i,l in trainLoader:
			i,l=i.to(device),l.to(device)

			optimizer.zero_grad()
			o=model(i)
			loss=criterion(o,l)
			loss.backward()
			optimizer.step()

			train_loss+=loss.item()*i.size(0)

		model.eval()
		val_loss=0.0

		correct=0
		total=0

		with torch.no_grad():
			for i,l in valLoader:
				i,l=i.to(device),l.to(device)

				o=model(i)
				loss=criterion(o,l)
				val_loss+=loss.item()*i.size(0)

				_,predicted=torch.max(o.data,1)
				total+=l.size(0)
				correct+=(predicted==l).sum().item()

		train_loss=train_loss/len(trainLoader.dataset)	
		val_loss=val_loss/len(valLoader.dataset)
		val_acc=correct/total

		log.info(f'Epoch: {e+1}/{num_epochs}')
		log.info(f'Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
	
	return model

def saveModel(model,classIdx,odir):
	torch.save(model,f'{odir}/cnnModel.pth')
	with open(f'{odir}/classIndex.json',mode='w',encoding='utf-8') as o:
		json.dump(classIdx,o)

def getCNNClassifier(idir):
	log.info(f'Training model from {idir}')
	trainLoader,valLoader,data=cnnSetup(idir)

	classIdx=data.class_to_idx
	log.info(f'Class index: {classIdx}')

	model=cnnTrain(len(classIdx),trainLoader,valLoader)
	return model,classIdx

def main(idir,odir):
	model,classIdx=getCNNClassifier(idir)
	#latest model will save within models folder
	saveModel(model,classIdx,odir)

    #save model within guid folder
	odir=f'{odir}/{GUID}/'
	os.makedirs(odir,exist_ok=True)
	saveModel(model,classIdx,odir)

	return model

if __name__ == "__main__":
	log.info(f'Running training {GUID}')
	main(tdata,'./captcha/models/')
	log.info(f'Training completed {GUID}')
