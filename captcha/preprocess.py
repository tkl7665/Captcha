from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .init import *

def cropImage(fname,idir,odir):
	ifile=f'{idir}/{fname}'
	if os.path.exists(ifile):
		img=Image.open(ifile)
		cimg=img.crop((5,11,49,21))

		ofile=f'c_{fname}'
		cimg.save(f'{odir}/{ofile}')
	else:
		ofile=None

	return ofile

def generateSingleFile(fname,label,odir):
	if os.path.exists(label):
		with open(label,mode='r',encoding='utf-8') as i:
			values=i.read().strip()

		x=0
		img=Image.open(f'{odir}/{fname}')

		ofolder=f'{odir}/singleChar/'
		for v in values:
			cimg=img.crop((x,0,x+8,img.height))
			x+=9

			sfolder=f'{ofolder}/{v}/'
			os.makedirs(sfolder,exist_ok=True)

			ofile=f'{sfolder}/{v}_{fname}'
			cimg.save(ofile)

			with open(f'{odir}/{v}.txt',mode='a+',encoding='utf-8') as o:
				o.write(f'{ofile}\n')
	else:
		log.warning(f'{label} not found')

def cropAllImages(idir,odir,sodir):
	for f in os.listdir(idir):
		if f.endswith('.jpg') or f.endswith('.JPG'):
			ofile=cropImage(f,idir,odir)
			if ofile:
				label=f.lower().replace('.jpg','.txt').replace('input','output')
				label=f'{sodir}/{label}'
				generateSingleFile(ofile,label,odir)
			else:
				log.warning(f'{f} not found')
		else:
			log.debug(f'{f} not an image file')

def getAugmentPipeline():
	pipeline=A.Compose([
		A.RandomBrightnessContrast(
			brightness_limit=(-0.1,0.1),
			contrast_limit=(-0.1,0.1),
			p=0.05
		),
		A.GaussNoise(
			var_limit=(5.0,10.0),
			p=0.2
		),
		A.PixelDropout(
			dropout_prob=0.01,
			per_channel=False,
			p=0.25
		),
		A.Normalize(mean=[0.5],std=[0.5]),
		ToTensorV2()
	])

	return pipeline

def augmentImage(ifile,pipeline,ofile):
	img=cv2.imread(ifile)
	aimg=np.array(pipeline(image=img)['image'])

	aimg=np.transpose(aimg,(1,2,0))
	aimg=(aimg*255).astype(np.uint8)
	cv2.imwrite(ofile,aimg)

def triggerAugment(idir,tCount=100):
	pipeline=getAugmentPipeline()

	for f in os.listdir(idir):
		ifolder=f'{idir}/{f}'
		#count number of files in the folder
		o=os.listdir(ifolder)
		for i in range(tCount-len(o)):
			ifile=f'{ifolder}/{o[0]}'
			ofile=f'{ifolder}/{f}_aug_{i}.jpg'
			augmentImage(ifile,pipeline,ofile)

if __name__ == "__main__":
	log.info(f'Running preprocessing {guid}')
	cropAllImages('./captcha/samples/input/','./captcha/trainingdata','./captcha/samples/output')
