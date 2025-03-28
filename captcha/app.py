import os
import json
from importlib.resources import files

import torch
from PIL import Image

from captcha.configs.shared import GUID
from captcha.configs.cleanup import CleanUpManager
from captcha.configs.logging import get_logger
from captcha.configs.cleanup import CleanUpManager

from captcha.ocr import ocrImage,checkTesseract
from captcha.trainCNN import cnnTransform

cleanup_mgr=CleanUpManager()
log=get_logger(__name__)

DEFAULT='CNN'
TESSERACT=False

class Captcha(object):
	def __init__(self):
		log.info('Initializing CNN model...')
		self.cidx=None
		self.model=None

		self.loadCNNClassifier()
		log.info(f'Loaded CNN model with {self.cidx}')

	def loadCNNClassifier(self,idir='captcha.models'):
		#to add in exception handling if possible
		cidxJSON=files(idir)/'classIndex.json'
		log.info(f'loading from {cidxJSON}')
		with open(cidxJSON,mode='r',encoding='utf-8') as i:
			self.cidx=json.load(i)

		mfile=files(idir)/'cnnModel.pth'
		log.info(f'loading from {mfile}')
		model=torch.load(mfile,weights_only=False)

		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model=model.to(device)

	def cnnPredict(self,img):
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		transform=cnnTransform()

		img=Image.open(img).convert('L')
		img=transform(img).unsqueeze(0).to(device)

		self.model.eval()
		with torch.no_grad():
			o=self.model(img)
			_,predicted=torch.max(o,1)

		label=[k for k,v in self.cidx.items() if v==predicted.item()]
		return label


	def __call__(self,im_path,save_path):
		'''Algo for inference
		args:
			im_path: .jpg image path to load and to infer
			save_path: output file path to save the one-line outcome
		'''
		if os.path.exists(im_path):
			ptext=self.runCNN(im_path)
			otext=ocrImage(im_path) if TESSERACT else ptext

			if TESSERACT:
				if otext==ptext:
					text=otext
					log.info('OCR and CNN matched')
				else:
					log.info(f'OCR and CNN mismatched: {otext} | {ptext}')
					log.info(f'Using {DEFAULT} as default')
					text=otext if DEFAULT=='OCR' else ptext
			else:
				text=ptext
				log.info(f'Using CNN result as Tesseract {TESSERACT}')
		else:
			log.warning(f'{im_path} not found')
			text='N/A'

		with open(save_path,mode='w',encoding='utf8') as f:
			f.write(text)

		log.info(f'Final Result: {text}')
		return text

	def runCNN(self,ifile,odir='./captcha/output/'):
		if os.path.exists(ifile):
			cimg=self.cropImage(ifile,odir)
			lfiles=self.cropLetters(cimg,odir)
			pred=self.predictLetters(lfiles)
		else:
			pred='N/A'
		return pred

	def predictLetters(self,lfiles):
		pred=[]
		for f in lfiles:
			p=self.cnnPredict(f)
			pred.append(p[0])

		return ''.join(pred)

	def cropImage(self,ifile,odir='./captcha/output/'):
		if os.path.exists(ifile):
			fname=os.path.basename(ifile)
			img=Image.open(ifile)
			cimg=img.crop((5,11,49,21))

			os.makedirs(odir,exist_ok=True)
			ofile=f'{odir}/{GUID}_{fname}'

			cimg.save(ofile)
			cleanup_mgr.add_temp_file(ofile)
		else:
			ofile=None

		return ofile

	def cropLetters(self,ifile,odir='./captcha/output/',length=5):
		lfiles=[]

		if os.path.exists(ifile):
			img=Image.open(ifile)
			x=0

			os.makedirs(odir,exist_ok=True)
			fname=os.path.basename(ifile)
			for i in range(length):
				cimg=img.crop((x,0,x+8,img.height))
				x+=9

				ofile=f'{odir}/{GUID}_{i}_{fname}'
				cimg.save(ofile)

				lfiles.append(ofile)
				cleanup_mgr.add_temp_file(ofile)
		else:
			save_path=None

		return lfiles

def initalize():
	log.info('Initializing...')

	log.info('Checking Tesseract...')
	TESSERACT=checkTesseract()

	log.info(f'Tesseract: {TESSERACT}')
	DEFAULT='OCR' if TESSERACT else 'CNN'
	log.info(f'Default: {DEFAULT}')

def changeDefault():
	global DEFAULT
	choice=input('Enter 1 for OCR, 2 for CNN:').lower().strip()

	if choice=='1':
		DEFAULT='OCR'
	elif choice=='2':
		DEFAULT='CNN'
	else:
		log.info('Invalid choice')
	log.info(f'Updated Default: {DEFAULT}')

def interactiveMode(odir):
	try:
		initalize()
		c=Captcha()
		odir=f'{odir}/{GUID}'

		os.makedirs(odir,exist_ok=True)
		fpath=os.path.abspath(odir).replace('\\','/')

		log.info(f'Saving Predictions at: {fpath}')

		run=0
		ifile=input(f'Enter image path (0 to quit <{DEFAULT}>): ').lower().strip()
		while ifile!='0':
			if len(ifile)>0:
				if ifile!='d':
					if os.path.exists(ifile):
						if os.path.isfile(ifile):
							if ifile.endswith('.jpg'):
								run+=1
								ofile=f'{odir}/prediction_{run}.txt'

								r=c(ifile,ofile)
								log.info(f'{r} written to {ofile}')
							else:
								log.info('Only JPG files are supported')
						else:
							log.info(f'{ifile} is not a file')
					else:
						log.warning(f'{ifile} not found')
				else:
					log.info(f'Current Default is {DEFAULT}')
					changeDefault()
			else:
				log.info('Empty input')

			ifile=input(f'Enter image path (0 to quit <{DEFAULT}>): ').lower().strip()

	except Exception as e:
		log.error(f'Error: {e}')

	log.info('Exiting...')

if __name__=='__main__':
	interactiveMode('./output/')
