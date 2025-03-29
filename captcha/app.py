import os
import json
import argparse
from importlib.resources import files

import torch
from torchvision import transforms
from PIL import Image

from captcha.configs.shared import GUID
from captcha.configs.cleanup import CleanUpManager
from captcha.configs.logging import get_logger
from captcha.configs.cleanup import CleanUpManager

from captcha.ocr import ocrImage,checkTesseract

cleanup_mgr=CleanUpManager()
log=get_logger(__name__)

DEFAULT='CNN'
TESSERACT=False

class Captcha(object):
	'''Captcha recognition model

	Handles the loading of the CNN mode, and any OCR engines that are available

	Attributes:
		model (torch.nn.Module): Loaded recognition model
		cidx (dict): Index and Character mapping for the CNN model
	'''
	def __init__(self):
		'''Initialize Captcha classifier with trained model
		'''
		log.info('Initializing CNN model...')
		self.cidx=None
		self.model=None

		self.loadCNNClassifier()
		log.info(f'Loaded CNN model with {self.cidx}')

	def __call__(self,im_path:str,save_path:str)->str:
		'''Calling the code for reference
		Args:
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

		dir_path=os.path.dirname(save_path)
		if dir_path:
			os.makedirs(dir_path,exist_ok=True)

		with open(save_path,mode='w',encoding='utf8') as f:
			f.write(text)

		log.info(f'Final Result: {text}')
		return text

	def loadCNNClassifier(self,idir:str='captcha.models')->None:
		'''Load the classifier from the specified folder

		Loads the model weights from a PyTorch checkpoint (.pth file) and corresponding class-to-index mapping from a JSON file.
		Args:
			idir(str):Python package path containing the model and index files.
		'''
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

	def cnnPredict(self,img:str)->str:
		'''Predict a single character from the given image file

		Loads the model weights from a PyTorch checkpoint (.pth file) and corresponding class-to-index mapping from a JSON file.
		Args:
			img(str):path to the given JPG file for prediction of a single character
		Returns:
			label(str):the predicted text by the model after mapping it to the class labels

		'''
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

	def runCNN(self,ifile:str,odir:str='./captcha/output/')->str:
		'''Trigger function to classify all text within the given captcha image file

		Load the given image, crop out individual letters based on the premise of 5 characters evenly spaced within the captcha, calls a prediction for each letter, finally returning the 5 characters prediction output

		Args:
			ifile(str):path to the given JPG file for prediction of a single captcha
			odir(str):output folder to hold interim cropped images that will be deleted upon exit of the program

		Returns:
			pred(str):all 5 predicted letters
		'''
		if os.path.exists(ifile):
			cimg=self.cropImage(ifile,odir)
			lfiles=self.cropLetters(cimg,odir)
			pred=self.predictLetters(lfiles)
		else:
			pred='N/A'
		return pred

	def predictLetters(self,lfiles:list)->str:
		'''Trigger function to call the classification for each of the letters and consolidate the results together

		Iterate through the list of given image files and call the classification for a single character

		Args:
			lfiles(list):List of strings containing the paths to the cropped images of individual letters

		Returns:
			str:sequential results of the prediction
		'''
		pred=[]
		for f in lfiles:
			p=self.cnnPredict(f)
			pred.append(p[0])

		return ''.join(pred)

	def cropImage(self,ifile:str,odir:str='./captcha/output/')->str:
		'''Crop the given image down to the region that contains the captcha text

		Args:
			ifile(str):Input image file the prediction is to be made
			odir(str):Output folder at which the cropped image should be saved at

		Returns:
			ofile(str):Path to the output cropped image
		'''
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

	def cropLetters(self,ifile:str,odir:str='./captcha/output/',length:int=5)->list:
		'''Crop the given image to get the individual letters within the captcha

		Args:
			ifile(str):Input image file at which the letters are to be cropped from
			odir(str):Output folder at which the cropped letters should be saved at
			length(int):Number of letters to be cropped. defaulted at 8

		Returns:
			lfile(list):List containing the path to the individually cropped letters
		'''
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

def cnnTransform():
	'''Form the transformer to be used 

	Returns:
		transform:to be used in transforming the image
	'''
	transform=transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.Resize((10,8)),
		transforms.ToTensor(),
		transforms.Normalize((0.5,),(0.5,))
	])
	return transform

def initalize():
	'''Initialization of basic items'''
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

def main():
	'''Entry point for calling the Captcha classifier to process a given image file (.JPG only) and writes the output into the specified text file
	'''
	parser=argparse.ArgumentParser(
		description='Captcha',
		add_help=True
	)

	parser.add_argument('img_path',help='Input image path')
	parser.add_argument('save_path',help='Output save path')

	args=parser.parse_args()
	if args.img_path and args.save_path:
		log.info(f'Processing {args.img_path}')
		c=Captcha()
		c(args.img_path,args.save_path)
	else:
		parser.error('Both img path and save path are required.')

if __name__=='__main__':
	main()
