from PIL import Image
from ocr import ocrImage
from trainCNN import *

from init import *

DEFAULT='OCR'

class Captcha(object):
	def __init__(self):
		m,cidx=loadCNNClassifier()
		self.model=m
		self.cidx=cidx

		log.info(f'Loaded CNN model with {cidx}')

	def __call__(self,im_path,save_path):
		'''
		Algo for inference
		args:
			im_path: .jpg image path to load and to infer
			save_path: output file path to save the one-line outcome
		'''
		if os.path.exists(im_path):
			ptext=self.runCNN(im_path)
			otext=ocrImage(im_path) if checkTesseract() else ptext

			if otext==ptext:
				text=otext
				log.info('OCR and CNN matched')
			else:
				log.info(f'OCR and CNN mismatched: {otext} | {ptext}')
				log.info(f'Using {DEFAULT} as default')
				text=otext if DEFAULT=='OCR' else ptext
		else:
			log.warning(f'{im_path} not found')
			text='N/A'

		with open(save_path,mode='w',encoding='utf8') as f:
			f.write(text)

		return text

	def runCNN(self,ifile,odir='./output/'):
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
			p=cnnPredict(self.model,self.cidx,f)
			pred.append(p[0])

		return ''.join(pred)

	def cropImage(self,ifile,odir='./output/'):
		if os.path.exists(ifile):
			fname=os.path.basename(ifile)
			img=Image.open(ifile)
			cimg=img.crop((5,11,49,21))

			os.makedirs(odir,exist_ok=True)
			ofile=f'{odir}/{guid}_{fname}'
			cimg.save(ofile)
		else:
			ofile=None

		return ofile

	def cropLetters(self,ifile,odir='./output/',length=5):
		lfiles=[]
		if os.path.exists(ifile):
			img=Image.open(ifile)
			x=0

			os.makedirs(odir,exist_ok=True)
			fname=os.path.basename(ifile)
			for i in range(length):
				cimg=img.crop((x,0,x+8,img.height))
				x+=9

				ofile=f'{odir}/{guid}_{i}_{fname}'
				cimg.save(ofile)
				lfiles.append(ofile)
		else:
			save_path=None

		return lfiles

if __name__=='__main__':
	c=Captcha()
