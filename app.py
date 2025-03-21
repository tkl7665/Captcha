#need to write how to import the model and also use the output
from ocr import ocrImage

from init import *

class Captcha(object):
	def __init__(self):
		pass
		#self.ocr=OCR()

	def __call__(self,im_path,save_path):
		'''
		Algo for inference
		args:
			im_path: .jpg image path to load and to infer
			save_path: output file path to save the one-line outcome
		'''
		text=ocrImage(im_path)

		with open(save_path,mode='w',encoding='utf8') as f:
			f.write(text)

