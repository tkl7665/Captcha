import cv2
import pytesseract

from init import *

OCR_CONFIG=r'--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJLKMNOPQRSTUVWXYZ0123456789'

def ocrImage(ifile):
	img=cv2.imread(ifile)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	text=pytesseract.image_to_string(gray,config=OCR_CONFIG)

	return text.strip()
