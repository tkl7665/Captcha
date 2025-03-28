import cv2
import pytesseract

from captcha.configs.logging import get_logger

OCR_CONFIG=r'--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJLKMNOPQRSTUVWXYZ0123456789'

log=get_logger(__name__)

def ocrImage(ifile):
	if checkTesseract():
		if os.path.exists(ifile):
			img=cv2.imread(ifile)
			gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			text=pytesseract.image_to_string(gray,config=OCR_CONFIG)
		else:
			log.warning(f'{ifile} not found')
			text='N/A'
	else:
		text='N/A'

	return text.strip()

def checkTesseract():
	try:
		pytesseract.get_tesseract_version()
		return True
	except Exception as e:
		log.warning(f'Tesseract not found: {e}')
		return False
