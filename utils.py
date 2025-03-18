from init import *
from PIL import Image

def extractPixels(ifile):
	results=[]
	if os.path.exists(ifile):
		img=Image.open(ifile)
		w,h=img.size

		for y in range(h):
			r=[]
			for x in range(w):
				r.append(img.getpixel((x,y)))
			results.append(r)
	else:
		log.warning(f'{ifile} not found')

	return results

