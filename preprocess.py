from PIL import Image
from init import *

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
		for v in values:
			cimg=img.crop((x,0,x+8,img.height))
			x+=9

			ofile=f'{odir}/{v}_{fname}'
			cimg.save(ofile)

			with open(f'{odir}/{v}.txt',mode='a+',encoding='utf-8') as o:
				o.write(f'{ofile}\n')
	else:
		log.warning(f'{label} not found')

def cropAlImages(idir,odir,sodir):
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

if __name__ == "__main__":
	log.info(f'Running preprocessing {guid}')
	cropAlImages('./samples/input/','./trainingdata','./samples/output')
