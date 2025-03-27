import sys,os,uuid,logging,json,time,random
import atexit,signal

import logging.config

ljson='./captcha/configs/logging.json'
if os.path.exists(ljson):
	with open(ljson,mode='r',encoding='utf-8') as i:
		logConfig=json.load(i)

	logging.config.dictConfig(logConfig)

log=logging.getLogger(__name__)

guid=uuid.uuid4().hex[:4]
OFILE_LIST=[]

def signal_handler(sig,frame):
	log.info('Ctrl+C caught. Closing')
	sys.exit(0)

def cleanup():
	log.info('Doing cleanup')
	log.info(f'Generated {OFILE_LIST}')

	for f in OFILE_LIST:
		try:
			log.info(f'Removing {f}')
			os.unlink(f)
		except Exception as e:
			log.info(f'Exception while trying to remove {f}')
			log.info(e)

	for h in logging.root.handlers[:]:
		logging.root.removeHandler(h)
		h.close()

atexit.register(lambda:cleanup())
signal.signal(signal.SIGINT,signal_handler)
