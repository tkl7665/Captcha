import sys,os,uuid,logging,json,time,random
import atexit,signal

import logging.config

ljson='./configs/logging.json'
with open(ljson,mode='r',encoding='utf-8') as i:
	logConfig=json.load(i)

logging.config.dictConfig(logConfig)
log=logging.getLogger(__name__)

guid=uuid.uuid4().hex[:4]

def signal_handler(sig,frame):
	log.info('Ctrl+C caught. Closing')
	sys.exit(0)

def cleanup():
	log.info('Doing cleanup')
	for h in logging.root.handlers[:]:
		logging.root.removeHandler(h)
		h.close()

atexit.register(lambda:cleanup())
signal.signal(signal.SIGINT,signal_handler)
