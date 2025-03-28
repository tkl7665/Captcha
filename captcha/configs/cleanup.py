import atexit
import signal
import os
import sys
import logging
from threading import Lock

from captcha.configs.logging import get_logger

log=get_logger(__name__)

class CleanUpManager:
	_instance=None
	_lock=Lock()

	'''Explicit cleanup handler'''
	def __new__(cls):
		with cls._lock:
			if not cls._instance:
				cls._instance=super().__new__(cls)
				cls._instance._initialized=False
			return cls._instance

	def __init__(self):
		if not self._initialized:
			self.temp_files=[]
			self._register_handlers()
			self._initialized=True

	def add_temp_file(self,path:str):
		self.temp_files.append(path)

	def _cleanup(self):
		log.info('Starting cleanup')
		for f in set(self.temp_files):
			try:
				log.info(f'Removing {f}')
				os.unlink(f)
			except Exception as e:
				log.info(f'Exception while trying to remove {f}')
				log.info(e)

	def _register_handlers(self):
		atexit.register(self._cleanup)
		signal.signal(signal.SIGINT,self._signal_handler)

	def _signal_handler(self,sig,frame):
		self.log.info('Received SIGINT')
		self._cleanup()
		sys.exit()

