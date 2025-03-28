import logging.config
import os
import json
from importlib.resources import files
from typing import Optional

_LOGGING_CONFIGURED=False

def configure_logging(log_dir:str='logs'):
	'''Explicite logging configuration

	Args:
		log_dir:folder to store log files
	'''
	config_path=files('captcha.configs')/'logging.json'

	#load logging config
	with open(config_path,'r',encoding='utf-8') as f:
		config=json.load(f)

	#ensure logging directory exists
	os.makedirs(log_dir,exist_ok=True)

	for handler in config['handlers'].values():
		if 'filename' in handler:
			handler['filename']=os.path.join(
				log_dir,
				os.path.basename(handler['filename'])
			)

	logging.config.dictConfig(config)
	_LOGGING_CONFIGURED=True

def get_logger(name:Optional[str]=None)->logging.Logger:
	'''Get logging instance

	Args:
		name: logger name (usually __name__)

	Returns:
		configured logger
	'''
	if not _LOGGING_CONFIGURED:
		configure_logging()

	return logging.getLogger(name)
