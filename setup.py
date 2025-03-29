from setuptools import setup,find_packages

setup(
	name='Captcha',
	version='1.0.8',
	packages=find_packages(),
	include_package_data=True,
	package_data={
		'captcha':[
			'configs/*',
			'models/*'
		]
	},
	install_requires=[
		'albucore',
		'albumentations',
		'annotated-types',
		'concurrent-log-handler',
		'filelock',
		'fsspec',
		'Jinja2',
		'MarkupSafe',
		'mpmath',
		'networkx',
		'numpy',
		'opencv-python-headless',
		'packaging',
		'pillow',
		'portalocker',
		'pydantic',
		'pydantic_core',
		'pytesseract',
		'pywin32',
		'PyYAML',
		'scipy',
		'simsimd',
		'stringzilla',
		'sympy',
		'torch',
		'torchvision',
		'typing-inspection',
		'typing_extensions'
	],
	entry_points={
		'console_scripts':[
			'captcha=captcha.app:main',
		],
	},
)
