import torch
import torch.nn as nn
import torch.nn.functional as F

class CharClassifier(nn.Module):
	def __init__(self,numClasses=36):
		super().__init__()
		self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)

		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)

		self.fc1=nn.Linear(64*2*2,128)
		self.fc2=nn.Linear(128,numClasses)

	def forward(self,x):
		x=self.pool(F.relu(self.conv1(x)))
		x=self.pool(F.relu(self.conv2(x)))

		x=torch.flatten(x,1)
		x=F.relu(self.fc1(x))
		x=self.fc2(x)

		return x
