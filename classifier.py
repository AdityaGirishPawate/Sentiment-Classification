import torch
import torch.nn as nn
import os
import json
import numpy as np
import pandas as po
from tqdm import tqdm
from transformers import AdamW
from utils import load_pretrained_tokenizer,load_pretrained_model

class Classifier(nn.Module):
	def __init__(self,model_name='distil_bert'):
		super(Classifier, self).__init__()
		self.pretrained_model 	= load_pretrained_model(model_name)
		self.dropout 			= nn.Dropout(0.2)
		self.linear		= nn.Linear(768, 2)

	def forward(self, input_ids, attention_mask, token_type_ids=None):
		outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
		pooled_output = outputs[0][:, 0, :]
		pooled_output = self.dropout(pooled_output)
		logits = self.linear(pooled_output)
		return logits
