from utils import load_pretrained_tokenizer
from classifier import Classifier
from tqdm import tqdm,trange
import torch
import torch.nn as nn
from transformers import AdamW

def train_model(train_x,train_y,model,tokenizer,BATCH_SIZE=64,num_epochs=10,device='cuda'):
	model.train()

	optimizer = AdamW(model.parameters(), lr=1e-5)
	loss_funk = nn.CrossEntropyLoss()
	for epoch in range(num_epochs):
		for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
			q_b = train_x[i:i+BATCH_SIZE]
			l_b = torch.tensor(train_y[i:i+BATCH_SIZE]).to(device)
			 
			encoded_input = tokenizer(q_b, padding=True, max_length=100,  truncation='longest_first', return_tensors="pt")
			input_ids = encoded_input['input_ids'].to(device)
			#token_type_ids = encoded_input['token_type_ids'].to(device)
			attention_mask = encoded_input['attention_mask'].to(device)

			pred = model(input_ids=input_ids, attention_mask=attention_mask)

			loss = loss_funk(pred, l_b)
			loss.backward()
			optimizer.step()
	
	return loss

	#torch.save(model.state_dict(), 'models/{}.pt'.format(model_name))

	#model.load_state_dict(torch.load('models/{}.pt'.format(model_name)))
