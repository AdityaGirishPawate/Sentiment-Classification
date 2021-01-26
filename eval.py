from utils import evaluate_classifier
from tqdm import tqdm
import torch
import numpy as np

def evaluate(val_x,val_y,model,tokenizer,model_name='distil_bert',val_BATCH_SIZE=16):
	y_pred = []
	y_true = []
	for i in tqdm(range(0, len(val_x) , val_BATCH_SIZE)):
		q_b = val_x[i:i+val_BATCH_SIZE]
		l_b = torch.tensor(val_y[i:i+val_BATCH_SIZE]).unsqueeze(0)
		
		encoded_input = tokenizer(q_b, padding=True, max_length=100,  truncation='longest_first', return_tensors="pt")
		input_ids = encoded_input['input_ids'].to(device)
		#token_type_ids = encoded_input['token_type_ids']
		attention_mask = encoded_input['attention_mask'].to(device)

		pred = model(input_ids=input_ids, attention_mask=attention_mask)

		y_pred += np.argmax(pred.cpu().detach().numpy(), axis = 1).tolist()
		y_true += l_b.detach().numpy().squeeze().tolist()

	return evaluate_classifier(model_name, y_true, y_pred)
