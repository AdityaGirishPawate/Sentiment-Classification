from utils import load_pretrained_tokenizer,load_pretrained_model
from get_data import get_data
from classifier import Classifier
from tqdm import tqdm,trange
import torch.nn as nn
from transformers import AdamW
from train import train_model
from eval import evaluate

def main():
	train_x,train_y,val_x,val_y=get_data(path='/media/aditya/450-GB-Disk/NLP warmup/train.csv',split=0.8)
	model = Classifier('distil_bert').to(device)
	tokenizer = load_pretrained_tokenizer('distil_bert')
	loss= train_model(train_x,train_y,model,tokenizer,BATCH_SIZE=32,num_epochs=10)
	score=evaluate(val_x,val_y,model,tokenizer,model_name='distil_bert')
	print(score)
	
if __name__ == "__main__":
	main()
	
	





