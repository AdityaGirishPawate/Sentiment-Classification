import numpy as np
import pandas as pd

def get_data(path,split=0.7):
	df=pd.read_csv(path)
	df=df.sample(frac=1)
	#(int)len(df)*split
	train_x=df['text'][:6500].to_list()
	train_y=np.array(df['target'][:6500])
	val_x=df['text'][6500:].to_list()
	val_y=np.array(df['target'][6500:])
	return [train_x,train_y,val_x,val_y]
