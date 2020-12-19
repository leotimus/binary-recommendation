  
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
import numpy as np
import time as time

# To do parallelism using multiprocessing, lambda functions aren't allowed
# and I can't find an easy way to accept multiple arguments, so the other 
# two just go in global variables.  ~ asg h
__currentModel = None
__currentUserId = None
var = {"__currentModel":None, "__currentUserId":None}



def topKRatings(k, model, usersId, itemsId, mtype=None):
	global __currentModel
	global __currentUserId
	global var
	topK = []
	isNFC = mtype == "NFC"
	count = 1
	itemslst = [i for i in itemsId]
	var["__currentModel"] = model
	for u in usersId:
		print("\rComputing top"+str(k)+": "+str(count)+"/"+str(len(usersId)), end="")
		ratings = []
		newratings = []
		if isNFC:
			candidate = pd.DataFrame({"customer-input":[u for i in range(len(itemsId))], "material-input":itemslst})
			ratings = list(model.predict(tf.data.Dataset.from_tensor_slices(dict(candidate)).batch(len(itemsId))))
			newratings = [(ratings[i], itemsId[i]) for i in range(len(itemsId))]
		else:
			var["__currentUserId"] = u
			for item in itemsId:
				ratings.append( __predictForCurrentUser(item))
		#	with Pool() as p:
		#		ratings = p.map(__predictForCurrentUser, itemsId)
		topK.append((u, __topk(newratings, k)))
		count += 1
	print("")
	return topK

def __predictForCurrentUser(i):
	global __currentModel
	global var
#	print([__currentUserId], [i])
	return (var["__currentModel"].predict([np.array([var["__currentUserId"]]), np.array([i])]), i)

def __topk(l,k):
	res = []
	for i in range(k):
		res.append(l[i])
		
	res.sort(reverse=True, key=(lambda x: x[0]))
	
	for i in range(k, len(l)):
		if l[i][0] > res[-1][0]:
			res.pop()
			__insertSorted(res, l[i])
	
	return res

def __insertSorted(l, val):
	i = len(l)
	current = l[-1][0]
	while i > 0 and current < val[0]:
		i -= 1
		current = l[i-1][0]
		
	l.insert(i,val)

def topKMetrics(predictions, positives, usersId, itemsId):
	nbrUser = len(usersId)
	nbrItem = len(itemsId)
	total = nbrUser * nbrItem
	
	real = set(positives)
	

	tp = 0
	fp = 0
	hits = 0	
	for u, topk in predictions:
		hit = False
		for r, i in topk:			
			if (u,i) in real:
				tp += 1
				hit = True
			else:
				fp += 1
		if(hit):
			hits+=1
#			print("hit by user: ",u)
	fn = len(real) - tp
	tn = total - tp - fp - fn
	hitRate =  hits/nbrUser
	return{"tp":tp, "tn":tn, "fp":fp, "fn":fn, "precision":tp/(tp+fp), "recall": tp/(tp+fn), "hitRate": hitRate}

def getAverage(results):
	average = {}

	for key in results[0]:
		average[key] = 0
		for result in results:
			average[key] += result[key]
		average[key] /= len(results)
	
	return average