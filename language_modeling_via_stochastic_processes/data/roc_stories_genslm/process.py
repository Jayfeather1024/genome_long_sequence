import pickle
import ilm

data = pickle.load(open("test.pkl", "rb"))
print(len(data))
data = data[0:10]
print(len(data))

data = pickle.load(open("train.pkl", "rb"))
print(len(data))
data = data[0:10]
print(len(data))
