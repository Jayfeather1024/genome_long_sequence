import sys, os, re
import numpy as np
import matplotlib.pyplot as plt

cosines = {}
distances = {}
with open('log.debug.2') as fin:
    for line in fin:
        m = re.match(r'id (\d+).*?cosine similarity: ([\.\d]+).*?distance: ([\.\d]+).*?', line)
        if m:
            #import pdb; pdb.set_trace()
            id = int(m.group(1))
            if id not in cosines:
                cosines[id] = []
                distances[id] = []
            cosines[id].append(float(m.group(2)))
            distances[id].append(float(m.group(3)))

for k in cosines:
    cosines[k] = np.array(cosines[k])
    cosines[k] = (cosines[k].mean(), np.std(cosines[k]))
    distances[k] = np.array(distances[k])
    distances[k] = (distances[k].mean(), np.std(distances[k]))

k_min = min(cosines.keys())
k_max = max(cosines.keys())

xs = []
ys = []
errs = []
for k in range(k_min, k_max+1):
    xs.append(k)
    ys.append(cosines[k][0])
    errs.append(cosines[k][1])
plt.xlabel('sentence id')
plt.ylabel('cosine similarity')


   
plt.errorbar(xs, ys, yerr=errs, fmt='r*-')
plt.savefig('cosines.png')
xs = []
ys = []
errs = []
for k in range(k_min, k_max+1):
    xs.append(k)
    ys.append(distances[k][0])
    errs.append(distances[k][1])

plt.figure()

   
plt.errorbar(xs, ys, yerr=errs, fmt='r*-')
plt.xlabel('sentence id')
plt.ylabel('Euclidean distance')
plt.savefig('distances.png')
