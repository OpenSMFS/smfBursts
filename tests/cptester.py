import numpy as np
from smfbursts.cfuncs import cpburstsearch, burstsearch

times = np.fromfile('cptimes.dat', dtype=np.int64)
dets = np.fromfile('cpdetsr.dat', dtype=np.uint8)
periods = np.fromfile('cpperiods.dat', dtype=np.int64)
bg = np.fromfile('cpbg.dat', dtype=np.float64)
sbr = np.fromfile('cpsbr.dat', dtype=np.float64)
det_ids = np.unique(dets)
print(det_ids)
start, stop = cpburstsearch(times, dets, periods, bg, sbr, 5e-8, alpha=0.0001, beta=0.01)
# del start, stop, times, dets, periods, bg, sbr, det_ids
print(f'num bursts: {start.size}')
size = stop - start
if start.size > 0:
    mm = np.array([np.min(size), np.max(size)])
    print("min max", mm)
print("starts")
print(start[:10])
print("sizes")
print(size[:10])
# print("min max", mm)
