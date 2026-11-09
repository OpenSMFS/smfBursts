import numpy as np
from numba import jit
import smfbursts.cfuncs as smc


times = [np.cumsum(np.random.poisson(90, size=np.random.randint(100, 300))) for _ in range(4)]
locs = [np.sort(np.random.randint(t.min()-100, t.max()+100, size=np.random.randint(100, 300))) for t in times]

@jit('f8(i8,f8)')
def exp(delta, tau):
    return np.exp(-np.abs(delta)/tau)

print("kde dist")
kde_single = smc.kde_dist(times[0], locs[0], 1000, 0, 100.0)
kde_multi = smc.kde_dist(times, locs, 1000, 0, 100.0)
kdef_single = smc.kde_dist(times[0], locs[0], 1000, exp, 100.0)
kdef_multi = smc.kde_dist(times, locs, 1000, exp, 100.0)

print(kde_single[:5])
print(kde_multi[0][:5])
print(kdef_single[:5])
print(kdef_multi[0][:5])

print("kde_photons")
kde = smc.kde_photons(times[0], 5e-8)
print(kde[:5])
kde = smc.kde_photons(times[0], 5e-8, drop_self=True)
print(kde[:5])
kde = smc.kde_photons(times[0], 5e-8, locs[0], drop_self=False)
print(kde[:5])
kde = smc.kde_photons(times[0], 5e-8, locs[0], drop_self=True)
print(kde[:5])
print('kde_photons with function')
kde = smc.kde_photons(times[0], 5e-8, func=exp)
print(kde[:5])
kde = smc.kde_photons(times[0], 5e-8, drop_self=True, func=exp)
print(kde[:5])
kde = smc.kde_photons(times[0], 5e-8, locs[0], drop_self=False, func=exp)
print(kde[:5])
kde = smc.kde_photons(times[0], 5e-8, locs[0], drop_self=True, func=exp)
print(kde[:5])
del kde_single, kde_multi, kdef_single, kdef_multi, kde

print('burst search')
times = np.fromfile('data/times.dat', dtype=np.int64)
dets = np.fromfile('data/dets.dat', dtype=np.uint8)
periods = np.fromfile('data/periods.dat', dtype=np.int64)
bg = np.fromfile('data/bg.dat', dtype=np.float64)
bgd = np.fromfile('data/bgd.dat', dtype=np.float64)
bga = np.fromfile('data/bga.dat', dtype=np.float64)
det_ids = np.unique(dets)
sbr = np.ones(bg.size)*20.0
print(det_ids)
start, stop = smc.cpburstsearch(times, dets, periods, bg, sbr, 5e-8, alpha=0.0001, beta=0.01)
print(f'num bursts: {start.size}')
size = stop - start
if start.size > 0:
    mm = np.array([np.min(size), np.max(size)])
    print("min max", mm)
print("starts")
print(start[:10])
print("sizes")
print(size[:10])
print("min max", mm)
print("cp for subset of dets")

strt, stp = smc.index_range(times, start[0], stop[0], 0)
istarts, istops = smc.index_ranges(times, start, stop)
print(strt, stp, istarts[0], istops[0])

start, stop = smc.cpburstsearch(times, dets, periods, bgd, sbr, 5e-8, det_ids=np.array([0,1], dtype=np.uint8), alpha=0.0001, beta=0.01)
print(f'num bursts: {start.size}')
size = stop - start
if start.size > 0:
    mm = np.array([np.min(size), np.max(size)])
    print("min max", mm)
print("starts")
print(start[:10])
print("sizes")
print(size[:10])
print("min max", mm)
print('standard burst search')
start, stop = smc.burstsearch(times, dets, periods, bg, 5e-8, m=10, F=6.0)
print(f'num bursts: {start.size}')
size = stop - start
if start.size > 0:
    mm = np.array([np.min(size), np.max(size)])
    print("min max", mm)
print("starts")
print(start[:10])
print("sizes")
print(size[:10])
print("min max", mm)

start, stop = smc.burstsearch(times, dets, periods, bgd, 5e-8, det_ids=np.array([0,1], dtype=np.uint8), m=10, F=6.0)
startb, stopb = smc.burstsearch(times, dets, periods, bga, 5e-8, det_ids=np.array([3], dtype=np.uint8), m=10, F=6.0)
print(f'num bursts: {start.size}')
size = stop - start
if start.size > 0:
    mm = np.array([np.min(size), np.max(size)])
    print("min max", mm)
print("starts")
print(start[:10])
print("sizes")
print(size[:10])
print("min max", mm)


print("max rate all dets")
mr = smc.maximum_rate(times, dets, istarts, istops, 5e-8)
print(mr[:10])
print('max rate some dets')
mr = smc.maximum_rate(times, dets, istarts, istops, 5e-8, det_ids=np.array([0,1], dtype=np.uint8))
print(mr[:10])

print('burst inverse gate')
invstart, invstop = smc.burstgate(start, stop, np.array([True, False]))
print(invstart[:5], invstop[:5])
print("burst gate and")
startg, stopg = smc.burstgate([start, startb], [stop, stopb], np.array([[False, False],[False, True]]))
print(startg[:5], stopg[:5])

print('bva')
bva = smc.burst_variance_analysis(dets, istarts, istops, np.array([0,1], dtype=np.uint8), np.array([1,], dtype=np.uint8))
print(bva[:5])