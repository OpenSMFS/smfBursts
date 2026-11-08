// Author: Paul David Harris
// email: harrid@gmail.com
// Puropse: Compute KDE of 1D arrays

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "smfbursts_burstwise.h"

double laplace_kdefunc(const int64_t delta, const double tau){
	return exp(-fabs((double) delta) / tau);
}

double gaussian_kdefunc(const int64_t delta, const double sigma){
	return exp(-((double)delta * (double) delta) / sigma);
}

double rect_kdefunc(const int64_t delta, const double tau){
	return (tau > (fabs((double)delta))) ? 0.0 : 1.0;
}

int kde_slow(const int64_t nphot, int64_t* times, const int64_t nloc, int64_t* locs, const double tau, const int64_t lim, double(*func)(int64_t, double), double* out){
	int64_t iloc, iphot, cloc, imin = 0, imax = 0, tmin, tmax;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = locs[iloc];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (times[imin] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (times[imax] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			out[iloc] += func(times[iphot] - cloc, lim);
		}
	}
	return FALSE;
}

int kde_slow_np(const int64_t nphot, const int64_t tstride, char* times, const int64_t nloc, const int64_t lstride, char* locs, const double tau, const int64_t lim, double(*func)(int64_t, double), double* out){
	int64_t iloc, iphot, cloc, imin = 0, imax = 0, tmin, tmax;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = *(int64_t*)&locs[iloc*lstride];
		tmin = cloc - lim;
		tmax = cloc + lim;
		while ((imin < nphot) && (*(int64_t*)&times[imin*tstride] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (*(int64_t*)&times[imax*tstride] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			out[iloc] += func((*(int64_t*)&times[iphot*tstride]) - cloc, tau);
		}
	}
	return FALSE;
}

int kde_slow_exclude_zero(const int64_t nphot, int64_t* times, const int64_t nloc, int64_t* locs, const double tau, const int64_t lim, double(*func)(int64_t, double), double* out){
	int64_t iloc, iphot, cloc, imin = 0, imax = 0, tmin, tmax;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = locs[iloc];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (times[imin] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (times[imax] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			if (cloc == times[iphot]){ 
				continue;
			}
			out[iloc] += func(times[iphot] - cloc, tau);
		}
	}
	return FALSE;
}

int kde_slow_exclude_zero_np(const int64_t nphot, const int64_t tstride, char* times, const int64_t nloc, const int64_t lstride, char* locs, const double tau, const int64_t lim, double(*func)(int64_t, double), double* out){
	int64_t iloc, iphot, cloc, imin = 0, imax = 0, tmin, tmax;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = *(int64_t*)&locs[iloc*lstride];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (*(int64_t*)&times[imin*tstride] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (*(int64_t*)&times[imax*tstride] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			if (cloc == *(int64_t*)&times[iphot*tstride]){ 
				continue;
			}
			out[iloc] += func((*(int64_t*)&times[iphot*tstride]) - cloc, lim);
		}
	}
	return FALSE;
}

int kde_array(const int64_t nphot, int64_t* times, const int64_t nloc, int64_t* locs, const int64_t lim, double *kernel, double* out){
	int64_t iloc, iphot, cloc, imin = 0, imax = 0, tmin, tmax;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = locs[iloc];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (times[imin] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (times[imax] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			out[iloc] += kernel[times[iphot] - cloc + lim];
		}
	}
	return FALSE;
}

int kde_array_np(const int64_t nphot, const int64_t tstride, char* times, const int64_t nloc, const int64_t lstride, char* locs, const int64_t lim, double *kernel, double* out){
	int64_t iloc, iphot, cloc, imin = 0, imax = 0, tmin, tmax;
	for (iloc = 0; iloc < nloc; iloc++){
		cloc = *(int64_t*)&locs[iloc*lstride];
		tmin = cloc - lim; // get minimum and maximum times of range to compute in KDE
		tmax = cloc + lim;
		while ((imin < nphot) && (*(int64_t*)&times[imin*tstride] < tmin)){ imin++; } // advance until imin is index in range
		while ((imax < nphot) && (*(int64_t*)&times[imax*tstride] < tmax)){ imax++; } // advance until imax is just out of range
		for (iphot = imin; iphot < imax; iphot++){
			out[iloc] += kernel[(*(int64_t*)&times[iphot*tstride]) - cloc + lim];
		}
	}
	return FALSE;
}
