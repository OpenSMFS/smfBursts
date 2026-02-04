#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "fretbursts_burstwise.h"

double laplace_kdefunc(int64_t loc, int64_t time, double tau){
	return exp(-fabs((double)(time - loc)) / tau);
}

double gaussian_kdefunc(int64_t loc, int64_t time, double tau){
	double diff = (double)(time - loc);
	return exp( -(diff*diff) / tau);
}

double rect_kdefunc(int64_t loc, int64_t time, double tau){
	return (tau > ((double) abs(loc - time))) ? 0.0 : 1.0;
}

int kde_self(int64_t nphot, int64_t* times, double tau, int64_t lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nphot; ic++){
		trail = times[ic] - lim;
		lead = times[ic] + lim;
		while ((itrail < nphot) && (times[itrail] < trail)){ itrail++;}
		while ((ilead < nphot) && (times[ilead] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			out[ic] += func(times[ic], times[ir], tau);
		}
	}
	return FALSE;
}

int kde_self_np(int64_t nphot, int64_t stride, char* times, double tau, int64_t lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nphot; ic++){
		trail = *(int64_t*)&times[ic*stride] - lim;
		lead = *(int64_t*)&times[ic*stride] + lim;
		while ((itrail < nphot) && ((int64_t*)&times[itrail*stride] < trail)){ itrail++;}
		while ((ilead < nphot) && ((int64_t*)&times[ilead*stride] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			out[ic] += func(*(int64_t*)&times[ic*stride], *(int64_t*)&times[ir*stride], tau);
		}
	}
	return FALSE;
}

int kde_self_exclude_zero(int64_t nphot, int64_t* times, double tau, double lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nphot; ic++){
		trail = times[ic] - lim;
		lead = times[ic] + lim;
		while ((itrail < nphot) && (times[itrail] < trail)){ itrail++;}
		while ((ilead < nphot) && (times[ilead] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			if (ir == ic){
				continue;
			}
			out[ic] += func(times[ic], times[ir], tau);
		}
	}
	return FALSE;
}

int kde_self_exclude_zero_np(int64_t nphot, int64_t stride, char* times, double tau, int64_t lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nphot; ic++){
		trail = *(int64_t*)&times[ic*stride] - lim;
		lead = *(int64_t*)&times[ic*stride] + lim;
		while ((itrail < nphot) && (*(int64_t*)&times[itrail*stride] < trail)){ itrail++;}
		while ((ilead < nphot) && (*(int64_t*)&times[ilead*stride] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			if (ir == ic){
				continue;
			}
			out[ic] += func(*(int64_t*)&times[ic*stride], *(int64_t*)&times[ir*stride], tau);
		}
	}
	return FALSE;
}

int kde_other(int64_t nphot, int64_t* times, int64_t nloc, int64_t* locs, double tau, double lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nloc; ic++){
		trail = locs[ic] - lim;
		lead = locs[ic] + lim;
		while ((itrail < nphot) && (times[itrail] < trail)){ itrail++;}
		while ((ilead < nphot) && (times[ilead] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			out[ic] += func(locs[ic], times[ir], tau);
		}
	}
	return FALSE;
}

int kde_other_np(int64_t nphot, int64_t tstride, char* times, int64_t nloc, int64_t lstride, char* locs, double tau, double lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nloc; ic++){
		trail = *(int64_t*)&locs[ic*lstride] - lim;
		lead = *(int64_t*)&locs[ic*lstride] + lim;
		while ((itrail < nphot) && (*(int64_t*)&times[itrail*tstride] < trail)){ itrail++;}
		while ((ilead < nphot) && (*(int64_t*)&times[ilead*tstride] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			out[ic] += func(*(int64_t*)&locs[ic*lstride], *(int64_t*)&times[ir*tstride], tau);
		}
	}
	return FALSE;
}

int kde_other_exclude_zero(int64_t nphot, int64_t* times, int64_t nloc, int64_t* locs, double tau, double lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nloc; ic++){
		trail = locs[ic] - lim;
		lead = locs[ic] + lim;
		while ((itrail < nphot) && (times[itrail] < trail)){ itrail++;}
		while ((ilead < nphot) && (times[ilead] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			if (locs[ic] == times[ir]){
				continue;
			}
			out[ir] += func(locs[ic], times[ir], tau);
		}
	}
	return FALSE;
}

int kde_other_exclude_zero_np(int64_t nphot, int64_t tstride, char* times, int64_t nloc, int64_t lstride, char* locs, double tau, double lim, double(*func)(int64_t, int64_t, double), double* out){
	int64_t ic, ir, itrail=0, ilead = 0;
	int64_t trail, lead;
	for (ic = 0; ic < nloc; ic++){
		trail = *(int64_t*)&locs[ic*lstride] - lim;
		lead = *(int64_t*)&locs[ic*lstride] + lim;
		while ((itrail < nphot) && (*(int64_t*)&times[itrail*tstride] < trail)){ itrail++;}
		while ((ilead < nphot) && (*(int64_t*)&times[ilead*tstride] < lead)){ ilead++;}
		for (ir = itrail; ir < ilead; ir++){
			if (*(int64_t*)&times[ir*tstride] == *(int64_t*)&locs[ic*lstride]){
				continue;
			}
			out[ic] += func(*(int64_t*)&locs[ic*lstride], *(int64_t*)&times[ir*tstride], tau);
		}
	}
	return FALSE;
}
