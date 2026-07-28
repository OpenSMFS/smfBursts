#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "smfbursts_burstwise.h"


// finds whether a given detector is in the given detset
static inline int in_set(uint8_t det, int64_t dsize, uint8_t *dset){
	for (int64_t i = 0; i < dsize; i++){ if (det == dset[i]) return TRUE;}
	return FALSE;
}


static inline int extend_bursts(Bursts *bursts, int64_t alloc_size){
	// NOTE: does NOT free data on failure, because old data is still valid
	// does NOT check if bursts, starts, or stops is NULL
	const int64_t new_size = bursts->size + alloc_size;
	int64_t *temp = (bursts->starts != NULL)? (int64_t*) realloc(bursts->starts, new_size*sizeof(int64_t)) : (int64_t*) malloc(new_size*sizeof(int64_t));
	if (temp == NULL) return TRUE;
	bursts->starts = temp;
	temp = (bursts->stops != NULL) ? (int64_t*) realloc(bursts->stops, new_size*sizeof(int64_t)) : (int64_t*) malloc(new_size*sizeof(int64_t));
	if (temp == NULL) return TRUE;
	bursts->stops = temp;
	bursts->size = new_size;
	return FALSE;
}


static inline int next_cpphoton(CPStream *photons, int64_t nper){
	int64_t i;
	for (i = photons->inext + 1; (i < photons->size) && (photons->times[i] < nper); i++){
		if (in_set(photons->dets[i], photons->dsize, photons->dset)){
			photons->iprev = photons->inext;
			photons->inext = i;
			photons->delta = photons->times[i] - photons->times[photons->iprev];
			return TRUE;
		}
	}
	return FALSE;
}


static inline int prev_cpphoton(CPStream *photons, int64_t imin){
	int64_t i;
	for (i = photons->iprev - 1; i >= imin; i--){
		if (in_set(photons->dets[i], photons->dsize, photons->dset)){
			photons->inext = photons->iprev;
			photons->iprev = i;
			photons->delta = photons->times[photons->inext] - photons->times[i];
			return TRUE;
		}
	}
	return FALSE;
}


static inline int SPRT(int64_t cA, int64_t cB, int64_t cC, int64_t nper, CPStream *photons){
	int64_t Dn = 0, nC = 0, n = 0;
	do {
		Dn += photons->delta;
		nC = n * cC;
		if ( Dn <= nC - cA ){ Dn = 0; n = 0;}
		else if ( Dn > nC - cB ){ return TRUE; }
		n++;
	} while (next_cpphoton(photons, nper));
	return FALSE;
}


static inline int fcusum(double Sa, double Sb, double h, int64_t nper, CPStream *photons, int64_t *knprev, int64_t *knnext, int64_t *deltanext){
	double Sn = 0.0;
	do{
		Sn += Sa - (Sb * (double) photons->delta);
		if ( Sn < 0.0 ) { Sn = 0.0; }
		else if ( Sn > h ){ 
			*knprev = photons->iprev; *knnext = photons->inext; *deltanext = photons->delta; 
			return TRUE; 
			}
	} while (next_cpphoton(photons, nper));
	return FALSE;
}


static inline int rcusum(double Sa, double Sb, double h, int64_t kl, int64_t nd, CPStream *photons, int64_t *kr){
	double Sn = 0.0;
	int64_t i;
	for (i = 0; i < nd; i++){ if ( !prev_photon(photons, kl) ) { return FALSE; } }
	do{
		Sn += Sa - (Sb * (double) photons->delta);
		if ( Sn < 0.0 ) { Sn = 0.0; }
		else if ( Sn > h ){ 
			*kr = photons->inext; return TRUE; 
		}
	} while ( prev_cpphoton(photons, kl) );
	return FALSE;
}


/* ********************************************************************
 *  Yang algorithm
 * name: cp_burst(alpha, beta, clk_p, photons, dsize, dset, bg, sbr, cper, nper, alloc_size, bursts)
 * @param
 * alpha: double, probability of false positive 
 * beta: double, probability of false negative
 * clk_p: double, clock rate, photons->times*clk_p will have units of times in seconds
 * photons: *CPStream, data set to perform burst search on
 * bg: double, background rate in counts/s
 * sbf: double, signal to background ratio
 * cper: int64_t, time in units of times of start of burst search
 * nper: int64_t, time in units of times of end of burst search
 * alloc_size: int64_t, buffer extension size for bursts
 * bursts: *Bursts, bursts to store
 * @return
 * int: boolean, TRUE if ERROR
 * *********************************************************************
 */
int cpt_burst(double alpha, double beta, double clk_p, CPStream *photons, 
			double bg, double sbr, int64_t cper, int64_t nper,
			int64_t alloc_size, Bursts *bursts){
	/* compute threshold constants */
	const double A = (1 - beta) / alpha;
	const double B = beta / (1 - alpha);
	const double Ibg = bg * clk_p;
	const double I0 = (sbr - 1.0) * Ibg;
	const double I1 = I0 / exp(2.0) + Ibg;
	const double Irat = I1 / Ibg;
	const double Idiff = I1 - Ibg;
	const double KL_disc = -Idiff/I1 + log(Irat);
	const double h = (-log(alpha/3.0/(KL_disc+1.0)/(KL_disc+1.0) * log(1.0/alpha)));
	const int64_t cA = (int64_t)(log(A) / Idiff);
	const int64_t cB = (int64_t) (log(B) / Idiff);
	const int64_t cC = (int64_t)(log(Irat) / Idiff);
	const double Sa = log(Irat);
	const double Sb = Idiff;
	const int64_t nd = (int64_t) round(log(1.0/alpha)/KL_disc);
	/* Temporary variables storing indexes of outputs to SPRT and f/r-CUSUM */
	int64_t kl, kr; // kl: start of burst, kr: end of burst
	int64_t knprev, knnext, deltanext; // locations of next "in det" pair for delta
	/* Advance until inside period */
	while ((photons->inext < photons->size) && (photons->times[photons->inext] < cper)){ photons->inext++; }
	/* initialization, find start of first burst */
	if ( ! fcusum(Sa, Sb, h, nper, photons, &knprev, &knnext, &deltanext) ) { printf("failed\n");goto final; }
	kl = knnext;
	SPRT(cA, cB, cC, nper, photons);
	/* ************************************************************** *
	 *                        Main loop                               *
	 * Note compared to Yang fig 2, this shifts to start with f-CUSUM *
	 * Which puts first SPRT out of loop, but allows while condition  *
	 * to be evaluated inside while call, instead of saving variable  *
	 * f-CUSUM finds location of start of next burst                  *
	 * ************************************************************** */
	while ( fcusum(Sa, Sb, h, nper, photons, &knprev, &knnext, &deltanext) ){
		/* r-CUSUM, locates true end of burst, returns TRUE if valid burst */
		if ( rcusum(Sa, Sb, h, kl, nd, photons, &kr) ){
			/* update burst */
			if (bursts->pos == bursts->size){ extend_bursts(bursts, alloc_size); }
			bursts->starts[bursts->pos] = photons->times[kl];
			bursts->stops[bursts->pos] = photons->times[kr] + 1;
			bursts->pos++;
		}
		/* set photon positions for next burst */
		kl = knprev;
		photons->iprev = knprev;
		photons->inext = knnext;
		photons->delta = deltanext;
		/* SPRT to find max end of next burst */
		SPRT(cA, cB, cC, nper, photons);
	}
	final:
	return finalize_bursts(bursts);
}

