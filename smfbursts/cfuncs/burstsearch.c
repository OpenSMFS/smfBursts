#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "smfbursts_burstwise.h"

static inline int Xfree(void *ptr){
	if (ptr != NULL){ free(ptr); return TRUE;}
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

// Reallocate arrays to match pos, used after burst search
int finalize_bursts(Bursts *bursts){
	// NOTE: does NOT check if bursts, starts or stops is NULL
	// NOTE: on error, frees starts, stops, and sets pos/size to 0
	if (bursts->pos == 0){
		Xfree(bursts->starts);
		bursts->starts = NULL;
		Xfree(bursts->stops);
		bursts->stops = NULL;
		bursts->size = 0;
		return FALSE;
	}
	if (bursts->pos == bursts->size) return FALSE;
	int64_t *temp = (int64_t*) realloc(bursts->starts, bursts->pos*sizeof(int64_t));
	if (temp == NULL) { 
		free(bursts->starts); 
		bursts->starts = NULL; 
		free(bursts->stops); 
		bursts->stops = NULL; 
		bursts->pos = 0; bursts->size = 0; 
		return TRUE;
		}
	bursts->starts = temp;
	temp = (int64_t*) realloc(bursts->stops, bursts->pos*sizeof(int64_t));
	if (temp == NULL) { 
		free(bursts->starts); bursts->starts = NULL; 
		free(bursts->stops); bursts->stops = NULL; 
		bursts->pos = 0; bursts->size = 0; 
		return TRUE;
		}
	bursts->stops = temp;
	bursts->size = bursts->pos;
	return FALSE;
}

// free arrays in bursts and set everything else to 0
int free_bursts_fields(Bursts *bursts){
	if (bursts == NULL) return FALSE;
	Xfree(bursts->starts);
	bursts->starts = NULL;
	Xfree(bursts->stops);
	bursts->stops = NULL;
	bursts->size = 0;
	bursts->pos = 0;
	return FALSE;
}

int alloc_Mpos(Mpos **pos, int64_t m){
	Mpos *out = (Mpos*) malloc(sizeof(Mpos));
	if (out== NULL) return 1;
	out->times = (int64_t*) calloc(m, sizeof(int64_t));
	if (out->times == NULL) {free(out); out = NULL; return 1;}
	out->m = m;
	out->pos = 0;
	*pos = out;
	return 0;
}

int free_Mpos(Mpos *pos){
	if (pos != NULL){
		Xfree(pos->times);
		pos->times = NULL;
		free(pos);
	}
	return FALSE;
}

//
Bursts* alloc_burst_array(int64_t n, int64_t alloc_size){
	Bursts *bursts = (Bursts*) calloc(n, sizeof(Bursts));
	if (bursts == NULL) return NULL;
	for (int64_t i = 0; i < n; i++){
		bursts[i].starts = (int64_t*) malloc(alloc_size*sizeof(int64_t));
		if (bursts[i].starts == NULL){goto error;}
		bursts[i].stops = (int64_t*) malloc(alloc_size*sizeof(int64_t));
		if (bursts[i].stops == NULL){goto error;}
		bursts[i].size = alloc_size;
		bursts[i].pos = 0;
	}
	return bursts;
	error:
	for (int64_t i = 0; i < n; i++){
		free_bursts_fields(&bursts[i]);
	}
	return NULL;
}

// concatenate n bursts arrays, realloc called many times, used when memory is limited
int sequential_concatenate_bursts(int64_t n, Bursts *bursts){
	if (n < 2) return FALSE;
	size_t new_size = bursts[0].size;
	int64_t *temp = NULL;
	for (int64_t i = 1; i < n; i++){
		new_size += bursts[i].size;
		temp = (bursts[0].starts == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].starts, new_size*sizeof(int64_t));
		if (temp == NULL){ return TRUE; }
		bursts[0].starts = temp;
		temp = (bursts[0].stops == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].stops, new_size*sizeof(int64_t));
		if (temp == NULL) { return TRUE; }
		bursts[0].stops = temp;
		for (int64_t pi = bursts[0].size, ni = 0; ni < bursts[i].size; pi++, ni++){
			bursts[0].starts[pi] = bursts[i].starts[ni];
			bursts[0].stops[pi] = bursts[i].stops[ni];
		}
		bursts[0].size = new_size;
		bursts[0].pos = new_size;
		free_bursts_fields(&bursts[i]);
	}
	return FALSE;
}

// concatenate n bursts arrays, realloc called once, best method
int combined_concatenate_bursts(int64_t n, Bursts *bursts){
	int64_t pi, ni, i, new_size = 0;
	for (i = 0; i < n; i++) { new_size += bursts[i].size; }
	int64_t *temp = NULL;
	temp = (bursts[0].starts == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].starts, new_size*sizeof(int64_t));
	if (temp == NULL) return TRUE;
	bursts[0].starts = temp;
	temp = (bursts[0].stops == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].stops, new_size*sizeof(int64_t));
	if (temp == NULL) return TRUE;
	bursts[0].stops = temp;
	pi = bursts[0].size;
	for ( i = 1; i < n; i++){
		for (ni=0; ni < bursts[i].size; pi++, ni++){
			bursts[0].starts[pi] = bursts[i].starts[ni];
			bursts[0].stops[pi] = bursts[i].stops[ni];
		}
		free_bursts_fields(&bursts[i]);
	}
	bursts[0].size = new_size;
	bursts[0].pos = new_size;
	return FALSE;
}

// concatenate n bursts arrays, tries the combined method first, if insufficient memory, calls sequential
int concatenate_bursts(int64_t n, Bursts *bursts){
	if (combined_concatenate_bursts(n, bursts)){
		return sequential_concatenate_bursts(n, bursts);
	}
	return FALSE;
}

// combine array of SEQUENTIAL bursts together, fusing bursts at begining and end if they overlap
// method uses more memeory but is faster than combined counterpart
int sequential_concatenate_bursts_fuse(int64_t n, Bursts *bursts){
	int64_t ni, new_size = bursts[0].size, pi = bursts[0].size;
	int64_t *temp = NULL;
	for (int64_t i = 1; i < n; i++){
		new_size += bursts[i].size;
		if (new_size == 0) continue;
		if ((pi != 0) && (bursts[i].size != 0)&&(bursts[0].stops[pi-1] >= bursts[i].starts[0])){ // fuse last and first bursts
			new_size--;
		}
		// reallocate starts
		temp = (bursts[0].starts == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].starts, new_size*sizeof(int64_t));
		if (temp == NULL) return TRUE;
		bursts[0].starts = temp;
		// reallocate stops
		temp = (bursts[0].stops == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].stops, new_size*sizeof(int64_t));
		if (temp == NULL){
			temp = realloc(bursts[0].starts, bursts[0].size*sizeof(int64_t));
			if (temp != NULL) bursts[0].starts = temp;
			return TRUE;
		}
		bursts[0].stops = temp;
		// place first burst of next bursts aray to concatenate
		if ((pi != 0) && (bursts[i].size != 0)&&(bursts[0].stops[pi-1] >= bursts[i].starts[0])){ // fuse last and first bursts
			bursts[0].stops[--pi] = bursts[i].stops[0];
		}
		else {
			bursts[0].starts[pi] = bursts[i].starts[0];
			bursts[0].stops[pi] = bursts[i].stops[0];
		}
		// copy values to rest of array
		for ( pi++, ni = 1; ni < bursts[i].size; pi++, ni++){
			bursts[0].starts[pi] = bursts[i].starts[ni];
			bursts[0].stops[pi] = bursts[i].stops[ni];
		}
		free_bursts_fields(&bursts[i]);
	}
	bursts[0].size = new_size;
	bursts[0].pos = new_size;
	return FALSE;
}

// combine array of SEQUENTIAL bursts together, fusing bursts at begining and end if they overlap
// method uses more memeory but is faster than sequential counterpart
int combined_concatenate_bursts_fuse(int64_t n, Bursts *bursts){
	int64_t pi, ni, new_size = bursts[0].size;
	for (pi = 0, ni = 1; ni < n;  pi++, ni++){
		new_size += bursts[ni].size;
		// check if can fuse bursts
		if ((bursts[pi].size != 0)&&(bursts[ni].size != 0)&&(bursts[pi].stops[bursts[pi].size-1] >= bursts[ni].starts[0])){
			new_size--;
		}
	}
	int64_t *temp = NULL;
	temp = (bursts[0].starts == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].starts, new_size*sizeof(int64_t));
	if (temp == NULL) return TRUE;
	bursts[0].starts = temp;
	temp = (bursts[0].stops == NULL) ? (int64_t*) malloc(new_size*sizeof(int64_t)) : (int64_t*) realloc(bursts[0].stops, new_size*sizeof(int64_t));
	if (temp == NULL) return TRUE;
	bursts[0].stops = temp;
	pi = bursts[0].size;
	for (int64_t i = 1; i < n; i++){
		if (bursts[i].size == 0){
			continue;
		}
		if (bursts[0].stops[pi-1] >= bursts[i].starts[0]){ // fuse last burst to first burst
			bursts[0].stops[--pi] = bursts[i].stops[0];
		}
		else{
			bursts[0].starts[pi] = bursts[i].starts[0];
			bursts[0].stops[pi] = bursts[i].stops[0];
		}
		for (pi++, ni=1; ni < bursts[i].size; pi++, ni++){
			bursts[0].starts[pi] = bursts[i].starts[ni];
			bursts[0].stops[pi] = bursts[i].stops[ni];
		}
		free_bursts_fields(&bursts[i]);
	}
	bursts[0].size = new_size;
	bursts[0].pos = new_size;
	return FALSE;
}

// combine array of SEQUENTIAL bursts together, fusing bursts at begining and end if they overlap
int concatenate_bursts_fuse(int64_t n, Bursts *bursts){
	if (combined_concatenate_bursts_fuse(n, bursts)){
		return sequential_concatenate_bursts_fuse(n, bursts);
	}
	return FALSE;
}

// finds whether a given detector is in the given detset
static inline int in_set(uint8_t det, int64_t dsize, uint8_t *dset){
	for (int64_t i = 0; i < dsize; i++){ if (det == dset[i]) return TRUE;}
	return FALSE;
}

static inline int init_mpos(PhStream *photons, Mpos *pos, int64_t dsize, uint8_t *dset){
	pos->pos = 0;
	const int64_t m_minus_two = pos->m - 2;
	while ( photons->pos < photons->size ){
		if (in_set(photons->dets[photons->pos], dsize, dset)){
			pos->times[pos->pos] = photons->times[photons->pos];
			if ( pos->pos == m_minus_two ){
				break;
			}
			pos->pos++;
		}
		photons->pos++;
	}
	return FALSE;
}

// advance to next photon pair in sequence
static inline int advance_photon_delta(PhStream *photons, Mpos *pos, int64_t dsize, uint8_t *dset, int64_t *delta){
	photons->pos++;
	int64_t new_pos = (pos->pos+1) % pos->m;
	int64_t prev_pos = (new_pos + 1) % pos->m;
	while (photons->pos < photons->size){
		if (in_set(photons->dets[photons->pos], dsize, dset)){
			pos->times[new_pos] = photons->times[photons->pos];
			*delta = pos->times[new_pos] - pos->times[prev_pos];
			pos->pos = new_pos;
			return TRUE;
		}
		photons->pos++;
	}
	return FALSE;
}

// sliding window burst search
int sliding_window_burst_search(int64_t m, double F, double clk_p, double c, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 double cbg, double nbg, 
							 int64_t alloc_size, Mpos *pos, Bursts *bursts){
	int err = 0;
	int64_t dT; 
	double mindTc = (((double)(m-1)) -c)/F/cbg/clk_p;
	double mindTn = (((double)(m-1)) -c)/F/nbg/clk_p;
	int64_t mindT = (int64_t) mindTc;
	int bstate = FALSE;
	// allocate structures
	// advance through burst background periods
	while ((photons->pos < photons->size) && (photons->times[photons->pos] < cper) ) { photons->pos++; }
	init_mpos(photons, pos, dsize, dset);
	// start main loop
	while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (photons->times[photons->pos] < nper)){
		if (!bstate && (dT < mindT)) { // transition into burst
			if (bursts->pos == bursts->size){ if (extend_bursts(bursts, alloc_size)) {err = 2; break;}}
			bursts->starts[bursts->pos] = (int64_t) (pos->times[pos->pos] - mindT);
			bstate = TRUE;
		}
		else if (bstate && (dT >= mindT)){ // transition out of burst
			bursts->stops[bursts->pos] = pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] + mindT;
			bursts->pos++;
			bstate = FALSE;
		}
	}
	// new loop for going into next period
	if ( !isnan(nbg) && (photons->pos != photons->size)) {
		while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] < nper)){
			mindT = (int64_t) ((mindTn*((double) (pos->times[pos->pos]-nper)))+(mindTc*((double) (nper - pos->times[(pos->pos == 0) ? m-1: pos->pos-1])))) / ((double) dT);
			if (!bstate && (dT < mindT)) { // transition into burst
				if (bursts->pos == bursts->size){ if (extend_bursts(bursts, alloc_size)) {err = 2; break;}}
				bursts->starts[bursts->pos] = pos->times[pos->pos] - mindT;
				bstate = TRUE;
			}
			else if (bstate && (dT >= mindT)){ // transition out of burst
				bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[m-1] + mindT : pos->times[pos->pos-1] + mindT;
				bursts->pos++;
				bstate = FALSE;
			}
		}
	}
	if ( bstate &&(photons->pos == photons->size)){
		bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[pos->pos-1] + mindT : pos->times[m-1] + mindT;
		bursts->pos++;
	}
	if (!err){
		err = finalize_bursts(bursts);
	}
	return err;
}

// slinding window bursts search, but automatically detects overlapping bursts and returns fused burst arrays
int sliding_window_burst_search_fuse(int64_t m, double F, double clk_p, double c, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 double cbg, double nbg, 
							 int64_t alloc_size, Mpos *pos, Bursts *bursts){
	int err = 0;
	int64_t dT; 
	double mindTc = (((double)(m-1)) -c)/F/cbg/clk_p;
	double mindTn = (((double)(m-1)) -c)/F/nbg/clk_p;
	int64_t mindT = (int64_t) mindTc;
	int64_t nstart;
	int bstate = FALSE;
	// allocate structures
	// advance through burst background periods
	while ((photons->pos < photons->size) && (photons->times[photons->pos] < cper) ) { photons->pos++; }
	// fill pos
	init_mpos(photons, pos, dsize, dset);
	// start main loop
	while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (photons->times[photons->pos] < nper)){
		if (!bstate && (dT < mindT)) { // transition into burst
			nstart = pos->times[pos->pos] - mindT;
			if ((bursts->pos == 0) || (nstart > (bursts->stops[bursts->pos -1]))){
				if (bursts->pos == bursts->size){
					if (extend_bursts(bursts, alloc_size)){
						err = 2; break;
					}
				}
				bursts->starts[bursts->pos] = nstart;
			}
			else {
				bursts->pos--;
			}
			bstate = TRUE;
		}
		else if (bstate && (dT >= mindT)){ // transition out of burst
			bursts->stops[bursts->pos] = pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] + mindT;
			bursts->pos++;
			bstate = FALSE;
		}
	}
	// new loop for going into next period
	if ( !isnan(nbg) && (photons->pos != photons->size)) { // iterate into next bg period
		while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] < nper)){
			mindT = (int64_t) ((mindTn*((double) (pos->times[pos->pos]-nper)))+(mindTc*((double) (nper - pos->times[(pos->pos == 0) ? m-1: pos->pos-1])))) / ((double) dT);
			if (!bstate && (dT < mindT)) { // transition into burst
				nstart = pos->times[pos->pos] - mindT;
				if ((bursts->pos == 0) || (nstart < (bursts->stops[bursts->pos -1 ]))){
					if (bursts->pos == bursts->size){ if (extend_bursts(bursts, alloc_size)) {err = 2; break;}}
					bursts->starts[bursts->pos] = nstart;
				}
				else {
					bursts->pos--;
				}
				bstate = TRUE;
			}
			else if (bstate && (dT >= mindT)){ // transition out of burst
				bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[m-1] + mindT : pos->times[pos->pos-1] + mindT;
				bursts->pos++;
				bstate = FALSE;
			}
		}
	}
	if ( bstate && (photons->pos == photons->size)){
		bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[pos->pos-1] + mindT : pos->times[m-1] + mindT;
		bursts->pos++;
	}
	if (!err){
		err = finalize_bursts(bursts);
	}
	return err;
}


int sliding_window_burst_search_T(int64_t m, double clk_p, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 double mindTc, double mindTn,
							 int64_t alloc_size, Mpos *pos, Bursts *bursts){
	int err = 0;
	int64_t dT; 
	int64_t mindT = (int64_t) mindTc;
	int bstate = FALSE;
	// allocate structures
	// advance through burst background periods
	while ((photons->pos < photons->size) && (photons->times[photons->pos] < cper) ) { photons->pos++; }
	init_mpos(photons, pos, dsize, dset);
	// start main loop
	while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (photons->times[photons->pos] < nper)){
		if (!bstate && (dT < mindT)) { // transition into burst
			if (bursts->pos == bursts->size){ if (extend_bursts(bursts, alloc_size)) {err = 2; break;}}
			bursts->starts[bursts->pos] = (int64_t) (pos->times[pos->pos] - mindT);
			bstate = TRUE;
		}
		else if (bstate && (dT >= mindT)){ // transition out of burst
			bursts->stops[bursts->pos] = pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] + mindT;
			bursts->pos++;
			bstate = FALSE;
		}
	}
	// new loop for going into next period
	if ( !isnan(mindTn) && (photons->pos != photons->size)) {
		while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] < nper)){
			mindT = (int64_t) ((mindTn*((double) (pos->times[pos->pos]-nper)))+(mindTc*((double) (nper - pos->times[(pos->pos == 0) ? m-1: pos->pos-1])))) / ((double) dT);
			if (!bstate && (dT < mindT)) { // transition into burst
				if (bursts->pos == bursts->size){ if (extend_bursts(bursts, alloc_size)) {err = 2; break;}}
				bursts->starts[bursts->pos] = pos->times[pos->pos] - mindT;
				bstate = TRUE;
			}
			else if (bstate && (dT >= mindT)){ // transition out of burst
				bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[m-1] + mindT : pos->times[pos->pos-1] + mindT;
				bursts->pos++;
				bstate = FALSE;
			}
		}
	}
	if ( bstate &&(photons->pos == photons->size)){
		bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[pos->pos-1] + mindT : pos->times[m-1] + mindT;
		bursts->pos++;
	}
	if (!err){
		err = finalize_bursts(bursts);
	}
	return err;
}

// slinding window bursts search, but automatically detects overlapping bursts and returns fused burst arrays
int sliding_window_burst_search_T_fuse(int64_t m, double clk_p, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 double mindTc, double mindTn, 
							 int64_t alloc_size, Mpos *pos, Bursts *bursts){
	int err = 0;
	int64_t dT; 
	int64_t mindT = (int64_t) mindTc;
	int64_t nstart;
	int bstate = FALSE;
	// allocate structures
	// advance through burst background periods
	while ((photons->pos < photons->size) && (photons->times[photons->pos] < cper) ) { photons->pos++; }
	// fill pos
	init_mpos(photons, pos, dsize, dset);
	// start main loop
	while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (photons->times[photons->pos] < nper)){
		if (!bstate && (dT < mindT)) { // transition into burst
			nstart = pos->times[pos->pos] - mindT;
			if ((bursts->pos == 0) || (nstart > (bursts->stops[bursts->pos -1]))){
				if (bursts->pos == bursts->size){
					if (extend_bursts(bursts, alloc_size)){
						err = 2; break;
					}
				}
				bursts->starts[bursts->pos] = nstart;
			}
			else {
				bursts->pos--;
			}
			bstate = TRUE;
		}
		else if (bstate && (dT >= mindT)){ // transition out of burst
			bursts->stops[bursts->pos] = pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] + mindT;
			bursts->pos++;
			bstate = FALSE;
		}
	}
	// new loop for going into next period
	if ( !isnan(mindTn) && (photons->pos != photons->size)) { // iterate into next bg period
		while (advance_photon_delta(photons, pos, dsize, dset, &dT) && (pos->times[(pos->pos == 0) ? m-1 : pos->pos -1] < nper)){
			mindT = (int64_t) ((mindTn*((double) (pos->times[pos->pos]-nper)))+(mindTc*((double) (nper - pos->times[(pos->pos == 0) ? m-1: pos->pos-1])))) / ((double) dT);
			if (!bstate && (dT < mindT)) { // transition into burst
				nstart = pos->times[pos->pos] - mindT;
				if ((bursts->pos == 0) || (nstart < (bursts->stops[bursts->pos -1 ]))){
					if (bursts->pos == bursts->size){ if (extend_bursts(bursts, alloc_size)) {err = 2; break;}}
					bursts->starts[bursts->pos] = nstart;
				}
				else {
					bursts->pos--;
				}
				bstate = TRUE;
			}
			else if (bstate && (dT >= mindT)){ // transition out of burst
				bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[m-1] + mindT : pos->times[pos->pos-1] + mindT;
				bursts->pos++;
				bstate = FALSE;
			}
		}
	}
	if ( bstate && (photons->pos == photons->size)){
		bursts->stops[bursts->pos] = (pos->pos == 0) ? pos->times[pos->pos-1] + mindT : pos->times[m-1] + mindT;
		bursts->pos++;
	}
	if (!err){
		err = finalize_bursts(bursts);
	}
	return err;
}
// fuse bursts with overlapping stop/start values
// reallocates (finalize) bursts at the end
int fuse_bursts_inplace(Bursts *bursts, int64_t max_gap){
	if (bursts->size == 0) return FALSE;
	int64_t dest = 0;
	for (int64_t origin = 1; origin < bursts->size; origin++){
		if ((bursts->stops[dest] + max_gap) > bursts->starts[origin]){ // overlap bewteen bursts
			bursts->stops[dest] = bursts->stops[origin]; // copy stop into stop of previous bursts
		} // Note: fusing is more defined by what is not done (advance destination position, copy start) rather than what is done
		else {
			dest++; // advance current destination bursts
			bursts->starts[dest] = bursts->starts[origin];
			bursts->stops[dest] = bursts->stops[origin];
		}
	}
	bursts->pos = ++dest;
	return finalize_bursts(bursts);
}

int fuse_bursts(Bursts *inbursts, int64_t max_sep, Bursts *outbursts){
	int64_t new_size = inbursts->size;
	// determine size of new array
	for (int64_t i = 0, ii = 1; ii < inbursts->size; i++, ii++){
		if ((inbursts->stops[i] + max_sep) > inbursts->starts[ii]){
			new_size--;
		}
	}
	if (new_size < 1){
		outbursts->size = 0;
		outbursts->pos = 0;
		outbursts->starts = NULL;
		outbursts->stops = NULL;
	}
	outbursts->starts = (int64_t*) malloc(new_size*sizeof(int64_t));
	if (outbursts->starts == NULL) { return TRUE; }
	outbursts->stops = (int64_t*) malloc(new_size*sizeof(int64_t));
	if (outbursts->stops == NULL){
		free(outbursts->starts);
		outbursts->starts = NULL;
		return TRUE;
	}
	outbursts->size = new_size;
	outbursts->pos = new_size;
	outbursts->starts[0] = inbursts->starts[0];
	outbursts->stops[0] = inbursts->stops[0];
	int64_t dest = 0;
	for (int64_t i = 0; i < inbursts->size; i++){
		if ((outbursts->stops[dest] + max_sep) > inbursts->starts[i]){
			outbursts->stops[dest] = inbursts->stops[i];
		}
		else{
			outbursts->starts[++dest] = inbursts->starts[i];
			outbursts->stops[dest] = inbursts->stops[i];
		}
	}
	return FALSE;
}

static inline int64_t evalstateidx(int64_t n, uint8_t *state){
	int64_t idx = 0;
	for (int64_t i = 0; i < n; i++) idx += state[i]*1<<(n-i-1);
	return idx;
}

// finds the next time in the array where the combination of states changes
// NOTE: so that can be used in while loop, evaluates to TRUE when not at end, FALSE when all pos at end
static inline int burst_array_next_state_change(int64_t n, Bursts *bursts, uint8_t *state, int64_t *curtime){
	int64_t min = INT64_MAX;
	for (int64_t i = 0; i < n; i++){
		if (bursts[i].pos == bursts[i].size){
			continue;
		}
		if (state[i] && (bursts[i].stops[bursts[i].pos] < min)){
			min = bursts[i].stops[bursts[i].pos];
		}
		else if ( (!state[i]) && (bursts[i].starts[bursts[i].pos] < min)){
			min = bursts[i].starts[bursts[i].pos];
		}
	}
	if (min == INT64_MAX){
		return FALSE;
	}
	*curtime = min;
	for (int64_t i = 0; i < n; i++){
		if (bursts[i].pos == bursts[i].size) { continue; }
		if (bursts[i].stops[bursts[i].pos] == min){
			state[i] = FALSE;
			bursts[i].pos++;
		}
		else if (bursts[i].starts[bursts[i].pos] == min){ // else if so that 0 size bursts automatically removed
			state[i] = TRUE;
		}
	}
	return TRUE;
}

// Perform logical operation in truthtable on bursts (array of Bursts)
// expects empty bursts array in out, will start adding bursts as position pos
// must check and fuse bursts before running this program
int burst_gate(int64_t n, Bursts *bursts, uint8_t *truthtable, int64_t start, int64_t stop, int64_t alloc_size, Bursts *out){
	uint8_t *state = (uint8_t*) calloc(n, sizeof(uint8_t));
	if (state == NULL) { return TRUE; }
	int64_t i = 0;
	uint8_t bstate = FALSE, nbstate = FALSE;
	int64_t curtime;
	for (i = 0 ; i < n ; i++){ bursts[i].pos = 0; } // initialize pos of all bursts to 0
	// initalize bursts, must check if bursts begin before start, and what start condition is
	if (!burst_array_next_state_change(n, bursts, state, &curtime)){ goto final; }
	// initialize bursts
	if (curtime <= start){
		bstate = truthtable[evalstateidx(n, state)];
		if (bstate){
			out->starts[out->pos] = curtime;
		}
	 }
	 else {
		// assign first start (and stop if relevant)
		bstate = truthtable[0];
		nbstate = truthtable[evalstateidx(n, state)];
		if (bstate){
			out->starts[out->pos] = start;
			if (!nbstate){
				out->stops[out->pos] = curtime;
				out->pos++;
			}
		}
		else if ( nbstate ){
			out->starts[out->pos] = curtime;
		}
		bstate = nbstate;
	 }
	// main loop
	while(burst_array_next_state_change(n, bursts, state, &curtime)){
		nbstate = truthtable[evalstateidx(n, state)];
		if ((!bstate) && nbstate){ // start of new burst
			if (out->pos == out->size){
				if (extend_bursts(out, alloc_size)){
					Xfree(state); 
					state = NULL; 
					return TRUE;
				}
			}
			out->starts[out->pos] = curtime;
		}
		else if ( bstate && (!nbstate) ){
			out->stops[out->pos] = curtime;
			out->pos++;
		}
		bstate = nbstate;
	}
	// termination: assign final stop if still in burst
	if (bstate && (curtime < stop)){
		out->stops[out->pos] = stop;
		out->pos++;
	}
	final:
	Xfree(state);
	state = NULL;
	finalize_bursts(out);
	return FALSE;
}

// check that values in bursts are valid, ie monotonically increasing and starts all before stops
int check_bursts_valid(Bursts *bursts){
	if (bursts->starts[0] >= bursts->stops[0]){
		return TRUE;
	}
	for (int64_t ip=0, i = 1; i < bursts->size; i++, ip++)
	{
		if (bursts->starts[ip] >= bursts->starts[i]){
			return TRUE;
		}
		if (bursts->stops[ip] >= bursts->stops[i]){
			return TRUE;
		}
		if (bursts->starts[i] >= bursts->stops[i]){
			return TRUE;
		}
	}
	return FALSE;
}

int check_bursts_fused(Bursts *bursts, int64_t max_delta){
	if (bursts->size == 0) {
		return FALSE; 
	}
	if (bursts->starts[0] >= bursts->stops[0]){
		return TRUE;
	}
	for (int64_t ip=0, i = 1; i < bursts->size; i++, ip++)
	{
		if (bursts->starts[ip] >= bursts->starts[i]){
			return TRUE;
		}
		if (bursts->stops[ip] >= bursts->stops[i]){
			return TRUE;
		}
		if (bursts->starts[i] >= bursts->stops[i]){
			return TRUE;
		}
		if ((bursts->stops[ip] + max_delta) > bursts->starts[i]){ // check if fused
			return TRUE;
		}
	}
	return FALSE;
}

// return maximum rate for a given range of photon times
double max_rate(PhStream *photons, int64_t dsize, uint8_t *dset, 
				int64_t istart, int64_t istop, double clk_p, Mpos *pos){
	int64_t mindT = INT64_MAX;
	int64_t cdelta = 0;
	photons->pos = istart;
	init_mpos(photons, pos, dsize, dset);
	while ( advance_photon_delta(photons, pos, dsize, dset, &cdelta )){
		if (photons->pos >= istop){
			break;
		}
		if (cdelta < mindT){
			mindT = cdelta;
		}
	}
	if (mindT == INT64_MAX)
		return 0.0;
	return ((double)pos->m /(double)mindT)/clk_p;
}


double bva(int64_t n, uint8_t *dets, int64_t dsizeAll, uint8_t *dsetAll, 
			int64_t dsizeSub, uint8_t *dsetSub, int64_t istart, int64_t istop, 
			int64_t *counts){
	int64_t c, pos = istart, isub = 0, runsum = 0;
	while (pos < istop){
		counts[isub] = 0;
		for (c = 0; (c < n)&&(pos < istop) ; pos++){
			if (in_set(dets[pos], dsizeAll, dsetAll)){
				if (in_set(dets[pos], dsizeSub, dsetSub)){
					counts[isub]++;
				}
				c++;
			}
		}
		if ( c == n){
			runsum += counts[isub++];
		}
	}
	double avg = (double) runsum / ((double)(isub*n));
	double chunk_delta, delta_sum = 0.0;
	for (int64_t i = 0; i < isub; i++){
		chunk_delta = (((double)counts[i]) / ((double) n)) - avg;
		delta_sum += chunk_delta*chunk_delta;
	}
	return sqrt(delta_sum/(double)isub);
}

