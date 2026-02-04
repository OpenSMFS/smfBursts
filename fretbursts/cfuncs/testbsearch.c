#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "fretbursts_burstwise.h"

//
// Functions for testing
//
int read_file(char *fname, long *size, void **out){
	FILE *fid = fopen(fname, "rb");
	if (fid == NULL) return TRUE;
	fseek(fid, 0, SEEK_END);
	*size = ftell(fid);
	rewind(fid);
	*out = malloc(*size);
	if (*out == NULL){fclose(fid); return TRUE;}
	fread(*out, sizeof(char), *size, fid);
	fclose(fid);
	return FALSE;
}

int main(int argc, char **argv){
	if (argc != 5) return 0;
	// load files
	int64_t *times = NULL, *periods = NULL;
	double *bg = NULL;
	uint8_t *dets = NULL;
	long ltimes = 0, ldets = 0, lperiods = 0, lbg = 0;
	int64_t dsize = 4;
	int64_t m = 10;
	double F = 6.0;
	double clk_p = 5e-8;
	int64_t c = -1;
	int64_t ncore = 1;
	int fuse = TRUE;
	if (read_file(argv[1], &ltimes, (void*) &times)) return 1;
	if (read_file(argv[2], &ldets, (void*) &dets)) {free(times); return 1;}
	if (read_file(argv[3], &lperiods, (void*) &periods)) {free(dets); free(times); return 1;}
	if (read_file(argv[4], &lbg, (void*) &bg)) {free(periods); free(dets); free(times); return 1;}
	// check matching lengths
	if ((ltimes % sizeof(int64_t) != 0)||(ldets % sizeof(uint8_t) != 0)||(lperiods % sizeof(int64_t) != 0)||(lbg % sizeof(double) != 0)){
		free(bg);
		free(periods);
		free(dets);
		free(times);
		return 1;
	}
	if (ltimes / sizeof(int64_t) != (ldets / sizeof(int8_t))){
		free(bg);
		free(periods);
		free(dets);
		free(times);
		return 1;
	}
	if (((lperiods / sizeof(int64_t)) - 1) != (lbg / sizeof(double))){
		free(bg);
		free(periods);
		free(dets);
		free(times);
		return 1;
	}
	int64_t nphot = ltimes / sizeof(uint64_t);
	int64_t nper = lbg / sizeof(double);
	
	uint8_t *dset = (uint8_t*) malloc(4*sizeof(uint8_t));
	int res = 0;
	for (int64_t i = 0; i < dsize; i++){
		dset[i] = i;
	}
	Bursts **obursts = (Bursts**) calloc(1, sizeof(Bursts*));
	res = burst_search_sliding_window(m, F, clk_p, c, nphot, times, dets,
										dsize, dset, nper, periods, bg, 
										500, ncore, fuse, obursts);
// //
// // Test for non-parallel burst search
// //
/*
	PhStream *photons = (PhStream*) malloc(sizeof(PhStream));
	if (photons == NULL){
		free(bg);
		free(periods);
		free(dets);
		free(times);
		return 1;
	}
	photons->size = nphot;
	photons->pos = 0;
	photons->times = times;
	photons->dets = dets;
	//prinf("allocating bursts\n");
	Bursts *bursts = alloc_burst_array(lbg, 1000);
		
	Mpos *pos;
	alloc_Mpos(&pos, m);
	for (int64_t i = 0; i < nper; i++){
		if ((i+1) != lbg){
			res = sliding_window_burst_search(m, F, clk_p, c, photons, dsize, dset, periods[i], periods[i+1], bg[i], NAN, 500, pos, &bursts[i]);
		}
		else{
			res = sliding_window_burst_search(m, F, clk_p, c, photons, dsize, dset, periods[i], periods[i+1], bg[i], bg[i+1], 500, pos, &bursts[i]);
		}
		finalize_bursts(&bursts[i]);
		//if (i < 3){
		//	printf("----- PERIODS %ld -----\n", i);
		//	for (size_t p = 0; (p < 10) && (p < bursts[i].size); p++){
		//		printf("start: %ld, stop: %ld\n", bursts[i].starts[p], bursts[i].stops[p]);
		//	}
		//}
		if (res){
			printf("ERROR in burst search\n");
			break;
		}
	} 
	free_Mpos(pos);
	if (res){
		goto frees;
	}
	concatenate_bursts_fuse(nper, bursts);
	printf("Before fuse check bursts valid is %d, is fused %d, pos: %ld\n", check_bursts_valid(bursts), check_bursts_fused(bursts, 0), bursts->pos);
	*/
	// End of single threaded type
	Bursts *bursts = obursts[0];
	
	//printf("bursts size: %ld, pos %ld\n", bursts->size, bursts->pos);
	//for (int64_t i = bursts->size - 10; (i < bursts->size) ; i++){
	//		printf("burst %ld: start: %ld, stop: %ld\n", i, bursts->starts[i], bursts->stops[i]);
	//}
	//printf("total bursts before fuse: %ld\n", bursts->size);
	//fuse_bursts(bursts, 1);
	//printf("After fuse check bursts valid is %d, is fused %d, pos: %ld\n", check_bursts_valid(bursts), check_bursts_fused(bursts, 1), bursts->pos);
	//printf("total bursts after fuse: %ld\n", bursts->size);
	//for (int64_t i = bursts->size-10; (i < bursts->size) ; i++){
	//		printf("burst %ld: start: %ld, stop: %ld\n", i, bursts->starts[i], bursts->stops[i]);
	//}
	int64_t *istarts = (int64_t*) malloc(bursts->size*sizeof(int64_t));
	int64_t *istops  = (int64_t*) malloc(bursts->size*sizeof(int64_t));
	double *maxrates = (double*) malloc(bursts->size*sizeof(int64_t));
	double *bvas = (double*) malloc(bursts->size*sizeof(int64_t));
	int64_t pos = 0;
	for (int64_t i = 0; i < bursts->size; i++){
		for( ; times[pos] < bursts->starts[i]; pos++){}
		istarts[i] = pos;
		for( ; (times[pos] < bursts->stops[i])&&(pos < nphot) ; pos++){}
		istops[i] = pos;
	}
	int64_t dsubsize = 1;
	int8_t *dsubset = (int8_t*) malloc(dsubsize*sizeof(int8_t));
	dsubset[0] = 0;
	int64_t dallsize = 2;
	int8_t *dallset = (int8_t*) malloc(dallsize*sizeof(int8_t));
	dallset[0] = 0;
	dallset[1] = 1;
	//printf("start burst max rate\n");
	bursts_max_rate(m, clk_p, nphot, times, dets, dsize, dset, bursts->size, istarts, istops, ncore, maxrates);
	//printf("start BVA\n");
	burst_variance_analysis(m, dets, bursts->size, istarts, istops, dallsize, dallset, dsubsize, dsubset, ncore, bvas);
	//printf("finish post-process\n");
	for (int64_t i = 0; (i < 10) && (i < bursts->size); i++){
		printf("burst %12ld | istart: %12ld, istop: %12ld, start: %12ld, stop: %12ld, maxrate: %4f, bva: %4f\n", i, istarts[i], istops[i], bursts->starts[i], bursts->stops[i], maxrates[i], bvas[i]);
	}
	for (int64_t i = bursts->size-10; i < bursts->size; i++){
		printf("burst %12ld | istart: %12ld, istop: %12ld, start: %12ld, stop: %12ld, maxrate: %4f, bva: %4f\n", i, istarts[i], istops[i], bursts->starts[i], bursts->stops[i], maxrates[i], bvas[i]);
	}
	Bursts fused;
	fuse_bursts(bursts, 1, &fused);
	printf("fused.size = %ld\n", fused.size);
	free(istarts);
	free(istops);
	free(maxrates);
	free(bvas);
	free(dsubset);
	free(sallset);
	frees:
	free(dset);
/*	for (int64_t i = 0; i < lbg; i++){
		free_bursts_fields(&bursts[i]);
	}*/
	free_bursts_fields(bursts);
	free_bursts_fields(&fused);
	free(obursts);
	free(bursts);
//	free(photons);
	free(bg);
	free(periods);
	free(dets);
	free(times);
	return 1;
}


