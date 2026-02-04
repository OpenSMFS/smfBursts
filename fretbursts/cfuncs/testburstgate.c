#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "fretbursts_burstwise.h"

int make_bursts(Bursts *bursts, int64_t n, int64_t starts[], int64_t stops[]){
	if (bursts->size < n){
		bursts->starts = (int64_t*) realloc(bursts->starts, n*sizeof(int64_t));
		bursts->stops = (int64_t*) realloc(bursts->stops, n*sizeof(int64_t));
		bursts->size = n;
	}
	for (int64_t i = 0; i < n; i++){
		bursts->starts[i] = starts[i];
		bursts->stops[i] = stops[i];
	}
	
	return FALSE;
}

uint8_t* copy_uint8(int64_t n, uint8_t arr[]){
	uint8_t *out = (uint8_t*) malloc(n*sizeof(uint8_t));
	for (int64_t i = 0; i < n; i++) out[i] = arr[i];
	return out;
}

int main(int argc, char **argv){
	int64_t starts0[] = {   1,   6,  11,  16,  25,  35};
	int64_t stops0[]  = {   5,  10,  15,  19,  30,  37};
	
	int64_t starts1[] = {   2,   6,  13,  17,  32};
	int64_t stops1[]  = {   4,   8,  15,  20,  38};
	
	int64_t starts2[] = {  36,  40,  44,  47,  51};
	int64_t stops2[]  = {  39,  42,  45,  49,  53};
	
	uint8_t ttand_[] = { FALSE, FALSE, FALSE,  TRUE};
	uint8_t ttor_[]  = { FALSE,  TRUE,  TRUE,  TRUE};
	uint8_t tteq_[]  = {  TRUE, FALSE, FALSE,  TRUE};
	uint8_t tton0_[] = { FALSE, FALSE,  TRUE, FALSE};
	uint8_t tton1_[] = { FALSE,  TRUE,  FALSE, FALSE};
	
	uint8_t *ttand = copy_uint8(4, ttand_);
	uint8_t *ttor  = copy_uint8(4, ttor_);
	uint8_t *tteq  = copy_uint8(4, tteq_);
	uint8_t *tton0 = copy_uint8(4, tton0_);
	uint8_t *tton1 = copy_uint8(4, tton1_);
	
	Bursts *bursts = alloc_burst_array(2, 4);
	Bursts *out = alloc_burst_array(2, 5);
	
	make_bursts(&bursts[0], 6, starts0, stops0);
	make_bursts(&bursts[1], 5, starts1, stops1);
	make_bursts(   &out[1], 5, starts2, stops2);
	//printf("size: %ld, pos: %ld, starts: %p, stops: %p\n", out->size, out->pos, out->starts, out->stops);
	burst_gate(2, bursts, ttor, 0, 36, 3, out);
	printf("%p\n", out);
	if (out != NULL ){
		printf("size: %ld, pos: %ld, starts: %p, stops: %p\n", out->size, out->pos, out->starts, out->stops);
	}
	printf("finihsed\n");
	sequential_concatenate_bursts_fuse(2, out);
	printf("size: %ld, pos: %ld, starts: %p, stops: %p\n", out->size, out->pos, out->starts, out->stops);
	out = realloc(out, sizeof(Bursts));
	for (int64_t i = 0; i < out->size; i++){
		printf("burst %2ld, start: %3ld, stop: %3ld\n", i, out->starts[i], out->stops[i]);
	}
	for (int64_t i = 0; i < 2; i++){
		free_bursts_fields(&bursts[i]);
	}
	for (int64_t i = 0; i < out->size; i++){
		printf("start: %4ld, stop: %4ld\n", out->starts[i], out->stops[i]);
	}
	free_bursts_fields(&out[0]);
	free(bursts);
	free(out);
	free(ttand);
	free(ttor);
	free(tteq);
	free(tton0);
	free(tton1);
}
