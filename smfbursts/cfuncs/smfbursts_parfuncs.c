#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#include "smfbursts_burstwise.h"


static inline int64_t get_next_comp(PoolMutex *pool){
	int64_t cur_comp = -1;
	// lock mutex to ensure threads to not retrieve same computation
#if defined(__linux__) || defined(__APPLE__)
	if(!pthread_mutex_lock(pool->comp_mutex)){
		cur_comp = pool->cur_comp++; // advance pool cur_comp after getting current comp
		pthread_mutex_unlock(pool->comp_mutex); // release for next thread to track
	}
#elif _WIN32
	if (WaitForSingleObject(pool->comp_mutex, INFINITE) == WAIT_OBJECT_0)
	{
		cur_comp = pool->cur_comp++;
		ReleaseMutex(pool->comp_mutex);
	}
#endif
	return cur_comp;
}

// function for starting a thread 
#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search(void *in)
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search(void *in)
#endif
{
	int64_t cur_period = 0;
	PeriodThrd *T = (PeriodThrd*) in;
	Mpos *pos = NULL;
	if ( alloc_Mpos(&pos, T->m) ){
		goto exit;
	}
	while((cur_period = get_next_comp(T->comp_lock)) < T->comp_lock->num_comp){
		if (cur_period == -1){
			break;
		}
		T->err = sliding_window_burst_search(T->m, T->F, T->clk_p, T->c, 
											T->photons, T->dsize, T->dset, 
											T->periods[cur_period], 
											T->periods[cur_period+1],
											T->bg[cur_period], 
											((cur_period+1) < T->comp_lock->num_comp) ? T->bg[cur_period] : NAN, 
											T->alloc_size, pos, &T->bursts[cur_period]);
		if (T->err != 0){
			break;
		}
	}
	free_Mpos(pos);
	exit:
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

// function for starting a thread
#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search_fuse(void *in)
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search_fuse(void *in)
#endif
{
	int64_t cur_period = 0;
	PeriodThrd *T = (PeriodThrd*) in;
	Mpos *pos = NULL;
	if ( alloc_Mpos(&pos, T->m) ){
		goto exit;
	}
	while ((cur_period = get_next_comp(T->comp_lock)) < T->comp_lock->num_comp){
		if (cur_period == -1){
			break;
		}
		T->err = sliding_window_burst_search_fuse(T->m, T->F, T->clk_p, T->c,
											T->photons, T->dsize, T->dset, 
											T->periods[cur_period], 
											T->periods[cur_period+1],  T->fuse,
											T->bg[cur_period], 
											((cur_period+1) < T->comp_lock->num_comp) ? T->bg[cur_period] : NAN, 
											T->alloc_size, pos, &T->bursts[cur_period]);
		if (T->err != 0){
			break;
		}
	}
	free_Mpos(pos);
	exit:
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

// function for starting a thread 
#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search_T(void *in)
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search_T(void *in)
#endif
{
	int64_t cur_period = 0;
	PeriodThrd *T = (PeriodThrd*) in;
	Mpos *pos = NULL;
	if ( alloc_Mpos(&pos, T->m) ){
		goto exit;
	}
	while((cur_period = get_next_comp(T->comp_lock)) < T->comp_lock->num_comp){
		if (cur_period == -1){
			break;
		}
		T->err = sliding_window_burst_search_T(T->m, T->clk_p, 
											T->photons, T->dsize, T->dset, 
											T->periods[cur_period], 
											T->periods[cur_period+1],
											T->bg[cur_period], 
											((cur_period+1) < T->comp_lock->num_comp) ? T->bg[cur_period] : NAN, 
											T->alloc_size, pos, &T->bursts[cur_period]);
		if (T->err != 0){
			break;
		}
	}
	free_Mpos(pos);
	exit:
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

// function for starting a thread
#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search_T_fuse(void *in)
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search_T_fuse(void *in)
#endif
{
	int64_t cur_period = 0;
	PeriodThrd *T = (PeriodThrd*) in;
	Mpos *pos = NULL;
	if ( alloc_Mpos(&pos, T->m) ){
		goto exit;
	}
	while ((cur_period = get_next_comp(T->comp_lock)) < T->comp_lock->num_comp){
		if (cur_period == -1){
			break;
		}
		T->err = sliding_window_burst_search_T_fuse(T->m, T->clk_p, 
											T->photons, T->dsize, T->dset, 
											T->periods[cur_period], 
											T->periods[cur_period+1],
											T->fuse,
											T->bg[cur_period], 
											((cur_period+1) < T->comp_lock->num_comp) ? T->bg[cur_period] : NAN, 
											T->alloc_size, pos, &T->bursts[cur_period]);
		if (T->err != 0){
			break;
		}
	}
	free_Mpos(pos);
	exit:
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

// probably static inline
static inline PhStream *alloc_PhStream(int64_t n, int64_t nphot, int64_t *times, uint8_t *dets){
	PhStream *photons = (PhStream*) malloc(n*sizeof(PhStream));
	if (photons == NULL) return photons;
	for (int64_t i = 0; i < n; i++){
		photons[i].size = nphot;
		photons[i].pos = 0;
		photons[i].times = times;
		photons[i].dets = dets;
	}
	return photons;
}

int burst_search_sliding_window(int64_t m, double F, double clk_p, double c,
								int64_t nphot, int64_t *times, uint8_t *dets,
								int64_t dsize, uint8_t *dset, 
								int64_t nperiods, int64_t *periods, double *bg, 
								int64_t alloc_size, int64_t ncore, double fuse, int asT, Bursts **bursts){
	int64_t i;
	int64_t fuse_p = (int64_t) (fuse / clk_p);
	int error = 0;
	if (nperiods < ncore){
		ncore = nperiods;
	}
	// create threadid and mutexes, different for linux/Windows
#if defined(__linux__) || defined(__APPLE__)
	// linux/mac
	pthread_t *tid = (pthread_t*) malloc(ncore*sizeof(pthread_t));
	if (tid == NULL){
		return -1;
	}
	pthread_mutex_t *comp_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (comp_mutex == NULL){
		free(tid);
		return -1;
	}
	pthread_mutex_init(comp_mutex, NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL){
		free(tid);
		return -1;
	}
	DWORD  windowsThreadId = 0;
	HANDLE comp_mutex = CreateMutex(NULL, FALSE, NULL);
#endif
	// build PoolMutex from created mutexes, this structure keeps track of which periods have been computed along with the mutex
	PoolMutex *comp_lock = malloc(sizeof(PoolMutex));
	comp_lock->comp_mutex = comp_mutex;
	comp_lock->num_comp = nperiods;
	comp_lock->cur_comp = 0;
	PhStream *photons = alloc_PhStream(ncore, nphot, times, dets);
	PeriodThrd *threads = (PeriodThrd*) malloc(ncore*sizeof(PeriodThrd));
	*bursts = alloc_burst_array(nperiods, alloc_size);
	for (i = 0; i < ncore; i++){
		threads[i].comp_lock = comp_lock;
		threads[i].m = m;
		threads[i].F = F;
		threads[i].clk_p = clk_p;
		threads[i].c = c;
		threads[i].fuse = fuse_p;
		threads[i].photons = &photons[i];
		threads[i].dsize = dsize;
		threads[i].dset = dset;
		threads[i].periods = periods;
		threads[i].bg = bg;
		threads[i].alloc_size = alloc_size;
		threads[i].bursts = *bursts;
		threads[i].err = 0;
	}
	// start threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++){
		if (asT){
			if (fuse >= 0.0){
				pthread_create(&tid[i], NULL, thread_sliding_window_burst_search_T_fuse, (void*) &threads[i]);
			}
			else{
				pthread_create(&tid[i], NULL, thread_sliding_window_burst_search_T, (void*) &threads[i]);
			}
		}
		else{
			if (fuse >= 0.0){
				pthread_create(&tid[i], NULL, thread_sliding_window_burst_search_fuse, (void*) &threads[i]);
			}
			else{
				pthread_create(&tid[i], NULL, thread_sliding_window_burst_search, (void*) &threads[i]);
			}
		}
	}
	for (i = 0; i < ncore; i++){
		pthread_join(tid[i], NULL);
	}
	if (comp_mutex != NULL){
		pthread_mutex_destroy(comp_mutex);
		free(comp_mutex);
		comp_mutex = NULL;
	}
#elif _WIN32
	for (i = 0; i < ncore; i++){
		if (asT){
			if ( fuse >= 0.0 ){
				tid[i] = CreateThread(NULL, 0, thread_sliding_window_burst_search_T_fuse, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
			}
			else{
				tid[i] = CreateThread(NULL, 0, thread_sliding_window_burst_search_T, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
			}
		}
		else{
			if ( fuse >= 0.0 ){
				tid[i] = CreateThread(NULL, 0, thread_sliding_window_burst_search_fuse, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
			}
			else{
				tid[i] = CreateThread(NULL, 0, thread_sliding_window_burst_search, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
			}
		}
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++){
		if (tid[i] != 0) {
			CloseHandle(tid[i]);
		}
	}
	if (comp_mutex){
		CloseHandle(comp_mutex);
	}
#endif
	if ( photons != NULL ){
		free(photons);
	}
	photons = NULL;
	if (threads != NULL){
		for ( i = 0; i < ncore; i++){
			if (threads[i].err < 0){
				error = -1;
			}
		}
		free(threads);
	}
	if (comp_lock != NULL){
		free(comp_lock);
		comp_lock = NULL;
	}
	if (tid != NULL){
		free(tid);
		tid = NULL;
	}
	if ( concatenate_bursts_fuse(nperiods, *bursts) ){
		error = 1;
	}
	*bursts = (Bursts*) realloc(*bursts, sizeof(Bursts));
	if (bursts == NULL){
		error = 1;
	}
	return error;
}


#if defined(__linux__) || defined(__APPLE__)
void* thread_max_rate(void *in)
#elif _WIN32
DWORD WINAPI thread_max_rate(void *in)
#endif
{
	int64_t cur_burst = 0;
	MaxRateThrd *T = (MaxRateThrd*) in;
	Mpos *pos = NULL;
	if ( alloc_Mpos(&pos, T->m) ){
		T->err = -1;
		goto exit;
	}
	while ((cur_burst = get_next_comp(T->comp_lock)) < T->comp_lock->num_comp){
		if (cur_burst == -1){
			break;
		}
		T->max_rates[cur_burst] = max_rate(T->photons, T->dsize, T->dset, 
							T->istarts[cur_burst], T->istops[cur_burst],
							T->clk_p, pos);
	}
	free_Mpos(pos);
	exit:
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

int bursts_max_rate(int64_t m, double clk_p, int64_t nphot, int64_t *times, uint8_t *dets,
					int64_t dsize, uint8_t *dset, 
					int64_t nbursts, int64_t *istarts, int64_t *istops, 
					int64_t ncore, double *max_rates){
	int64_t i;
	int error = 0;
	if (nbursts < ncore){
		ncore = nbursts;
	}
	// create threadid and mutexes, different for linux/Windows
#if defined(__linux__) || defined(__APPLE__)
	// linux/mac
	pthread_t *tid = (pthread_t*) malloc(ncore*sizeof(pthread_t));
	if (tid == NULL){
		return -1;
	}
	pthread_mutex_t *comp_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (comp_mutex == NULL){
		free(tid);
		return -1;
	}
	pthread_mutex_init(comp_mutex, NULL);
#elif _WIN32
	// Windows
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL){
		free(tid);
		return -1;
	}
	DWORD  windowsThreadId = 0;
	HANDLE comp_mutex = CreateMutex(NULL, FALSE, NULL);
#endif
	// build PoolMutex from created mutexes, this structure keeps track of which periods have been computed along with the mutex
	PoolMutex *comp_lock = malloc(sizeof(PoolMutex));
	comp_lock->comp_mutex = comp_mutex;
	comp_lock->num_comp = nbursts;
	comp_lock->cur_comp = 0;
	PhStream *photons = alloc_PhStream(ncore, nphot, times, dets);
	MaxRateThrd *threads = (MaxRateThrd*) malloc(ncore*sizeof(MaxRateThrd));
	for (i = 0; i < ncore; i++){
		threads[i].comp_lock = comp_lock;
		threads[i].photons = &photons[i];
		threads[i].istarts = istarts;
		threads[i].istops = istops;
		threads[i].dsize = dsize;
		threads[i].dset = dset;
		threads[i].m = m;
		threads[i].clk_p = clk_p;
		threads[i].max_rates = max_rates;
		threads[i].err = 0;
	}
	// start threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++){
		pthread_create(&tid[i], NULL, thread_max_rate, (void*) &threads[i]);
	}
	for (i = 0; i < ncore; i++){
		pthread_join(tid[i], NULL);
	}
	if (comp_mutex != NULL){
		pthread_mutex_destroy(comp_mutex);
		free(comp_mutex);
		comp_mutex = NULL;
	}
#elif _WIN32
	for (i = 0; i < ncore; i++){
			tid[i] = CreateThread(NULL, 0, thread_max_rate, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++){
		if (tid[i] != 0) {
			CloseHandle(tid[i]);
		}
	}
	if (comp_mutex){
		CloseHandle(comp_mutex);
	}
#endif
	if ( photons != NULL ){
		free(photons);
	}
	photons = NULL;
	if (threads != NULL){
		for ( i = 0; i < ncore; i++){
			if (threads[i].err < 0){
				error = -1;
			}
		}
		free(threads);
	}
	if (comp_lock != NULL){
		free(comp_lock);
		comp_lock = NULL;
	}
	if (tid != NULL){
		free(tid);
		tid = NULL;
	}
	return error;
}


#if defined(__linux__) || defined(__APPLE__)
void* thread_bva(void *in)
#elif _WIN32
DWORD WINAPI thread_bva(void *in)
#endif
{
	int64_t cur_burst = 0;
	BVAThrd *T = (BVAThrd*) in;
	int64_t *counts = (int64_t*) malloc(T->max_subs*sizeof(int64_t));
	if ( counts == NULL){
		T->err = -1;
		goto exit;
	}
	while ((cur_burst = get_next_comp(T->comp_lock)) < T->comp_lock->num_comp){
		if (cur_burst == -1){
			break;
		}
		T->bvas[cur_burst] = bva(T->n, T->dets, T->dsizeAll, T->dsetAll,
									T->dsizeSub, T->dsetSub, 
									T->istarts[cur_burst], T->istops[cur_burst],
									counts);
	}
	if (counts != NULL){
		free(counts);
	}
	counts = NULL;
	exit:
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

int burst_variance_analysis(int64_t n, uint8_t *dets, 
							int64_t nbursts, int64_t *istarts, int64_t *istops,
							int64_t dsizeAll, uint8_t *dsetAll, int64_t dsizeSub, uint8_t *dsetSub,
							int64_t ncore, double *bvas){
	int64_t i;
	int error = 0;
	if (nbursts < ncore){
		ncore = nbursts;
	}
	// create threadid and mutexes, different for linux/Windows
#if defined(__linux__) || defined(__APPLE__)
	// linux/mac
	pthread_t *tid = (pthread_t*) malloc(ncore*sizeof(pthread_t));
	if (tid == NULL){
		return -1;
	}
	pthread_mutex_t *comp_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (comp_mutex == NULL){
		free(tid);
		return -1;
	}
	pthread_mutex_init(comp_mutex, NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL){
		free(tid);
		return -1;
	}
	DWORD  windowsThreadId = 0;
	HANDLE comp_mutex = CreateMutex(NULL, FALSE, NULL);
#endif
	// build PoolMutex from created mutexes, this structure keeps track of which periods have been computed along with the mutex
	PoolMutex *comp_lock = malloc(sizeof(PoolMutex));
	comp_lock->comp_mutex = comp_mutex;
	comp_lock->num_comp = nbursts;
	comp_lock->cur_comp = 0;
	int64_t max_subs = 0;
	for (i = 0; i < nbursts; i++){
		if ( max_subs < ((istops[i] - istarts[i]) / n) ){
			max_subs = (istops[i] - istarts[i]) / n;
		}
	}
	max_subs++;
	BVAThrd *threads = (BVAThrd*) malloc(ncore*sizeof(BVAThrd));
	for (i = 0; i < ncore; i++){
		threads[i].comp_lock = comp_lock;
		threads[i].dets = dets;
		threads[i].istarts = istarts;
		threads[i].istops = istops;
		threads[i].dsizeAll = dsizeAll;
		threads[i].dsizeSub = dsizeSub;
		threads[i].dsetAll = dsetAll;
		threads[i].dsetSub = dsetSub;
		threads[i].n = n;
		threads[i].bvas = bvas;
		threads[i].max_subs = max_subs;
		threads[i].err = 0;
	}
	// start threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++){
		pthread_create(&tid[i], NULL, thread_bva, (void*) &threads[i]);
	}
	for (i = 0; i < ncore; i++){
		pthread_join(tid[i], NULL);
	}
	if (comp_mutex != NULL){
		pthread_mutex_destroy(comp_mutex);
		free(comp_mutex);
		comp_mutex = NULL;
	}
#elif _WIN32
	for (i = 0; i < ncore; i++){
			tid[i] = CreateThread(NULL, 0, thread_bva, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++){
		if (tid[i] != 0) {
			CloseHandle(tid[i]);
		}
	}
	if (comp_mutex){
		CloseHandle(comp_mutex);
	}
#endif
	if (threads != NULL){
		for ( i = 0; i < ncore; i++){
			if (threads[i].err < 0){
				error = -1;
			}
		}
		free(threads);
	}
	if (comp_lock != NULL){
		free(comp_lock);
		comp_lock = NULL;
	}
	if (tid != NULL){
		free(tid);
		tid = NULL;
	}
	return error;
}


#if defined(__linux__) || defined(__APPLE__)
void* thread_cp_burst_search(void *in)
#elif _WIN32
DWORD WINAPI thread_cp_burst_search(void *in)
#endif
{
	int64_t cur_period = 0;
	CPThrd *T = (CPThrd*) in;
	while((cur_period = get_next_comp(T->comp_lock)) < T->comp_lock->num_comp){
		if (cur_period == -1){
			break;
		}
		T->err = cp_burst_search(T->alpha, T->beta, T->clk_p, T->photons,
								T->bg[cur_period], T->sbr[cur_period],
								T->periods[cur_period], T->periods[cur_period + 1],
								T->minsep, T->alloc_size, &T->bursts[cur_period]);
		if (T->err != 0){
			break;
		}
	}
#if defined(__linux__) || defined(__APPLE__)
	pthread_exit(NULL);
#elif _WIN32
	ExitThread(0);
#endif
}

// probably static inline
static inline CPStream *alloc_CPStream(int64_t n, int64_t nphot, int64_t *times, uint8_t *dets, int64_t dsize, uint8_t *dset){
	CPStream *photons = (CPStream*) malloc(n*sizeof(CPStream));
	if (photons == NULL) return photons;
	for (int64_t i = 0; i < n; i++){
		photons[i].size = nphot;
		photons[i].iprev = 0;
		photons[i].inext = 0;
		photons[i].delta = 0;
		photons[i].dsize = dsize;
		photons[i].times = times;
		photons[i].dets = dets;
		photons[i].dset = dset;
	}
	return photons;
}


int burst_search_cp(double alpha, double beta, double clk_p,
								int64_t nphot, int64_t *times, uint8_t *dets,
								int64_t dsize, uint8_t *dset, 
								int64_t nperiods, int64_t *periods, double *bg, double *sbr,
								double fuse, int64_t alloc_size, int64_t ncore, Bursts **bursts){
	int64_t i;
	int error = 0;
	if (nperiods < ncore){
		ncore = nperiods;
	}
	int64_t minsep = (int64_t) lround(fuse / clk_p);
	// create threadid and mutexes, different for linux/Windows
#if defined(__linux__) || defined(__APPLE__)
	// linux/mac
	pthread_t *tid = (pthread_t*) malloc(ncore*sizeof(pthread_t));
	if (tid == NULL){
		return -1;
	}
	pthread_mutex_t *comp_mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
	if (comp_mutex == NULL){
		free(tid);
		return -1;
	}
	pthread_mutex_init(comp_mutex, NULL);
#elif _WIN32
	HANDLE* tid = (HANDLE*)calloc(ncore, sizeof(HANDLE));
	if (tid == NULL){
		free(tid);
		return -1;
	}
	DWORD  windowsThreadId = 0;
	HANDLE comp_mutex = CreateMutex(NULL, FALSE, NULL);
#endif
	// build PoolMutex from created mutexes, this structure keeps track of which periods have been computed along with the mutex
	PoolMutex *comp_lock = malloc(sizeof(PoolMutex));
	comp_lock->comp_mutex = comp_mutex;
	comp_lock->num_comp = nperiods;
	comp_lock->cur_comp = 0;
	CPStream *photons = alloc_CPStream(ncore, nphot, times, dets, dsize, dset);
	CPThrd *threads = (CPThrd*) malloc(ncore*sizeof(CPThrd));
	*bursts = alloc_burst_array(nperiods, alloc_size);
	for (i = 0; i < ncore; i++){
		threads[i].comp_lock = comp_lock;
		threads[i].photons = &photons[i];
		threads[i].bursts = *bursts;
		threads[i].periods = periods;
		threads[i].bg = bg;
		threads[i].sbr = sbr;
		threads[i].alpha = alpha;
		threads[i].beta = beta;
		threads[i].clk_p = clk_p;
		threads[i].minsep = minsep;
		threads[i].alloc_size = alloc_size;
		threads[i].err = 0;
	}
	// start threads
#if defined(__linux__) || defined(__APPLE__)
	for (i = 0; i < ncore; i++){
		pthread_create(&tid[i], NULL, thread_cp_burst_search, (void*) &threads[i]);
	}
	for (i = 0; i < ncore; i++){
		pthread_join(tid[i], NULL);
	}
	if (comp_mutex != NULL){
		pthread_mutex_destroy(comp_mutex);
		free(comp_mutex);
		comp_mutex = NULL;
	}
#elif _WIN32
	for (i = 0; i < ncore; i++){
		tid[i] = CreateThread(NULL, 0, thread_cp_burst_search, (LPVOID) &threads[i], 0, (LPDWORD)&windowsThreadId);
	}
	WaitForMultipleObjects((DWORD)ncore, tid, TRUE, INFINITE);
	for (i = 0;  i < ncore; i++){
		if (tid[i] != 0) {
			CloseHandle(tid[i]);
		}
	}
	if (comp_mutex){
		CloseHandle(comp_mutex);
	}
#endif
	if ( photons != NULL ){
		free(photons);
	}
	photons = NULL;
	if (threads != NULL){
		for ( i = 0; i < ncore; i++){
			if (threads[i].err < 0){
				error = -1;
			}
		}
		free(threads);
	}
	if (comp_lock != NULL){
		free(comp_lock);
		comp_lock = NULL;
	}
	if (tid != NULL){
		free(tid);
		tid = NULL;
	}
	if ( concatenate_bursts_fuse(nperiods, *bursts) ){
		error = 1;
	}
	*bursts = (Bursts*) realloc(*bursts, sizeof(Bursts));
	if (bursts == NULL){
		error = 1;
	}
	return error;
}
