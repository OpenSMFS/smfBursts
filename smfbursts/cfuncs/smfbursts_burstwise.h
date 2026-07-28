#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <pthread.h>
#elif _WIN32
#include <windows.h>
#endif

#define TRUE 1
#define FALSE 0


typedef struct{
	int64_t size; // number of photons in the stream
	int64_t pos; // current positon in photon stream being analyzed, should be initialized to 0
	int64_t *times; // arrival times of photons, has size nphot
	uint8_t *dets; // detector index of photons, has size nphot
} PhStream;

typedef struct{
	int64_t size;
	int64_t iprev;
	int64_t inext;
	int64_t delta;
	int64_t dsize;
	int64_t *times;
	uint8_t *dets;
	uint8_t *dset;
}CPStream;


typedef struct{
	int64_t m;
	int64_t pos;
	int64_t *times;
} Mpos;

typedef struct{
	int64_t size; // 
	int64_t pos;
	int64_t *starts;
	int64_t *stops;
} Bursts;

typedef struct
{
	int64_t cur_comp; // next burst to work on
	int64_t num_comp; // total number of bursts in set
#if defined(__linux__) || defined(__APPLE__)
	pthread_mutex_t *comp_mutex; // mutex for checking on cur_burst
#elif _WIN32
	HANDLE comp_mutex; // mutex for checking on cur_burst
#endif
} PoolMutex;


typedef struct{
	PoolMutex *comp_lock;
	PhStream *photons;
	Bursts *bursts;
	int64_t *periods;
	double *bg;
	uint8_t *dset;
	int64_t m;
	double F;
	double clk_p;
	double c;
	int64_t fuse;
	int64_t dsize;
	int64_t alloc_size;
	int err;
	
} PeriodThrd;

typedef struct{
	PoolMutex *comp_lock;
	CPStream *photons;
	Bursts *bursts;
	int64_t *periods;
	double *bg;
	double *sbr;
	double alpha;
	double beta;
	double clk_p;
	int64_t minsep;
	int64_t alloc_size;
	int err;
}CPThrd;

typedef struct{
	PoolMutex *comp_lock;
	PhStream *photons;
	int64_t *istarts;
	int64_t *istops;
	uint8_t *dset;
	double *max_rates;
	int64_t dsize;
	int64_t m;
	double clk_p;
	int err;
} MaxRateThrd;

typedef struct{
	PoolMutex *comp_lock;
	uint8_t *dets;
	int64_t *istarts;
	int64_t *istops;
	uint8_t *dsetAll;
	uint8_t *dsetSub;
	double *bvas;
	int64_t n;
	int64_t dsizeAll;
	int64_t dsizeSub;
	int64_t max_subs;
	int err;
} BVAThrd;

int finalize_bursts(Bursts *bursts);
int free_bursts_fields(Bursts *bursts);
int alloc_Mpos(Mpos **pos, int64_t m);
int free_Mpos(Mpos *pos);
Bursts* alloc_burst_array(int64_t n, int64_t alloc_size);
int sequential_concatenate_bursts(int64_t n, Bursts *bursts);
int combined_concatenate_bursts(int64_t n, Bursts *bursts);
int concatenate_bursts(int64_t n, Bursts *bursts);
int sequential_concatenate_bursts_fuse(int64_t n, Bursts *bursts);
int combined_concatenate_bursts_fuse(int64_t n, Bursts *bursts);
int concatenate_bursts_fuse(int64_t n, Bursts *bursts);

int sliding_window_burst_search(int64_t m, double F, double clk_p, double c, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 double cbg, double nbg, 
							 int64_t alloc_size, Mpos *pos, Bursts *bursts);
							 
int sliding_window_burst_search_fuse(int64_t m, double F, double clk_p, double c, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 int64_t fuse, double cbg, double nbg, 
							 int64_t alloc_size, Mpos *pos, Bursts *bursts);

int sliding_window_burst_search_T(int64_t m, double clk_p, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 double mindTc, double mindTn,
							 int64_t alloc_size, Mpos *pos, Bursts *bursts);

int sliding_window_burst_search_T_fuse(int64_t m, double clk_p, PhStream *photons, 
							 int64_t dsize, uint8_t *dset, int64_t cper, int64_t nper,
							 int64_t fuse, double mindTc, double mindTn, 
							 int64_t alloc_size, Mpos *pos, Bursts *bursts);


int cp_burst_search(double alpha, double beta, double clk_p, CPStream *photons, 
			double bg, double sbr, int64_t cper, int64_t nper, int64_t minsep,
			int64_t alloc_size, Bursts *bursts);

int fuse_bursts_inplace(Bursts *bursts, int64_t max_gap);
int fuse_bursts(Bursts *inbursts, int64_t max_sep, Bursts *outbursts);
int burst_gate(int64_t n, Bursts *bursts, uint8_t *truthtable, int64_t start, int64_t stop, int64_t alloc_size, Bursts *out);
int check_bursts_valid(Bursts *bursts);
int check_bursts_fused(Bursts *bursts, int64_t max_delta);

double max_rate(PhStream *photons, int64_t dsize, uint8_t *dset, 
				int64_t istart, int64_t istop, double clk_p, Mpos *pos);

double bva(int64_t n, uint8_t *dets, int64_t dsizeAll, uint8_t *dsetAll, 
			int64_t dsizeSub, uint8_t *dsetSub, int64_t istart, int64_t istop, 
			int64_t *counts);

#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search(void *in);
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search(void *in);
#endif

#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search_fuse(void *in);
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search_fuse(void *in);
#endif


#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search_T(void *in);
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search_T(void *in);
#endif

#if defined(__linux__) || defined(__APPLE__)
void* thread_sliding_window_burst_search_T_fuse(void *in);
#elif _WIN32
DWORD WINAPI thread_sliding_window_burst_search_T_fuse(void *in);
#endif

int burst_search_sliding_window(int64_t m, double F, double clk_p, double c,
								int64_t nphot, int64_t *times, uint8_t *dets,
								int64_t dsize, uint8_t *dset, 
								int64_t nperiods, int64_t *periods, double *bg, 
								int64_t alloc_size, int64_t ncore, double fuse, int asT, Bursts **bursts);

#if defined(__linux__) || defined(__APPLE__)
void* thread_max_rate(void *in);
#elif _WIN32
DWORD WINAPI thread_max_rate(void *in);
#endif

int bursts_max_rate(int64_t m, double clk_p, int64_t nphot, int64_t *times, uint8_t *dets,
					int64_t dsize, uint8_t *dset, 
					int64_t nbursts, int64_t *istarts, int64_t *istops, 
					int64_t ncore, double *max_rates);

#if defined(__linux__) || defined(__APPLE__)
void* thread_cp_burst_search(void *in);
#elif _WIN32
DWORD WINAPI thread_cp_burst_search(void *in);
#endif

int burst_search_cp(double alpha, double beta, double clk_p,
								int64_t nphot, int64_t *times, uint8_t *dets,
								int64_t dsize, uint8_t *dset, 
								int64_t nperiods, int64_t *periods, double *bg, double *sbr,
								double fuse, int64_t alloc_size, int64_t ncore, Bursts **bursts);


#if defined(__linux__) || defined(__APPLE__)
void* thread_bva(void *in);
#elif _WIN32
DWORD WINAPI thread_bva(void *in);
#endif

int burst_variance_analysis(int64_t n, uint8_t *dets, 
							int64_t nbursts, int64_t *istarts, int64_t *istops,
							int64_t dsizeAll, uint8_t *dsetAll, int64_t dsizeSub, uint8_t *dsetSub,
							int64_t ncore, double *bvas);

double laplace_kdefunc(const int64_t loc, const int64_t time, const double tau);
double gaussian_kdefunc(const int64_t loc, const int64_t time, const double tau);
double rect_kdefunc(const int64_t loc, const int64_t time, const double tau);
int kde_self(const int64_t nphot, int64_t* times, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
int kde_self_np(const int64_t nphot, const int64_t stride, char* times, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
int kde_self_exclude_zero(const int64_t nphot, int64_t* times, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
int kde_self_exclude_zero_np(const int64_t nphot, const int64_t stride, char* times, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
int kde_other(const int64_t nphot, int64_t* times, const int64_t nloc, int64_t* locs, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
int kde_other_np(const int64_t nphot, const int64_t tstride, char* times, const int64_t nloc, const int64_t lstride, char* locs, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
int kde_other_exclude_zero(const int64_t nphot, int64_t* times, const int64_t nloc, int64_t* locs, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
int kde_other_exclude_zero_np(const int64_t nphot, const int64_t tstride, char* times, const int64_t nloc, const int64_t lstride, char* locs, const double tau, const int64_t lim, double(*func)(int64_t, int64_t, double), double* out);
