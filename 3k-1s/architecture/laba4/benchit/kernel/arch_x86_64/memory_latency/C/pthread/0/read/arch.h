/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
#ifndef __ARCH_H
#define __ARCH_H

/* cpu-feature definition */
#define X86_64       0x00000001
#define FPU          0x00000002
#define MMX          0x00000004
#define MMX_EXT      0x00000008
#define _3DNOW       0x00000010
#define _3DNOW_EXT   0x00000020
#define SSE          0x00000040
#define SSE2         0x00000080
#define SSE3         0x00000100
#define SSSE3        0x00000200
#define SSE4_1       0x00000400
#define SSE4_2       0x00000800
#define SSE4A        0x00001000
#define SSE5         0x00002000
#define ABM          0x00004000
#define POPCNT       0x00008000
#define CX8          0x00010000
#define CX16         0x00020000
#define CLFLUSH      0x00040000
#define TSC          0x00080000
#define FREQ_SCALING 0x00100000
#define MONITOR      0x00200000
#define MTRR         0x00400000
#define NX           0x00800000
#define CPUID        0x01000000
#define AVX          0x02000000
#define HAP          0x04000000

#define MAX_CACHELEVELS 3
#define MAX_TLBLEVELS   3
#define MAX_PAGESIZES   3

typedef struct cpu_info
{
  char vendor[13];
  char model_str[48];
  char architecture[10];
  unsigned int features;
  unsigned int clflush_linesize;
  unsigned int num_cores;
  unsigned int num_cores_per_package;
  unsigned int rdtsc_latency;
  unsigned int tsc_invariant;
  unsigned int Cachelevels;
  unsigned int Cache_unified[MAX_CACHELEVELS];
  unsigned int Cache_shared[MAX_CACHELEVELS];
  unsigned int Cacheline_size[MAX_CACHELEVELS];
  unsigned int I_Cache_Size[MAX_CACHELEVELS];
  unsigned int I_Cache_Sets[MAX_CACHELEVELS];
  unsigned int D_Cache_Size[MAX_CACHELEVELS];
  unsigned int D_Cache_Sets[MAX_CACHELEVELS];
  unsigned int U_Cache_Size[MAX_CACHELEVELS];
  unsigned int U_Cache_Sets[MAX_CACHELEVELS];
  unsigned int Total_D_Cache_Size;
  unsigned int D_Cache_Size_per_Core;
  unsigned int num_pagesizes;
  unsigned long long pagesizes[MAX_PAGESIZES];
  unsigned int virt_addr_length;
  unsigned int phys_addr_length;
  unsigned int tlblevels;
  unsigned int I_TLB_Size[MAX_TLBLEVELS][MAX_PAGESIZES];
  unsigned int I_TLB_Sets[MAX_TLBLEVELS][MAX_PAGESIZES];
  unsigned int D_TLB_Size[MAX_TLBLEVELS][MAX_PAGESIZES];
  unsigned int D_TLB_Sets[MAX_TLBLEVELS][MAX_PAGESIZES];
  unsigned int U_TLB_Size[MAX_TLBLEVELS][MAX_PAGESIZES];
  unsigned int U_TLB_Sets[MAX_TLBLEVELS][MAX_PAGESIZES];
  unsigned int Cacheflushsize;
  unsigned int family,model,stepping;
  unsigned long long clockrate;
} cpu_info_t;

extern void init_cpuinfo(cpu_info_t *cpuinfo, int print);

extern int cpu_set(int id);
extern int cpu_allowed(int id);

extern int clflush(void* buffer,unsigned long long size, cpu_info_t cpuinfo);

// currently not used
#if 0
extern int write_nt(void* buffer,unsigned long long size, cpu_info_t cpuinfo);
extern int prefetch(void* buffer,unsigned long long size, cpu_info_t cpuinfo);
#endif

#endif
