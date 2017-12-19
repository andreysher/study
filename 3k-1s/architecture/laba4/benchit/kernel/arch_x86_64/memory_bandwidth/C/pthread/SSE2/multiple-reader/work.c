/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures aggregate read bandwidth of multiple parallel threads.
 *******************************************************************/

/*
 * TODO  - share data between threads (modified and unmodified)
 *       - adopt cache and TLB parameters to refer to identifiers returned by 
 *         the hardware detection
 *       - AVX and MIC (Knights Ferry) support
 *       - support low level Performance Counter APIs to get access to uncore/NB events
 *       - remove unnecessary variables from performance counter implementation
 *       - improve cacheflush algorithm to take the minimal cachesize per core into acount
 *         (e.g. 2 Threads on 1 Package have 8 MB in Nehalem, 2 Threads on 2 Packages 16 MB,
 *          Shanghai has 8 MB for 4 Threads, 7 MB for 2 Threads in one package)
 *       - local alloc of flush buffer
 *       - memory layout improvements (as for single-r1w1)
 *       - Assembler implementation of use_memory() to make it independent of compiler optimization
 */
 
#include "work.h"
#include "shared.h"
#include "interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#ifdef USE_PAPI
#include <papi.h>
#endif

/*
 * use a block of memory to ensure it is in the caches afterwards
 */
void inline use_memory(void* buffer,unsigned long long memsize,int mode,int direction,int repeat,cpu_info_t cpuinfo)
{
   int i,j,tmp=0xd08a721b;
   unsigned long long stride = 128;

   for (i=cpuinfo.Cachelevels;i>0;i--)
   {
     if (cpuinfo.Cacheline_size[i-1]<stride) stride=cpuinfo.Cacheline_size[i-1];
   }

   j=repeat;
   while(j--)
   {
       if (direction==FIFO) for (i=0;i<memsize-stride;i+=stride)
       {
         //tmp|=*((int*)((unsigned long long)buffer+i));
         *((int*)((unsigned long long)buffer+i))=tmp;
       }
       if (direction==LIFO) for (i=(memsize-1)-stride;i>=0;i-=stride)
       {
         //tmp|=*((int*)((unsigned long long)buffer+i));
         *((int*)((unsigned long long)buffer+i))=tmp;     
       }
   }
   //now buffer is invalid in other caches, modified in local cache
   if (mode==MODE_EXCLUSIVE) 
   {
     clflush(buffer,memsize,cpuinfo);
     //now buffer is invalid in local cache too
     j=repeat;
     while(j--)
     {
      if (direction==FIFO) for (i=0;i<memsize-stride;i+=stride)
      {
        tmp|=*((int*)((unsigned long long)buffer+i));
      }
      if (direction==LIFO) for (i=(memsize-1)-stride;i>=0;i-=stride)
      {
        tmp|=*((int*)((unsigned long long)buffer+i));
      }
      //result has to be stored somewhere to prevent the compiler from deleting the hole loop
      //if compiler optimisation is disabled, the following command is not needed
      //((int*)((unsigned long long)buffer+i))=tmp;
     }
     //now buffer is exclusive (except 1 cacheline) in local cache
   }
   if (mode==MODE_INVALID)
   {
     clflush(buffer,memsize,cpuinfo);
     //now buffer is invalid in local cache too
   }
}

/*
 * assembler implementation of bandwidth measurement
 */
double asm_work_movdqa(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_movdqa(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif

   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqa_1:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqa_1;"         //|<
                "_sync1_movdqa_1:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqa_1;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqa_1;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqa_1;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqa_1;"         //|<
                "_wait_movdqa_1:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqa_1;"          //|<
                "_sync2_movdqa_1:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqa_1;"         //|<
                "_skip_tsc_sync_movdqa_1:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_1:"
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm0;"
                "movdqa 32(%%r9), %%xmm0;"
                "movdqa 48(%%r9), %%xmm0;"
                "movdqa 64(%%r9), %%xmm0;"
                "movdqa 80(%%r9), %%xmm0;"
                "movdqa 96(%%r9), %%xmm0;"
                "movdqa 112(%%r9), %%xmm0;"
                "movdqa 128(%%r9), %%xmm0;"
                "movdqa 144(%%r9), %%xmm0;"
                "movdqa 160(%%r9), %%xmm0;"
                "movdqa 176(%%r9), %%xmm0;"
                "movdqa 192(%%r9), %%xmm0;"
                "movdqa 208(%%r9), %%xmm0;"
                "movdqa 224(%%r9), %%xmm0;"
                "movdqa 240(%%r9), %%xmm0;"
                "movdqa 256(%%r9), %%xmm0;"
                "movdqa 272(%%r9), %%xmm0;"
                "movdqa 288(%%r9), %%xmm0;"
                "movdqa 304(%%r9), %%xmm0;"
                "movdqa 320(%%r9), %%xmm0;"
                "movdqa 336(%%r9), %%xmm0;"
                "movdqa 352(%%r9), %%xmm0;"
                "movdqa 368(%%r9), %%xmm0;"
                "movdqa 384(%%r9), %%xmm0;"
                "movdqa 400(%%r9), %%xmm0;"
                "movdqa 416(%%r9), %%xmm0;"
                "movdqa 432(%%r9), %%xmm0;"
                "movdqa 448(%%r9), %%xmm0;"
                "movdqa 464(%%r9), %%xmm0;"
                "movdqa 480(%%r9), %%xmm0;"
                "movdqa 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqa_1;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqa_2:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqa_2;"         //|<
                "_sync1_movdqa_2:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqa_2;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqa_2;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqa_2;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqa_2;"         //|<
                "_wait_movdqa_2:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqa_2;"          //|<
                "_sync2_movdqa_2:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqa_2;"         //|<
                "_skip_tsc_sync_movdqa_2:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_2:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm0;movdqa 48(%%r9), %%xmm1;"
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;"
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;"
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;"
                "movdqa 160(%%r9), %%xmm0;movdqa 176(%%r9), %%xmm1;"
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;"
                "movdqa 224(%%r9), %%xmm0;movdqa 240(%%r9), %%xmm1;"
                "movdqa 256(%%r9), %%xmm0;movdqa 272(%%r9), %%xmm1;"
                "movdqa 288(%%r9), %%xmm0;movdqa 304(%%r9), %%xmm1;"
                "movdqa 320(%%r9), %%xmm0;movdqa 336(%%r9), %%xmm1;"
                "movdqa 352(%%r9), %%xmm0;movdqa 368(%%r9), %%xmm1;"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;"
                "movdqa 416(%%r9), %%xmm0;movdqa 432(%%r9), %%xmm1;"
                "movdqa 448(%%r9), %%xmm0;movdqa 464(%%r9), %%xmm1;"
                "movdqa 480(%%r9), %%xmm0;movdqa 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqa_2;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqa_3:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqa_3;"         //|<
                "_sync1_movdqa_3:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqa_3;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqa_3;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqa_3;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqa_3;"         //|<
                "_wait_movdqa_3:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqa_3;"          //|<
                "_sync2_movdqa_3:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqa_3;"         //|<
                "_skip_tsc_sync_movdqa_3:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_3:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;"
                "movdqa 48(%%r9), %%xmm0;movdqa 64(%%r9), %%xmm1;movdqa 80(%%r9), %%xmm2;"
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;movdqa 128(%%r9), %%xmm2;"
                "movdqa 144(%%r9), %%xmm0;movdqa 160(%%r9), %%xmm1;movdqa 176(%%r9), %%xmm2;"
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;"
                "movdqa 240(%%r9), %%xmm0;movdqa 256(%%r9), %%xmm1;movdqa 272(%%r9), %%xmm2;"
                "movdqa 288(%%r9), %%xmm0;movdqa 304(%%r9), %%xmm1;movdqa 320(%%r9), %%xmm2;"
                "movdqa 336(%%r9), %%xmm0;movdqa 352(%%r9), %%xmm1;movdqa 368(%%r9), %%xmm2;"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;"
                "movdqa 432(%%r9), %%xmm0;movdqa 448(%%r9), %%xmm1;movdqa 464(%%r9), %%xmm2;"
                "movdqa 480(%%r9), %%xmm0;movdqa 496(%%r9), %%xmm1;movdqa 512(%%r9), %%xmm2;"
                "movdqa 528(%%r9), %%xmm0;movdqa 544(%%r9), %%xmm1;movdqa 560(%%r9), %%xmm2;"
                "movdqa 576(%%r9), %%xmm0;movdqa 592(%%r9), %%xmm1;movdqa 608(%%r9), %%xmm2;"
                "movdqa 624(%%r9), %%xmm0;movdqa 640(%%r9), %%xmm1;movdqa 656(%%r9), %%xmm2;"
                "movdqa 672(%%r9), %%xmm0;movdqa 688(%%r9), %%xmm1;movdqa 704(%%r9), %%xmm2;"
                "movdqa 720(%%r9), %%xmm0;movdqa 736(%%r9), %%xmm1;movdqa 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqa_3;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*48*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqa_4:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqa_4;"         //|<
                "_sync1_movdqa_4:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqa_4;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqa_4;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqa_4;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqa_4;"         //|<
                "_wait_movdqa_4:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqa_4;"          //|<
                "_sync2_movdqa_4:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqa_4;"         //|<
                "_skip_tsc_sync_movdqa_4:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_4:"
                //sequentiell
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;"
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;"
                "movdqa 256(%%r9), %%xmm0;movdqa 272(%%r9), %%xmm1;movdqa 288(%%r9), %%xmm2;movdqa 304(%%r9), %%xmm3;"
                "movdqa 320(%%r9), %%xmm0;movdqa 336(%%r9), %%xmm1;movdqa 352(%%r9), %%xmm2;movdqa 368(%%r9), %%xmm3;"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;movdqa 432(%%r9), %%xmm3;"
                "movdqa 448(%%r9), %%xmm0;movdqa 464(%%r9), %%xmm1;movdqa 480(%%r9), %%xmm2;movdqa 496(%%r9), %%xmm3;"
                /*
                //transponiert (8x4 Matrix)
                "movdqa (%%r9), %%xmm0;movdqa 64(%%r9), %%xmm1;movdqa 128(%%r9), %%xmm2;movdqa 192(%%r9), %%xmm3;"
                "movdqa 256(%%r9), %%xmm0;movdqa 320(%%r9), %%xmm1;movdqa 384(%%r9), %%xmm2;movdqa 448(%%r9), %%xmm3;"
                "movdqa 16(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 144(%%r9), %%xmm2;movdqa 208(%%r9), %%xmm3;"
                "movdqa 272(%%r9), %%xmm0;movdqa 336(%%r9), %%xmm1;movdqa 400(%%r9), %%xmm2;movdqa 464(%%r9), %%xmm3;"
                "movdqa 32(%%r9), %%xmm0;movdqa 96(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 224(%%r9), %%xmm3;"
                "movdqa 288(%%r9), %%xmm0;movdqa 352(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;movdqa 480(%%r9), %%xmm3;"
                "movdqa 48(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;movdqa 176(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;"
                "movdqa 304(%%r9), %%xmm0;movdqa 368(%%r9), %%xmm1;movdqa 432(%%r9), %%xmm2;movdqa 496(%%r9), %%xmm3;"

                 //prefetch +8 cachelines
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;prefetcht0 512(%%r9);"
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;prefetcht0 576(%%r9);"
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;prefetcht0 640(%%r9);"
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;prefetcht0 704(%%r9);"
                "movdqa 256(%%r9), %%xmm0;movdqa 272(%%r9), %%xmm1;movdqa 288(%%r9), %%xmm2;movdqa 304(%%r9), %%xmm3;prefetcht0 768(%%r9);"
                "movdqa 320(%%r9), %%xmm0;movdqa 336(%%r9), %%xmm1;movdqa 352(%%r9), %%xmm2;movdqa 368(%%r9), %%xmm3;prefetcht0 832(%%r9);"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;movdqa 432(%%r9), %%xmm3;prefetcht0 896(%%r9);"
                "movdqa 448(%%r9), %%xmm0;movdqa 464(%%r9), %%xmm1;movdqa 480(%%r9), %%xmm2;movdqa 496(%%r9), %%xmm3;prefetcht0 960(%%r9);"

                //prefetch +16 cachelines
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;prefetcht0 1024(%%r9);"
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;prefetcht0 1088(%%r9);"
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;prefetcht0 1152(%%r9);"
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;prefetcht0 1216(%%r9);"
                "movdqa 256(%%r9), %%xmm0;movdqa 272(%%r9), %%xmm1;movdqa 288(%%r9), %%xmm2;movdqa 304(%%r9), %%xmm3;prefetcht0 1280(%%r9);"
                "movdqa 320(%%r9), %%xmm0;movdqa 336(%%r9), %%xmm1;movdqa 352(%%r9), %%xmm2;movdqa 368(%%r9), %%xmm3;prefetcht0 1344(%%r9);"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;movdqa 432(%%r9), %%xmm3;prefetcht0 1408(%%r9);"
                "movdqa 448(%%r9), %%xmm0;movdqa 464(%%r9), %%xmm1;movdqa 480(%%r9), %%xmm2;movdqa 496(%%r9), %%xmm3;prefetcht0 1472(%%r9);"

                 //prefetch +32 cachelines
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;prefetcht0 2048(%%r9);"
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;prefetcht0 2112(%%r9);"
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;prefetcht0 2176(%%r9);"
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;prefetcht0 2240(%%r9);"
                "movdqa 256(%%r9), %%xmm0;movdqa 272(%%r9), %%xmm1;movdqa 288(%%r9), %%xmm2;movdqa 304(%%r9), %%xmm3;prefetcht0 2304(%%r9);"
                "movdqa 320(%%r9), %%xmm0;movdqa 336(%%r9), %%xmm1;movdqa 352(%%r9), %%xmm2;movdqa 368(%%r9), %%xmm3;prefetcht0 2368(%%r9);"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;movdqa 432(%%r9), %%xmm3;prefetcht0 2432(%%r9);"
                "movdqa 448(%%r9), %%xmm0;movdqa 464(%%r9), %%xmm1;movdqa 480(%%r9), %%xmm2;movdqa 496(%%r9), %%xmm3;prefetcht0 2496(%%r9);"

                // 1 accsses per cache line
                "movdqa (%%r9), %%xmm0;"
                "movdqa 64(%%r9), %%xmm0;"
                "movdqa 128(%%r9), %%xmm0;"
                "movdqa 192(%%r9), %%xmm0;"
                "movdqa 256(%%r9), %%xmm0;"
                "movdqa 320(%%r9), %%xmm0;"
                "movdqa 384(%%r9), %%xmm0;"
                "movdqa 448(%%r9), %%xmm0;"
                */
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqa_4;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
      default: ret=0.0;break;
   }  
  //printf ("id: %i, %llu - %llu : %llu\n",id,data->start_ts,data->end_ts,data->end_ts-data->start_ts);
  //printf("end asm\n");fflush(stdout);
	
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if ((!id)&&(data->num_events))
    { 
      PAPI_read(data->Eventset,data->values);
      for (i=0;i<data->num_events;i++)
      {
      
         if (burst_length!=3) data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
         else data->papi_results[i]=(double)data->values[i]/(double)(passes*48);

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
	return ret;
}

double asm_work_movdqu(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_movdqu(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   int i;   

   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif

   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqu_1:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqu_1;"         //|<
                "_sync1_movdqu_1:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqu_1;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqu_1;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqu_1;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqu_1;"         //|<
                "_wait_movdqu_1:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqu_1;"          //|<
                "_sync2_movdqu_1:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqu_1;"         //|<
                "_skip_tsc_sync_movdqu_1:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_1:"
                "movdqu (%%r9), %%xmm0;"
                "movdqu 16(%%r9), %%xmm0;"
                "movdqu 32(%%r9), %%xmm0;"
                "movdqu 48(%%r9), %%xmm0;"
                "movdqu 64(%%r9), %%xmm0;"
                "movdqu 80(%%r9), %%xmm0;"
                "movdqu 96(%%r9), %%xmm0;"
                "movdqu 112(%%r9), %%xmm0;"
                "movdqu 128(%%r9), %%xmm0;"
                "movdqu 144(%%r9), %%xmm0;"
                "movdqu 160(%%r9), %%xmm0;"
                "movdqu 176(%%r9), %%xmm0;"
                "movdqu 192(%%r9), %%xmm0;"
                "movdqu 208(%%r9), %%xmm0;"
                "movdqu 224(%%r9), %%xmm0;"
                "movdqu 240(%%r9), %%xmm0;"
                "movdqu 256(%%r9), %%xmm0;"
                "movdqu 272(%%r9), %%xmm0;"
                "movdqu 288(%%r9), %%xmm0;"
                "movdqu 304(%%r9), %%xmm0;"
                "movdqu 320(%%r9), %%xmm0;"
                "movdqu 336(%%r9), %%xmm0;"
                "movdqu 352(%%r9), %%xmm0;"
                "movdqu 368(%%r9), %%xmm0;"
                "movdqu 384(%%r9), %%xmm0;"
                "movdqu 400(%%r9), %%xmm0;"
                "movdqu 416(%%r9), %%xmm0;"
                "movdqu 432(%%r9), %%xmm0;"
                "movdqu 448(%%r9), %%xmm0;"
                "movdqu 464(%%r9), %%xmm0;"
                "movdqu 480(%%r9), %%xmm0;"
                "movdqu 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqu_1;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqu_2:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqu_2;"         //|<
                "_sync1_movdqu_2:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqu_2;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqu_2;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqu_2;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqu_2;"         //|<
                "_wait_movdqu_2:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqu_2;"          //|<
                "_sync2_movdqu_2:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqu_2;"         //|<
                "_skip_tsc_sync_movdqu_2:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_2:"
                "movdqu (%%r9), %%xmm0;movdqu 16(%%r9), %%xmm1;"
                "movdqu 32(%%r9), %%xmm0;movdqu 48(%%r9), %%xmm1;"
                "movdqu 64(%%r9), %%xmm0;movdqu 80(%%r9), %%xmm1;"
                "movdqu 96(%%r9), %%xmm0;movdqu 112(%%r9), %%xmm1;"
                "movdqu 128(%%r9), %%xmm0;movdqu 144(%%r9), %%xmm1;"
                "movdqu 160(%%r9), %%xmm0;movdqu 176(%%r9), %%xmm1;"
                "movdqu 192(%%r9), %%xmm0;movdqu 208(%%r9), %%xmm1;"
                "movdqu 224(%%r9), %%xmm0;movdqu 240(%%r9), %%xmm1;"
                "movdqu 256(%%r9), %%xmm0;movdqu 272(%%r9), %%xmm1;"
                "movdqu 288(%%r9), %%xmm0;movdqu 304(%%r9), %%xmm1;"
                "movdqu 320(%%r9), %%xmm0;movdqu 336(%%r9), %%xmm1;"
                "movdqu 352(%%r9), %%xmm0;movdqu 368(%%r9), %%xmm1;"
                "movdqu 384(%%r9), %%xmm0;movdqu 400(%%r9), %%xmm1;"
                "movdqu 416(%%r9), %%xmm0;movdqu 432(%%r9), %%xmm1;"
                "movdqu 448(%%r9), %%xmm0;movdqu 464(%%r9), %%xmm1;"
                "movdqu 480(%%r9), %%xmm0;movdqu 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqu_2;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqu_3:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqu_3;"         //|<
                "_sync1_movdqu_3:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqu_3;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqu_3;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqu_3;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqu_3;"         //|<
                "_wait_movdqu_3:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqu_3;"          //|<
                "_sync2_movdqu_3:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqu_3;"         //|<
                "_skip_tsc_sync_movdqu_3:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_3:"
                "movdqu (%%r9), %%xmm0;movdqu 16(%%r9), %%xmm1;movdqu 32(%%r9), %%xmm2;"
                "movdqu 48(%%r9), %%xmm0;movdqu 64(%%r9), %%xmm1;movdqu 80(%%r9), %%xmm2;"
                "movdqu 96(%%r9), %%xmm0;movdqu 112(%%r9), %%xmm1;movdqu 128(%%r9), %%xmm2;"
                "movdqu 144(%%r9), %%xmm0;movdqu 160(%%r9), %%xmm1;movdqu 176(%%r9), %%xmm2;"
                "movdqu 192(%%r9), %%xmm0;movdqu 208(%%r9), %%xmm1;movdqu 224(%%r9), %%xmm2;"
                "movdqu 240(%%r9), %%xmm0;movdqu 256(%%r9), %%xmm1;movdqu 272(%%r9), %%xmm2;"
                "movdqu 288(%%r9), %%xmm0;movdqu 304(%%r9), %%xmm1;movdqu 320(%%r9), %%xmm2;"
                "movdqu 336(%%r9), %%xmm0;movdqu 352(%%r9), %%xmm1;movdqu 368(%%r9), %%xmm2;"
                "movdqu 384(%%r9), %%xmm0;movdqu 400(%%r9), %%xmm1;movdqu 416(%%r9), %%xmm2;"
                "movdqu 432(%%r9), %%xmm0;movdqu 448(%%r9), %%xmm1;movdqu 464(%%r9), %%xmm2;"
                "movdqu 480(%%r9), %%xmm0;movdqu 496(%%r9), %%xmm1;movdqu 512(%%r9), %%xmm2;"
                "movdqu 528(%%r9), %%xmm0;movdqu 544(%%r9), %%xmm1;movdqu 560(%%r9), %%xmm2;"
                "movdqu 576(%%r9), %%xmm0;movdqu 592(%%r9), %%xmm1;movdqu 608(%%r9), %%xmm2;"
                "movdqu 624(%%r9), %%xmm0;movdqu 640(%%r9), %%xmm1;movdqu 656(%%r9), %%xmm2;"
                "movdqu 672(%%r9), %%xmm0;movdqu 688(%%r9), %%xmm1;movdqu 704(%%r9), %%xmm2;"
                "movdqu 720(%%r9), %%xmm0;movdqu 736(%%r9), %%xmm1;movdqu 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqu_3;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*48*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_movdqu_4:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movdqu_4;"         //|<
                "_sync1_movdqu_4:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movdqu_4;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movdqu_4;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movdqu_4;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movdqu_4;"         //|<
                "_wait_movdqu_4:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movdqu_4;"          //|<
                "_sync2_movdqu_4:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movdqu_4;"         //|<
                "_skip_tsc_sync_movdqu_4:"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_4:"
                "movdqu (%%r9), %%xmm0;movdqu 16(%%r9), %%xmm1;movdqu 32(%%r9), %%xmm2;movdqu 48(%%r9), %%xmm3;"
                "movdqu 64(%%r9), %%xmm0;movdqu 80(%%r9), %%xmm1;movdqu 96(%%r9), %%xmm2;movdqu 112(%%r9), %%xmm3;"
                "movdqu 128(%%r9), %%xmm0;movdqu 144(%%r9), %%xmm1;movdqu 160(%%r9), %%xmm2;movdqu 176(%%r9), %%xmm3;"
                "movdqu 192(%%r9), %%xmm0;movdqu 208(%%r9), %%xmm1;movdqu 224(%%r9), %%xmm2;movdqu 240(%%r9), %%xmm3;"
                "movdqu 256(%%r9), %%xmm0;movdqu 272(%%r9), %%xmm1;movdqu 288(%%r9), %%xmm2;movdqu 304(%%r9), %%xmm3;"
                "movdqu 320(%%r9), %%xmm0;movdqu 336(%%r9), %%xmm1;movdqu 352(%%r9), %%xmm2;movdqu 368(%%r9), %%xmm3;"
                "movdqu 384(%%r9), %%xmm0;movdqu 400(%%r9), %%xmm1;movdqu 416(%%r9), %%xmm2;movdqu 432(%%r9), %%xmm3;"
                "movdqu 448(%%r9), %%xmm0;movdqu 464(%%r9), %%xmm1;movdqu 480(%%r9), %%xmm2;movdqu 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movdqu_4;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
      default: ret=0.0;break;
   }  
  //printf ("id: %i, %llu - %llu : %llu\n",id,data->start_ts,data->end_ts,data->end_ts-data->start_ts);
  //printf("end asm\n");fflush(stdout);
	
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if ((!id)&&(data->num_events))
    { 
      PAPI_read(data->Eventset,data->values);
      for (i=0;i<data->num_events;i++)
      {
      
         if (burst_length!=3) data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
         else data->papi_results[i]=(double)data->values[i]/(double)(passes*48);

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
	return ret;
}

double asm_work_mov(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mov(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   int i;

   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif

   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_mov_1:"                //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_mov_1;"            //|<
                "_sync1_mov_1:"                //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_mov_1;"            //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_mov_1;"     //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_mov_1;"             //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_mov_1;"            //|<
                "_wait_mov_1:"                 //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_mov_1;"             //|<
                "_sync2_mov_1:"                //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_mov_1;"            //|<
                "_skip_tsc_sync_mov_1:"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_1:"
                "mov (%%r9), %%r12;"
                "mov 8(%%r9), %%r12;"
                "mov 16(%%r9), %%r12;"
                "mov 24(%%r9), %%r12;"
                "mov 32(%%r9), %%r12;"
                "mov 40(%%r9), %%r12;"
                "mov 48(%%r9), %%r12;"
                "mov 56(%%r9), %%r12;"
                "mov 64(%%r9), %%r12;"
                "mov 72(%%r9), %%r12;"
                "mov 80(%%r9), %%r12;"
                "mov 88(%%r9), %%r12;"
                "mov 96(%%r9), %%r12;"
                "mov 104(%%r9), %%r12;"
                "mov 112(%%r9), %%r12;"
                "mov 120(%%r9), %%r12;"
                "mov 128(%%r9), %%r12;"
                "mov 136(%%r9), %%r12;"
                "mov 144(%%r9), %%r12;"
                "mov 152(%%r9), %%r12;"
                "mov 160(%%r9), %%r12;"
                "mov 168(%%r9), %%r12;"
                "mov 176(%%r9), %%r12;"
                "mov 184(%%r9), %%r12;"
                "mov 192(%%r9), %%r12;"
                "mov 200(%%r9), %%r12;"
                "mov 208(%%r9), %%r12;"
                "mov 216(%%r9), %%r12;"
                "mov 224(%%r9), %%r12;"
                "mov 232(%%r9), %%r12;"
                "mov 240(%%r9), %%r12;"
                "mov 248(%%r9), %%r12;"
                "mov 256(%%r9), %%r12;"
                "mov 264(%%r9), %%r12;"
                "mov 272(%%r9), %%r12;"
                "mov 280(%%r9), %%r12;"
                "mov 288(%%r9), %%r12;"
                "mov 296(%%r9), %%r12;"
                "mov 304(%%r9), %%r12;"
                "mov 312(%%r9), %%r12;"
                "mov 320(%%r9), %%r12;"
                "mov 328(%%r9), %%r12;"
                "mov 336(%%r9), %%r12;"
                "mov 344(%%r9), %%r12;"
                "mov 352(%%r9), %%r12;"
                "mov 360(%%r9), %%r12;"
                "mov 368(%%r9), %%r12;"
                "mov 376(%%r9), %%r12;"
                "mov 384(%%r9), %%r12;"
                "mov 392(%%r9), %%r12;"
                "mov 400(%%r9), %%r12;"
                "mov 408(%%r9), %%r12;"
                "mov 416(%%r9), %%r12;"
                "mov 424(%%r9), %%r12;"
                "mov 432(%%r9), %%r12;"
                "mov 440(%%r9), %%r12;"
                "mov 448(%%r9), %%r12;"
                "mov 456(%%r9), %%r12;"
                "mov 464(%%r9), %%r12;"
                "mov 472(%%r9), %%r12;"
                "mov 480(%%r9), %%r12;"
                "mov 488(%%r9), %%r12;"
                "mov 496(%%r9), %%r12;"
                "mov 504(%%r9), %%r12;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_mov_1;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_mov_2:"                //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_mov_2;"            //|<
                "_sync1_mov_2:"                //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_mov_2;"            //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_mov_2;"     //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_mov_2;"             //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_mov_2;"            //|<
                "_wait_mov_2:"                 //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_mov_2;"             //|<
                "_sync2_mov_2:"                //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_mov_2;"            //|<
                "_skip_tsc_sync_mov_2:"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_2:"
                "mov (%%r9), %%r12;mov 8(%%r9), %%r13;"
                "mov 16(%%r9), %%r12;mov 24(%%r9), %%r13;"
                "mov 32(%%r9), %%r12;mov 40(%%r9), %%r13;"
                "mov 48(%%r9), %%r12;mov 56(%%r9), %%r13;"
                "mov 64(%%r9), %%r12;mov 72(%%r9), %%r13;"
                "mov 80(%%r9), %%r12;mov 88(%%r9), %%r13;"
                "mov 96(%%r9), %%r12;mov 104(%%r9), %%r13;"
                "mov 112(%%r9), %%r12;mov 120(%%r9), %%r13;"
                "mov 128(%%r9), %%r12;mov 136(%%r9), %%r13;"
                "mov 144(%%r9), %%r12;mov 152(%%r9), %%r13;"
                "mov 160(%%r9), %%r12;mov 168(%%r9), %%r13;"
                "mov 176(%%r9), %%r12;mov 184(%%r9), %%r13;"
                "mov 192(%%r9), %%r12;mov 200(%%r9), %%r13;"
                "mov 208(%%r9), %%r12;mov 216(%%r9), %%r13;"
                "mov 224(%%r9), %%r12;mov 232(%%r9), %%r13;"
                "mov 240(%%r9), %%r12;mov 248(%%r9), %%r13;"
                "mov 256(%%r9), %%r12;mov 264(%%r9), %%r13;"
                "mov 272(%%r9), %%r12;mov 280(%%r9), %%r13;"
                "mov 288(%%r9), %%r12;mov 296(%%r9), %%r13;"
                "mov 304(%%r9), %%r12;mov 312(%%r9), %%r13;"
                "mov 320(%%r9), %%r12;mov 328(%%r9), %%r13;"
                "mov 336(%%r9), %%r12;mov 344(%%r9), %%r13;"
                "mov 352(%%r9), %%r12;mov 360(%%r9), %%r13;"
                "mov 368(%%r9), %%r12;mov 376(%%r9), %%r13;"
                "mov 384(%%r9), %%r12;mov 392(%%r9), %%r13;"
                "mov 400(%%r9), %%r12;mov 408(%%r9), %%r13;"
                "mov 416(%%r9), %%r12;mov 424(%%r9), %%r13;"
                "mov 432(%%r9), %%r12;mov 440(%%r9), %%r13;"
                "mov 448(%%r9), %%r12;mov 456(%%r9), %%r13;"
                "mov 464(%%r9), %%r12;mov 472(%%r9), %%r13;"
                "mov 480(%%r9), %%r12;mov 488(%%r9), %%r13;"
                "mov 496(%%r9), %%r12;mov 504(%%r9), %%r13;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_mov_2;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_mov_3:"                //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_mov_3;"            //|<
                "_sync1_mov_3:"                //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_mov_3;"            //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_mov_3;"     //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_mov_3;"             //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_mov_3;"            //|<
                "_wait_mov_3:"                 //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_mov_3;"             //|<
                "_sync2_mov_3:"                //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_mov_3;"            //|<
                "_skip_tsc_sync_mov_3:"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_3:"
                "mov (%%r9), %%r12;mov 8(%%r9), %%r13;mov 16(%%r9), %%r14;"
                "mov 24(%%r9), %%r12;mov 32(%%r9), %%r13;mov 40(%%r9), %%r14;"
                "mov 48(%%r9), %%r12;mov 56(%%r9), %%r13;mov 64(%%r9), %%r14;"
                "mov 72(%%r9), %%r12;mov 80(%%r9), %%r13;mov 88(%%r9), %%r14;"
                "mov 96(%%r9), %%r12;mov 104(%%r9), %%r13;mov 112(%%r9), %%r14;"
                "mov 120(%%r9), %%r12;mov 128(%%r9), %%r13;mov 136(%%r9), %%r14;"
                "mov 144(%%r9), %%r12;mov 152(%%r9), %%r13;mov 160(%%r9), %%r14;"
                "mov 168(%%r9), %%r12;mov 176(%%r9), %%r13;mov 184(%%r9), %%r14;"
                "mov 192(%%r9), %%r12;mov 200(%%r9), %%r13;mov 208(%%r9), %%r14;"
                "mov 216(%%r9), %%r12;mov 224(%%r9), %%r13;mov 232(%%r9), %%r14;"
                "mov 240(%%r9), %%r12;mov 248(%%r9), %%r13;mov 256(%%r9), %%r14;"
                "mov 264(%%r9), %%r12;mov 272(%%r9), %%r13;mov 280(%%r9), %%r14;"
                "mov 288(%%r9), %%r12;mov 296(%%r9), %%r13;mov 304(%%r9), %%r14;"
                "mov 312(%%r9), %%r12;mov 320(%%r9), %%r13;mov 328(%%r9), %%r14;"
                "mov 336(%%r9), %%r12;mov 344(%%r9), %%r13;mov 352(%%r9), %%r14;"
                "mov 360(%%r9), %%r12;mov 368(%%r9), %%r13;mov 376(%%r9), %%r14;"
                "mov 384(%%r9), %%r12;mov 392(%%r9), %%r13;mov 400(%%r9), %%r14;"
                "mov 408(%%r9), %%r12;mov 416(%%r9), %%r13;mov 424(%%r9), %%r14;"
                "mov 432(%%r9), %%r12;mov 440(%%r9), %%r13;mov 448(%%r9), %%r14;"
                "mov 456(%%r9), %%r12;mov 464(%%r9), %%r13;mov 472(%%r9), %%r14;"
                "mov 480(%%r9), %%r12;mov 488(%%r9), %%r13;mov 496(%%r9), %%r14;"
                "mov 504(%%r9), %%r12;mov 512(%%r9), %%r13;mov 520(%%r9), %%r14;"
                "mov 528(%%r9), %%r12;mov 536(%%r9), %%r13;mov 544(%%r9), %%r14;"
                "mov 552(%%r9), %%r12;mov 560(%%r9), %%r13;mov 568(%%r9), %%r14;"
                "mov 576(%%r9), %%r12;mov 584(%%r9), %%r13;mov 592(%%r9), %%r14;"
                "mov 600(%%r9), %%r12;mov 608(%%r9), %%r13;mov 616(%%r9), %%r14;"
                "mov 624(%%r9), %%r12;mov 632(%%r9), %%r13;mov 640(%%r9), %%r14;"
                "mov 648(%%r9), %%r12;mov 656(%%r9), %%r13;mov 664(%%r9), %%r14;"
                "mov 672(%%r9), %%r12;mov 680(%%r9), %%r13;mov 688(%%r9), %%r14;"
                "mov 696(%%r9), %%r12;mov 704(%%r9), %%r13;mov 712(%%r9), %%r14;"
                "mov 720(%%r9), %%r12;mov 728(%%r9), %%r13;mov 736(%%r9), %%r14;"
                "mov 744(%%r9), %%r12;mov 752(%%r9), %%r13;mov 760(%%r9), %%r14;" 
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_mov_3;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "r14"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*48*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;   
      /*
       * Input:  RAX: addr (pointer to the buffer)
       *         RBX: passes (number of iterations)
       *         RCX: runing_threads (number of threads)
       *         RDX: id (thread ID)
       *         %8:  sync_ptr (pointer to sync buffer for cmpxchg and TSC sync)
       * Output: RAX: stop timestamp 
       *         RBX: start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "mov %8,%%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"                                                
                 //sync
                "mov %%r12,%%rbx;"             //Synchronisation
                "add $1,%%rbx;"                //Phase 1: Barrier >>
                "mov 8(%%r8),%%r13;"           //|load TSC feature flag for Phase 2
                "_sync0_mov_4:"                //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_mov_4;"            //|<
                "_sync1_mov_4:"                //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_mov_4;"            //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_mov_4;"     //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_mov_4;"             //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_mov_4;"            //|<
                "_wait_mov_4:"                 //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_mov_4;"             //|<
                "_sync2_mov_4:"                //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_mov_4;"            //|<
                "_skip_tsc_sync_mov_4:"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_4:"
                "mov (%%r9), %%r12;mov 8(%%r9), %%r13;mov 16(%%r9), %%r14;mov 24(%%r9), %%r15;"
                "mov 32(%%r9), %%r12;mov 40(%%r9), %%r13;mov 48(%%r9), %%r14;mov 56(%%r9), %%r15;"
                "mov 64(%%r9), %%r12;mov 72(%%r9), %%r13;mov 80(%%r9), %%r14;mov 88(%%r9), %%r15;"
                "mov 96(%%r9), %%r12;mov 104(%%r9), %%r13;mov 112(%%r9), %%r14;mov 120(%%r9), %%r15;"
                "mov 128(%%r9), %%r12;mov 136(%%r9), %%r13;mov 144(%%r9), %%r14;mov 152(%%r9), %%r15;"
                "mov 160(%%r9), %%r12;mov 168(%%r9), %%r13;mov 176(%%r9), %%r14;mov 184(%%r9), %%r15;"
                "mov 192(%%r9), %%r12;mov 200(%%r9), %%r13;mov 208(%%r9), %%r14;mov 216(%%r9), %%r15;"
                "mov 224(%%r9), %%r12;mov 232(%%r9), %%r13;mov 240(%%r9), %%r14;mov 248(%%r9), %%r15;"
                "mov 256(%%r9), %%r12;mov 264(%%r9), %%r13;mov 272(%%r9), %%r14;mov 280(%%r9), %%r15;"
                "mov 288(%%r9), %%r12;mov 296(%%r9), %%r13;mov 304(%%r9), %%r14;mov 312(%%r9), %%r15;"
                "mov 320(%%r9), %%r12;mov 328(%%r9), %%r13;mov 336(%%r9), %%r14;mov 344(%%r9), %%r15;"
                "mov 352(%%r9), %%r12;mov 360(%%r9), %%r13;mov 368(%%r9), %%r14;mov 376(%%r9), %%r15;"
                "mov 384(%%r9), %%r12;mov 392(%%r9), %%r13;mov 400(%%r9), %%r14;mov 408(%%r9), %%r15;"
                "mov 416(%%r9), %%r12;mov 424(%%r9), %%r13;mov 432(%%r9), %%r14;mov 440(%%r9), %%r15;"
                "mov 448(%%r9), %%r12;mov 456(%%r9), %%r13;mov 464(%%r9), %%r14;mov 472(%%r9), %%r15;"
                "mov 480(%%r9), %%r12;mov 488(%%r9), %%r13;mov 496(%%r9), %%r14;mov 504(%%r9), %%r15;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_mov_4;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(passes*32*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
      default: ret=0.0;break;
   }  
  //printf ("id: %i, %llu - %llu : %llu\n",id,data->start_ts,data->end_ts,data->end_ts-data->start_ts);
  //printf("end asm\n");fflush(stdout);
	
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if ((!id)&&(data->num_events))
    { 
      PAPI_read(data->Eventset,data->values);
      for (i=0;i<data->num_events;i++)
      {
      
         if (burst_length!=3) data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
         else data->papi_results[i]=(double)data->values[i]/(double)(passes*48);

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
	return ret;
}

/*
 * function that does the measurement
 */
void inline _work( unsigned long long memsize, int offset, int function, int burst_length, int runs, volatile mydata_t* data, double **results)
{
  int latency,i,j,t;
  double tmax;
  double tmp=(double)0;
  unsigned long long tmp2,tmp3;

	/* aligned address */
	unsigned long long aligned_addr,accesses;
	
  aligned_addr=(unsigned long long)(data->buffer) + offset;

  accesses=memsize/(2*sizeof(unsigned long long));
  accesses=(accesses>>5)*32;

   // printf("starting measurment %i accesses in %i Bytes of memory\n",accesses,memsize);
   t=data->num_threads-1;
   tmax=0;
  
   if (((accesses/32)/(t+1))>16) 
   {
    data->running_threads=t+1;

    //init threaddata
    if (t)
    {
     for (j=0;j<t;j++)
     {
      data->threaddata[j+1].memsize=memsize/(t+1);
      data->threaddata[j+1].accesses=((accesses/(t+1))/32)*32;
      }
    }   

   latency=data->cpuinfo->rdtsc_latency;

    
    for (i=0;i<runs*(t+1);i++)
    {     
     data->synch[0]=0;
     /* copy invariant TSC flag to sync area, asm work functions will skip TSC based synchronization if not set */ 
     data->synch[1]=data->cpuinfo->tsc_invariant;

     //tell other threads to touch memory 
     if (t)
     {
       for (j=0;j<t;j++)
       {
        data->thread_comm[j+1]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        //printf("per thread: memsize: %i, accesses: %i\n",data->threaddata[j+1].memsize,data->threaddata[j+1].accesses);
       }
     }
          
     //access whole buffer to warm up cache and tlb
     use_memory((void*)aligned_addr,memsize/(t+1),data->USE_MODE,data->USE_DIRECTION,data->NUM_USES,*(data->cpuinfo));
     //flush cachelevels as specified in PARAMETERS
     flush_caches((void*) aligned_addr,memsize/(t+1),data->settings,data->NUM_FLUSHES,data->FLUSH_MODE,data->cache_flush_area,data->cpuinfo,0,data->running_threads);

     //wait for other threads touching the memory
     if (t)
     {
       for (j=0;j<t;j++)
       {
        data->thread_comm[j+1]=THREAD_WAIT;       
        while (!data->ack);
        data->ack=0;
       }
     }

     //tell other threads to start
     if (t)
     {
       for (j=0;j<t;j++)
       {
         data->thread_comm[j+1]=THREAD_WORK;
         while (!data->ack);
         data->ack=0;
       }
     }      
      
      /* call ASM implementation */
      //printf("call asm impl. latency: %i cycles\n",latency);
      switch(function)
      {
        case USE_MOVDQA: tmp=asm_work_movdqa(aligned_addr,((accesses/(t+1))),burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MOVDQU: tmp=asm_work_movdqu(aligned_addr,((accesses/(t+1))),burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MOV: tmp=asm_work_mov(aligned_addr,((accesses/(t+1))),burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        default: break;
      }
      
      tmp2=data->threaddata[0].start_ts;
      tmp3=data->threaddata[0].end_ts;
      
     // printf (":id %i,%llu - %llu : %llu\n",0,tmp2,tmp3,tmp3-tmp2);
      
     //wait for other THREADS
     if (t)
     {
       for (j=0;j<t;j++)
       {
         data->thread_comm[j+1]=THREAD_WAIT;       
         while (!data->ack);
         data->ack=0;

         /* find earliest start timestamp and latest end timestamp if timestamps are comparable between CPUs, otherwise choose longest interval (assume concurrent start) */
         if (data->cpuinfo->tsc_invariant==1)
         {
           if (data->threaddata[j+1].start_ts<tmp2) tmp2=data->threaddata[j+1].start_ts;       
           if (data->threaddata[j+1].end_ts>tmp3) tmp3=data->threaddata[j+1].end_ts; 
         }
         else if ((data->threaddata[j+1].end_ts-data->threaddata[j+1].start_ts)>(tmp3-tmp2))
         {
           tmp2=data->threaddata[j+1].start_ts;
           tmp3=data->threaddata[j+1].end_ts; 
         }
         //printf (":id %i,%llu - %llu : %llu\n",j+1,data->threaddata[j+1].start_ts,data->threaddata[j+1].end_ts,data->threaddata[j+1].end_ts-data->threaddata[j+1].start_ts);      
       }
     } 

     //printf ("%llu - %llu : %llu\n\n",tmp2,tmp3,tmp3-tmp2);
     tmp2=(tmp3-tmp2)-latency;
      
     if ((int)tmp!=-1)
      {
       tmp=((((double)(((accesses/(t+1)))*(t+1)*16)))/ ((double)(tmp2)/data->cpuinfo->clockrate)) *0.000000001;
      
       if ((int)tmax==0)  tmax=tmp;
       if (tmp>tmax) tmax=tmp;
      }
     
     
    }
   }
   else tmax=0;
  
   if (tmax)
   {
     (*results)[0]=(double)tmax;
     #ifdef USE_PAPI
     for (j=0;j<data->num_events;j++)
     {
       (*results)[j+1]=0;
       for (t=0;t<data->num_threads;t++)
       {
         (*results)[j+1]+=data->threaddata[t].papi_results[j];
       }
       //if (data->num_threads) (*results)[j+1]=(double)(*results)[j+1]/(double)data->num_threads;
     }
     #endif
   }
   else 
   {
     (*results)[0]=INVALID_MEASUREMENT;
     #ifdef USE_PAPI
     for (j=0;j<data->num_events;j++)
     {
       (*results)[j+1]=INVALID_MEASUREMENT;
     }
     #endif    
   }
}
