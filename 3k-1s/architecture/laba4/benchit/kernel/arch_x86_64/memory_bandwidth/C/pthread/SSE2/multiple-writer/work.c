/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures aggregate write bandwidth of multiple parallel threads.
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
                "movdqa %%xmm0,(%%r9);"
                "movdqa %%xmm0,16(%%r9);"
                "movdqa %%xmm0,32(%%r9);"
                "movdqa %%xmm0,48(%%r9);"
                "movdqa %%xmm0,64(%%r9);"
                "movdqa %%xmm0,80(%%r9);"
                "movdqa %%xmm0,96(%%r9);"
                "movdqa %%xmm0,112(%%r9);"
                "movdqa %%xmm0,128(%%r9);"
                "movdqa %%xmm0,144(%%r9);"
                "movdqa %%xmm0,160(%%r9);"
                "movdqa %%xmm0,176(%%r9);"
                "movdqa %%xmm0,192(%%r9);"
                "movdqa %%xmm0,208(%%r9);"
                "movdqa %%xmm0,224(%%r9);"
                "movdqa %%xmm0,240(%%r9);"
                "movdqa %%xmm0,256(%%r9);"
                "movdqa %%xmm0,272(%%r9);"
                "movdqa %%xmm0,288(%%r9);"
                "movdqa %%xmm0,304(%%r9);"
                "movdqa %%xmm0,320(%%r9);"
                "movdqa %%xmm0,336(%%r9);"
                "movdqa %%xmm0,352(%%r9);"
                "movdqa %%xmm0,368(%%r9);"
                "movdqa %%xmm0,384(%%r9);"
                "movdqa %%xmm0,400(%%r9);"
                "movdqa %%xmm0,416(%%r9);"
                "movdqa %%xmm0,432(%%r9);"
                "movdqa %%xmm0,448(%%r9);"
                "movdqa %%xmm0,464(%%r9);"
                "movdqa %%xmm0,480(%%r9);"
                "movdqa %%xmm0,496(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "memory"
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
                "movdqa %%xmm0,(%%r9);movdqa %%xmm1,16(%%r9);"
                "movdqa %%xmm0,32(%%r9);movdqa %%xmm1,48(%%r9);"
                "movdqa %%xmm0,64(%%r9);movdqa %%xmm1,80(%%r9);"
                "movdqa %%xmm0,96(%%r9);movdqa %%xmm1,112(%%r9);"
                "movdqa %%xmm0,128(%%r9);movdqa %%xmm1,144(%%r9);"
                "movdqa %%xmm0,160(%%r9);movdqa %%xmm1,176(%%r9);"
                "movdqa %%xmm0,192(%%r9);movdqa %%xmm1,208(%%r9);"
                "movdqa %%xmm0,224(%%r9);movdqa %%xmm1,240(%%r9);"
                "movdqa %%xmm0,256(%%r9);movdqa %%xmm1,272(%%r9);"
                "movdqa %%xmm0,288(%%r9);movdqa %%xmm1,304(%%r9);"
                "movdqa %%xmm0,320(%%r9);movdqa %%xmm1,336(%%r9);"
                "movdqa %%xmm0,352(%%r9);movdqa %%xmm1,368(%%r9);"
                "movdqa %%xmm0,384(%%r9);movdqa %%xmm1,400(%%r9);"
                "movdqa %%xmm0,416(%%r9);movdqa %%xmm1,432(%%r9);"
                "movdqa %%xmm0,448(%%r9);movdqa %%xmm1,464(%%r9);"
                "movdqa %%xmm0,480(%%r9);movdqa %%xmm1,496(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "memory"
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
                "movdqa %%xmm0,(%%r9);movdqa %%xmm1,16(%%r9);movdqa %%xmm2,32(%%r9);"
                "movdqa %%xmm0,48(%%r9);movdqa %%xmm1,64(%%r9);movdqa %%xmm2,80(%%r9);"
                "movdqa %%xmm0,96(%%r9);movdqa %%xmm1,112(%%r9);movdqa %%xmm2,128(%%r9);"
                "movdqa %%xmm0,144(%%r9);movdqa %%xmm1,160(%%r9);movdqa %%xmm2,176(%%r9);"
                "movdqa %%xmm0,192(%%r9);movdqa %%xmm1,208(%%r9);movdqa %%xmm2,224(%%r9);"
                "movdqa %%xmm0,240(%%r9);movdqa %%xmm1,256(%%r9);movdqa %%xmm2,272(%%r9);"
                "movdqa %%xmm0,288(%%r9);movdqa %%xmm1,304(%%r9);movdqa %%xmm2,320(%%r9);"
                "movdqa %%xmm0,336(%%r9);movdqa %%xmm1,352(%%r9);movdqa %%xmm2,368(%%r9);"
                "movdqa %%xmm0,384(%%r9);movdqa %%xmm1,400(%%r9);movdqa %%xmm2,416(%%r9);"
                "movdqa %%xmm0,432(%%r9);movdqa %%xmm1,448(%%r9);movdqa %%xmm2,464(%%r9);"
                "movdqa %%xmm0,480(%%r9);movdqa %%xmm1,496(%%r9);movdqa %%xmm2,512(%%r9);"
                "movdqa %%xmm0,528(%%r9);movdqa %%xmm1,544(%%r9);movdqa %%xmm2,560(%%r9);"
                "movdqa %%xmm0,576(%%r9);movdqa %%xmm1,592(%%r9);movdqa %%xmm2,608(%%r9);"
                "movdqa %%xmm0,624(%%r9);movdqa %%xmm1,640(%%r9);movdqa %%xmm2,656(%%r9);"
                "movdqa %%xmm0,672(%%r9);movdqa %%xmm1,688(%%r9);movdqa %%xmm2,704(%%r9);"
                "movdqa %%xmm0,720(%%r9);movdqa %%xmm1,736(%%r9);movdqa %%xmm2,752(%%r9);"     
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "memory"
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
                "movdqa %%xmm0,(%%r9);movdqa %%xmm1,16(%%r9);movdqa %%xmm2,32(%%r9);movdqa %%xmm3,48(%%r9);"
                "movdqa %%xmm0,64(%%r9);movdqa %%xmm1,80(%%r9);movdqa %%xmm2,96(%%r9);movdqa %%xmm3,112(%%r9);"
                "movdqa %%xmm0,128(%%r9);movdqa %%xmm1,144(%%r9);movdqa %%xmm2,160(%%r9);movdqa %%xmm3,176(%%r9);"
                "movdqa %%xmm0,192(%%r9);movdqa %%xmm1,208(%%r9);movdqa %%xmm2,224(%%r9);movdqa %%xmm3,240(%%r9);"
                "movdqa %%xmm0,256(%%r9);movdqa %%xmm1,272(%%r9);movdqa %%xmm2,288(%%r9);movdqa %%xmm3,304(%%r9);"
                "movdqa %%xmm0,320(%%r9);movdqa %%xmm1,336(%%r9);movdqa %%xmm2,352(%%r9);movdqa %%xmm3,368(%%r9);"
                "movdqa %%xmm0,384(%%r9);movdqa %%xmm1,400(%%r9);movdqa %%xmm2,416(%%r9);movdqa %%xmm3,432(%%r9);"
                "movdqa %%xmm0,448(%%r9);movdqa %%xmm1,464(%%r9);movdqa %%xmm2,480(%%r9);movdqa %%xmm3,496(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
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
                "movdqu %%xmm0,(%%r9);"
                "movdqu %%xmm0,16(%%r9);"
                "movdqu %%xmm0,32(%%r9);"
                "movdqu %%xmm0,48(%%r9);"
                "movdqu %%xmm0,64(%%r9);"
                "movdqu %%xmm0,80(%%r9);"
                "movdqu %%xmm0,96(%%r9);"
                "movdqu %%xmm0,112(%%r9);"
                "movdqu %%xmm0,128(%%r9);"
                "movdqu %%xmm0,144(%%r9);"
                "movdqu %%xmm0,160(%%r9);"
                "movdqu %%xmm0,176(%%r9);"
                "movdqu %%xmm0,192(%%r9);"
                "movdqu %%xmm0,208(%%r9);"
                "movdqu %%xmm0,224(%%r9);"
                "movdqu %%xmm0,240(%%r9);"
                "movdqu %%xmm0,256(%%r9);"
                "movdqu %%xmm0,272(%%r9);"
                "movdqu %%xmm0,288(%%r9);"
                "movdqu %%xmm0,304(%%r9);"
                "movdqu %%xmm0,320(%%r9);"
                "movdqu %%xmm0,336(%%r9);"
                "movdqu %%xmm0,352(%%r9);"
                "movdqu %%xmm0,368(%%r9);"
                "movdqu %%xmm0,384(%%r9);"
                "movdqu %%xmm0,400(%%r9);"
                "movdqu %%xmm0,416(%%r9);"
                "movdqu %%xmm0,432(%%r9);"
                "movdqu %%xmm0,448(%%r9);"
                "movdqu %%xmm0,464(%%r9);"
                "movdqu %%xmm0,480(%%r9);"
                "movdqu %%xmm0,496(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "memory"
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
                "movdqu %%xmm0,(%%r9);movdqu %%xmm1,16(%%r9);"
                "movdqu %%xmm0,32(%%r9);movdqu %%xmm1,48(%%r9);"
                "movdqu %%xmm0,64(%%r9);movdqu %%xmm1,80(%%r9);"
                "movdqu %%xmm0,96(%%r9);movdqu %%xmm1,112(%%r9);"
                "movdqu %%xmm0,128(%%r9);movdqu %%xmm1,144(%%r9);"
                "movdqu %%xmm0,160(%%r9);movdqu %%xmm1,176(%%r9);"
                "movdqu %%xmm0,192(%%r9);movdqu %%xmm1,208(%%r9);"
                "movdqu %%xmm0,224(%%r9);movdqu %%xmm1,240(%%r9);"
                "movdqu %%xmm0,256(%%r9);movdqu %%xmm1,272(%%r9);"
                "movdqu %%xmm0,288(%%r9);movdqu %%xmm1,304(%%r9);"
                "movdqu %%xmm0,320(%%r9);movdqu %%xmm1,336(%%r9);"
                "movdqu %%xmm0,352(%%r9);movdqu %%xmm1,368(%%r9);"
                "movdqu %%xmm0,384(%%r9);movdqu %%xmm1,400(%%r9);"
                "movdqu %%xmm0,416(%%r9);movdqu %%xmm1,432(%%r9);"
                "movdqu %%xmm0,448(%%r9);movdqu %%xmm1,464(%%r9);"
                "movdqu %%xmm0,480(%%r9);movdqu %%xmm1,496(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "memory"
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
                "movdqu %%xmm0,(%%r9);movdqu %%xmm1,16(%%r9);movdqu %%xmm2,32(%%r9);"
                "movdqu %%xmm0,48(%%r9);movdqu %%xmm1,64(%%r9);movdqu %%xmm2,80(%%r9);"
                "movdqu %%xmm0,96(%%r9);movdqu %%xmm1,112(%%r9);movdqu %%xmm2,128(%%r9);"
                "movdqu %%xmm0,144(%%r9);movdqu %%xmm1,160(%%r9);movdqu %%xmm2,176(%%r9);"
                "movdqu %%xmm0,192(%%r9);movdqu %%xmm1,208(%%r9);movdqu %%xmm2,224(%%r9);"
                "movdqu %%xmm0,240(%%r9);movdqu %%xmm1,256(%%r9);movdqu %%xmm2,272(%%r9);"
                "movdqu %%xmm0,288(%%r9);movdqu %%xmm1,304(%%r9);movdqu %%xmm2,320(%%r9);"
                "movdqu %%xmm0,336(%%r9);movdqu %%xmm1,352(%%r9);movdqu %%xmm2,368(%%r9);"
                "movdqu %%xmm0,384(%%r9);movdqu %%xmm1,400(%%r9);movdqu %%xmm2,416(%%r9);"
                "movdqu %%xmm0,432(%%r9);movdqu %%xmm1,448(%%r9);movdqu %%xmm2,464(%%r9);"
                "movdqu %%xmm0,480(%%r9);movdqu %%xmm1,496(%%r9);movdqu %%xmm2,512(%%r9);"
                "movdqu %%xmm0,528(%%r9);movdqu %%xmm1,544(%%r9);movdqu %%xmm2,560(%%r9);"
                "movdqu %%xmm0,576(%%r9);movdqu %%xmm1,592(%%r9);movdqu %%xmm2,608(%%r9);"
                "movdqu %%xmm0,624(%%r9);movdqu %%xmm1,640(%%r9);movdqu %%xmm2,656(%%r9);"
                "movdqu %%xmm0,672(%%r9);movdqu %%xmm1,688(%%r9);movdqu %%xmm2,704(%%r9);"
                "movdqu %%xmm0,720(%%r9);movdqu %%xmm1,736(%%r9);movdqu %%xmm2,752(%%r9);"     
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "memory"
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
                "movdqu %%xmm0,(%%r9);movdqu %%xmm1,16(%%r9);movdqu %%xmm2,32(%%r9);movdqu %%xmm3,48(%%r9);"
                "movdqu %%xmm0,64(%%r9);movdqu %%xmm1,80(%%r9);movdqu %%xmm2,96(%%r9);movdqu %%xmm3,112(%%r9);"
                "movdqu %%xmm0,128(%%r9);movdqu %%xmm1,144(%%r9);movdqu %%xmm2,160(%%r9);movdqu %%xmm3,176(%%r9);"
                "movdqu %%xmm0,192(%%r9);movdqu %%xmm1,208(%%r9);movdqu %%xmm2,224(%%r9);movdqu %%xmm3,240(%%r9);"
                "movdqu %%xmm0,256(%%r9);movdqu %%xmm1,272(%%r9);movdqu %%xmm2,288(%%r9);movdqu %%xmm3,304(%%r9);"
                "movdqu %%xmm0,320(%%r9);movdqu %%xmm1,336(%%r9);movdqu %%xmm2,352(%%r9);movdqu %%xmm3,368(%%r9);"
                "movdqu %%xmm0,384(%%r9);movdqu %%xmm1,400(%%r9);movdqu %%xmm2,416(%%r9);movdqu %%xmm3,432(%%r9);"
                "movdqu %%xmm0,448(%%r9);movdqu %%xmm1,464(%%r9);movdqu %%xmm2,480(%%r9);movdqu %%xmm3,496(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
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

double asm_work_movntdq(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_movntdq(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
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
                "_sync0_movntdq_1:"            //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movntdq_1;"        //|<
                "_sync1_movntdq_1:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movntdq_1;"        //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movntdq_1;" //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movntdq_1;"         //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movntdq_1;"        //|<
                "_wait_movntdq_1:"             //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movntdq_1;"         //|<
                "_sync2_movntdq_1:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movntdq_1;"        //|<
                "_skip_tsc_sync_movntdq_1:"    //<<
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
                "_work_loop_movntdq_1:"
                "movntdq %%xmm0,(%%r9);"
                "movntdq %%xmm0,16(%%r9);"
                "movntdq %%xmm0,32(%%r9);"
                "movntdq %%xmm0,48(%%r9);"
                "movntdq %%xmm0,64(%%r9);"
                "movntdq %%xmm0,80(%%r9);"
                "movntdq %%xmm0,96(%%r9);"
                "movntdq %%xmm0,112(%%r9);"
                "movntdq %%xmm0,128(%%r9);"
                "movntdq %%xmm0,144(%%r9);"
                "movntdq %%xmm0,160(%%r9);"
                "movntdq %%xmm0,176(%%r9);"
                "movntdq %%xmm0,192(%%r9);"
                "movntdq %%xmm0,208(%%r9);"
                "movntdq %%xmm0,224(%%r9);"
                "movntdq %%xmm0,240(%%r9);"
                "movntdq %%xmm0,256(%%r9);"
                "movntdq %%xmm0,272(%%r9);"
                "movntdq %%xmm0,288(%%r9);"
                "movntdq %%xmm0,304(%%r9);"
                "movntdq %%xmm0,320(%%r9);"
                "movntdq %%xmm0,336(%%r9);"
                "movntdq %%xmm0,352(%%r9);"
                "movntdq %%xmm0,368(%%r9);"
                "movntdq %%xmm0,384(%%r9);"
                "movntdq %%xmm0,400(%%r9);"
                "movntdq %%xmm0,416(%%r9);"
                "movntdq %%xmm0,432(%%r9);"
                "movntdq %%xmm0,448(%%r9);"
                "movntdq %%xmm0,464(%%r9);"
                "movntdq %%xmm0,480(%%r9);"
                "movntdq %%xmm0,496(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movntdq_1;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "memory"
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
                "_sync0_movntdq_2:"            //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movntdq_2;"        //|<
                "_sync1_movntdq_2:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movntdq_2;"        //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movntdq_2;" //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movntdq_2;"         //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movntdq_2;"        //|<
                "_wait_movntdq_2:"             //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movntdq_2;"         //|<
                "_sync2_movntdq_2:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movntdq_2;"        //|<
                "_skip_tsc_sync_movntdq_2:"    //<<
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
                "_work_loop_movntdq_2:"
                "movntdq %%xmm0,(%%r9);movntdq %%xmm1,16(%%r9);"
                "movntdq %%xmm0,32(%%r9);movntdq %%xmm1,48(%%r9);"
                "movntdq %%xmm0,64(%%r9);movntdq %%xmm1,80(%%r9);"
                "movntdq %%xmm0,96(%%r9);movntdq %%xmm1,112(%%r9);"
                "movntdq %%xmm0,128(%%r9);movntdq %%xmm1,144(%%r9);"
                "movntdq %%xmm0,160(%%r9);movntdq %%xmm1,176(%%r9);"
                "movntdq %%xmm0,192(%%r9);movntdq %%xmm1,208(%%r9);"
                "movntdq %%xmm0,224(%%r9);movntdq %%xmm1,240(%%r9);"
                "movntdq %%xmm0,256(%%r9);movntdq %%xmm1,272(%%r9);"
                "movntdq %%xmm0,288(%%r9);movntdq %%xmm1,304(%%r9);"
                "movntdq %%xmm0,320(%%r9);movntdq %%xmm1,336(%%r9);"
                "movntdq %%xmm0,352(%%r9);movntdq %%xmm1,368(%%r9);"
                "movntdq %%xmm0,384(%%r9);movntdq %%xmm1,400(%%r9);"
                "movntdq %%xmm0,416(%%r9);movntdq %%xmm1,432(%%r9);"
                "movntdq %%xmm0,448(%%r9);movntdq %%xmm1,464(%%r9);"
                "movntdq %%xmm0,480(%%r9);movntdq %%xmm1,496(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movntdq_2;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "memory"
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
                "_sync0_movntdq_3:"            //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movntdq_3;"        //|<
                "_sync1_movntdq_3:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movntdq_3;"        //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movntdq_3;" //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movntdq_3;"         //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movntdq_3;"        //|<
                "_wait_movntdq_3:"             //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movntdq_3;"         //|<
                "_sync2_movntdq_3:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movntdq_3;"        //|<
                "_skip_tsc_sync_movntdq_3:"    //<<
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
                "_work_loop_movntdq_3:"
                "movntdq %%xmm0,(%%r9);movntdq %%xmm1,16(%%r9);movntdq %%xmm2,32(%%r9);"
                "movntdq %%xmm0,48(%%r9);movntdq %%xmm1,64(%%r9);movntdq %%xmm2,80(%%r9);"
                "movntdq %%xmm0,96(%%r9);movntdq %%xmm1,112(%%r9);movntdq %%xmm2,128(%%r9);"
                "movntdq %%xmm0,144(%%r9);movntdq %%xmm1,160(%%r9);movntdq %%xmm2,176(%%r9);"
                "movntdq %%xmm0,192(%%r9);movntdq %%xmm1,208(%%r9);movntdq %%xmm2,224(%%r9);"
                "movntdq %%xmm0,240(%%r9);movntdq %%xmm1,256(%%r9);movntdq %%xmm2,272(%%r9);"
                "movntdq %%xmm0,288(%%r9);movntdq %%xmm1,304(%%r9);movntdq %%xmm2,320(%%r9);"
                "movntdq %%xmm0,336(%%r9);movntdq %%xmm1,352(%%r9);movntdq %%xmm2,368(%%r9);"
                "movntdq %%xmm0,384(%%r9);movntdq %%xmm1,400(%%r9);movntdq %%xmm2,416(%%r9);"
                "movntdq %%xmm0,432(%%r9);movntdq %%xmm1,448(%%r9);movntdq %%xmm2,464(%%r9);"
                "movntdq %%xmm0,480(%%r9);movntdq %%xmm1,496(%%r9);movntdq %%xmm2,512(%%r9);"
                "movntdq %%xmm0,528(%%r9);movntdq %%xmm1,544(%%r9);movntdq %%xmm2,560(%%r9);"
                "movntdq %%xmm0,576(%%r9);movntdq %%xmm1,592(%%r9);movntdq %%xmm2,608(%%r9);"
                "movntdq %%xmm0,624(%%r9);movntdq %%xmm1,640(%%r9);movntdq %%xmm2,656(%%r9);"
                "movntdq %%xmm0,672(%%r9);movntdq %%xmm1,688(%%r9);movntdq %%xmm2,704(%%r9);"
                "movntdq %%xmm0,720(%%r9);movntdq %%xmm1,736(%%r9);movntdq %%xmm2,752(%%r9);"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movntdq_3;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "memory"
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
                "_sync0_movntdq_4:"            //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movntdq_4;"        //|<
                "_sync1_movntdq_4:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movntdq_4;"        //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movntdq_4;" //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movntdq_4;"         //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movntdq_4;"        //|<
                "_wait_movntdq_4:"             //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movntdq_4;"         //|<
                "_sync2_movntdq_4:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movntdq_4;"        //|<
                "_skip_tsc_sync_movntdq_4:"    //<<
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
                "_work_loop_movntdq_4:"
                "movntdq %%xmm0,(%%r9);movntdq %%xmm1,16(%%r9);movntdq %%xmm2,32(%%r9);movntdq %%xmm3,48(%%r9);"
                "movntdq %%xmm0,64(%%r9);movntdq %%xmm1,80(%%r9);movntdq %%xmm2,96(%%r9);movntdq %%xmm3,112(%%r9);"
                "movntdq %%xmm0,128(%%r9);movntdq %%xmm1,144(%%r9);movntdq %%xmm2,160(%%r9);movntdq %%xmm3,176(%%r9);"
                "movntdq %%xmm0,192(%%r9);movntdq %%xmm1,208(%%r9);movntdq %%xmm2,224(%%r9);movntdq %%xmm3,240(%%r9);"
                "movntdq %%xmm0,256(%%r9);movntdq %%xmm1,272(%%r9);movntdq %%xmm2,288(%%r9);movntdq %%xmm3,304(%%r9);"
                "movntdq %%xmm0,320(%%r9);movntdq %%xmm1,336(%%r9);movntdq %%xmm2,352(%%r9);movntdq %%xmm3,368(%%r9);"
                "movntdq %%xmm0,384(%%r9);movntdq %%xmm1,400(%%r9);movntdq %%xmm2,416(%%r9);movntdq %%xmm3,432(%%r9);"
                "movntdq %%xmm0,448(%%r9);movntdq %%xmm1,464(%%r9);movntdq %%xmm2,480(%%r9);movntdq %%xmm3,496(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movntdq_4;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
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

double asm_work_movnti(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_movnti(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
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
                "_sync0_movnti_1:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movnti_1;"         //|<
                "_sync1_movnti_1:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movnti_1;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movnti_1;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movnti_1;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movnti_1;"         //|<
                "_wait_movnti_1:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movnti_1;"          //|<
                "_sync2_movnti_1:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movnti_1;"         //|<
                "_skip_tsc_sync_movnti_1:"     //<<
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
                "_work_loop_movnti_1:"
                "movnti %%r12, (%%r9);"
                "movnti %%r12, 8(%%r9);"
                "movnti %%r12, 16(%%r9);"
                "movnti %%r12, 24(%%r9);"
                "movnti %%r12, 32(%%r9);"
                "movnti %%r12, 40(%%r9);"
                "movnti %%r12, 48(%%r9);"
                "movnti %%r12, 56(%%r9);"
                "movnti %%r12, 64(%%r9);"
                "movnti %%r12, 72(%%r9);"
                "movnti %%r12, 80(%%r9);"
                "movnti %%r12, 88(%%r9);"
                "movnti %%r12, 96(%%r9);"
                "movnti %%r12, 104(%%r9);"
                "movnti %%r12, 112(%%r9);"
                "movnti %%r12, 120(%%r9);"
                "movnti %%r12, 128(%%r9);"
                "movnti %%r12, 136(%%r9);"
                "movnti %%r12, 144(%%r9);"
                "movnti %%r12, 152(%%r9);"
                "movnti %%r12, 160(%%r9);"
                "movnti %%r12, 168(%%r9);"
                "movnti %%r12, 176(%%r9);"
                "movnti %%r12, 184(%%r9);"
                "movnti %%r12, 192(%%r9);"
                "movnti %%r12, 200(%%r9);"
                "movnti %%r12, 208(%%r9);"
                "movnti %%r12, 216(%%r9);"
                "movnti %%r12, 224(%%r9);"
                "movnti %%r12, 232(%%r9);"
                "movnti %%r12, 240(%%r9);"
                "movnti %%r12, 248(%%r9);"
                "movnti %%r12, 256(%%r9);"
                "movnti %%r12, 264(%%r9);"
                "movnti %%r12, 272(%%r9);"
                "movnti %%r12, 280(%%r9);"
                "movnti %%r12, 288(%%r9);"
                "movnti %%r12, 296(%%r9);"
                "movnti %%r12, 304(%%r9);"
                "movnti %%r12, 312(%%r9);"
                "movnti %%r12, 320(%%r9);"
                "movnti %%r12, 328(%%r9);"
                "movnti %%r12, 336(%%r9);"
                "movnti %%r12, 344(%%r9);"
                "movnti %%r12, 352(%%r9);"
                "movnti %%r12, 360(%%r9);"
                "movnti %%r12, 368(%%r9);"
                "movnti %%r12, 376(%%r9);"
                "movnti %%r12, 384(%%r9);"
                "movnti %%r12, 392(%%r9);"
                "movnti %%r12, 400(%%r9);"
                "movnti %%r12, 408(%%r9);"
                "movnti %%r12, 416(%%r9);"
                "movnti %%r12, 424(%%r9);"
                "movnti %%r12, 432(%%r9);"
                "movnti %%r12, 440(%%r9);"
                "movnti %%r12, 448(%%r9);"
                "movnti %%r12, 456(%%r9);"
                "movnti %%r12, 464(%%r9);"
                "movnti %%r12, 472(%%r9);"
                "movnti %%r12, 480(%%r9);"
                "movnti %%r12, 488(%%r9);"
                "movnti %%r12, 496(%%r9);"
                "movnti %%r12, 504(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movnti_1;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "memory"
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
                "_sync0_movnti_2:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movnti_2;"         //|<
                "_sync1_movnti_2:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movnti_2;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movnti_2;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movnti_2;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movnti_2;"         //|<
                "_wait_movnti_2:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movnti_2;"          //|<
                "_sync2_movnti_2:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movnti_2;"         //|<
                "_skip_tsc_sync_movnti_2:"     //<<
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
                "_work_loop_movnti_2:"
                "movnti %%r12, (%%r9);movnti %%r13, 8(%%r9);"
                "movnti %%r12, 16(%%r9);movnti %%r13, 24(%%r9);"
                "movnti %%r12, 32(%%r9);movnti %%r13, 40(%%r9);"
                "movnti %%r12, 48(%%r9);movnti %%r13, 56(%%r9);"
                "movnti %%r12, 64(%%r9);movnti %%r13, 72(%%r9);"
                "movnti %%r12, 80(%%r9);movnti %%r13, 88(%%r9);"
                "movnti %%r12, 96(%%r9);movnti %%r13, 104(%%r9);"
                "movnti %%r12, 112(%%r9);movnti %%r13, 120(%%r9);"
                "movnti %%r12, 128(%%r9);movnti %%r13, 136(%%r9);"
                "movnti %%r12, 144(%%r9);movnti %%r13, 152(%%r9);"
                "movnti %%r12, 160(%%r9);movnti %%r13, 168(%%r9);"
                "movnti %%r12, 176(%%r9);movnti %%r13, 184(%%r9);"
                "movnti %%r12, 192(%%r9);movnti %%r13, 200(%%r9);"
                "movnti %%r12, 208(%%r9);movnti %%r13, 216(%%r9);"
                "movnti %%r12, 224(%%r9);movnti %%r13, 232(%%r9);"
                "movnti %%r12, 240(%%r9);movnti %%r13, 248(%%r9);"
                "movnti %%r12, 256(%%r9);movnti %%r13, 264(%%r9);"
                "movnti %%r12, 272(%%r9);movnti %%r13, 280(%%r9);"
                "movnti %%r12, 288(%%r9);movnti %%r13, 296(%%r9);"
                "movnti %%r12, 304(%%r9);movnti %%r13, 312(%%r9);"
                "movnti %%r12, 320(%%r9);movnti %%r13, 328(%%r9);"
                "movnti %%r12, 336(%%r9);movnti %%r13, 344(%%r9);"
                "movnti %%r12, 352(%%r9);movnti %%r13, 360(%%r9);"
                "movnti %%r12, 368(%%r9);movnti %%r13, 376(%%r9);"
                "movnti %%r12, 384(%%r9);movnti %%r13, 392(%%r9);"
                "movnti %%r12, 400(%%r9);movnti %%r13, 408(%%r9);"
                "movnti %%r12, 416(%%r9);movnti %%r13, 424(%%r9);"
                "movnti %%r12, 432(%%r9);movnti %%r13, 440(%%r9);"
                "movnti %%r12, 448(%%r9);movnti %%r13, 456(%%r9);"
                "movnti %%r12, 464(%%r9);movnti %%r13, 472(%%r9);"
                "movnti %%r12, 480(%%r9);movnti %%r13, 488(%%r9);"
                "movnti %%r12, 496(%%r9);movnti %%r13, 504(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movnti_2;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "memory"
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
                "_sync0_movnti_3:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movnti_3;"         //|<
                "_sync1_movnti_3:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movnti_3;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movnti_3;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movnti_3;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movnti_3;"         //|<
                "_wait_movnti_3:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movnti_3;"          //|<
                "_sync2_movnti_3:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movnti_3;"         //|<
                "_skip_tsc_sync_movnti_3:"     //<<
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
                "_work_loop_movnti_3:"
                "movnti %%r12, (%%r9);movnti %%r13, 8(%%r9);movnti %%r14, 16(%%r9);"
                "movnti %%r12, 24(%%r9);movnti %%r13, 32(%%r9);movnti %%r14, 40(%%r9);"
                "movnti %%r12, 48(%%r9);movnti %%r13, 56(%%r9);movnti %%r14, 64(%%r9);"
                "movnti %%r12, 72(%%r9);movnti %%r13, 80(%%r9);movnti %%r14, 88(%%r9);"
                "movnti %%r12, 96(%%r9);movnti %%r13, 104(%%r9);movnti %%r14, 112(%%r9);"
                "movnti %%r12, 120(%%r9);movnti %%r13, 128(%%r9);movnti %%r14, 136(%%r9);"
                "movnti %%r12, 144(%%r9);movnti %%r13, 152(%%r9);movnti %%r14, 160(%%r9);"
                "movnti %%r12, 168(%%r9);movnti %%r13, 176(%%r9);movnti %%r14, 184(%%r9);"
                "movnti %%r12, 192(%%r9);movnti %%r13, 200(%%r9);movnti %%r14, 208(%%r9);"
                "movnti %%r12, 216(%%r9);movnti %%r13, 224(%%r9);movnti %%r14, 232(%%r9);"
                "movnti %%r12, 240(%%r9);movnti %%r13, 248(%%r9);movnti %%r14, 256(%%r9);"
                "movnti %%r12, 264(%%r9);movnti %%r13, 272(%%r9);movnti %%r14, 280(%%r9);"
                "movnti %%r12, 288(%%r9);movnti %%r13, 296(%%r9);movnti %%r14, 304(%%r9);"
                "movnti %%r12, 312(%%r9);movnti %%r13, 320(%%r9);movnti %%r14, 328(%%r9);"
                "movnti %%r12, 336(%%r9);movnti %%r13, 344(%%r9);movnti %%r14, 352(%%r9);"
                "movnti %%r12, 360(%%r9);movnti %%r13, 368(%%r9);movnti %%r14, 376(%%r9);"
                "movnti %%r12, 384(%%r9);movnti %%r13, 392(%%r9);movnti %%r14, 400(%%r9);"
                "movnti %%r12, 408(%%r9);movnti %%r13, 416(%%r9);movnti %%r14, 424(%%r9);"
                "movnti %%r12, 432(%%r9);movnti %%r13, 440(%%r9);movnti %%r14, 448(%%r9);"
                "movnti %%r12, 456(%%r9);movnti %%r13, 464(%%r9);movnti %%r14, 472(%%r9);"
                "movnti %%r12, 480(%%r9);movnti %%r13, 488(%%r9);movnti %%r14, 496(%%r9);"
                "movnti %%r12, 504(%%r9);movnti %%r13, 512(%%r9);movnti %%r14, 520(%%r9);"
                "movnti %%r12, 528(%%r9);movnti %%r13, 536(%%r9);movnti %%r14, 544(%%r9);"
                "movnti %%r12, 552(%%r9);movnti %%r13, 560(%%r9);movnti %%r14, 568(%%r9);"
                "movnti %%r12, 576(%%r9);movnti %%r13, 584(%%r9);movnti %%r14, 592(%%r9);"
                "movnti %%r12, 600(%%r9);movnti %%r13, 608(%%r9);movnti %%r14, 616(%%r9);"
                "movnti %%r12, 624(%%r9);movnti %%r13, 632(%%r9);movnti %%r14, 640(%%r9);"
                "movnti %%r12, 648(%%r9);movnti %%r13, 656(%%r9);movnti %%r14, 664(%%r9);"
                "movnti %%r12, 672(%%r9);movnti %%r13, 680(%%r9);movnti %%r14, 688(%%r9);"
                "movnti %%r12, 696(%%r9);movnti %%r13, 704(%%r9);movnti %%r14, 712(%%r9);"
                "movnti %%r12, 720(%%r9);movnti %%r13, 728(%%r9);movnti %%r14, 736(%%r9);"
                "movnti %%r12, 744(%%r9);movnti %%r13, 752(%%r9);movnti %%r14, 760(%%r9);" 
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movnti_3;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "r14", "memory"
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
                "_sync0_movnti_4:"             //|atomically replace thread_id (r12) with thread_id+1 (rbx)>
                  "mov %%r12,%%rax;"           //|
                  "lock cmpxchg %%bl,(%%r8);"  //|
                "jnz _sync0_movnti_4;"         //|<
                "_sync1_movnti_4:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"          //|
                "jne _sync1_movnti_4;"         //<< 
                "cmp $0,%%r13;"                //Phase 2: TSC (optimization for concurrent start of all threads) >>
                "je _skip_tsc_sync_movnti_4;"  //|skip if not available
                "cmp $0,%%r12;"                //|master thread selects start time in future >
                "jne _wait_movnti_4;"          //|
                "rdtsc;"                       //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                "add $10000,%%rax;"            //|
                "mov %%rax,8(%%r8);"           //|
                "mov %%rax,%%r13;"             //|
                "mfence;"                      //|
                "jmp _sync2_movnti_4;"         //|<
                "_wait_movnti_4:"              //|other threads wait until start time is defined  >
                  "mov 8(%%r8),%%r13;"         //|
                  "cmp $1,%%r13;"              //|
                "jle _wait_movnti_4;"          //|<
                "_sync2_movnti_4:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                     //|
                "shl $32,%%rdx;"               //|
                "add %%rdx,%%rax;"             //|
                  "cmp %%rax,%%r13;"           //|
                "jge _sync2_movnti_4;"         //|<
                "_skip_tsc_sync_movnti_4:"     //<<
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
                "_work_loop_movnti_4:"
                "movnti %%r12, (%%r9);movnti %%r13, 8(%%r9);movnti %%r14, 16(%%r9);movnti %%r15, 24(%%r9);"
                "movnti %%r12, 32(%%r9);movnti %%r13, 40(%%r9);movnti %%r14, 48(%%r9);movnti %%r15, 56(%%r9);"
                "movnti %%r12, 64(%%r9);movnti %%r13, 72(%%r9);movnti %%r14, 80(%%r9);movnti %%r15, 88(%%r9);"
                "movnti %%r12, 96(%%r9);movnti %%r13, 104(%%r9);movnti %%r14, 112(%%r9);movnti %%r15, 120(%%r9);"
                "movnti %%r12, 128(%%r9);movnti %%r13, 136(%%r9);movnti %%r14, 144(%%r9);movnti %%r15, 152(%%r9);"
                "movnti %%r12, 160(%%r9);movnti %%r13, 168(%%r9);movnti %%r14, 176(%%r9);movnti %%r15, 184(%%r9);"
                "movnti %%r12, 192(%%r9);movnti %%r13, 200(%%r9);movnti %%r14, 208(%%r9);movnti %%r15, 216(%%r9);"
                "movnti %%r12, 224(%%r9);movnti %%r13, 232(%%r9);movnti %%r14, 240(%%r9);movnti %%r15, 248(%%r9);"
                "movnti %%r12, 256(%%r9);movnti %%r13, 264(%%r9);movnti %%r14, 272(%%r9);movnti %%r15, 280(%%r9);"
                "movnti %%r12, 288(%%r9);movnti %%r13, 296(%%r9);movnti %%r14, 304(%%r9);movnti %%r15, 312(%%r9);"
                "movnti %%r12, 320(%%r9);movnti %%r13, 328(%%r9);movnti %%r14, 336(%%r9);movnti %%r15, 344(%%r9);"
                "movnti %%r12, 352(%%r9);movnti %%r13, 360(%%r9);movnti %%r14, 368(%%r9);movnti %%r15, 376(%%r9);"
                "movnti %%r12, 384(%%r9);movnti %%r13, 392(%%r9);movnti %%r14, 400(%%r9);movnti %%r15, 408(%%r9);"
                "movnti %%r12, 416(%%r9);movnti %%r13, 424(%%r9);movnti %%r14, 432(%%r9);movnti %%r15, 440(%%r9);"
                "movnti %%r12, 448(%%r9);movnti %%r13, 456(%%r9);movnti %%r14, 464(%%r9);movnti %%r15, 472(%%r9);"
                "movnti %%r12, 480(%%r9);movnti %%r13, 488(%%r9);movnti %%r14, 496(%%r9);movnti %%r15, 504(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _work_loop_movnti_4;"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "memory"
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
                "mov %%r12, (%%r9);"
                "mov %%r12, 8(%%r9);"
                "mov %%r12, 16(%%r9);"
                "mov %%r12, 24(%%r9);"
                "mov %%r12, 32(%%r9);"
                "mov %%r12, 40(%%r9);"
                "mov %%r12, 48(%%r9);"
                "mov %%r12, 56(%%r9);"
                "mov %%r12, 64(%%r9);"
                "mov %%r12, 72(%%r9);"
                "mov %%r12, 80(%%r9);"
                "mov %%r12, 88(%%r9);"
                "mov %%r12, 96(%%r9);"
                "mov %%r12, 104(%%r9);"
                "mov %%r12, 112(%%r9);"
                "mov %%r12, 120(%%r9);"
                "mov %%r12, 128(%%r9);"
                "mov %%r12, 136(%%r9);"
                "mov %%r12, 144(%%r9);"
                "mov %%r12, 152(%%r9);"
                "mov %%r12, 160(%%r9);"
                "mov %%r12, 168(%%r9);"
                "mov %%r12, 176(%%r9);"
                "mov %%r12, 184(%%r9);"
                "mov %%r12, 192(%%r9);"
                "mov %%r12, 200(%%r9);"
                "mov %%r12, 208(%%r9);"
                "mov %%r12, 216(%%r9);"
                "mov %%r12, 224(%%r9);"
                "mov %%r12, 232(%%r9);"
                "mov %%r12, 240(%%r9);"
                "mov %%r12, 248(%%r9);"
                "mov %%r12, 256(%%r9);"
                "mov %%r12, 264(%%r9);"
                "mov %%r12, 272(%%r9);"
                "mov %%r12, 280(%%r9);"
                "mov %%r12, 288(%%r9);"
                "mov %%r12, 296(%%r9);"
                "mov %%r12, 304(%%r9);"
                "mov %%r12, 312(%%r9);"
                "mov %%r12, 320(%%r9);"
                "mov %%r12, 328(%%r9);"
                "mov %%r12, 336(%%r9);"
                "mov %%r12, 344(%%r9);"
                "mov %%r12, 352(%%r9);"
                "mov %%r12, 360(%%r9);"
                "mov %%r12, 368(%%r9);"
                "mov %%r12, 376(%%r9);"
                "mov %%r12, 384(%%r9);"
                "mov %%r12, 392(%%r9);"
                "mov %%r12, 400(%%r9);"
                "mov %%r12, 408(%%r9);"
                "mov %%r12, 416(%%r9);"
                "mov %%r12, 424(%%r9);"
                "mov %%r12, 432(%%r9);"
                "mov %%r12, 440(%%r9);"
                "mov %%r12, 448(%%r9);"
                "mov %%r12, 456(%%r9);"
                "mov %%r12, 464(%%r9);"
                "mov %%r12, 472(%%r9);"
                "mov %%r12, 480(%%r9);"
                "mov %%r12, 488(%%r9);"
                "mov %%r12, 496(%%r9);"
                "mov %%r12, 504(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "memory"
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
                "mov %%r12, (%%r9);mov %%r13, 8(%%r9);"
                "mov %%r12, 16(%%r9);mov %%r13, 24(%%r9);"
                "mov %%r12, 32(%%r9);mov %%r13, 40(%%r9);"
                "mov %%r12, 48(%%r9);mov %%r13, 56(%%r9);"
                "mov %%r12, 64(%%r9);mov %%r13, 72(%%r9);"
                "mov %%r12, 80(%%r9);mov %%r13, 88(%%r9);"
                "mov %%r12, 96(%%r9);mov %%r13, 104(%%r9);"
                "mov %%r12, 112(%%r9);mov %%r13, 120(%%r9);"
                "mov %%r12, 128(%%r9);mov %%r13, 136(%%r9);"
                "mov %%r12, 144(%%r9);mov %%r13, 152(%%r9);"
                "mov %%r12, 160(%%r9);mov %%r13, 168(%%r9);"
                "mov %%r12, 176(%%r9);mov %%r13, 184(%%r9);"
                "mov %%r12, 192(%%r9);mov %%r13, 200(%%r9);"
                "mov %%r12, 208(%%r9);mov %%r13, 216(%%r9);"
                "mov %%r12, 224(%%r9);mov %%r13, 232(%%r9);"
                "mov %%r12, 240(%%r9);mov %%r13, 248(%%r9);"
                "mov %%r12, 256(%%r9);mov %%r13, 264(%%r9);"
                "mov %%r12, 272(%%r9);mov %%r13, 280(%%r9);"
                "mov %%r12, 288(%%r9);mov %%r13, 296(%%r9);"
                "mov %%r12, 304(%%r9);mov %%r13, 312(%%r9);"
                "mov %%r12, 320(%%r9);mov %%r13, 328(%%r9);"
                "mov %%r12, 336(%%r9);mov %%r13, 344(%%r9);"
                "mov %%r12, 352(%%r9);mov %%r13, 360(%%r9);"
                "mov %%r12, 368(%%r9);mov %%r13, 376(%%r9);"
                "mov %%r12, 384(%%r9);mov %%r13, 392(%%r9);"
                "mov %%r12, 400(%%r9);mov %%r13, 408(%%r9);"
                "mov %%r12, 416(%%r9);mov %%r13, 424(%%r9);"
                "mov %%r12, 432(%%r9);mov %%r13, 440(%%r9);"
                "mov %%r12, 448(%%r9);mov %%r13, 456(%%r9);"
                "mov %%r12, 464(%%r9);mov %%r13, 472(%%r9);"
                "mov %%r12, 480(%%r9);mov %%r13, 488(%%r9);"
                "mov %%r12, 496(%%r9);mov %%r13, 504(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "memory"
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
                "mov %%r12, (%%r9);mov %%r13, 8(%%r9);mov %%r14, 16(%%r9);"
                "mov %%r12, 24(%%r9);mov %%r13, 32(%%r9);mov %%r14, 40(%%r9);"
                "mov %%r12, 48(%%r9);mov %%r13, 56(%%r9);mov %%r14, 64(%%r9);"
                "mov %%r12, 72(%%r9);mov %%r13, 80(%%r9);mov %%r14, 88(%%r9);"
                "mov %%r12, 96(%%r9);mov %%r13, 104(%%r9);mov %%r14, 112(%%r9);"
                "mov %%r12, 120(%%r9);mov %%r13, 128(%%r9);mov %%r14, 136(%%r9);"
                "mov %%r12, 144(%%r9);mov %%r13, 152(%%r9);mov %%r14, 160(%%r9);"
                "mov %%r12, 168(%%r9);mov %%r13, 176(%%r9);mov %%r14, 184(%%r9);"
                "mov %%r12, 192(%%r9);mov %%r13, 200(%%r9);mov %%r14, 208(%%r9);"
                "mov %%r12, 216(%%r9);mov %%r13, 224(%%r9);mov %%r14, 232(%%r9);"
                "mov %%r12, 240(%%r9);mov %%r13, 248(%%r9);mov %%r14, 256(%%r9);"
                "mov %%r12, 264(%%r9);mov %%r13, 272(%%r9);mov %%r14, 280(%%r9);"
                "mov %%r12, 288(%%r9);mov %%r13, 296(%%r9);mov %%r14, 304(%%r9);"
                "mov %%r12, 312(%%r9);mov %%r13, 320(%%r9);mov %%r14, 328(%%r9);"
                "mov %%r12, 336(%%r9);mov %%r13, 344(%%r9);mov %%r14, 352(%%r9);"
                "mov %%r12, 360(%%r9);mov %%r13, 368(%%r9);mov %%r14, 376(%%r9);"
                "mov %%r12, 384(%%r9);mov %%r13, 392(%%r9);mov %%r14, 400(%%r9);"
                "mov %%r12, 408(%%r9);mov %%r13, 416(%%r9);mov %%r14, 424(%%r9);"
                "mov %%r12, 432(%%r9);mov %%r13, 440(%%r9);mov %%r14, 448(%%r9);"
                "mov %%r12, 456(%%r9);mov %%r13, 464(%%r9);mov %%r14, 472(%%r9);"
                "mov %%r12, 480(%%r9);mov %%r13, 488(%%r9);mov %%r14, 496(%%r9);"
                "mov %%r12, 504(%%r9);mov %%r13, 512(%%r9);mov %%r14, 520(%%r9);"
                "mov %%r12, 528(%%r9);mov %%r13, 536(%%r9);mov %%r14, 544(%%r9);"
                "mov %%r12, 552(%%r9);mov %%r13, 560(%%r9);mov %%r14, 568(%%r9);"
                "mov %%r12, 576(%%r9);mov %%r13, 584(%%r9);mov %%r14, 592(%%r9);"
                "mov %%r12, 600(%%r9);mov %%r13, 608(%%r9);mov %%r14, 616(%%r9);"
                "mov %%r12, 624(%%r9);mov %%r13, 632(%%r9);mov %%r14, 640(%%r9);"
                "mov %%r12, 648(%%r9);mov %%r13, 656(%%r9);mov %%r14, 664(%%r9);"
                "mov %%r12, 672(%%r9);mov %%r13, 680(%%r9);mov %%r14, 688(%%r9);"
                "mov %%r12, 696(%%r9);mov %%r13, 704(%%r9);mov %%r14, 712(%%r9);"
                "mov %%r12, 720(%%r9);mov %%r13, 728(%%r9);mov %%r14, 736(%%r9);"
                "mov %%r12, 744(%%r9);mov %%r13, 752(%%r9);mov %%r14, 760(%%r9);" 
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "r14", "memory"
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
                "mov %%r12, (%%r9);mov %%r13, 8(%%r9);mov %%r14, 16(%%r9);mov %%r15, 24(%%r9);"
                "mov %%r12, 32(%%r9);mov %%r13, 40(%%r9);mov %%r14, 48(%%r9);mov %%r15, 56(%%r9);"
                "mov %%r12, 64(%%r9);mov %%r13, 72(%%r9);mov %%r14, 80(%%r9);mov %%r15, 88(%%r9);"
                "mov %%r12, 96(%%r9);mov %%r13, 104(%%r9);mov %%r14, 112(%%r9);mov %%r15, 120(%%r9);"
                "mov %%r12, 128(%%r9);mov %%r13, 136(%%r9);mov %%r14, 144(%%r9);mov %%r15, 152(%%r9);"
                "mov %%r12, 160(%%r9);mov %%r13, 168(%%r9);mov %%r14, 176(%%r9);mov %%r15, 184(%%r9);"
                "mov %%r12, 192(%%r9);mov %%r13, 200(%%r9);mov %%r14, 208(%%r9);mov %%r15, 216(%%r9);"
                "mov %%r12, 224(%%r9);mov %%r13, 232(%%r9);mov %%r14, 240(%%r9);mov %%r15, 248(%%r9);"
                "mov %%r12, 256(%%r9);mov %%r13, 264(%%r9);mov %%r14, 272(%%r9);mov %%r15, 280(%%r9);"
                "mov %%r12, 288(%%r9);mov %%r13, 296(%%r9);mov %%r14, 304(%%r9);mov %%r15, 312(%%r9);"
                "mov %%r12, 320(%%r9);mov %%r13, 328(%%r9);mov %%r14, 336(%%r9);mov %%r15, 344(%%r9);"
                "mov %%r12, 352(%%r9);mov %%r13, 360(%%r9);mov %%r14, 368(%%r9);mov %%r15, 376(%%r9);"
                "mov %%r12, 384(%%r9);mov %%r13, 392(%%r9);mov %%r14, 400(%%r9);mov %%r15, 408(%%r9);"
                "mov %%r12, 416(%%r9);mov %%r13, 424(%%r9);mov %%r14, 432(%%r9);mov %%r15, 440(%%r9);"
                "mov %%r12, 448(%%r9);mov %%r13, 456(%%r9);mov %%r14, 464(%%r9);mov %%r15, 472(%%r9);"
                "mov %%r12, 480(%%r9);mov %%r13, 488(%%r9);mov %%r14, 496(%%r9);mov %%r15, 504(%%r9);"
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
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "memory"
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

     //wait for other THREAD touching the memory
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
        case USE_MOVNTDQ: tmp=asm_work_movntdq(aligned_addr,((accesses/(t+1))),burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MOVDQA: tmp=asm_work_movdqa(aligned_addr,((accesses/(t+1))),burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MOVDQU: tmp=asm_work_movdqu(aligned_addr,((accesses/(t+1))),burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MOVNTI: tmp=asm_work_movnti(aligned_addr,((accesses/(t+1))),burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
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
  
   if ((int)tmax)
   {
     (*results)[0]=tmax;
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
