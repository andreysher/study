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
 *       - implement alternative synchronisation, that does not rely on
 *         synchronus Time Sramp Counters
 *       - adopt cache and TLB parameters to refer to identifiers returned by 
 *         the hardware detection
 *       - add serialization parameter to include additional cpuid calls
 *         (Requiered to match specification of serializing instructions, but 
 *         actually only causing additional overhead on todays systems. In current
 *         CPUs mfence seems to be serializing, however this might change in 
 *         future processors)
 *       - AVX and Larrabee support
 *       - support low level Performance Counter APIs to get access to uncore/NB events
 *       - remove unnecessary variables from performance counter implementation
 *       - improve cacheflush algorithm to take the minimal cachesize per core into acount
 *         (e.g. 2 Threads on 1 Package have 8 MB in Nehalem, 2 Threads on 2 Packages 16 MB,
 *          Shanghai has 8 MB for 4 Threads, 7 MB for 2 Threads in one package)
 *       - local alloc of flush buffer
 *       - memory layout improvements (as for single-r1w1)
 */
 
#include "work.h"
#include "interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>

#ifdef USE_PAPI
#include <papi.h>
#endif

#ifdef USE_VTRACE
#include "vt_user.h"
#endif

static int min(int a, int b)
{
 if (a<b) return a;
 else return b;
}

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
         tmp=*((int*)((unsigned long long)buffer+i));
         *((int*)((unsigned long long)buffer+i))=tmp;
       }
       if (direction==LIFO) for (i=(memsize-1)-stride;i>=0;i-=stride)
       {
         tmp=*((int*)((unsigned long long)buffer+i));
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
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_load_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_load_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pi_1;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pi_1:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pi_1;"      //|<
                "_sync1_load_pi_1:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pi_1;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pi_1;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pi_1;"      //|<
                "_wait_load_pi_1:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pi_1;"        //|<
                "_sync2_load_pi_1:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pi_1;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pi_1:"
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
                "jnz _skip_reset_load_pi_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pi_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pi_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pi_2;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pi_2:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pi_2;"      //|<
                "_sync1_load_pi_2:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pi_2;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pi_2;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pi_2;"      //|<
                "_wait_load_pi_2:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pi_2;"        //|<
                "_sync2_load_pi_2:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pi_2;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pi_2:"
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
                "jnz _skip_reset_load_pi_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pi_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pi_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pi_3;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pi_3:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pi_3;"      //|<
                "_sync1_load_pi_3:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pi_3;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pi_3;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pi_3;"      //|<
                "_wait_load_pi_3:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pi_3;"        //|<
                "_sync2_load_pi_3:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pi_3;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pi_3:"
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
                "jnz _skip_reset_load_pi_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pi_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_load_pi_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pi_4;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pi_4:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pi_4;"      //|<
                "_sync1_load_pi_4:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pi_4;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pi_4;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pi_4;"      //|<
                "_wait_load_pi_4:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pi_4;"        //|<
                "_sync2_load_pi_4:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pi_4;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pi_4:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;"
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;"
                "movdqa 256(%%r9), %%xmm0;movdqa 272(%%r9), %%xmm1;movdqa 288(%%r9), %%xmm2;movdqa 304(%%r9), %%xmm3;"
                "movdqa 320(%%r9), %%xmm0;movdqa 336(%%r9), %%xmm1;movdqa 352(%%r9), %%xmm2;movdqa 368(%%r9), %%xmm3;"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;movdqa 432(%%r9), %%xmm3;"
                "movdqa 448(%%r9), %%xmm0;movdqa 464(%%r9), %%xmm1;movdqa 480(%%r9), %%xmm2;movdqa 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_pi_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pi_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pi_4;"
                /*"sub $1,%%r10;"
                "jz _reset_load_pi_4;"
                "_skip_reset_load_pi_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pi_4;"
                "jmp end_load_pi_4;"
                "_reset_load_pi_4:"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "jmp _skip_reset_load_pi_4;"
                "end_load_pi_4:"*/
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pi_8;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pi_8:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pi_8;"      //|<
                "_sync1_load_pi_8:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pi_8;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pi_8;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pi_8;"      //|<
                "_wait_load_pi_8:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pi_8;"        //|<
                "_sync2_load_pi_8:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pi_8;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pi_8:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm4;movdqa 80(%%r9), %%xmm5;movdqa 96(%%r9), %%xmm6;movdqa 112(%%r9), %%xmm7;"
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movdqa 192(%%r9), %%xmm4;movdqa 208(%%r9), %%xmm5;movdqa 224(%%r9), %%xmm6;movdqa 240(%%r9), %%xmm7;"
                "movdqa 256(%%r9), %%xmm0;movdqa 272(%%r9), %%xmm1;movdqa 288(%%r9), %%xmm2;movdqa 304(%%r9), %%xmm3;"
                "movdqa 320(%%r9), %%xmm4;movdqa 336(%%r9), %%xmm5;movdqa 352(%%r9), %%xmm6;movdqa 368(%%r9), %%xmm7;"
                "movdqa 384(%%r9), %%xmm0;movdqa 400(%%r9), %%xmm1;movdqa 416(%%r9), %%xmm2;movdqa 432(%%r9), %%xmm3;"
                "movdqa 448(%%r9), %%xmm4;movdqa 464(%%r9), %%xmm5;movdqa 480(%%r9), %%xmm6;movdqa 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_pi_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pi_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pi_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_load_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_load_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pd_1;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pd_1:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pd_1;"      //|<
                "_sync1_load_pd_1:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pd_1;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pd_1;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pd_1;"      //|<
                "_wait_load_pd_1:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pd_1;"        //|<
                "_sync2_load_pd_1:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pd_1;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pd_1:"
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm0;"
                "movapd 32(%%r9), %%xmm0;"
                "movapd 48(%%r9), %%xmm0;"
                "movapd 64(%%r9), %%xmm0;"
                "movapd 80(%%r9), %%xmm0;"
                "movapd 96(%%r9), %%xmm0;"
                "movapd 112(%%r9), %%xmm0;"
                "movapd 128(%%r9), %%xmm0;"
                "movapd 144(%%r9), %%xmm0;"
                "movapd 160(%%r9), %%xmm0;"
                "movapd 176(%%r9), %%xmm0;"
                "movapd 192(%%r9), %%xmm0;"
                "movapd 208(%%r9), %%xmm0;"
                "movapd 224(%%r9), %%xmm0;"
                "movapd 240(%%r9), %%xmm0;"
                "movapd 256(%%r9), %%xmm0;"
                "movapd 272(%%r9), %%xmm0;"
                "movapd 288(%%r9), %%xmm0;"
                "movapd 304(%%r9), %%xmm0;"
                "movapd 320(%%r9), %%xmm0;"
                "movapd 336(%%r9), %%xmm0;"
                "movapd 352(%%r9), %%xmm0;"
                "movapd 368(%%r9), %%xmm0;"
                "movapd 384(%%r9), %%xmm0;"
                "movapd 400(%%r9), %%xmm0;"
                "movapd 416(%%r9), %%xmm0;"
                "movapd 432(%%r9), %%xmm0;"
                "movapd 448(%%r9), %%xmm0;"
                "movapd 464(%%r9), %%xmm0;"
                "movapd 480(%%r9), %%xmm0;"
                "movapd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pd_2;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pd_2:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pd_2;"      //|<
                "_sync1_load_pd_2:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pd_2;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pd_2;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pd_2;"      //|<
                "_wait_load_pd_2:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pd_2;"        //|<
                "_sync2_load_pd_2:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pd_2;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pd_2:"
                "movapd (%%r9), %%xmm0;movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm0;movapd 48(%%r9), %%xmm1;"
                "movapd 64(%%r9), %%xmm0;movapd 80(%%r9), %%xmm1;"
                "movapd 96(%%r9), %%xmm0;movapd 112(%%r9), %%xmm1;"
                "movapd 128(%%r9), %%xmm0;movapd 144(%%r9), %%xmm1;"
                "movapd 160(%%r9), %%xmm0;movapd 176(%%r9), %%xmm1;"
                "movapd 192(%%r9), %%xmm0;movapd 208(%%r9), %%xmm1;"
                "movapd 224(%%r9), %%xmm0;movapd 240(%%r9), %%xmm1;"
                "movapd 256(%%r9), %%xmm0;movapd 272(%%r9), %%xmm1;"
                "movapd 288(%%r9), %%xmm0;movapd 304(%%r9), %%xmm1;"
                "movapd 320(%%r9), %%xmm0;movapd 336(%%r9), %%xmm1;"
                "movapd 352(%%r9), %%xmm0;movapd 368(%%r9), %%xmm1;"
                "movapd 384(%%r9), %%xmm0;movapd 400(%%r9), %%xmm1;"
                "movapd 416(%%r9), %%xmm0;movapd 432(%%r9), %%xmm1;"
                "movapd 448(%%r9), %%xmm0;movapd 464(%%r9), %%xmm1;"
                "movapd 480(%%r9), %%xmm0;movapd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pd_3;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pd_3:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pd_3;"      //|<
                "_sync1_load_pd_3:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pd_3;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pd_3;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pd_3;"      //|<
                "_wait_load_pd_3:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pd_3;"        //|<
                "_sync2_load_pd_3:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pd_3;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pd_3:"
                "movapd (%%r9), %%xmm0;movapd 16(%%r9), %%xmm1;movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm0;movapd 64(%%r9), %%xmm1;movapd 80(%%r9), %%xmm2;"
                "movapd 96(%%r9), %%xmm0;movapd 112(%%r9), %%xmm1;movapd 128(%%r9), %%xmm2;"
                "movapd 144(%%r9), %%xmm0;movapd 160(%%r9), %%xmm1;movapd 176(%%r9), %%xmm2;"
                "movapd 192(%%r9), %%xmm0;movapd 208(%%r9), %%xmm1;movapd 224(%%r9), %%xmm2;"
                "movapd 240(%%r9), %%xmm0;movapd 256(%%r9), %%xmm1;movapd 272(%%r9), %%xmm2;"
                "movapd 288(%%r9), %%xmm0;movapd 304(%%r9), %%xmm1;movapd 320(%%r9), %%xmm2;"
                "movapd 336(%%r9), %%xmm0;movapd 352(%%r9), %%xmm1;movapd 368(%%r9), %%xmm2;"
                "movapd 384(%%r9), %%xmm0;movapd 400(%%r9), %%xmm1;movapd 416(%%r9), %%xmm2;"
                "movapd 432(%%r9), %%xmm0;movapd 448(%%r9), %%xmm1;movapd 464(%%r9), %%xmm2;"
                "movapd 480(%%r9), %%xmm0;movapd 496(%%r9), %%xmm1;movapd 512(%%r9), %%xmm2;"
                "movapd 528(%%r9), %%xmm0;movapd 544(%%r9), %%xmm1;movapd 560(%%r9), %%xmm2;"
                "movapd 576(%%r9), %%xmm0;movapd 592(%%r9), %%xmm1;movapd 608(%%r9), %%xmm2;"
                "movapd 624(%%r9), %%xmm0;movapd 640(%%r9), %%xmm1;movapd 656(%%r9), %%xmm2;"
                "movapd 672(%%r9), %%xmm0;movapd 688(%%r9), %%xmm1;movapd 704(%%r9), %%xmm2;"
                "movapd 720(%%r9), %%xmm0;movapd 736(%%r9), %%xmm1;movapd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_load_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pd_4;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pd_4:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pd_4;"      //|<
                "_sync1_load_pd_4:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pd_4;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pd_4;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pd_4;"      //|<
                "_wait_load_pd_4:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pd_4;"        //|<
                "_sync2_load_pd_4:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pd_4;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pd_4:"
                "movapd (%%r9), %%xmm0;movapd 16(%%r9), %%xmm1;movapd 32(%%r9), %%xmm2;movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm0;movapd 80(%%r9), %%xmm1;movapd 96(%%r9), %%xmm2;movapd 112(%%r9), %%xmm3;"
                "movapd 128(%%r9), %%xmm0;movapd 144(%%r9), %%xmm1;movapd 160(%%r9), %%xmm2;movapd 176(%%r9), %%xmm3;"
                "movapd 192(%%r9), %%xmm0;movapd 208(%%r9), %%xmm1;movapd 224(%%r9), %%xmm2;movapd 240(%%r9), %%xmm3;"
                "movapd 256(%%r9), %%xmm0;movapd 272(%%r9), %%xmm1;movapd 288(%%r9), %%xmm2;movapd 304(%%r9), %%xmm3;"
                "movapd 320(%%r9), %%xmm0;movapd 336(%%r9), %%xmm1;movapd 352(%%r9), %%xmm2;movapd 368(%%r9), %%xmm3;"
                "movapd 384(%%r9), %%xmm0;movapd 400(%%r9), %%xmm1;movapd 416(%%r9), %%xmm2;movapd 432(%%r9), %%xmm3;"
                "movapd 448(%%r9), %%xmm0;movapd 464(%%r9), %%xmm1;movapd 480(%%r9), %%xmm2;movapd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
"mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_pd_8;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_pd_8:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_pd_8;"      //|<
                "_sync1_load_pd_8:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_pd_8;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_pd_8;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_pd_8;"      //|<
                "_wait_load_pd_8:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_pd_8;"        //|<
                "_sync2_load_pd_8:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_pd_8;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_pd_8:"
                "movapd (%%r9), %%xmm0;movapd 16(%%r9), %%xmm1;movapd 32(%%r9), %%xmm2;movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;movapd 80(%%r9), %%xmm5;movapd 96(%%r9), %%xmm6;movapd 112(%%r9), %%xmm7;"
                "movapd 128(%%r9), %%xmm0;movapd 144(%%r9), %%xmm1;movapd 160(%%r9), %%xmm2;movapd 176(%%r9), %%xmm3;"
                "movapd 192(%%r9), %%xmm4;movapd 208(%%r9), %%xmm5;movapd 224(%%r9), %%xmm6;movapd 240(%%r9), %%xmm7;"
                "movapd 256(%%r9), %%xmm0;movapd 272(%%r9), %%xmm1;movapd 288(%%r9), %%xmm2;movapd 304(%%r9), %%xmm3;"
                "movapd 320(%%r9), %%xmm4;movapd 336(%%r9), %%xmm5;movapd 352(%%r9), %%xmm6;movapd 368(%%r9), %%xmm7;"
                "movapd 384(%%r9), %%xmm0;movapd 400(%%r9), %%xmm1;movapd 416(%%r9), %%xmm2;movapd 432(%%r9), %%xmm3;"
                "movapd 448(%%r9), %%xmm4;movapd 464(%%r9), %%xmm5;movapd 480(%%r9), %%xmm6;movapd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_load_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_load_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_ps_1;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_ps_1:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_ps_1;"      //|<
                "_sync1_load_ps_1:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_ps_1;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_ps_1;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_ps_1;"      //|<
                "_wait_load_ps_1:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_ps_1;"        //|<
                "_sync2_load_ps_1:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_ps_1;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_ps_1:"
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm0;"
                "movaps 32(%%r9), %%xmm0;"
                "movaps 48(%%r9), %%xmm0;"
                "movaps 64(%%r9), %%xmm0;"
                "movaps 80(%%r9), %%xmm0;"
                "movaps 96(%%r9), %%xmm0;"
                "movaps 112(%%r9), %%xmm0;"
                "movaps 128(%%r9), %%xmm0;"
                "movaps 144(%%r9), %%xmm0;"
                "movaps 160(%%r9), %%xmm0;"
                "movaps 176(%%r9), %%xmm0;"
                "movaps 192(%%r9), %%xmm0;"
                "movaps 208(%%r9), %%xmm0;"
                "movaps 224(%%r9), %%xmm0;"
                "movaps 240(%%r9), %%xmm0;"
                "movaps 256(%%r9), %%xmm0;"
                "movaps 272(%%r9), %%xmm0;"
                "movaps 288(%%r9), %%xmm0;"
                "movaps 304(%%r9), %%xmm0;"
                "movaps 320(%%r9), %%xmm0;"
                "movaps 336(%%r9), %%xmm0;"
                "movaps 352(%%r9), %%xmm0;"
                "movaps 368(%%r9), %%xmm0;"
                "movaps 384(%%r9), %%xmm0;"
                "movaps 400(%%r9), %%xmm0;"
                "movaps 416(%%r9), %%xmm0;"
                "movaps 432(%%r9), %%xmm0;"
                "movaps 448(%%r9), %%xmm0;"
                "movaps 464(%%r9), %%xmm0;"
                "movaps 480(%%r9), %%xmm0;"
                "movaps 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_ps_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_ps_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_ps_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_ps_2;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_ps_2:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_ps_2;"      //|<
                "_sync1_load_ps_2:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_ps_2;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_ps_2;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_ps_2;"      //|<
                "_wait_load_ps_2:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_ps_2;"        //|<
                "_sync2_load_ps_2:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_ps_2;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_ps_2:"
                "movaps (%%r9), %%xmm0;movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm0;movaps 48(%%r9), %%xmm1;"
                "movaps 64(%%r9), %%xmm0;movaps 80(%%r9), %%xmm1;"
                "movaps 96(%%r9), %%xmm0;movaps 112(%%r9), %%xmm1;"
                "movaps 128(%%r9), %%xmm0;movaps 144(%%r9), %%xmm1;"
                "movaps 160(%%r9), %%xmm0;movaps 176(%%r9), %%xmm1;"
                "movaps 192(%%r9), %%xmm0;movaps 208(%%r9), %%xmm1;"
                "movaps 224(%%r9), %%xmm0;movaps 240(%%r9), %%xmm1;"
                "movaps 256(%%r9), %%xmm0;movaps 272(%%r9), %%xmm1;"
                "movaps 288(%%r9), %%xmm0;movaps 304(%%r9), %%xmm1;"
                "movaps 320(%%r9), %%xmm0;movaps 336(%%r9), %%xmm1;"
                "movaps 352(%%r9), %%xmm0;movaps 368(%%r9), %%xmm1;"
                "movaps 384(%%r9), %%xmm0;movaps 400(%%r9), %%xmm1;"
                "movaps 416(%%r9), %%xmm0;movaps 432(%%r9), %%xmm1;"
                "movaps 448(%%r9), %%xmm0;movaps 464(%%r9), %%xmm1;"
                "movaps 480(%%r9), %%xmm0;movaps 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_ps_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_ps_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_ps_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_ps_3;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_ps_3:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_ps_3;"      //|<
                "_sync1_load_ps_3:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_ps_3;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_ps_3;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_ps_3;"      //|<
                "_wait_load_ps_3:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_ps_3;"        //|<
                "_sync2_load_ps_3:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_ps_3;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_ps_3:"
                "movaps (%%r9), %%xmm0;movaps 16(%%r9), %%xmm1;movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm0;movaps 64(%%r9), %%xmm1;movaps 80(%%r9), %%xmm2;"
                "movaps 96(%%r9), %%xmm0;movaps 112(%%r9), %%xmm1;movaps 128(%%r9), %%xmm2;"
                "movaps 144(%%r9), %%xmm0;movaps 160(%%r9), %%xmm1;movaps 176(%%r9), %%xmm2;"
                "movaps 192(%%r9), %%xmm0;movaps 208(%%r9), %%xmm1;movaps 224(%%r9), %%xmm2;"
                "movaps 240(%%r9), %%xmm0;movaps 256(%%r9), %%xmm1;movaps 272(%%r9), %%xmm2;"
                "movaps 288(%%r9), %%xmm0;movaps 304(%%r9), %%xmm1;movaps 320(%%r9), %%xmm2;"
                "movaps 336(%%r9), %%xmm0;movaps 352(%%r9), %%xmm1;movaps 368(%%r9), %%xmm2;"
                "movaps 384(%%r9), %%xmm0;movaps 400(%%r9), %%xmm1;movaps 416(%%r9), %%xmm2;"
                "movaps 432(%%r9), %%xmm0;movaps 448(%%r9), %%xmm1;movaps 464(%%r9), %%xmm2;"
                "movaps 480(%%r9), %%xmm0;movaps 496(%%r9), %%xmm1;movaps 512(%%r9), %%xmm2;"
                "movaps 528(%%r9), %%xmm0;movaps 544(%%r9), %%xmm1;movaps 560(%%r9), %%xmm2;"
                "movaps 576(%%r9), %%xmm0;movaps 592(%%r9), %%xmm1;movaps 608(%%r9), %%xmm2;"
                "movaps 624(%%r9), %%xmm0;movaps 640(%%r9), %%xmm1;movaps 656(%%r9), %%xmm2;"
                "movaps 672(%%r9), %%xmm0;movaps 688(%%r9), %%xmm1;movaps 704(%%r9), %%xmm2;"
                "movaps 720(%%r9), %%xmm0;movaps 736(%%r9), %%xmm1;movaps 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_ps_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_ps_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_load_ps_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_ps_4;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_ps_4:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_ps_4;"      //|<
                "_sync1_load_ps_4:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_ps_4;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_ps_4;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_ps_4;"      //|<
                "_wait_load_ps_4:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_ps_4;"        //|<
                "_sync2_load_ps_4:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_ps_4;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_ps_4:"
                "movaps (%%r9), %%xmm0;movaps 16(%%r9), %%xmm1;movaps 32(%%r9), %%xmm2;movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm0;movaps 80(%%r9), %%xmm1;movaps 96(%%r9), %%xmm2;movaps 112(%%r9), %%xmm3;"
                "movaps 128(%%r9), %%xmm0;movaps 144(%%r9), %%xmm1;movaps 160(%%r9), %%xmm2;movaps 176(%%r9), %%xmm3;"
                "movaps 192(%%r9), %%xmm0;movaps 208(%%r9), %%xmm1;movaps 224(%%r9), %%xmm2;movaps 240(%%r9), %%xmm3;"
                "movaps 256(%%r9), %%xmm0;movaps 272(%%r9), %%xmm1;movaps 288(%%r9), %%xmm2;movaps 304(%%r9), %%xmm3;"
                "movaps 320(%%r9), %%xmm0;movaps 336(%%r9), %%xmm1;movaps 352(%%r9), %%xmm2;movaps 368(%%r9), %%xmm3;"
                "movaps 384(%%r9), %%xmm0;movaps 400(%%r9), %%xmm1;movaps 416(%%r9), %%xmm2;movaps 432(%%r9), %%xmm3;"
                "movaps 448(%%r9), %%xmm0;movaps 464(%%r9), %%xmm1;movaps 480(%%r9), %%xmm2;movaps 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_ps_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_ps_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_ps_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_load_ps_8;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_load_ps_8:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_load_ps_8;"      //|<
                "_sync1_load_ps_8:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_load_ps_8;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_load_ps_8;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_load_ps_8;"      //|<
                "_wait_load_ps_8:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_load_ps_8;"        //|<
                "_sync2_load_ps_8:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_load_ps_8;"      //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_load_ps_8:"
                "movaps (%%r9), %%xmm0;movaps 16(%%r9), %%xmm1;movaps 32(%%r9), %%xmm2;movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;movaps 80(%%r9), %%xmm5;movaps 96(%%r9), %%xmm6;movaps 112(%%r9), %%xmm7;"
                "movaps 128(%%r9), %%xmm0;movaps 144(%%r9), %%xmm1;movaps 160(%%r9), %%xmm2;movaps 176(%%r9), %%xmm3;"
                "movaps 192(%%r9), %%xmm4;movaps 208(%%r9), %%xmm5;movaps 224(%%r9), %%xmm6;movaps 240(%%r9), %%xmm7;"
                "movaps 256(%%r9), %%xmm0;movaps 272(%%r9), %%xmm1;movaps 288(%%r9), %%xmm2;movaps 304(%%r9), %%xmm3;"
                "movaps 320(%%r9), %%xmm4;movaps 336(%%r9), %%xmm5;movaps 352(%%r9), %%xmm6;movaps 368(%%r9), %%xmm7;"
                "movaps 384(%%r9), %%xmm0;movaps 400(%%r9), %%xmm1;movaps 416(%%r9), %%xmm2;movaps 432(%%r9), %%xmm3;"
                "movaps 448(%%r9), %%xmm4;movaps 464(%%r9), %%xmm5;movaps 480(%%r9), %%xmm6;movaps 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_load_ps_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_load_ps_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_load_ps_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_add_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_add_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pi_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pi_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pi_1;"       //|<
                "_sync1_add_pi_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pi_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pi_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pi_1;"       //|<
                "_wait_add_pi_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pi_1;"         //|<
                "_sync2_add_pi_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pi_1;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pi_1:"
                "paddq (%%r9), %%xmm0;"
                "paddq 16(%%r9), %%xmm0;"
                "paddq 32(%%r9), %%xmm0;"
                "paddq 48(%%r9), %%xmm0;"
                "paddq 64(%%r9), %%xmm0;"
                "paddq 80(%%r9), %%xmm0;"
                "paddq 96(%%r9), %%xmm0;"
                "paddq 112(%%r9), %%xmm0;"
                "paddq 128(%%r9), %%xmm0;"
                "paddq 144(%%r9), %%xmm0;"
                "paddq 160(%%r9), %%xmm0;"
                "paddq 176(%%r9), %%xmm0;"
                "paddq 192(%%r9), %%xmm0;"
                "paddq 208(%%r9), %%xmm0;"
                "paddq 224(%%r9), %%xmm0;"
                "paddq 240(%%r9), %%xmm0;"
                "paddq 256(%%r9), %%xmm0;"
                "paddq 272(%%r9), %%xmm0;"
                "paddq 288(%%r9), %%xmm0;"
                "paddq 304(%%r9), %%xmm0;"
                "paddq 320(%%r9), %%xmm0;"
                "paddq 336(%%r9), %%xmm0;"
                "paddq 352(%%r9), %%xmm0;"
                "paddq 368(%%r9), %%xmm0;"
                "paddq 384(%%r9), %%xmm0;"
                "paddq 400(%%r9), %%xmm0;"
                "paddq 416(%%r9), %%xmm0;"
                "paddq 432(%%r9), %%xmm0;"
                "paddq 448(%%r9), %%xmm0;"
                "paddq 464(%%r9), %%xmm0;"
                "paddq 480(%%r9), %%xmm0;"
                "paddq 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pi_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pi_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pi_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pi_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pi_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pi_2;"       //|<
                "_sync1_add_pi_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pi_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pi_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pi_2;"       //|<
                "_wait_add_pi_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pi_2;"         //|<
                "_sync2_add_pi_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pi_2;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pi_2:"
                "paddq (%%r9), %%xmm0;paddq 16(%%r9), %%xmm1;"
                "paddq 32(%%r9), %%xmm0;paddq 48(%%r9), %%xmm1;"
                "paddq 64(%%r9), %%xmm0;paddq 80(%%r9), %%xmm1;"
                "paddq 96(%%r9), %%xmm0;paddq 112(%%r9), %%xmm1;"
                "paddq 128(%%r9), %%xmm0;paddq 144(%%r9), %%xmm1;"
                "paddq 160(%%r9), %%xmm0;paddq 176(%%r9), %%xmm1;"
                "paddq 192(%%r9), %%xmm0;paddq 208(%%r9), %%xmm1;"
                "paddq 224(%%r9), %%xmm0;paddq 240(%%r9), %%xmm1;"
                "paddq 256(%%r9), %%xmm0;paddq 272(%%r9), %%xmm1;"
                "paddq 288(%%r9), %%xmm0;paddq 304(%%r9), %%xmm1;"
                "paddq 320(%%r9), %%xmm0;paddq 336(%%r9), %%xmm1;"
                "paddq 352(%%r9), %%xmm0;paddq 368(%%r9), %%xmm1;"
                "paddq 384(%%r9), %%xmm0;paddq 400(%%r9), %%xmm1;"
                "paddq 416(%%r9), %%xmm0;paddq 432(%%r9), %%xmm1;"
                "paddq 448(%%r9), %%xmm0;paddq 464(%%r9), %%xmm1;"
                "paddq 480(%%r9), %%xmm0;paddq 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pi_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pi_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pi_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pi_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pi_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pi_3;"       //|<
                "_sync1_add_pi_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pi_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pi_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pi_3;"       //|<
                "_wait_add_pi_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pi_3;"         //|<
                "_sync2_add_pi_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pi_3;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pi_3:"
                "paddq (%%r9), %%xmm0;paddq 16(%%r9), %%xmm1;paddq 32(%%r9), %%xmm2;"
                "paddq 48(%%r9), %%xmm0;paddq 64(%%r9), %%xmm1;paddq 80(%%r9), %%xmm2;"
                "paddq 96(%%r9), %%xmm0;paddq 112(%%r9), %%xmm1;paddq 128(%%r9), %%xmm2;"
                "paddq 144(%%r9), %%xmm0;paddq 160(%%r9), %%xmm1;paddq 176(%%r9), %%xmm2;"
                "paddq 192(%%r9), %%xmm0;paddq 208(%%r9), %%xmm1;paddq 224(%%r9), %%xmm2;"
                "paddq 240(%%r9), %%xmm0;paddq 256(%%r9), %%xmm1;paddq 272(%%r9), %%xmm2;"
                "paddq 288(%%r9), %%xmm0;paddq 304(%%r9), %%xmm1;paddq 320(%%r9), %%xmm2;"
                "paddq 336(%%r9), %%xmm0;paddq 352(%%r9), %%xmm1;paddq 368(%%r9), %%xmm2;"
                "paddq 384(%%r9), %%xmm0;paddq 400(%%r9), %%xmm1;paddq 416(%%r9), %%xmm2;"
                "paddq 432(%%r9), %%xmm0;paddq 448(%%r9), %%xmm1;paddq 464(%%r9), %%xmm2;"
                "paddq 480(%%r9), %%xmm0;paddq 496(%%r9), %%xmm1;paddq 512(%%r9), %%xmm2;"
                "paddq 528(%%r9), %%xmm0;paddq 544(%%r9), %%xmm1;paddq 560(%%r9), %%xmm2;"
                "paddq 576(%%r9), %%xmm0;paddq 592(%%r9), %%xmm1;paddq 608(%%r9), %%xmm2;"
                "paddq 624(%%r9), %%xmm0;paddq 640(%%r9), %%xmm1;paddq 656(%%r9), %%xmm2;"
                "paddq 672(%%r9), %%xmm0;paddq 688(%%r9), %%xmm1;paddq 704(%%r9), %%xmm2;"
                "paddq 720(%%r9), %%xmm0;paddq 736(%%r9), %%xmm1;paddq 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pi_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pi_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_add_pi_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pi_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pi_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pi_4;"       //|<
                "_sync1_add_pi_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pi_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pi_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pi_4;"       //|<
                "_wait_add_pi_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pi_4;"         //|<
                "_sync2_add_pi_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pi_4;"       //<<
                //initialize registers
                "movdqa 0(%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                "movdqa 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pi_4:"
                "paddq (%%r9), %%xmm0;paddq 16(%%r9), %%xmm1;paddq 32(%%r9), %%xmm2;paddq 48(%%r9), %%xmm3;"
                "paddq 64(%%r9), %%xmm0;paddq 80(%%r9), %%xmm1;paddq 96(%%r9), %%xmm2;paddq 112(%%r9), %%xmm3;"
                "paddq 128(%%r9), %%xmm0;paddq 144(%%r9), %%xmm1;paddq 160(%%r9), %%xmm2;paddq 176(%%r9), %%xmm3;"
                "paddq 192(%%r9), %%xmm0;paddq 208(%%r9), %%xmm1;paddq 224(%%r9), %%xmm2;paddq 240(%%r9), %%xmm3;"
                "paddq 256(%%r9), %%xmm0;paddq 272(%%r9), %%xmm1;paddq 288(%%r9), %%xmm2;paddq 304(%%r9), %%xmm3;"
                "paddq 320(%%r9), %%xmm0;paddq 336(%%r9), %%xmm1;paddq 352(%%r9), %%xmm2;paddq 368(%%r9), %%xmm3;"
                "paddq 384(%%r9), %%xmm0;paddq 400(%%r9), %%xmm1;paddq 416(%%r9), %%xmm2;paddq 432(%%r9), %%xmm3;"
                "paddq 448(%%r9), %%xmm0;paddq 464(%%r9), %%xmm1;paddq 480(%%r9), %%xmm2;paddq 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pi_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pi_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pi_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pi_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pi_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pi_8;"       //|<
                "_sync1_add_pi_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pi_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pi_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pi_8;"       //|<
                "_wait_add_pi_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pi_8;"         //|<
                "_sync2_add_pi_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pi_8;"       //<<
                //initialize registers
                "movdqa 0(%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                "movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm4;"
                "movdqa 80(%%r9), %%xmm5;"
                "movdqa 96(%%r9), %%xmm6;"
                "movdqa 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pi_8:"
                "paddq (%%r9), %%xmm0;paddq 16(%%r9), %%xmm1;paddq 32(%%r9), %%xmm2;paddq 48(%%r9), %%xmm3;"
                "paddq 64(%%r9), %%xmm4;paddq 80(%%r9), %%xmm5;paddq 96(%%r9), %%xmm6;paddq 112(%%r9), %%xmm7;"
                "paddq 128(%%r9), %%xmm0;paddq 144(%%r9), %%xmm1;paddq 160(%%r9), %%xmm2;paddq 176(%%r9), %%xmm3;"
                "paddq 192(%%r9), %%xmm4;paddq 208(%%r9), %%xmm5;paddq 224(%%r9), %%xmm6;paddq 240(%%r9), %%xmm7;"
                "paddq 256(%%r9), %%xmm0;paddq 272(%%r9), %%xmm1;paddq 288(%%r9), %%xmm2;paddq 304(%%r9), %%xmm3;"
                "paddq 320(%%r9), %%xmm4;paddq 336(%%r9), %%xmm5;paddq 352(%%r9), %%xmm6;paddq 368(%%r9), %%xmm7;"
                "paddq 384(%%r9), %%xmm0;paddq 400(%%r9), %%xmm1;paddq 416(%%r9), %%xmm2;paddq 432(%%r9), %%xmm3;"
                "paddq 448(%%r9), %%xmm4;paddq 464(%%r9), %%xmm5;paddq 480(%%r9), %%xmm6;paddq 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pi_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pi_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pi_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;   }  
  //printf ("id: %i, %llu - %llu : %llu\n",id,data->start_ts,data->end_ts,data->end_ts-data->start_ts);
  //printf("end asm\n");fflush(stdout);
	
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if ((!id)&&(data->num_events))
    { 
      PAPI_read(data->Eventset,data->values);
      for (i=0;i<data->num_events;i++)
      {
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_add_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_add_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pd_1;"       //|<
                "_sync1_add_pd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pd_1;"       //|<
                "_wait_add_pd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pd_1;"         //|<
                "_sync2_add_pd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pd_1:"
                "addpd (%%r9), %%xmm0;"
                "addpd 16(%%r9), %%xmm0;"
                "addpd 32(%%r9), %%xmm0;"
                "addpd 48(%%r9), %%xmm0;"
                "addpd 64(%%r9), %%xmm0;"
                "addpd 80(%%r9), %%xmm0;"
                "addpd 96(%%r9), %%xmm0;"
                "addpd 112(%%r9), %%xmm0;"
                "addpd 128(%%r9), %%xmm0;"
                "addpd 144(%%r9), %%xmm0;"
                "addpd 160(%%r9), %%xmm0;"
                "addpd 176(%%r9), %%xmm0;"
                "addpd 192(%%r9), %%xmm0;"
                "addpd 208(%%r9), %%xmm0;"
                "addpd 224(%%r9), %%xmm0;"
                "addpd 240(%%r9), %%xmm0;"
                "addpd 256(%%r9), %%xmm0;"
                "addpd 272(%%r9), %%xmm0;"
                "addpd 288(%%r9), %%xmm0;"
                "addpd 304(%%r9), %%xmm0;"
                "addpd 320(%%r9), %%xmm0;"
                "addpd 336(%%r9), %%xmm0;"
                "addpd 352(%%r9), %%xmm0;"
                "addpd 368(%%r9), %%xmm0;"
                "addpd 384(%%r9), %%xmm0;"
                "addpd 400(%%r9), %%xmm0;"
                "addpd 416(%%r9), %%xmm0;"
                "addpd 432(%%r9), %%xmm0;"
                "addpd 448(%%r9), %%xmm0;"
                "addpd 464(%%r9), %%xmm0;"
                "addpd 480(%%r9), %%xmm0;"
                "addpd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pd_2;"       //|<
                "_sync1_add_pd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pd_2;"       //|<
                "_wait_add_pd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pd_2;"         //|<
                "_sync2_add_pd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pd_2:"
                "addpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;"
                "addpd 32(%%r9), %%xmm0;addpd 48(%%r9), %%xmm1;"
                "addpd 64(%%r9), %%xmm0;addpd 80(%%r9), %%xmm1;"
                "addpd 96(%%r9), %%xmm0;addpd 112(%%r9), %%xmm1;"
                "addpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;"
                "addpd 160(%%r9), %%xmm0;addpd 176(%%r9), %%xmm1;"
                "addpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;"
                "addpd 224(%%r9), %%xmm0;addpd 240(%%r9), %%xmm1;"
                "addpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;"
                "addpd 288(%%r9), %%xmm0;addpd 304(%%r9), %%xmm1;"
                "addpd 320(%%r9), %%xmm0;addpd 336(%%r9), %%xmm1;"
                "addpd 352(%%r9), %%xmm0;addpd 368(%%r9), %%xmm1;"
                "addpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;"
                "addpd 416(%%r9), %%xmm0;addpd 432(%%r9), %%xmm1;"
                "addpd 448(%%r9), %%xmm0;addpd 464(%%r9), %%xmm1;"
                "addpd 480(%%r9), %%xmm0;addpd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pd_3;"       //|<
                "_sync1_add_pd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pd_3;"       //|<
                "_wait_add_pd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pd_3;"         //|<
                "_sync2_add_pd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pd_3:"
                "addpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;addpd 32(%%r9), %%xmm2;"
                "addpd 48(%%r9), %%xmm0;addpd 64(%%r9), %%xmm1;addpd 80(%%r9), %%xmm2;"
                "addpd 96(%%r9), %%xmm0;addpd 112(%%r9), %%xmm1;addpd 128(%%r9), %%xmm2;"
                "addpd 144(%%r9), %%xmm0;addpd 160(%%r9), %%xmm1;addpd 176(%%r9), %%xmm2;"
                "addpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;addpd 224(%%r9), %%xmm2;"
                "addpd 240(%%r9), %%xmm0;addpd 256(%%r9), %%xmm1;addpd 272(%%r9), %%xmm2;"
                "addpd 288(%%r9), %%xmm0;addpd 304(%%r9), %%xmm1;addpd 320(%%r9), %%xmm2;"
                "addpd 336(%%r9), %%xmm0;addpd 352(%%r9), %%xmm1;addpd 368(%%r9), %%xmm2;"
                "addpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;addpd 416(%%r9), %%xmm2;"
                "addpd 432(%%r9), %%xmm0;addpd 448(%%r9), %%xmm1;addpd 464(%%r9), %%xmm2;"
                "addpd 480(%%r9), %%xmm0;addpd 496(%%r9), %%xmm1;addpd 512(%%r9), %%xmm2;"
                "addpd 528(%%r9), %%xmm0;addpd 544(%%r9), %%xmm1;addpd 560(%%r9), %%xmm2;"
                "addpd 576(%%r9), %%xmm0;addpd 592(%%r9), %%xmm1;addpd 608(%%r9), %%xmm2;"
                "addpd 624(%%r9), %%xmm0;addpd 640(%%r9), %%xmm1;addpd 656(%%r9), %%xmm2;"
                "addpd 672(%%r9), %%xmm0;addpd 688(%%r9), %%xmm1;addpd 704(%%r9), %%xmm2;"
                "addpd 720(%%r9), %%xmm0;addpd 736(%%r9), %%xmm1;addpd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_add_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pd_4;"       //|<
                "_sync1_add_pd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pd_4;"       //|<
                "_wait_add_pd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pd_4;"         //|<
                "_sync2_add_pd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pd_4:"
                "addpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;addpd 32(%%r9), %%xmm2;addpd 48(%%r9), %%xmm3;"
                "addpd 64(%%r9), %%xmm0;addpd 80(%%r9), %%xmm1;addpd 96(%%r9), %%xmm2;addpd 112(%%r9), %%xmm3;"
                "addpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;addpd 160(%%r9), %%xmm2;addpd 176(%%r9), %%xmm3;"
                "addpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;addpd 224(%%r9), %%xmm2;addpd 240(%%r9), %%xmm3;"
                "addpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;addpd 288(%%r9), %%xmm2;addpd 304(%%r9), %%xmm3;"
                "addpd 320(%%r9), %%xmm0;addpd 336(%%r9), %%xmm1;addpd 352(%%r9), %%xmm2;addpd 368(%%r9), %%xmm3;"
                "addpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;addpd 416(%%r9), %%xmm2;addpd 432(%%r9), %%xmm3;"
                "addpd 448(%%r9), %%xmm0;addpd 464(%%r9), %%xmm1;addpd 480(%%r9), %%xmm2;addpd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_pd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_pd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_pd_8;"       //|<
                "_sync1_add_pd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_pd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_pd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_pd_8;"       //|<
                "_wait_add_pd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_pd_8;"         //|<
                "_sync2_add_pd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_pd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_pd_8:"
                "addpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;addpd 32(%%r9), %%xmm2;addpd 48(%%r9), %%xmm3;"
                "addpd 64(%%r9), %%xmm4;addpd 80(%%r9), %%xmm5;addpd 96(%%r9), %%xmm6;addpd 112(%%r9), %%xmm7;"
                "addpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;addpd 160(%%r9), %%xmm2;addpd 176(%%r9), %%xmm3;"
                "addpd 192(%%r9), %%xmm4;addpd 208(%%r9), %%xmm5;addpd 224(%%r9), %%xmm6;addpd 240(%%r9), %%xmm7;"
                "addpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;addpd 288(%%r9), %%xmm2;addpd 304(%%r9), %%xmm3;"
                "addpd 320(%%r9), %%xmm4;addpd 336(%%r9), %%xmm5;addpd 352(%%r9), %%xmm6;addpd 368(%%r9), %%xmm7;"
                "addpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;addpd 416(%%r9), %%xmm2;addpd 432(%%r9), %%xmm3;"
                "addpd 448(%%r9), %%xmm4;addpd 464(%%r9), %%xmm5;addpd 480(%%r9), %%xmm6;addpd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;   }  
  //printf ("id: %i, %llu - %llu : %llu\n",id,data->start_ts,data->end_ts,data->end_ts-data->start_ts);
  //printf("end asm\n");fflush(stdout);
	
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if ((!id)&&(data->num_events))
    { 
      PAPI_read(data->Eventset,data->values);
      for (i=0;i<data->num_events;i++)
      {
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_add_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_add_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ps_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ps_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ps_1;"       //|<
                "_sync1_add_ps_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ps_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ps_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ps_1;"       //|<
                "_wait_add_ps_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ps_1;"         //|<
                "_sync2_add_ps_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ps_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ps_1:"
                "addps (%%r9), %%xmm0;"
                "addps 16(%%r9), %%xmm0;"
                "addps 32(%%r9), %%xmm0;"
                "addps 48(%%r9), %%xmm0;"
                "addps 64(%%r9), %%xmm0;"
                "addps 80(%%r9), %%xmm0;"
                "addps 96(%%r9), %%xmm0;"
                "addps 112(%%r9), %%xmm0;"
                "addps 128(%%r9), %%xmm0;"
                "addps 144(%%r9), %%xmm0;"
                "addps 160(%%r9), %%xmm0;"
                "addps 176(%%r9), %%xmm0;"
                "addps 192(%%r9), %%xmm0;"
                "addps 208(%%r9), %%xmm0;"
                "addps 224(%%r9), %%xmm0;"
                "addps 240(%%r9), %%xmm0;"
                "addps 256(%%r9), %%xmm0;"
                "addps 272(%%r9), %%xmm0;"
                "addps 288(%%r9), %%xmm0;"
                "addps 304(%%r9), %%xmm0;"
                "addps 320(%%r9), %%xmm0;"
                "addps 336(%%r9), %%xmm0;"
                "addps 352(%%r9), %%xmm0;"
                "addps 368(%%r9), %%xmm0;"
                "addps 384(%%r9), %%xmm0;"
                "addps 400(%%r9), %%xmm0;"
                "addps 416(%%r9), %%xmm0;"
                "addps 432(%%r9), %%xmm0;"
                "addps 448(%%r9), %%xmm0;"
                "addps 464(%%r9), %%xmm0;"
                "addps 480(%%r9), %%xmm0;"
                "addps 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ps_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ps_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ps_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ps_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ps_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ps_2;"       //|<
                "_sync1_add_ps_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ps_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ps_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ps_2;"       //|<
                "_wait_add_ps_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ps_2;"         //|<
                "_sync2_add_ps_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ps_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ps_2:"
                "addps (%%r9), %%xmm0;addps 16(%%r9), %%xmm1;"
                "addps 32(%%r9), %%xmm0;addps 48(%%r9), %%xmm1;"
                "addps 64(%%r9), %%xmm0;addps 80(%%r9), %%xmm1;"
                "addps 96(%%r9), %%xmm0;addps 112(%%r9), %%xmm1;"
                "addps 128(%%r9), %%xmm0;addps 144(%%r9), %%xmm1;"
                "addps 160(%%r9), %%xmm0;addps 176(%%r9), %%xmm1;"
                "addps 192(%%r9), %%xmm0;addps 208(%%r9), %%xmm1;"
                "addps 224(%%r9), %%xmm0;addps 240(%%r9), %%xmm1;"
                "addps 256(%%r9), %%xmm0;addps 272(%%r9), %%xmm1;"
                "addps 288(%%r9), %%xmm0;addps 304(%%r9), %%xmm1;"
                "addps 320(%%r9), %%xmm0;addps 336(%%r9), %%xmm1;"
                "addps 352(%%r9), %%xmm0;addps 368(%%r9), %%xmm1;"
                "addps 384(%%r9), %%xmm0;addps 400(%%r9), %%xmm1;"
                "addps 416(%%r9), %%xmm0;addps 432(%%r9), %%xmm1;"
                "addps 448(%%r9), %%xmm0;addps 464(%%r9), %%xmm1;"
                "addps 480(%%r9), %%xmm0;addps 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ps_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ps_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ps_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ps_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ps_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ps_3;"       //|<
                "_sync1_add_ps_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ps_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ps_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ps_3;"       //|<
                "_wait_add_ps_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ps_3;"         //|<
                "_sync2_add_ps_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ps_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ps_3:"
                "addps (%%r9), %%xmm0;addps 16(%%r9), %%xmm1;addps 32(%%r9), %%xmm2;"
                "addps 48(%%r9), %%xmm0;addps 64(%%r9), %%xmm1;addps 80(%%r9), %%xmm2;"
                "addps 96(%%r9), %%xmm0;addps 112(%%r9), %%xmm1;addps 128(%%r9), %%xmm2;"
                "addps 144(%%r9), %%xmm0;addps 160(%%r9), %%xmm1;addps 176(%%r9), %%xmm2;"
                "addps 192(%%r9), %%xmm0;addps 208(%%r9), %%xmm1;addps 224(%%r9), %%xmm2;"
                "addps 240(%%r9), %%xmm0;addps 256(%%r9), %%xmm1;addps 272(%%r9), %%xmm2;"
                "addps 288(%%r9), %%xmm0;addps 304(%%r9), %%xmm1;addps 320(%%r9), %%xmm2;"
                "addps 336(%%r9), %%xmm0;addps 352(%%r9), %%xmm1;addps 368(%%r9), %%xmm2;"
                "addps 384(%%r9), %%xmm0;addps 400(%%r9), %%xmm1;addps 416(%%r9), %%xmm2;"
                "addps 432(%%r9), %%xmm0;addps 448(%%r9), %%xmm1;addps 464(%%r9), %%xmm2;"
                "addps 480(%%r9), %%xmm0;addps 496(%%r9), %%xmm1;addps 512(%%r9), %%xmm2;"
                "addps 528(%%r9), %%xmm0;addps 544(%%r9), %%xmm1;addps 560(%%r9), %%xmm2;"
                "addps 576(%%r9), %%xmm0;addps 592(%%r9), %%xmm1;addps 608(%%r9), %%xmm2;"
                "addps 624(%%r9), %%xmm0;addps 640(%%r9), %%xmm1;addps 656(%%r9), %%xmm2;"
                "addps 672(%%r9), %%xmm0;addps 688(%%r9), %%xmm1;addps 704(%%r9), %%xmm2;"
                "addps 720(%%r9), %%xmm0;addps 736(%%r9), %%xmm1;addps 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ps_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ps_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_add_ps_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ps_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ps_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ps_4;"       //|<
                "_sync1_add_ps_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ps_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ps_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ps_4;"       //|<
                "_wait_add_ps_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ps_4;"         //|<
                "_sync2_add_ps_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ps_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ps_4:"
                "addps (%%r9), %%xmm0;addps 16(%%r9), %%xmm1;addps 32(%%r9), %%xmm2;addps 48(%%r9), %%xmm3;"
                "addps 64(%%r9), %%xmm0;addps 80(%%r9), %%xmm1;addps 96(%%r9), %%xmm2;addps 112(%%r9), %%xmm3;"
                "addps 128(%%r9), %%xmm0;addps 144(%%r9), %%xmm1;addps 160(%%r9), %%xmm2;addps 176(%%r9), %%xmm3;"
                "addps 192(%%r9), %%xmm0;addps 208(%%r9), %%xmm1;addps 224(%%r9), %%xmm2;addps 240(%%r9), %%xmm3;"
                "addps 256(%%r9), %%xmm0;addps 272(%%r9), %%xmm1;addps 288(%%r9), %%xmm2;addps 304(%%r9), %%xmm3;"
                "addps 320(%%r9), %%xmm0;addps 336(%%r9), %%xmm1;addps 352(%%r9), %%xmm2;addps 368(%%r9), %%xmm3;"
                "addps 384(%%r9), %%xmm0;addps 400(%%r9), %%xmm1;addps 416(%%r9), %%xmm2;addps 432(%%r9), %%xmm3;"
                "addps 448(%%r9), %%xmm0;addps 464(%%r9), %%xmm1;addps 480(%%r9), %%xmm2;addps 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ps_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ps_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ps_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ps_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ps_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ps_8;"       //|<
                "_sync1_add_ps_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ps_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ps_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ps_8;"       //|<
                "_wait_add_ps_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ps_8;"         //|<
                "_sync2_add_ps_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ps_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ps_8:"
                "addps (%%r9), %%xmm0;addps 16(%%r9), %%xmm1;addps 32(%%r9), %%xmm2;addps 48(%%r9), %%xmm3;"
                "addps 64(%%r9), %%xmm4;addps 80(%%r9), %%xmm5;addps 96(%%r9), %%xmm6;addps 112(%%r9), %%xmm7;"
                "addps 128(%%r9), %%xmm0;addps 144(%%r9), %%xmm1;addps 160(%%r9), %%xmm2;addps 176(%%r9), %%xmm3;"
                "addps 192(%%r9), %%xmm4;addps 208(%%r9), %%xmm5;addps 224(%%r9), %%xmm6;addps 240(%%r9), %%xmm7;"
                "addps 256(%%r9), %%xmm0;addps 272(%%r9), %%xmm1;addps 288(%%r9), %%xmm2;addps 304(%%r9), %%xmm3;"
                "addps 320(%%r9), %%xmm4;addps 336(%%r9), %%xmm5;addps 352(%%r9), %%xmm6;addps 368(%%r9), %%xmm7;"
                "addps 384(%%r9), %%xmm0;addps 400(%%r9), %%xmm1;addps 416(%%r9), %%xmm2;addps 432(%%r9), %%xmm3;"
                "addps 448(%%r9), %%xmm4;addps 464(%%r9), %%xmm5;addps 480(%%r9), %%xmm6;addps 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ps_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ps_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ps_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;   }  
  //printf ("id: %i, %llu - %llu : %llu\n",id,data->start_ts,data->end_ts,data->end_ts-data->start_ts);
  //printf("end asm\n");fflush(stdout);
	
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if ((!id)&&(data->num_events))
    { 
      PAPI_read(data->Eventset,data->values);
      for (i=0;i<data->num_events;i++)
      {
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_add_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_add_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_sd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_sd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_sd_1;"       //|<
                "_sync1_add_sd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_sd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_sd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_sd_1;"       //|<
                "_wait_add_sd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_sd_1;"         //|<
                "_sync2_add_sd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_sd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_sd_1:"
                "addsd (%%r9), %%xmm0;"
                "addsd 16(%%r9), %%xmm0;"
                "addsd 32(%%r9), %%xmm0;"
                "addsd 48(%%r9), %%xmm0;"
                "addsd 64(%%r9), %%xmm0;"
                "addsd 80(%%r9), %%xmm0;"
                "addsd 96(%%r9), %%xmm0;"
                "addsd 112(%%r9), %%xmm0;"
                "addsd 128(%%r9), %%xmm0;"
                "addsd 144(%%r9), %%xmm0;"
                "addsd 160(%%r9), %%xmm0;"
                "addsd 176(%%r9), %%xmm0;"
                "addsd 192(%%r9), %%xmm0;"
                "addsd 208(%%r9), %%xmm0;"
                "addsd 224(%%r9), %%xmm0;"
                "addsd 240(%%r9), %%xmm0;"
                "addsd 256(%%r9), %%xmm0;"
                "addsd 272(%%r9), %%xmm0;"
                "addsd 288(%%r9), %%xmm0;"
                "addsd 304(%%r9), %%xmm0;"
                "addsd 320(%%r9), %%xmm0;"
                "addsd 336(%%r9), %%xmm0;"
                "addsd 352(%%r9), %%xmm0;"
                "addsd 368(%%r9), %%xmm0;"
                "addsd 384(%%r9), %%xmm0;"
                "addsd 400(%%r9), %%xmm0;"
                "addsd 416(%%r9), %%xmm0;"
                "addsd 432(%%r9), %%xmm0;"
                "addsd 448(%%r9), %%xmm0;"
                "addsd 464(%%r9), %%xmm0;"
                "addsd 480(%%r9), %%xmm0;"
                "addsd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_sd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_sd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_sd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_sd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_sd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_sd_2;"       //|<
                "_sync1_add_sd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_sd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_sd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_sd_2;"       //|<
                "_wait_add_sd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_sd_2;"         //|<
                "_sync2_add_sd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_sd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_sd_2:"
                "addsd (%%r9), %%xmm0;addsd 16(%%r9), %%xmm1;"
                "addsd 32(%%r9), %%xmm0;addsd 48(%%r9), %%xmm1;"
                "addsd 64(%%r9), %%xmm0;addsd 80(%%r9), %%xmm1;"
                "addsd 96(%%r9), %%xmm0;addsd 112(%%r9), %%xmm1;"
                "addsd 128(%%r9), %%xmm0;addsd 144(%%r9), %%xmm1;"
                "addsd 160(%%r9), %%xmm0;addsd 176(%%r9), %%xmm1;"
                "addsd 192(%%r9), %%xmm0;addsd 208(%%r9), %%xmm1;"
                "addsd 224(%%r9), %%xmm0;addsd 240(%%r9), %%xmm1;"
                "addsd 256(%%r9), %%xmm0;addsd 272(%%r9), %%xmm1;"
                "addsd 288(%%r9), %%xmm0;addsd 304(%%r9), %%xmm1;"
                "addsd 320(%%r9), %%xmm0;addsd 336(%%r9), %%xmm1;"
                "addsd 352(%%r9), %%xmm0;addsd 368(%%r9), %%xmm1;"
                "addsd 384(%%r9), %%xmm0;addsd 400(%%r9), %%xmm1;"
                "addsd 416(%%r9), %%xmm0;addsd 432(%%r9), %%xmm1;"
                "addsd 448(%%r9), %%xmm0;addsd 464(%%r9), %%xmm1;"
                "addsd 480(%%r9), %%xmm0;addsd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_sd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_sd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_sd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_sd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_sd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_sd_3;"       //|<
                "_sync1_add_sd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_sd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_sd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_sd_3;"       //|<
                "_wait_add_sd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_sd_3;"         //|<
                "_sync2_add_sd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_sd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_sd_3:"
                "addsd (%%r9), %%xmm0;addsd 16(%%r9), %%xmm1;addsd 32(%%r9), %%xmm2;"
                "addsd 48(%%r9), %%xmm0;addsd 64(%%r9), %%xmm1;addsd 80(%%r9), %%xmm2;"
                "addsd 96(%%r9), %%xmm0;addsd 112(%%r9), %%xmm1;addsd 128(%%r9), %%xmm2;"
                "addsd 144(%%r9), %%xmm0;addsd 160(%%r9), %%xmm1;addsd 176(%%r9), %%xmm2;"
                "addsd 192(%%r9), %%xmm0;addsd 208(%%r9), %%xmm1;addsd 224(%%r9), %%xmm2;"
                "addsd 240(%%r9), %%xmm0;addsd 256(%%r9), %%xmm1;addsd 272(%%r9), %%xmm2;"
                "addsd 288(%%r9), %%xmm0;addsd 304(%%r9), %%xmm1;addsd 320(%%r9), %%xmm2;"
                "addsd 336(%%r9), %%xmm0;addsd 352(%%r9), %%xmm1;addsd 368(%%r9), %%xmm2;"
                "addsd 384(%%r9), %%xmm0;addsd 400(%%r9), %%xmm1;addsd 416(%%r9), %%xmm2;"
                "addsd 432(%%r9), %%xmm0;addsd 448(%%r9), %%xmm1;addsd 464(%%r9), %%xmm2;"
                "addsd 480(%%r9), %%xmm0;addsd 496(%%r9), %%xmm1;addsd 512(%%r9), %%xmm2;"
                "addsd 528(%%r9), %%xmm0;addsd 544(%%r9), %%xmm1;addsd 560(%%r9), %%xmm2;"
                "addsd 576(%%r9), %%xmm0;addsd 592(%%r9), %%xmm1;addsd 608(%%r9), %%xmm2;"
                "addsd 624(%%r9), %%xmm0;addsd 640(%%r9), %%xmm1;addsd 656(%%r9), %%xmm2;"
                "addsd 672(%%r9), %%xmm0;addsd 688(%%r9), %%xmm1;addsd 704(%%r9), %%xmm2;"
                "addsd 720(%%r9), %%xmm0;addsd 736(%%r9), %%xmm1;addsd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_sd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_sd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_add_sd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_sd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_sd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_sd_4;"       //|<
                "_sync1_add_sd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_sd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_sd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_sd_4;"       //|<
                "_wait_add_sd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_sd_4;"         //|<
                "_sync2_add_sd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_sd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_sd_4:"
                "addsd (%%r9), %%xmm0;addsd 16(%%r9), %%xmm1;addsd 32(%%r9), %%xmm2;addsd 48(%%r9), %%xmm3;"
                "addsd 64(%%r9), %%xmm0;addsd 80(%%r9), %%xmm1;addsd 96(%%r9), %%xmm2;addsd 112(%%r9), %%xmm3;"
                "addsd 128(%%r9), %%xmm0;addsd 144(%%r9), %%xmm1;addsd 160(%%r9), %%xmm2;addsd 176(%%r9), %%xmm3;"
                "addsd 192(%%r9), %%xmm0;addsd 208(%%r9), %%xmm1;addsd 224(%%r9), %%xmm2;addsd 240(%%r9), %%xmm3;"
                "addsd 256(%%r9), %%xmm0;addsd 272(%%r9), %%xmm1;addsd 288(%%r9), %%xmm2;addsd 304(%%r9), %%xmm3;"
                "addsd 320(%%r9), %%xmm0;addsd 336(%%r9), %%xmm1;addsd 352(%%r9), %%xmm2;addsd 368(%%r9), %%xmm3;"
                "addsd 384(%%r9), %%xmm0;addsd 400(%%r9), %%xmm1;addsd 416(%%r9), %%xmm2;addsd 432(%%r9), %%xmm3;"
                "addsd 448(%%r9), %%xmm0;addsd 464(%%r9), %%xmm1;addsd 480(%%r9), %%xmm2;addsd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_sd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_sd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_sd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_sd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_sd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_sd_8;"       //|<
                "_sync1_add_sd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_sd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_sd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_sd_8;"       //|<
                "_wait_add_sd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_sd_8;"         //|<
                "_sync2_add_sd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_sd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_sd_8:"
                "addsd (%%r9), %%xmm0;addsd 16(%%r9), %%xmm1;addsd 32(%%r9), %%xmm2;addsd 48(%%r9), %%xmm3;"
                "addsd 64(%%r9), %%xmm4;addsd 80(%%r9), %%xmm5;addsd 96(%%r9), %%xmm6;addsd 112(%%r9), %%xmm7;"
                "addsd 128(%%r9), %%xmm0;addsd 144(%%r9), %%xmm1;addsd 160(%%r9), %%xmm2;addsd 176(%%r9), %%xmm3;"
                "addsd 192(%%r9), %%xmm4;addsd 208(%%r9), %%xmm5;addsd 224(%%r9), %%xmm6;addsd 240(%%r9), %%xmm7;"
                "addsd 256(%%r9), %%xmm0;addsd 272(%%r9), %%xmm1;addsd 288(%%r9), %%xmm2;addsd 304(%%r9), %%xmm3;"
                "addsd 320(%%r9), %%xmm4;addsd 336(%%r9), %%xmm5;addsd 352(%%r9), %%xmm6;addsd 368(%%r9), %%xmm7;"
                "addsd 384(%%r9), %%xmm0;addsd 400(%%r9), %%xmm1;addsd 416(%%r9), %%xmm2;addsd 432(%%r9), %%xmm3;"
                "addsd 448(%%r9), %%xmm4;addsd 464(%%r9), %%xmm5;addsd 480(%%r9), %%xmm6;addsd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_sd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_sd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_sd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;   }  
  //printf ("id: %i, %llu - %llu : %llu\n",id,data->start_ts,data->end_ts,data->end_ts-data->start_ts);
  //printf("end asm\n");fflush(stdout);
	
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if ((!id)&&(data->num_events))
    { 
      PAPI_read(data->Eventset,data->values);
      for (i=0;i<data->num_events;i++)
      {
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_add_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_add_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ss_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ss_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ss_1;"       //|<
                "_sync1_add_ss_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ss_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ss_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ss_1;"       //|<
                "_wait_add_ss_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ss_1;"         //|<
                "_sync2_add_ss_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ss_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ss_1:"
                "addss (%%r9), %%xmm0;"
                "addss 16(%%r9), %%xmm0;"
                "addss 32(%%r9), %%xmm0;"
                "addss 48(%%r9), %%xmm0;"
                "addss 64(%%r9), %%xmm0;"
                "addss 80(%%r9), %%xmm0;"
                "addss 96(%%r9), %%xmm0;"
                "addss 112(%%r9), %%xmm0;"
                "addss 128(%%r9), %%xmm0;"
                "addss 144(%%r9), %%xmm0;"
                "addss 160(%%r9), %%xmm0;"
                "addss 176(%%r9), %%xmm0;"
                "addss 192(%%r9), %%xmm0;"
                "addss 208(%%r9), %%xmm0;"
                "addss 224(%%r9), %%xmm0;"
                "addss 240(%%r9), %%xmm0;"
                "addss 256(%%r9), %%xmm0;"
                "addss 272(%%r9), %%xmm0;"
                "addss 288(%%r9), %%xmm0;"
                "addss 304(%%r9), %%xmm0;"
                "addss 320(%%r9), %%xmm0;"
                "addss 336(%%r9), %%xmm0;"
                "addss 352(%%r9), %%xmm0;"
                "addss 368(%%r9), %%xmm0;"
                "addss 384(%%r9), %%xmm0;"
                "addss 400(%%r9), %%xmm0;"
                "addss 416(%%r9), %%xmm0;"
                "addss 432(%%r9), %%xmm0;"
                "addss 448(%%r9), %%xmm0;"
                "addss 464(%%r9), %%xmm0;"
                "addss 480(%%r9), %%xmm0;"
                "addss 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ss_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ss_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ss_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ss_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ss_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ss_2;"       //|<
                "_sync1_add_ss_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ss_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ss_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ss_2;"       //|<
                "_wait_add_ss_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ss_2;"         //|<
                "_sync2_add_ss_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ss_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ss_2:"
                "addss (%%r9), %%xmm0;addss 16(%%r9), %%xmm1;"
                "addss 32(%%r9), %%xmm0;addss 48(%%r9), %%xmm1;"
                "addss 64(%%r9), %%xmm0;addss 80(%%r9), %%xmm1;"
                "addss 96(%%r9), %%xmm0;addss 112(%%r9), %%xmm1;"
                "addss 128(%%r9), %%xmm0;addss 144(%%r9), %%xmm1;"
                "addss 160(%%r9), %%xmm0;addss 176(%%r9), %%xmm1;"
                "addss 192(%%r9), %%xmm0;addss 208(%%r9), %%xmm1;"
                "addss 224(%%r9), %%xmm0;addss 240(%%r9), %%xmm1;"
                "addss 256(%%r9), %%xmm0;addss 272(%%r9), %%xmm1;"
                "addss 288(%%r9), %%xmm0;addss 304(%%r9), %%xmm1;"
                "addss 320(%%r9), %%xmm0;addss 336(%%r9), %%xmm1;"
                "addss 352(%%r9), %%xmm0;addss 368(%%r9), %%xmm1;"
                "addss 384(%%r9), %%xmm0;addss 400(%%r9), %%xmm1;"
                "addss 416(%%r9), %%xmm0;addss 432(%%r9), %%xmm1;"
                "addss 448(%%r9), %%xmm0;addss 464(%%r9), %%xmm1;"
                "addss 480(%%r9), %%xmm0;addss 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ss_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ss_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ss_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ss_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ss_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ss_3;"       //|<
                "_sync1_add_ss_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ss_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ss_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ss_3;"       //|<
                "_wait_add_ss_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ss_3;"         //|<
                "_sync2_add_ss_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ss_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ss_3:"
                "addss (%%r9), %%xmm0;addss 16(%%r9), %%xmm1;addss 32(%%r9), %%xmm2;"
                "addss 48(%%r9), %%xmm0;addss 64(%%r9), %%xmm1;addss 80(%%r9), %%xmm2;"
                "addss 96(%%r9), %%xmm0;addss 112(%%r9), %%xmm1;addss 128(%%r9), %%xmm2;"
                "addss 144(%%r9), %%xmm0;addss 160(%%r9), %%xmm1;addss 176(%%r9), %%xmm2;"
                "addss 192(%%r9), %%xmm0;addss 208(%%r9), %%xmm1;addss 224(%%r9), %%xmm2;"
                "addss 240(%%r9), %%xmm0;addss 256(%%r9), %%xmm1;addss 272(%%r9), %%xmm2;"
                "addss 288(%%r9), %%xmm0;addss 304(%%r9), %%xmm1;addss 320(%%r9), %%xmm2;"
                "addss 336(%%r9), %%xmm0;addss 352(%%r9), %%xmm1;addss 368(%%r9), %%xmm2;"
                "addss 384(%%r9), %%xmm0;addss 400(%%r9), %%xmm1;addss 416(%%r9), %%xmm2;"
                "addss 432(%%r9), %%xmm0;addss 448(%%r9), %%xmm1;addss 464(%%r9), %%xmm2;"
                "addss 480(%%r9), %%xmm0;addss 496(%%r9), %%xmm1;addss 512(%%r9), %%xmm2;"
                "addss 528(%%r9), %%xmm0;addss 544(%%r9), %%xmm1;addss 560(%%r9), %%xmm2;"
                "addss 576(%%r9), %%xmm0;addss 592(%%r9), %%xmm1;addss 608(%%r9), %%xmm2;"
                "addss 624(%%r9), %%xmm0;addss 640(%%r9), %%xmm1;addss 656(%%r9), %%xmm2;"
                "addss 672(%%r9), %%xmm0;addss 688(%%r9), %%xmm1;addss 704(%%r9), %%xmm2;"
                "addss 720(%%r9), %%xmm0;addss 736(%%r9), %%xmm1;addss 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ss_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ss_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_add_ss_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ss_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ss_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ss_4;"       //|<
                "_sync1_add_ss_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ss_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ss_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ss_4;"       //|<
                "_wait_add_ss_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ss_4;"         //|<
                "_sync2_add_ss_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ss_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ss_4:"
                "addss (%%r9), %%xmm0;addss 16(%%r9), %%xmm1;addss 32(%%r9), %%xmm2;addss 48(%%r9), %%xmm3;"
                "addss 64(%%r9), %%xmm0;addss 80(%%r9), %%xmm1;addss 96(%%r9), %%xmm2;addss 112(%%r9), %%xmm3;"
                "addss 128(%%r9), %%xmm0;addss 144(%%r9), %%xmm1;addss 160(%%r9), %%xmm2;addss 176(%%r9), %%xmm3;"
                "addss 192(%%r9), %%xmm0;addss 208(%%r9), %%xmm1;addss 224(%%r9), %%xmm2;addss 240(%%r9), %%xmm3;"
                "addss 256(%%r9), %%xmm0;addss 272(%%r9), %%xmm1;addss 288(%%r9), %%xmm2;addss 304(%%r9), %%xmm3;"
                "addss 320(%%r9), %%xmm0;addss 336(%%r9), %%xmm1;addss 352(%%r9), %%xmm2;addss 368(%%r9), %%xmm3;"
                "addss 384(%%r9), %%xmm0;addss 400(%%r9), %%xmm1;addss 416(%%r9), %%xmm2;addss 432(%%r9), %%xmm3;"
                "addss 448(%%r9), %%xmm0;addss 464(%%r9), %%xmm1;addss 480(%%r9), %%xmm2;addss 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ss_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ss_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ss_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_add_ss_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_add_ss_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_add_ss_8;"       //|<
                "_sync1_add_ss_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_add_ss_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_add_ss_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_add_ss_8;"       //|<
                "_wait_add_ss_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_add_ss_8;"         //|<
                "_sync2_add_ss_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_add_ss_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_add_ss_8:"
                "addss (%%r9), %%xmm0;addss 16(%%r9), %%xmm1;addss 32(%%r9), %%xmm2;addss 48(%%r9), %%xmm3;"
                "addss 64(%%r9), %%xmm4;addss 80(%%r9), %%xmm5;addss 96(%%r9), %%xmm6;addss 112(%%r9), %%xmm7;"
                "addss 128(%%r9), %%xmm0;addss 144(%%r9), %%xmm1;addss 160(%%r9), %%xmm2;addss 176(%%r9), %%xmm3;"
                "addss 192(%%r9), %%xmm4;addss 208(%%r9), %%xmm5;addss 224(%%r9), %%xmm6;addss 240(%%r9), %%xmm7;"
                "addss 256(%%r9), %%xmm0;addss 272(%%r9), %%xmm1;addss 288(%%r9), %%xmm2;addss 304(%%r9), %%xmm3;"
                "addss 320(%%r9), %%xmm4;addss 336(%%r9), %%xmm5;addss 352(%%r9), %%xmm6;addss 368(%%r9), %%xmm7;"
                "addss 384(%%r9), %%xmm0;addss 400(%%r9), %%xmm1;addss 416(%%r9), %%xmm2;addss 432(%%r9), %%xmm3;"
                "addss 448(%%r9), %%xmm4;addss 464(%%r9), %%xmm5;addss 480(%%r9), %%xmm6;addss 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_add_ss_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_add_ss_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_add_ss_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_mul_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mul_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pi_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pi_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pi_1;"       //|<
                "_sync1_mul_pi_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pi_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pi_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pi_1;"       //|<
                "_wait_mul_pi_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pi_1;"         //|<
                "_sync2_mul_pi_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pi_1;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pi_1:"
                "pmuldq (%%r9), %%xmm0;"
                "pmuldq 16(%%r9), %%xmm0;"
                "pmuldq 32(%%r9), %%xmm0;"
                "pmuldq 48(%%r9), %%xmm0;"
                "pmuldq 64(%%r9), %%xmm0;"
                "pmuldq 80(%%r9), %%xmm0;"
                "pmuldq 96(%%r9), %%xmm0;"
                "pmuldq 112(%%r9), %%xmm0;"
                "pmuldq 128(%%r9), %%xmm0;"
                "pmuldq 144(%%r9), %%xmm0;"
                "pmuldq 160(%%r9), %%xmm0;"
                "pmuldq 176(%%r9), %%xmm0;"
                "pmuldq 192(%%r9), %%xmm0;"
                "pmuldq 208(%%r9), %%xmm0;"
                "pmuldq 224(%%r9), %%xmm0;"
                "pmuldq 240(%%r9), %%xmm0;"
                "pmuldq 256(%%r9), %%xmm0;"
                "pmuldq 272(%%r9), %%xmm0;"
                "pmuldq 288(%%r9), %%xmm0;"
                "pmuldq 304(%%r9), %%xmm0;"
                "pmuldq 320(%%r9), %%xmm0;"
                "pmuldq 336(%%r9), %%xmm0;"
                "pmuldq 352(%%r9), %%xmm0;"
                "pmuldq 368(%%r9), %%xmm0;"
                "pmuldq 384(%%r9), %%xmm0;"
                "pmuldq 400(%%r9), %%xmm0;"
                "pmuldq 416(%%r9), %%xmm0;"
                "pmuldq 432(%%r9), %%xmm0;"
                "pmuldq 448(%%r9), %%xmm0;"
                "pmuldq 464(%%r9), %%xmm0;"
                "pmuldq 480(%%r9), %%xmm0;"
                "pmuldq 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pi_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pi_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pi_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pi_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pi_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pi_2;"       //|<
                "_sync1_mul_pi_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pi_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pi_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pi_2;"       //|<
                "_wait_mul_pi_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pi_2;"         //|<
                "_sync2_mul_pi_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pi_2;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pi_2:"
                "pmuldq (%%r9), %%xmm0;pmuldq 16(%%r9), %%xmm1;"
                "pmuldq 32(%%r9), %%xmm0;pmuldq 48(%%r9), %%xmm1;"
                "pmuldq 64(%%r9), %%xmm0;pmuldq 80(%%r9), %%xmm1;"
                "pmuldq 96(%%r9), %%xmm0;pmuldq 112(%%r9), %%xmm1;"
                "pmuldq 128(%%r9), %%xmm0;pmuldq 144(%%r9), %%xmm1;"
                "pmuldq 160(%%r9), %%xmm0;pmuldq 176(%%r9), %%xmm1;"
                "pmuldq 192(%%r9), %%xmm0;pmuldq 208(%%r9), %%xmm1;"
                "pmuldq 224(%%r9), %%xmm0;pmuldq 240(%%r9), %%xmm1;"
                "pmuldq 256(%%r9), %%xmm0;pmuldq 272(%%r9), %%xmm1;"
                "pmuldq 288(%%r9), %%xmm0;pmuldq 304(%%r9), %%xmm1;"
                "pmuldq 320(%%r9), %%xmm0;pmuldq 336(%%r9), %%xmm1;"
                "pmuldq 352(%%r9), %%xmm0;pmuldq 368(%%r9), %%xmm1;"
                "pmuldq 384(%%r9), %%xmm0;pmuldq 400(%%r9), %%xmm1;"
                "pmuldq 416(%%r9), %%xmm0;pmuldq 432(%%r9), %%xmm1;"
                "pmuldq 448(%%r9), %%xmm0;pmuldq 464(%%r9), %%xmm1;"
                "pmuldq 480(%%r9), %%xmm0;pmuldq 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pi_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pi_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pi_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pi_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pi_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pi_3;"       //|<
                "_sync1_mul_pi_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pi_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pi_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pi_3;"       //|<
                "_wait_mul_pi_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pi_3;"         //|<
                "_sync2_mul_pi_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pi_3;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pi_3:"
                "pmuldq (%%r9), %%xmm0;pmuldq 16(%%r9), %%xmm1;pmuldq 32(%%r9), %%xmm2;"
                "pmuldq 48(%%r9), %%xmm0;pmuldq 64(%%r9), %%xmm1;pmuldq 80(%%r9), %%xmm2;"
                "pmuldq 96(%%r9), %%xmm0;pmuldq 112(%%r9), %%xmm1;pmuldq 128(%%r9), %%xmm2;"
                "pmuldq 144(%%r9), %%xmm0;pmuldq 160(%%r9), %%xmm1;pmuldq 176(%%r9), %%xmm2;"
                "pmuldq 192(%%r9), %%xmm0;pmuldq 208(%%r9), %%xmm1;pmuldq 224(%%r9), %%xmm2;"
                "pmuldq 240(%%r9), %%xmm0;pmuldq 256(%%r9), %%xmm1;pmuldq 272(%%r9), %%xmm2;"
                "pmuldq 288(%%r9), %%xmm0;pmuldq 304(%%r9), %%xmm1;pmuldq 320(%%r9), %%xmm2;"
                "pmuldq 336(%%r9), %%xmm0;pmuldq 352(%%r9), %%xmm1;pmuldq 368(%%r9), %%xmm2;"
                "pmuldq 384(%%r9), %%xmm0;pmuldq 400(%%r9), %%xmm1;pmuldq 416(%%r9), %%xmm2;"
                "pmuldq 432(%%r9), %%xmm0;pmuldq 448(%%r9), %%xmm1;pmuldq 464(%%r9), %%xmm2;"
                "pmuldq 480(%%r9), %%xmm0;pmuldq 496(%%r9), %%xmm1;pmuldq 512(%%r9), %%xmm2;"
                "pmuldq 528(%%r9), %%xmm0;pmuldq 544(%%r9), %%xmm1;pmuldq 560(%%r9), %%xmm2;"
                "pmuldq 576(%%r9), %%xmm0;pmuldq 592(%%r9), %%xmm1;pmuldq 608(%%r9), %%xmm2;"
                "pmuldq 624(%%r9), %%xmm0;pmuldq 640(%%r9), %%xmm1;pmuldq 656(%%r9), %%xmm2;"
                "pmuldq 672(%%r9), %%xmm0;pmuldq 688(%%r9), %%xmm1;pmuldq 704(%%r9), %%xmm2;"
                "pmuldq 720(%%r9), %%xmm0;pmuldq 736(%%r9), %%xmm1;pmuldq 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pi_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pi_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_mul_pi_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pi_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pi_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pi_4;"       //|<
                "_sync1_mul_pi_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pi_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pi_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pi_4;"       //|<
                "_wait_mul_pi_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pi_4;"         //|<
                "_sync2_mul_pi_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pi_4;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                "movdqa 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pi_4:"
                "pmuldq (%%r9), %%xmm0;pmuldq 16(%%r9), %%xmm1;pmuldq 32(%%r9), %%xmm2;pmuldq 48(%%r9), %%xmm3;"
                "pmuldq 64(%%r9), %%xmm0;pmuldq 80(%%r9), %%xmm1;pmuldq 96(%%r9), %%xmm2;pmuldq 112(%%r9), %%xmm3;"
                "pmuldq 128(%%r9), %%xmm0;pmuldq 144(%%r9), %%xmm1;pmuldq 160(%%r9), %%xmm2;pmuldq 176(%%r9), %%xmm3;"
                "pmuldq 192(%%r9), %%xmm0;pmuldq 208(%%r9), %%xmm1;pmuldq 224(%%r9), %%xmm2;pmuldq 240(%%r9), %%xmm3;"
                "pmuldq 256(%%r9), %%xmm0;pmuldq 272(%%r9), %%xmm1;pmuldq 288(%%r9), %%xmm2;pmuldq 304(%%r9), %%xmm3;"
                "pmuldq 320(%%r9), %%xmm0;pmuldq 336(%%r9), %%xmm1;pmuldq 352(%%r9), %%xmm2;pmuldq 368(%%r9), %%xmm3;"
                "pmuldq 384(%%r9), %%xmm0;pmuldq 400(%%r9), %%xmm1;pmuldq 416(%%r9), %%xmm2;pmuldq 432(%%r9), %%xmm3;"
                "pmuldq 448(%%r9), %%xmm0;pmuldq 464(%%r9), %%xmm1;pmuldq 480(%%r9), %%xmm2;pmuldq 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pi_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pi_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pi_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pi_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pi_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pi_8;"       //|<
                "_sync1_mul_pi_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pi_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pi_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pi_8;"       //|<
                "_wait_mul_pi_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pi_8;"         //|<
                "_sync2_mul_pi_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pi_8;"       //<<
                //initialize registers
                "movdqa 0(%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                "movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm4;"
                "movdqa 80(%%r9), %%xmm5;"
                "movdqa 96(%%r9), %%xmm6;"
                "movdqa 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pi_8:"
                "pmuldq (%%r9), %%xmm0;pmuldq 16(%%r9), %%xmm1;pmuldq 32(%%r9), %%xmm2;pmuldq 48(%%r9), %%xmm3;"
                "pmuldq 64(%%r9), %%xmm4;pmuldq 80(%%r9), %%xmm5;pmuldq 96(%%r9), %%xmm6;pmuldq 112(%%r9), %%xmm7;"
                "pmuldq 128(%%r9), %%xmm0;pmuldq 144(%%r9), %%xmm1;pmuldq 160(%%r9), %%xmm2;pmuldq 176(%%r9), %%xmm3;"
                "pmuldq 192(%%r9), %%xmm4;pmuldq 208(%%r9), %%xmm5;pmuldq 224(%%r9), %%xmm6;pmuldq 240(%%r9), %%xmm7;"
                "pmuldq 256(%%r9), %%xmm0;pmuldq 272(%%r9), %%xmm1;pmuldq 288(%%r9), %%xmm2;pmuldq 304(%%r9), %%xmm3;"
                "pmuldq 320(%%r9), %%xmm4;pmuldq 336(%%r9), %%xmm5;pmuldq 352(%%r9), %%xmm6;pmuldq 368(%%r9), %%xmm7;"
                "pmuldq 384(%%r9), %%xmm0;pmuldq 400(%%r9), %%xmm1;pmuldq 416(%%r9), %%xmm2;pmuldq 432(%%r9), %%xmm3;"
                "pmuldq 448(%%r9), %%xmm4;pmuldq 464(%%r9), %%xmm5;pmuldq 480(%%r9), %%xmm6;pmuldq 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pi_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pi_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pi_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}


/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_mul_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mul_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pd_1;"       //|<
                "_sync1_mul_pd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pd_1;"       //|<
                "_wait_mul_pd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pd_1;"         //|<
                "_sync2_mul_pd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pd_1:"
                "mulpd (%%r9), %%xmm0;"
                "mulpd 16(%%r9), %%xmm0;"
                "mulpd 32(%%r9), %%xmm0;"
                "mulpd 48(%%r9), %%xmm0;"
                "mulpd 64(%%r9), %%xmm0;"
                "mulpd 80(%%r9), %%xmm0;"
                "mulpd 96(%%r9), %%xmm0;"
                "mulpd 112(%%r9), %%xmm0;"
                "mulpd 128(%%r9), %%xmm0;"
                "mulpd 144(%%r9), %%xmm0;"
                "mulpd 160(%%r9), %%xmm0;"
                "mulpd 176(%%r9), %%xmm0;"
                "mulpd 192(%%r9), %%xmm0;"
                "mulpd 208(%%r9), %%xmm0;"
                "mulpd 224(%%r9), %%xmm0;"
                "mulpd 240(%%r9), %%xmm0;"
                "mulpd 256(%%r9), %%xmm0;"
                "mulpd 272(%%r9), %%xmm0;"
                "mulpd 288(%%r9), %%xmm0;"
                "mulpd 304(%%r9), %%xmm0;"
                "mulpd 320(%%r9), %%xmm0;"
                "mulpd 336(%%r9), %%xmm0;"
                "mulpd 352(%%r9), %%xmm0;"
                "mulpd 368(%%r9), %%xmm0;"
                "mulpd 384(%%r9), %%xmm0;"
                "mulpd 400(%%r9), %%xmm0;"
                "mulpd 416(%%r9), %%xmm0;"
                "mulpd 432(%%r9), %%xmm0;"
                "mulpd 448(%%r9), %%xmm0;"
                "mulpd 464(%%r9), %%xmm0;"
                "mulpd 480(%%r9), %%xmm0;"
                "mulpd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pd_2;"       //|<
                "_sync1_mul_pd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pd_2;"       //|<
                "_wait_mul_pd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pd_2;"         //|<
                "_sync2_mul_pd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pd_2:"
                "mulpd (%%r9), %%xmm0;mulpd 16(%%r9), %%xmm1;"
                "mulpd 32(%%r9), %%xmm0;mulpd 48(%%r9), %%xmm1;"
                "mulpd 64(%%r9), %%xmm0;mulpd 80(%%r9), %%xmm1;"
                "mulpd 96(%%r9), %%xmm0;mulpd 112(%%r9), %%xmm1;"
                "mulpd 128(%%r9), %%xmm0;mulpd 144(%%r9), %%xmm1;"
                "mulpd 160(%%r9), %%xmm0;mulpd 176(%%r9), %%xmm1;"
                "mulpd 192(%%r9), %%xmm0;mulpd 208(%%r9), %%xmm1;"
                "mulpd 224(%%r9), %%xmm0;mulpd 240(%%r9), %%xmm1;"
                "mulpd 256(%%r9), %%xmm0;mulpd 272(%%r9), %%xmm1;"
                "mulpd 288(%%r9), %%xmm0;mulpd 304(%%r9), %%xmm1;"
                "mulpd 320(%%r9), %%xmm0;mulpd 336(%%r9), %%xmm1;"
                "mulpd 352(%%r9), %%xmm0;mulpd 368(%%r9), %%xmm1;"
                "mulpd 384(%%r9), %%xmm0;mulpd 400(%%r9), %%xmm1;"
                "mulpd 416(%%r9), %%xmm0;mulpd 432(%%r9), %%xmm1;"
                "mulpd 448(%%r9), %%xmm0;mulpd 464(%%r9), %%xmm1;"
                "mulpd 480(%%r9), %%xmm0;mulpd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pd_3;"       //|<
                "_sync1_mul_pd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pd_3;"       //|<
                "_wait_mul_pd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pd_3;"         //|<
                "_sync2_mul_pd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pd_3:"
                "mulpd (%%r9), %%xmm0;mulpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;"
                "mulpd 48(%%r9), %%xmm0;mulpd 64(%%r9), %%xmm1;mulpd 80(%%r9), %%xmm2;"
                "mulpd 96(%%r9), %%xmm0;mulpd 112(%%r9), %%xmm1;mulpd 128(%%r9), %%xmm2;"
                "mulpd 144(%%r9), %%xmm0;mulpd 160(%%r9), %%xmm1;mulpd 176(%%r9), %%xmm2;"
                "mulpd 192(%%r9), %%xmm0;mulpd 208(%%r9), %%xmm1;mulpd 224(%%r9), %%xmm2;"
                "mulpd 240(%%r9), %%xmm0;mulpd 256(%%r9), %%xmm1;mulpd 272(%%r9), %%xmm2;"
                "mulpd 288(%%r9), %%xmm0;mulpd 304(%%r9), %%xmm1;mulpd 320(%%r9), %%xmm2;"
                "mulpd 336(%%r9), %%xmm0;mulpd 352(%%r9), %%xmm1;mulpd 368(%%r9), %%xmm2;"
                "mulpd 384(%%r9), %%xmm0;mulpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;"
                "mulpd 432(%%r9), %%xmm0;mulpd 448(%%r9), %%xmm1;mulpd 464(%%r9), %%xmm2;"
                "mulpd 480(%%r9), %%xmm0;mulpd 496(%%r9), %%xmm1;mulpd 512(%%r9), %%xmm2;"
                "mulpd 528(%%r9), %%xmm0;mulpd 544(%%r9), %%xmm1;mulpd 560(%%r9), %%xmm2;"
                "mulpd 576(%%r9), %%xmm0;mulpd 592(%%r9), %%xmm1;mulpd 608(%%r9), %%xmm2;"
                "mulpd 624(%%r9), %%xmm0;mulpd 640(%%r9), %%xmm1;mulpd 656(%%r9), %%xmm2;"
                "mulpd 672(%%r9), %%xmm0;mulpd 688(%%r9), %%xmm1;mulpd 704(%%r9), %%xmm2;"
                "mulpd 720(%%r9), %%xmm0;mulpd 736(%%r9), %%xmm1;mulpd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_mul_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
 			/*printf("%f - %f - %f - %f - %f - %f - %f - %f\n",((double*)addr)[0],((double*)addr)[1],((double*)addr)[2],((double*)addr)[3],((double*)addr)[4],((double*)addr)[5],((double*)addr)[6],((double*)addr)[7]);
 			addr-=64;
 			((double*)addr)[0]=0.0;
 			((double*)addr)[1]=0.0;
 			((double*)addr)[2]=0.0;
 			((double*)addr)[3]=0.0;
 			((double*)addr)[4]=0.0;
 			((double*)addr)[5]=0.0;
 			((double*)addr)[6]=0.0;
 			((double*)addr)[7]=0.0;
 			addr+=64;*/
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pd_4;"       //|<
                "_sync1_mul_pd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pd_4;"       //|<
                "_wait_mul_pd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pd_4;"         //|<
                "_sync2_mul_pd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pd_4:"
                "mulpd (%%r9), %%xmm0;mulpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;mulpd 48(%%r9), %%xmm3;"
                "mulpd 64(%%r9), %%xmm0;mulpd 80(%%r9), %%xmm1;mulpd 96(%%r9), %%xmm2;mulpd 112(%%r9), %%xmm3;"
                "mulpd 128(%%r9), %%xmm0;mulpd 144(%%r9), %%xmm1;mulpd 160(%%r9), %%xmm2;mulpd 176(%%r9), %%xmm3;"
                "mulpd 192(%%r9), %%xmm0;mulpd 208(%%r9), %%xmm1;mulpd 224(%%r9), %%xmm2;mulpd 240(%%r9), %%xmm3;"
                "mulpd 256(%%r9), %%xmm0;mulpd 272(%%r9), %%xmm1;mulpd 288(%%r9), %%xmm2;mulpd 304(%%r9), %%xmm3;"
                "mulpd 320(%%r9), %%xmm0;mulpd 336(%%r9), %%xmm1;mulpd 352(%%r9), %%xmm2;mulpd 368(%%r9), %%xmm3;"
                "mulpd 384(%%r9), %%xmm0;mulpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;mulpd 432(%%r9), %%xmm3;"
                "mulpd 448(%%r9), %%xmm0;mulpd 464(%%r9), %%xmm1;mulpd 480(%%r9), %%xmm2;mulpd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                /*"sub $64,%%r14;"                
                "movapd %%xmm0,(%%r14);"
                "movapd %%xmm1,16(%%r14);"
                "movapd %%xmm2,32(%%r14);"
                "movapd %%xmm3,48(%%r14);"*/
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
 			/*printf("%f - %f - %f - %f - %f - %f - %f - %f\n",((double*)addr)[0],((double*)addr)[1],((double*)addr)[2],((double*)addr)[3],((double*)addr)[4],((double*)addr)[5],((double*)addr)[6],((double*)addr)[7]);
				addr-=64;
 			printf("%f - %f - %f - %f - %f - %f - %f - %f\n",((double*)addr)[0],((double*)addr)[1],((double*)addr)[2],((double*)addr)[3],((double*)addr)[4],((double*)addr)[5],((double*)addr)[6],((double*)addr)[7]);
 			fflush(stdout);*/
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
 			/*printf("%f - %f - %f - %f - %f - %f - %f - %f\n",((double*)addr)[0],((double*)addr)[1],((double*)addr)[2],((double*)addr)[3],((double*)addr)[4],((double*)addr)[5],((double*)addr)[6],((double*)addr)[7]);
 			addr-=64;
 			((double*)addr)[0]=0.0;
 			((double*)addr)[1]=0.0;
 			((double*)addr)[2]=0.0;
 			((double*)addr)[3]=0.0;
 			((double*)addr)[4]=0.0;
 			((double*)addr)[5]=0.0;
 			((double*)addr)[6]=0.0;
 			((double*)addr)[7]=0.0;
 			addr+=64;*/
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_pd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_pd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_pd_8;"       //|<
                "_sync1_mul_pd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_pd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_pd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_pd_8;"       //|<
                "_wait_mul_pd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_pd_8;"         //|<
                "_sync2_mul_pd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_pd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_pd_8:"
                "mulpd (%%r9), %%xmm0;mulpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;mulpd 48(%%r9), %%xmm3;"
                "mulpd 64(%%r9), %%xmm4;mulpd 80(%%r9), %%xmm5;mulpd 96(%%r9), %%xmm6;mulpd 112(%%r9), %%xmm7;"
                "mulpd 128(%%r9), %%xmm0;mulpd 144(%%r9), %%xmm1;mulpd 160(%%r9), %%xmm2;mulpd 176(%%r9), %%xmm3;"
                "mulpd 192(%%r9), %%xmm4;mulpd 208(%%r9), %%xmm5;mulpd 224(%%r9), %%xmm6;mulpd 240(%%r9), %%xmm7;"
                "mulpd 256(%%r9), %%xmm0;mulpd 272(%%r9), %%xmm1;mulpd 288(%%r9), %%xmm2;mulpd 304(%%r9), %%xmm3;"
                "mulpd 320(%%r9), %%xmm4;mulpd 336(%%r9), %%xmm5;mulpd 352(%%r9), %%xmm6;mulpd 368(%%r9), %%xmm7;"
                "mulpd 384(%%r9), %%xmm0;mulpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;mulpd 432(%%r9), %%xmm3;"
                "mulpd 448(%%r9), %%xmm4;mulpd 464(%%r9), %%xmm5;mulpd 480(%%r9), %%xmm6;mulpd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                /*"sub $64,%%r14;"                
                "movapd %%xmm0,(%%r14);"
                "movapd %%xmm1,16(%%r14);"
                "movapd %%xmm2,32(%%r14);"
                "movapd %%xmm3,48(%%r14);"*/
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
 			/*printf("%f - %f - %f - %f - %f - %f - %f - %f\n",((double*)addr)[0],((double*)addr)[1],((double*)addr)[2],((double*)addr)[3],((double*)addr)[4],((double*)addr)[5],((double*)addr)[6],((double*)addr)[7]);
				addr-=64;
 			printf("%f - %f - %f - %f - %f - %f - %f - %f\n",((double*)addr)[0],((double*)addr)[1],((double*)addr)[2],((double*)addr)[3],((double*)addr)[4],((double*)addr)[5],((double*)addr)[6],((double*)addr)[7]);
 			fflush(stdout);*/
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_mul_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mul_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ps_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ps_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ps_1;"       //|<
                "_sync1_mul_ps_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ps_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ps_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ps_1;"       //|<
                "_wait_mul_ps_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ps_1;"         //|<
                "_sync2_mul_ps_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ps_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ps_1:"
                "mulps (%%r9), %%xmm0;"
                "mulps 16(%%r9), %%xmm0;"
                "mulps 32(%%r9), %%xmm0;"
                "mulps 48(%%r9), %%xmm0;"
                "mulps 64(%%r9), %%xmm0;"
                "mulps 80(%%r9), %%xmm0;"
                "mulps 96(%%r9), %%xmm0;"
                "mulps 112(%%r9), %%xmm0;"
                "mulps 128(%%r9), %%xmm0;"
                "mulps 144(%%r9), %%xmm0;"
                "mulps 160(%%r9), %%xmm0;"
                "mulps 176(%%r9), %%xmm0;"
                "mulps 192(%%r9), %%xmm0;"
                "mulps 208(%%r9), %%xmm0;"
                "mulps 224(%%r9), %%xmm0;"
                "mulps 240(%%r9), %%xmm0;"
                "mulps 256(%%r9), %%xmm0;"
                "mulps 272(%%r9), %%xmm0;"
                "mulps 288(%%r9), %%xmm0;"
                "mulps 304(%%r9), %%xmm0;"
                "mulps 320(%%r9), %%xmm0;"
                "mulps 336(%%r9), %%xmm0;"
                "mulps 352(%%r9), %%xmm0;"
                "mulps 368(%%r9), %%xmm0;"
                "mulps 384(%%r9), %%xmm0;"
                "mulps 400(%%r9), %%xmm0;"
                "mulps 416(%%r9), %%xmm0;"
                "mulps 432(%%r9), %%xmm0;"
                "mulps 448(%%r9), %%xmm0;"
                "mulps 464(%%r9), %%xmm0;"
                "mulps 480(%%r9), %%xmm0;"
                "mulps 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ps_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ps_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ps_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ps_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ps_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ps_2;"       //|<
                "_sync1_mul_ps_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ps_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ps_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ps_2;"       //|<
                "_wait_mul_ps_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ps_2;"         //|<
                "_sync2_mul_ps_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ps_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ps_2:"
                "mulps (%%r9), %%xmm0;mulps 16(%%r9), %%xmm1;"
                "mulps 32(%%r9), %%xmm0;mulps 48(%%r9), %%xmm1;"
                "mulps 64(%%r9), %%xmm0;mulps 80(%%r9), %%xmm1;"
                "mulps 96(%%r9), %%xmm0;mulps 112(%%r9), %%xmm1;"
                "mulps 128(%%r9), %%xmm0;mulps 144(%%r9), %%xmm1;"
                "mulps 160(%%r9), %%xmm0;mulps 176(%%r9), %%xmm1;"
                "mulps 192(%%r9), %%xmm0;mulps 208(%%r9), %%xmm1;"
                "mulps 224(%%r9), %%xmm0;mulps 240(%%r9), %%xmm1;"
                "mulps 256(%%r9), %%xmm0;mulps 272(%%r9), %%xmm1;"
                "mulps 288(%%r9), %%xmm0;mulps 304(%%r9), %%xmm1;"
                "mulps 320(%%r9), %%xmm0;mulps 336(%%r9), %%xmm1;"
                "mulps 352(%%r9), %%xmm0;mulps 368(%%r9), %%xmm1;"
                "mulps 384(%%r9), %%xmm0;mulps 400(%%r9), %%xmm1;"
                "mulps 416(%%r9), %%xmm0;mulps 432(%%r9), %%xmm1;"
                "mulps 448(%%r9), %%xmm0;mulps 464(%%r9), %%xmm1;"
                "mulps 480(%%r9), %%xmm0;mulps 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ps_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ps_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ps_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ps_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ps_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ps_3;"       //|<
                "_sync1_mul_ps_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ps_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ps_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ps_3;"       //|<
                "_wait_mul_ps_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ps_3;"         //|<
                "_sync2_mul_ps_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ps_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ps_3:"
                "mulps (%%r9), %%xmm0;mulps 16(%%r9), %%xmm1;mulps 32(%%r9), %%xmm2;"
                "mulps 48(%%r9), %%xmm0;mulps 64(%%r9), %%xmm1;mulps 80(%%r9), %%xmm2;"
                "mulps 96(%%r9), %%xmm0;mulps 112(%%r9), %%xmm1;mulps 128(%%r9), %%xmm2;"
                "mulps 144(%%r9), %%xmm0;mulps 160(%%r9), %%xmm1;mulps 176(%%r9), %%xmm2;"
                "mulps 192(%%r9), %%xmm0;mulps 208(%%r9), %%xmm1;mulps 224(%%r9), %%xmm2;"
                "mulps 240(%%r9), %%xmm0;mulps 256(%%r9), %%xmm1;mulps 272(%%r9), %%xmm2;"
                "mulps 288(%%r9), %%xmm0;mulps 304(%%r9), %%xmm1;mulps 320(%%r9), %%xmm2;"
                "mulps 336(%%r9), %%xmm0;mulps 352(%%r9), %%xmm1;mulps 368(%%r9), %%xmm2;"
                "mulps 384(%%r9), %%xmm0;mulps 400(%%r9), %%xmm1;mulps 416(%%r9), %%xmm2;"
                "mulps 432(%%r9), %%xmm0;mulps 448(%%r9), %%xmm1;mulps 464(%%r9), %%xmm2;"
                "mulps 480(%%r9), %%xmm0;mulps 496(%%r9), %%xmm1;mulps 512(%%r9), %%xmm2;"
                "mulps 528(%%r9), %%xmm0;mulps 544(%%r9), %%xmm1;mulps 560(%%r9), %%xmm2;"
                "mulps 576(%%r9), %%xmm0;mulps 592(%%r9), %%xmm1;mulps 608(%%r9), %%xmm2;"
                "mulps 624(%%r9), %%xmm0;mulps 640(%%r9), %%xmm1;mulps 656(%%r9), %%xmm2;"
                "mulps 672(%%r9), %%xmm0;mulps 688(%%r9), %%xmm1;mulps 704(%%r9), %%xmm2;"
                "mulps 720(%%r9), %%xmm0;mulps 736(%%r9), %%xmm1;mulps 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ps_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ps_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_mul_ps_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ps_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ps_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ps_4;"       //|<
                "_sync1_mul_ps_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ps_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ps_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ps_4;"       //|<
                "_wait_mul_ps_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ps_4;"         //|<
                "_sync2_mul_ps_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ps_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ps_4:"
                "mulps (%%r9), %%xmm0;mulps 16(%%r9), %%xmm1;mulps 32(%%r9), %%xmm2;mulps 48(%%r9), %%xmm3;"
                "mulps 64(%%r9), %%xmm0;mulps 80(%%r9), %%xmm1;mulps 96(%%r9), %%xmm2;mulps 112(%%r9), %%xmm3;"
                "mulps 128(%%r9), %%xmm0;mulps 144(%%r9), %%xmm1;mulps 160(%%r9), %%xmm2;mulps 176(%%r9), %%xmm3;"
                "mulps 192(%%r9), %%xmm0;mulps 208(%%r9), %%xmm1;mulps 224(%%r9), %%xmm2;mulps 240(%%r9), %%xmm3;"
                "mulps 256(%%r9), %%xmm0;mulps 272(%%r9), %%xmm1;mulps 288(%%r9), %%xmm2;mulps 304(%%r9), %%xmm3;"
                "mulps 320(%%r9), %%xmm0;mulps 336(%%r9), %%xmm1;mulps 352(%%r9), %%xmm2;mulps 368(%%r9), %%xmm3;"
                "mulps 384(%%r9), %%xmm0;mulps 400(%%r9), %%xmm1;mulps 416(%%r9), %%xmm2;mulps 432(%%r9), %%xmm3;"
                "mulps 448(%%r9), %%xmm0;mulps 464(%%r9), %%xmm1;mulps 480(%%r9), %%xmm2;mulps 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ps_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ps_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ps_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ps_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ps_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ps_8;"       //|<
                "_sync1_mul_ps_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ps_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ps_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ps_8;"       //|<
                "_wait_mul_ps_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ps_8;"         //|<
                "_sync2_mul_ps_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ps_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ps_8:"
                "mulps (%%r9), %%xmm0;mulps 16(%%r9), %%xmm1;mulps 32(%%r9), %%xmm2;mulps 48(%%r9), %%xmm3;"
                "mulps 64(%%r9), %%xmm4;mulps 80(%%r9), %%xmm5;mulps 96(%%r9), %%xmm6;mulps 112(%%r9), %%xmm7;"
                "mulps 128(%%r9), %%xmm0;mulps 144(%%r9), %%xmm1;mulps 160(%%r9), %%xmm2;mulps 176(%%r9), %%xmm3;"
                "mulps 192(%%r9), %%xmm4;mulps 208(%%r9), %%xmm5;mulps 224(%%r9), %%xmm6;mulps 240(%%r9), %%xmm7;"
                "mulps 256(%%r9), %%xmm0;mulps 272(%%r9), %%xmm1;mulps 288(%%r9), %%xmm2;mulps 304(%%r9), %%xmm3;"
                "mulps 320(%%r9), %%xmm4;mulps 336(%%r9), %%xmm5;mulps 352(%%r9), %%xmm6;mulps 368(%%r9), %%xmm7;"
                "mulps 384(%%r9), %%xmm0;mulps 400(%%r9), %%xmm1;mulps 416(%%r9), %%xmm2;mulps 432(%%r9), %%xmm3;"
                "mulps 448(%%r9), %%xmm4;mulps 464(%%r9), %%xmm5;mulps 480(%%r9), %%xmm6;mulps 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ps_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ps_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ps_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_mul_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mul_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_sd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_sd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_sd_1;"       //|<
                "_sync1_mul_sd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_sd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_sd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_sd_1;"       //|<
                "_wait_mul_sd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_sd_1;"         //|<
                "_sync2_mul_sd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_sd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_sd_1:"
                "mulsd (%%r9), %%xmm0;"
                "mulsd 16(%%r9), %%xmm0;"
                "mulsd 32(%%r9), %%xmm0;"
                "mulsd 48(%%r9), %%xmm0;"
                "mulsd 64(%%r9), %%xmm0;"
                "mulsd 80(%%r9), %%xmm0;"
                "mulsd 96(%%r9), %%xmm0;"
                "mulsd 112(%%r9), %%xmm0;"
                "mulsd 128(%%r9), %%xmm0;"
                "mulsd 144(%%r9), %%xmm0;"
                "mulsd 160(%%r9), %%xmm0;"
                "mulsd 176(%%r9), %%xmm0;"
                "mulsd 192(%%r9), %%xmm0;"
                "mulsd 208(%%r9), %%xmm0;"
                "mulsd 224(%%r9), %%xmm0;"
                "mulsd 240(%%r9), %%xmm0;"
                "mulsd 256(%%r9), %%xmm0;"
                "mulsd 272(%%r9), %%xmm0;"
                "mulsd 288(%%r9), %%xmm0;"
                "mulsd 304(%%r9), %%xmm0;"
                "mulsd 320(%%r9), %%xmm0;"
                "mulsd 336(%%r9), %%xmm0;"
                "mulsd 352(%%r9), %%xmm0;"
                "mulsd 368(%%r9), %%xmm0;"
                "mulsd 384(%%r9), %%xmm0;"
                "mulsd 400(%%r9), %%xmm0;"
                "mulsd 416(%%r9), %%xmm0;"
                "mulsd 432(%%r9), %%xmm0;"
                "mulsd 448(%%r9), %%xmm0;"
                "mulsd 464(%%r9), %%xmm0;"
                "mulsd 480(%%r9), %%xmm0;"
                "mulsd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_sd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_sd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_sd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_sd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_sd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_sd_2;"       //|<
                "_sync1_mul_sd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_sd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_sd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_sd_2;"       //|<
                "_wait_mul_sd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_sd_2;"         //|<
                "_sync2_mul_sd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_sd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_sd_2:"
                "mulsd (%%r9), %%xmm0;mulsd 16(%%r9), %%xmm1;"
                "mulsd 32(%%r9), %%xmm0;mulsd 48(%%r9), %%xmm1;"
                "mulsd 64(%%r9), %%xmm0;mulsd 80(%%r9), %%xmm1;"
                "mulsd 96(%%r9), %%xmm0;mulsd 112(%%r9), %%xmm1;"
                "mulsd 128(%%r9), %%xmm0;mulsd 144(%%r9), %%xmm1;"
                "mulsd 160(%%r9), %%xmm0;mulsd 176(%%r9), %%xmm1;"
                "mulsd 192(%%r9), %%xmm0;mulsd 208(%%r9), %%xmm1;"
                "mulsd 224(%%r9), %%xmm0;mulsd 240(%%r9), %%xmm1;"
                "mulsd 256(%%r9), %%xmm0;mulsd 272(%%r9), %%xmm1;"
                "mulsd 288(%%r9), %%xmm0;mulsd 304(%%r9), %%xmm1;"
                "mulsd 320(%%r9), %%xmm0;mulsd 336(%%r9), %%xmm1;"
                "mulsd 352(%%r9), %%xmm0;mulsd 368(%%r9), %%xmm1;"
                "mulsd 384(%%r9), %%xmm0;mulsd 400(%%r9), %%xmm1;"
                "mulsd 416(%%r9), %%xmm0;mulsd 432(%%r9), %%xmm1;"
                "mulsd 448(%%r9), %%xmm0;mulsd 464(%%r9), %%xmm1;"
                "mulsd 480(%%r9), %%xmm0;mulsd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_sd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_sd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_sd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_sd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_sd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_sd_3;"       //|<
                "_sync1_mul_sd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_sd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_sd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_sd_3;"       //|<
                "_wait_mul_sd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_sd_3;"         //|<
                "_sync2_mul_sd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_sd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_sd_3:"
                "mulsd (%%r9), %%xmm0;mulsd 16(%%r9), %%xmm1;mulsd 32(%%r9), %%xmm2;"
                "mulsd 48(%%r9), %%xmm0;mulsd 64(%%r9), %%xmm1;mulsd 80(%%r9), %%xmm2;"
                "mulsd 96(%%r9), %%xmm0;mulsd 112(%%r9), %%xmm1;mulsd 128(%%r9), %%xmm2;"
                "mulsd 144(%%r9), %%xmm0;mulsd 160(%%r9), %%xmm1;mulsd 176(%%r9), %%xmm2;"
                "mulsd 192(%%r9), %%xmm0;mulsd 208(%%r9), %%xmm1;mulsd 224(%%r9), %%xmm2;"
                "mulsd 240(%%r9), %%xmm0;mulsd 256(%%r9), %%xmm1;mulsd 272(%%r9), %%xmm2;"
                "mulsd 288(%%r9), %%xmm0;mulsd 304(%%r9), %%xmm1;mulsd 320(%%r9), %%xmm2;"
                "mulsd 336(%%r9), %%xmm0;mulsd 352(%%r9), %%xmm1;mulsd 368(%%r9), %%xmm2;"
                "mulsd 384(%%r9), %%xmm0;mulsd 400(%%r9), %%xmm1;mulsd 416(%%r9), %%xmm2;"
                "mulsd 432(%%r9), %%xmm0;mulsd 448(%%r9), %%xmm1;mulsd 464(%%r9), %%xmm2;"
                "mulsd 480(%%r9), %%xmm0;mulsd 496(%%r9), %%xmm1;mulsd 512(%%r9), %%xmm2;"
                "mulsd 528(%%r9), %%xmm0;mulsd 544(%%r9), %%xmm1;mulsd 560(%%r9), %%xmm2;"
                "mulsd 576(%%r9), %%xmm0;mulsd 592(%%r9), %%xmm1;mulsd 608(%%r9), %%xmm2;"
                "mulsd 624(%%r9), %%xmm0;mulsd 640(%%r9), %%xmm1;mulsd 656(%%r9), %%xmm2;"
                "mulsd 672(%%r9), %%xmm0;mulsd 688(%%r9), %%xmm1;mulsd 704(%%r9), %%xmm2;"
                "mulsd 720(%%r9), %%xmm0;mulsd 736(%%r9), %%xmm1;mulsd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_sd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_sd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_mul_sd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_sd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_sd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_sd_4;"       //|<
                "_sync1_mul_sd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_sd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_sd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_sd_4;"       //|<
                "_wait_mul_sd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_sd_4;"         //|<
                "_sync2_mul_sd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_sd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_sd_4:"
                "mulsd (%%r9), %%xmm0;mulsd 16(%%r9), %%xmm1;mulsd 32(%%r9), %%xmm2;mulsd 48(%%r9), %%xmm3;"
                "mulsd 64(%%r9), %%xmm0;mulsd 80(%%r9), %%xmm1;mulsd 96(%%r9), %%xmm2;mulsd 112(%%r9), %%xmm3;"
                "mulsd 128(%%r9), %%xmm0;mulsd 144(%%r9), %%xmm1;mulsd 160(%%r9), %%xmm2;mulsd 176(%%r9), %%xmm3;"
                "mulsd 192(%%r9), %%xmm0;mulsd 208(%%r9), %%xmm1;mulsd 224(%%r9), %%xmm2;mulsd 240(%%r9), %%xmm3;"
                "mulsd 256(%%r9), %%xmm0;mulsd 272(%%r9), %%xmm1;mulsd 288(%%r9), %%xmm2;mulsd 304(%%r9), %%xmm3;"
                "mulsd 320(%%r9), %%xmm0;mulsd 336(%%r9), %%xmm1;mulsd 352(%%r9), %%xmm2;mulsd 368(%%r9), %%xmm3;"
                "mulsd 384(%%r9), %%xmm0;mulsd 400(%%r9), %%xmm1;mulsd 416(%%r9), %%xmm2;mulsd 432(%%r9), %%xmm3;"
                "mulsd 448(%%r9), %%xmm0;mulsd 464(%%r9), %%xmm1;mulsd 480(%%r9), %%xmm2;mulsd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_sd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_sd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_sd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_sd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_sd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_sd_8;"       //|<
                "_sync1_mul_sd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_sd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_sd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_sd_8;"       //|<
                "_wait_mul_sd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_sd_8;"         //|<
                "_sync2_mul_sd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_sd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_sd_8:"
                "mulsd (%%r9), %%xmm0;mulsd 16(%%r9), %%xmm1;mulsd 32(%%r9), %%xmm2;mulsd 48(%%r9), %%xmm3;"
                "mulsd 64(%%r9), %%xmm4;mulsd 80(%%r9), %%xmm5;mulsd 96(%%r9), %%xmm6;mulsd 112(%%r9), %%xmm7;"
                "mulsd 128(%%r9), %%xmm0;mulsd 144(%%r9), %%xmm1;mulsd 160(%%r9), %%xmm2;mulsd 176(%%r9), %%xmm3;"
                "mulsd 192(%%r9), %%xmm4;mulsd 208(%%r9), %%xmm5;mulsd 224(%%r9), %%xmm6;mulsd 240(%%r9), %%xmm7;"
                "mulsd 256(%%r9), %%xmm0;mulsd 272(%%r9), %%xmm1;mulsd 288(%%r9), %%xmm2;mulsd 304(%%r9), %%xmm3;"
                "mulsd 320(%%r9), %%xmm4;mulsd 336(%%r9), %%xmm5;mulsd 352(%%r9), %%xmm6;mulsd 368(%%r9), %%xmm7;"
                "mulsd 384(%%r9), %%xmm0;mulsd 400(%%r9), %%xmm1;mulsd 416(%%r9), %%xmm2;mulsd 432(%%r9), %%xmm3;"
                "mulsd 448(%%r9), %%xmm4;mulsd 464(%%r9), %%xmm5;mulsd 480(%%r9), %%xmm6;mulsd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_sd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_sd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_sd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_mul_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mul_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ss_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ss_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ss_1;"       //|<
                "_sync1_mul_ss_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ss_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ss_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ss_1;"       //|<
                "_wait_mul_ss_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ss_1;"         //|<
                "_sync2_mul_ss_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ss_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ss_1:"
                "mulss (%%r9), %%xmm0;"
                "mulss 16(%%r9), %%xmm0;"
                "mulss 32(%%r9), %%xmm0;"
                "mulss 48(%%r9), %%xmm0;"
                "mulss 64(%%r9), %%xmm0;"
                "mulss 80(%%r9), %%xmm0;"
                "mulss 96(%%r9), %%xmm0;"
                "mulss 112(%%r9), %%xmm0;"
                "mulss 128(%%r9), %%xmm0;"
                "mulss 144(%%r9), %%xmm0;"
                "mulss 160(%%r9), %%xmm0;"
                "mulss 176(%%r9), %%xmm0;"
                "mulss 192(%%r9), %%xmm0;"
                "mulss 208(%%r9), %%xmm0;"
                "mulss 224(%%r9), %%xmm0;"
                "mulss 240(%%r9), %%xmm0;"
                "mulss 256(%%r9), %%xmm0;"
                "mulss 272(%%r9), %%xmm0;"
                "mulss 288(%%r9), %%xmm0;"
                "mulss 304(%%r9), %%xmm0;"
                "mulss 320(%%r9), %%xmm0;"
                "mulss 336(%%r9), %%xmm0;"
                "mulss 352(%%r9), %%xmm0;"
                "mulss 368(%%r9), %%xmm0;"
                "mulss 384(%%r9), %%xmm0;"
                "mulss 400(%%r9), %%xmm0;"
                "mulss 416(%%r9), %%xmm0;"
                "mulss 432(%%r9), %%xmm0;"
                "mulss 448(%%r9), %%xmm0;"
                "mulss 464(%%r9), %%xmm0;"
                "mulss 480(%%r9), %%xmm0;"
                "mulss 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ss_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ss_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ss_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ss_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ss_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ss_2;"       //|<
                "_sync1_mul_ss_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ss_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ss_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ss_2;"       //|<
                "_wait_mul_ss_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ss_2;"         //|<
                "_sync2_mul_ss_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ss_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ss_2:"
                "mulss (%%r9), %%xmm0;mulss 16(%%r9), %%xmm1;"
                "mulss 32(%%r9), %%xmm0;mulss 48(%%r9), %%xmm1;"
                "mulss 64(%%r9), %%xmm0;mulss 80(%%r9), %%xmm1;"
                "mulss 96(%%r9), %%xmm0;mulss 112(%%r9), %%xmm1;"
                "mulss 128(%%r9), %%xmm0;mulss 144(%%r9), %%xmm1;"
                "mulss 160(%%r9), %%xmm0;mulss 176(%%r9), %%xmm1;"
                "mulss 192(%%r9), %%xmm0;mulss 208(%%r9), %%xmm1;"
                "mulss 224(%%r9), %%xmm0;mulss 240(%%r9), %%xmm1;"
                "mulss 256(%%r9), %%xmm0;mulss 272(%%r9), %%xmm1;"
                "mulss 288(%%r9), %%xmm0;mulss 304(%%r9), %%xmm1;"
                "mulss 320(%%r9), %%xmm0;mulss 336(%%r9), %%xmm1;"
                "mulss 352(%%r9), %%xmm0;mulss 368(%%r9), %%xmm1;"
                "mulss 384(%%r9), %%xmm0;mulss 400(%%r9), %%xmm1;"
                "mulss 416(%%r9), %%xmm0;mulss 432(%%r9), %%xmm1;"
                "mulss 448(%%r9), %%xmm0;mulss 464(%%r9), %%xmm1;"
                "mulss 480(%%r9), %%xmm0;mulss 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ss_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ss_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ss_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ss_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ss_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ss_3;"       //|<
                "_sync1_mul_ss_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ss_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ss_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ss_3;"       //|<
                "_wait_mul_ss_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ss_3;"         //|<
                "_sync2_mul_ss_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ss_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ss_3:"
                "mulss (%%r9), %%xmm0;mulss 16(%%r9), %%xmm1;mulss 32(%%r9), %%xmm2;"
                "mulss 48(%%r9), %%xmm0;mulss 64(%%r9), %%xmm1;mulss 80(%%r9), %%xmm2;"
                "mulss 96(%%r9), %%xmm0;mulss 112(%%r9), %%xmm1;mulss 128(%%r9), %%xmm2;"
                "mulss 144(%%r9), %%xmm0;mulss 160(%%r9), %%xmm1;mulss 176(%%r9), %%xmm2;"
                "mulss 192(%%r9), %%xmm0;mulss 208(%%r9), %%xmm1;mulss 224(%%r9), %%xmm2;"
                "mulss 240(%%r9), %%xmm0;mulss 256(%%r9), %%xmm1;mulss 272(%%r9), %%xmm2;"
                "mulss 288(%%r9), %%xmm0;mulss 304(%%r9), %%xmm1;mulss 320(%%r9), %%xmm2;"
                "mulss 336(%%r9), %%xmm0;mulss 352(%%r9), %%xmm1;mulss 368(%%r9), %%xmm2;"
                "mulss 384(%%r9), %%xmm0;mulss 400(%%r9), %%xmm1;mulss 416(%%r9), %%xmm2;"
                "mulss 432(%%r9), %%xmm0;mulss 448(%%r9), %%xmm1;mulss 464(%%r9), %%xmm2;"
                "mulss 480(%%r9), %%xmm0;mulss 496(%%r9), %%xmm1;mulss 512(%%r9), %%xmm2;"
                "mulss 528(%%r9), %%xmm0;mulss 544(%%r9), %%xmm1;mulss 560(%%r9), %%xmm2;"
                "mulss 576(%%r9), %%xmm0;mulss 592(%%r9), %%xmm1;mulss 608(%%r9), %%xmm2;"
                "mulss 624(%%r9), %%xmm0;mulss 640(%%r9), %%xmm1;mulss 656(%%r9), %%xmm2;"
                "mulss 672(%%r9), %%xmm0;mulss 688(%%r9), %%xmm1;mulss 704(%%r9), %%xmm2;"
                "mulss 720(%%r9), %%xmm0;mulss 736(%%r9), %%xmm1;mulss 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ss_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ss_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_mul_ss_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ss_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ss_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ss_4;"       //|<
                "_sync1_mul_ss_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ss_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ss_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ss_4;"       //|<
                "_wait_mul_ss_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ss_4;"         //|<
                "_sync2_mul_ss_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ss_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ss_4:"
                "mulss (%%r9), %%xmm0;mulss 16(%%r9), %%xmm1;mulss 32(%%r9), %%xmm2;mulss 48(%%r9), %%xmm3;"
                "mulss 64(%%r9), %%xmm0;mulss 80(%%r9), %%xmm1;mulss 96(%%r9), %%xmm2;mulss 112(%%r9), %%xmm3;"
                "mulss 128(%%r9), %%xmm0;mulss 144(%%r9), %%xmm1;mulss 160(%%r9), %%xmm2;mulss 176(%%r9), %%xmm3;"
                "mulss 192(%%r9), %%xmm0;mulss 208(%%r9), %%xmm1;mulss 224(%%r9), %%xmm2;mulss 240(%%r9), %%xmm3;"
                "mulss 256(%%r9), %%xmm0;mulss 272(%%r9), %%xmm1;mulss 288(%%r9), %%xmm2;mulss 304(%%r9), %%xmm3;"
                "mulss 320(%%r9), %%xmm0;mulss 336(%%r9), %%xmm1;mulss 352(%%r9), %%xmm2;mulss 368(%%r9), %%xmm3;"
                "mulss 384(%%r9), %%xmm0;mulss 400(%%r9), %%xmm1;mulss 416(%%r9), %%xmm2;mulss 432(%%r9), %%xmm3;"
                "mulss 448(%%r9), %%xmm0;mulss 464(%%r9), %%xmm1;mulss 480(%%r9), %%xmm2;mulss 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ss_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ss_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ss_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_ss_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_ss_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_ss_8;"       //|<
                "_sync1_mul_ss_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_ss_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_ss_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_ss_8;"       //|<
                "_wait_mul_ss_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_ss_8;"         //|<
                "_sync2_mul_ss_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_ss_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_ss_8:"
                "mulss (%%r9), %%xmm0;mulss 16(%%r9), %%xmm1;mulss 32(%%r9), %%xmm2;mulss 48(%%r9), %%xmm3;"
                "mulss 64(%%r9), %%xmm4;mulss 80(%%r9), %%xmm5;mulss 96(%%r9), %%xmm6;mulss 112(%%r9), %%xmm7;"
                "mulss 128(%%r9), %%xmm0;mulss 144(%%r9), %%xmm1;mulss 160(%%r9), %%xmm2;mulss 176(%%r9), %%xmm3;"
                "mulss 192(%%r9), %%xmm4;mulss 208(%%r9), %%xmm5;mulss 224(%%r9), %%xmm6;mulss 240(%%r9), %%xmm7;"
                "mulss 256(%%r9), %%xmm0;mulss 272(%%r9), %%xmm1;mulss 288(%%r9), %%xmm2;mulss 304(%%r9), %%xmm3;"
                "mulss 320(%%r9), %%xmm4;mulss 336(%%r9), %%xmm5;mulss 352(%%r9), %%xmm6;mulss 368(%%r9), %%xmm7;"
                "mulss 384(%%r9), %%xmm0;mulss 400(%%r9), %%xmm1;mulss 416(%%r9), %%xmm2;mulss 432(%%r9), %%xmm3;"
                "mulss 448(%%r9), %%xmm4;mulss 464(%%r9), %%xmm5;mulss 480(%%r9), %%xmm6;mulss 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_ss_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_ss_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_ss_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_mul_add_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mul_add_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_add_pd_1;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_add_pd_1:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_add_pd_1;"   //|<
                "_sync1_mul_add_pd_1:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_add_pd_1;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_add_pd_1;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_add_pd_1;"   //|<
                "_wait_mul_add_pd_1:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_add_pd_1;"     //|<
                "_sync2_mul_add_pd_1:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_add_pd_1;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 32(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_add_pd_1:"
                "mulpd (%%r9), %%xmm0;"
                "mulpd 16(%%r9), %%xmm0;"
                "addpd 32(%%r9), %%xmm1;"
                "addpd 48(%%r9), %%xmm1;"
                "mulpd 64(%%r9), %%xmm0;"
                "mulpd 80(%%r9), %%xmm0;"
                "addpd 96(%%r9), %%xmm1;"
                "addpd 112(%%r9), %%xmm1;"
                "mulpd 128(%%r9), %%xmm0;"
                "mulpd 144(%%r9), %%xmm0;"
                "addpd 160(%%r9), %%xmm1;"
                "addpd 176(%%r9), %%xmm1;"
                "mulpd 192(%%r9), %%xmm0;"
                "mulpd 208(%%r9), %%xmm0;"
                "addpd 224(%%r9), %%xmm1;"
                "addpd 240(%%r9), %%xmm1;"
                "mulpd 256(%%r9), %%xmm0;"
                "mulpd 272(%%r9), %%xmm0;"
                "addpd 288(%%r9), %%xmm1;"
                "addpd 304(%%r9), %%xmm1;"
                "mulpd 320(%%r9), %%xmm0;"
                "mulpd 336(%%r9), %%xmm0;"
                "addpd 352(%%r9), %%xmm1;"
                "addpd 368(%%r9), %%xmm1;"
                "mulpd 384(%%r9), %%xmm0;"
                "mulpd 400(%%r9), %%xmm0;"
                "addpd 416(%%r9), %%xmm1;"
                "addpd 432(%%r9), %%xmm1;"
                "mulpd 448(%%r9), %%xmm0;"
                "mulpd 464(%%r9), %%xmm0;"
                "addpd 480(%%r9), %%xmm1;"
                "addpd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_add_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_add_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_add_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_add_pd_2;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_add_pd_2:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_add_pd_2;"   //|<
                "_sync1_mul_add_pd_2:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_add_pd_2;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_add_pd_2;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_add_pd_2;"   //|<
                "_wait_mul_add_pd_2:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_add_pd_2;"     //|<
                "_sync2_mul_add_pd_2:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_add_pd_2;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_add_pd_2:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;"
                "mulpd 32(%%r9), %%xmm2;addpd 48(%%r9), %%xmm3;"
                "mulpd 64(%%r9), %%xmm0;addpd 80(%%r9), %%xmm1;"
                "mulpd 96(%%r9), %%xmm2;addpd 112(%%r9), %%xmm3;"
                "mulpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;"
                "mulpd 160(%%r9), %%xmm2;addpd 176(%%r9), %%xmm3;"
                "mulpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;"
                "mulpd 224(%%r9), %%xmm2;addpd 240(%%r9), %%xmm3;"
                "mulpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;"
                "mulpd 288(%%r9), %%xmm2;addpd 304(%%r9), %%xmm3;"
                "mulpd 320(%%r9), %%xmm0;addpd 336(%%r9), %%xmm1;"
                "mulpd 352(%%r9), %%xmm2;addpd 368(%%r9), %%xmm3;"
                "mulpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;"
                "mulpd 416(%%r9), %%xmm2;addpd 432(%%r9), %%xmm3;"
                "mulpd 448(%%r9), %%xmm0;addpd 464(%%r9), %%xmm1;"
                "mulpd 480(%%r9), %%xmm2;addpd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_add_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_add_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_add_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_add_pd_3;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_add_pd_3:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_add_pd_3;"   //|<
                "_sync1_mul_add_pd_3:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_add_pd_3;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_add_pd_3;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_add_pd_3;"   //|<
                "_wait_mul_add_pd_3:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_add_pd_3;"     //|<
                "_sync2_mul_add_pd_3:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_add_pd_3;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_add_pd_3:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;"
                "addpd 48(%%r9), %%xmm3;mulpd 64(%%r9), %%xmm4;addpd 80(%%r9), %%xmm5;"                             
                "addpd 96(%%r9), %%xmm3;mulpd 112(%%r9), %%xmm4;addpd 128(%%r9), %%xmm5;"
                "mulpd 144(%%r9), %%xmm0;addpd 160(%%r9), %%xmm1;mulpd 176(%%r9), %%xmm2;"
                
                "mulpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;mulpd 224(%%r9), %%xmm2;"
                "addpd 240(%%r9), %%xmm3;mulpd 256(%%r9), %%xmm4;addpd 272(%%r9), %%xmm5;"                
                "addpd 288(%%r9), %%xmm3;mulpd 304(%%r9), %%xmm4;addpd 320(%%r9), %%xmm5;"
                "mulpd 336(%%r9), %%xmm0;addpd 352(%%r9), %%xmm1;mulpd 368(%%r9), %%xmm2;"
                                
                "mulpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;"
                "addpd 432(%%r9), %%xmm3;mulpd 448(%%r9), %%xmm4;addpd 464(%%r9), %%xmm5;"                
                "addpd 480(%%r9), %%xmm3;mulpd 496(%%r9), %%xmm4;addpd 512(%%r9), %%xmm5;"
                "mulpd 528(%%r9), %%xmm0;addpd 544(%%r9), %%xmm1;mulpd 560(%%r9), %%xmm2;"
                                
                "mulpd 576(%%r9), %%xmm0;addpd 592(%%r9), %%xmm1;mulpd 608(%%r9), %%xmm2;"
                "addpd 624(%%r9), %%xmm3;mulpd 640(%%r9), %%xmm4;addpd 656(%%r9), %%xmm5;"
                "addpd 672(%%r9), %%xmm3;mulpd 688(%%r9), %%xmm4;addpd 704(%%r9), %%xmm5;" 
                "mulpd 720(%%r9), %%xmm0;addpd 736(%%r9), %%xmm1;mulpd 752(%%r9), %%xmm2;"                    
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_add_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_add_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_mul_add_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_add_pd_4;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_add_pd_4:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_add_pd_4;"   //|<
                "_sync1_mul_add_pd_4:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_add_pd_4;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_add_pd_4;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_add_pd_4;"   //|<
                "_wait_mul_add_pd_4:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_add_pd_4;"     //|<
                "_sync2_mul_add_pd_4:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_add_pd_4;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_add_pd_4:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;addpd 48(%%r9), %%xmm3;"
                "mulpd 64(%%r9), %%xmm4;addpd 80(%%r9), %%xmm5;mulpd 96(%%r9), %%xmm6;addpd 112(%%r9), %%xmm7;"
                "mulpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;mulpd 160(%%r9), %%xmm2;addpd 176(%%r9), %%xmm3;"
                "mulpd 192(%%r9), %%xmm4;addpd 208(%%r9), %%xmm5;mulpd 224(%%r9), %%xmm6;addpd 240(%%r9), %%xmm7;"
                "mulpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;mulpd 288(%%r9), %%xmm2;addpd 304(%%r9), %%xmm3;"
                "mulpd 320(%%r9), %%xmm4;addpd 336(%%r9), %%xmm5;mulpd 352(%%r9), %%xmm6;addpd 368(%%r9), %%xmm7;"
                "mulpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;addpd 432(%%r9), %%xmm3;"
                "mulpd 448(%%r9), %%xmm4;addpd 464(%%r9), %%xmm5;mulpd 480(%%r9), %%xmm6;addpd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_add_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_add_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_add_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_add_pd_8;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_add_pd_8:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_add_pd_8;"   //|<
                "_sync1_mul_add_pd_8:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_add_pd_8;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_add_pd_8;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_add_pd_8;"   //|<
                "_wait_mul_add_pd_8:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_add_pd_8;"     //|<
                "_sync2_mul_add_pd_8:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_add_pd_8;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                "movapd 128(%%r9), %%xmm8;"
                "movapd 144(%%r9), %%xmm9;"
                "movapd 160(%%r9), %%xmm10;"
                "movapd 176(%%r9), %%xmm11;"
                "movapd 192(%%r9), %%xmm12;"
                "movapd 208(%%r9), %%xmm13;"
                "movapd 224(%%r9), %%xmm14;"
                "movapd 240(%%r9), %%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_add_pd_8:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;addpd 48(%%r9), %%xmm3;"
                "mulpd 64(%%r9), %%xmm4;addpd 80(%%r9), %%xmm5;mulpd 96(%%r9), %%xmm6;addpd 112(%%r9), %%xmm7;"
                "mulpd 128(%%r9), %%xmm8;addpd 144(%%r9), %%xmm9;mulpd 160(%%r9), %%xmm10;addpd 176(%%r9), %%xmm11;"
                "mulpd 192(%%r9), %%xmm12;addpd 208(%%r9), %%xmm13;mulpd 224(%%r9), %%xmm14;addpd 240(%%r9), %%xmm15;"
                "mulpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;mulpd 288(%%r9), %%xmm2;addpd 304(%%r9), %%xmm3;"
                "mulpd 320(%%r9), %%xmm4;addpd 336(%%r9), %%xmm5;mulpd 352(%%r9), %%xmm6;addpd 368(%%r9), %%xmm7;"
                "mulpd 384(%%r9), %%xmm8;addpd 400(%%r9), %%xmm9;mulpd 416(%%r9), %%xmm10;addpd 432(%%r9), %%xmm11;"
                "mulpd 448(%%r9), %%xmm12;addpd 464(%%r9), %%xmm13;mulpd 480(%%r9), %%xmm14;addpd 496(%%r9), %%xmm15;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_add_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_add_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_add_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
		);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_mul_plus_add_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_mul_plus_add_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_plus_add_pd_1;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_plus_add_pd_1:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_plus_add_pd_1;"   //|<
                "_sync1_mul_plus_add_pd_1:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_plus_add_pd_1;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_plus_add_pd_1;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_plus_add_pd_1;"   //|<
                "_wait_mul_plus_add_pd_1:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_plus_add_pd_1;"     //|<
                "_sync2_mul_plus_add_pd_1:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_plus_add_pd_1;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 32(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_plus_add_pd_1:"
                "mulpd (%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 16(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 32(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 48(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "mulpd 64(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 80(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 96(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 112(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "mulpd 128(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 144(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 160(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 176(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "mulpd 192(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 208(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 224(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 240(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "mulpd 256(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 272(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 288(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 304(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "mulpd 320(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 336(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 352(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 368(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "mulpd 384(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 400(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 416(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 432(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "mulpd 448(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "mulpd 464(%%r9), %%xmm0;addpd %%xmm0, %%xmm1;"
                "addpd 480(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "addpd 496(%%r9), %%xmm0;mulpd %%xmm0, %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_plus_add_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_plus_add_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_plus_add_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_plus_add_pd_2;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_plus_add_pd_2:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_plus_add_pd_2;"   //|<
                "_sync1_mul_plus_add_pd_2:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_plus_add_pd_2;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_plus_add_pd_2;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_plus_add_pd_2;"   //|<
                "_wait_mul_plus_add_pd_2:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_plus_add_pd_2;"     //|<
                "_sync2_mul_plus_add_pd_2:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_plus_add_pd_2;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_plus_add_pd_2:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 32(%%r9), %%xmm0;addpd 48(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 64(%%r9), %%xmm0;addpd 80(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 96(%%r9), %%xmm0;addpd 112(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 160(%%r9), %%xmm0;addpd 176(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 224(%%r9), %%xmm0;addpd 240(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 288(%%r9), %%xmm0;addpd 304(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 320(%%r9), %%xmm0;addpd 336(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 352(%%r9), %%xmm0;addpd 368(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 416(%%r9), %%xmm0;addpd 432(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 448(%%r9), %%xmm0;addpd 464(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "mulpd 480(%%r9), %%xmm0;addpd 496(%%r9), %%xmm1;"
                "addpd %%xmm0,%%xmm2;mulpd %%xmm1,%%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_plus_add_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_plus_add_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_plus_add_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_plus_add_pd_3;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_plus_add_pd_3:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_plus_add_pd_3;"   //|<
                "_sync1_mul_plus_add_pd_3:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_plus_add_pd_3;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_plus_add_pd_3;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_plus_add_pd_3;"   //|<
                "_wait_mul_plus_add_pd_3:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_plus_add_pd_3;"     //|<
                "_sync2_mul_plus_add_pd_3:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_plus_add_pd_3;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_plus_add_pd_3:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;"
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"
                "mulpd 48(%%r9), %%xmm0;addpd 64(%%r9), %%xmm1;mulpd 80(%%r9), %%xmm2;"
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"               
                "addpd 96(%%r9), %%xmm0;mulpd 112(%%r9), %%xmm1;addpd 128(%%r9), %%xmm2;"
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                "addpd 144(%%r9), %%xmm0;mulpd 160(%%r9), %%xmm1;addpd 176(%%r9), %%xmm2;"
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                
                "mulpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;mulpd 224(%%r9), %%xmm2;"
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"
                "mulpd 240(%%r9), %%xmm0;addpd 256(%%r9), %%xmm1;mulpd 272(%%r9), %%xmm2;"                
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"
                "addpd 288(%%r9), %%xmm0;mulpd 304(%%r9), %%xmm1;addpd 320(%%r9), %%xmm2;"
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                "addpd 336(%%r9), %%xmm0;mulpd 352(%%r9), %%xmm1;addpd 368(%%r9), %%xmm2;"
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                
                "mulpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;"
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"
                "mulpd 432(%%r9), %%xmm0;addpd 448(%%r9), %%xmm1;mulpd 464(%%r9), %%xmm2;"                
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"
                "addpd 480(%%r9), %%xmm0;mulpd 496(%%r9), %%xmm1;addpd 512(%%r9), %%xmm2;"
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                "addpd 528(%%r9), %%xmm0;mulpd 544(%%r9), %%xmm1;addpd 560(%%r9), %%xmm2;"
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                                
                "mulpd 576(%%r9), %%xmm0;addpd 592(%%r9), %%xmm1;mulpd 608(%%r9), %%xmm2;"
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"
                "mulpd 624(%%r9), %%xmm0;addpd 640(%%r9), %%xmm1;mulpd 656(%%r9), %%xmm2;"
                "addpd %%xmm0,%%xmm3;mulpd %%xmm1,%%xmm4;addpd %%xmm2,%%xmm5;"
                "addpd 672(%%r9), %%xmm0;mulpd 688(%%r9), %%xmm1;addpd 704(%%r9), %%xmm2;" 
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                "addpd 720(%%r9), %%xmm0;mulpd 736(%%r9), %%xmm1;addpd 752(%%r9), %%xmm2;"                    
                "mulpd %%xmm0,%%xmm3;addpd %%xmm1,%%xmm4;mulpd %%xmm2,%%xmm5;"
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_plus_add_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_plus_add_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_mul_plus_add_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_plus_add_pd_4;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_plus_add_pd_4:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_plus_add_pd_4;"   //|<
                "_sync1_mul_plus_add_pd_4:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_plus_add_pd_4;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_plus_add_pd_4;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_plus_add_pd_4;"   //|<
                "_wait_mul_plus_add_pd_4:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_plus_add_pd_4;"     //|<
                "_sync2_mul_plus_add_pd_4:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_plus_add_pd_4;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_plus_add_pd_4:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;addpd 48(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "mulpd 64(%%r9), %%xmm0;addpd 80(%%r9), %%xmm1;mulpd 96(%%r9), %%xmm2;addpd 112(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "mulpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;mulpd 160(%%r9), %%xmm2;addpd 176(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "mulpd 192(%%r9), %%xmm0;addpd 208(%%r9), %%xmm1;mulpd 224(%%r9), %%xmm2;addpd 240(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "mulpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;mulpd 288(%%r9), %%xmm2;addpd 304(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "mulpd 320(%%r9), %%xmm0;addpd 336(%%r9), %%xmm1;mulpd 352(%%r9), %%xmm2;addpd 368(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "mulpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;addpd 432(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "mulpd 448(%%r9), %%xmm0;addpd 464(%%r9), %%xmm1;mulpd 480(%%r9), %%xmm2;addpd 496(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm4;mulpd %%xmm1,%%xmm5;addpd %%xmm2,%%xmm6;mulpd %%xmm3,%%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_plus_add_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_plus_add_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_plus_add_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_mul_plus_add_pd_8;"   //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_mul_plus_add_pd_8:"       //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_mul_plus_add_pd_8;"   //|<
                "_sync1_mul_plus_add_pd_8:"       //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_mul_plus_add_pd_8;"   //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_mul_plus_add_pd_8;"    //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_mul_plus_add_pd_8;"   //|<
                "_wait_mul_plus_add_pd_8:"        //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_mul_plus_add_pd_8;"     //|<
                "_sync2_mul_plus_add_pd_8:"       //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_mul_plus_add_pd_8;"   //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                "movapd 128(%%r9), %%xmm8;"
                "movapd 144(%%r9), %%xmm9;"
                "movapd 160(%%r9), %%xmm10;"
                "movapd 176(%%r9), %%xmm11;"
                "movapd 192(%%r9), %%xmm12;"
                "movapd 208(%%r9), %%xmm13;"
                "movapd 224(%%r9), %%xmm14;"
                "movapd 240(%%r9), %%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_mul_plus_add_pd_8:"
                "mulpd (%%r9), %%xmm0;addpd 16(%%r9), %%xmm1;mulpd 32(%%r9), %%xmm2;addpd 48(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm8;mulpd %%xmm1,%%xmm9;addpd %%xmm2,%%xmm10;mulpd %%xmm3,%%xmm11;"
                "mulpd 64(%%r9), %%xmm4;addpd 80(%%r9), %%xmm5;mulpd 96(%%r9), %%xmm6;addpd 112(%%r9), %%xmm7;"
                "addpd %%xmm4,%%xmm12;mulpd %%xmm5,%%xmm13;addpd %%xmm6,%%xmm14;mulpd %%xmm7,%%xmm15;"
                "mulpd 128(%%r9), %%xmm0;addpd 144(%%r9), %%xmm1;mulpd 160(%%r9), %%xmm2;addpd 176(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm8;mulpd %%xmm1,%%xmm9;addpd %%xmm2,%%xmm10;mulpd %%xmm3,%%xmm11;"
                "mulpd 192(%%r9), %%xmm4;addpd 208(%%r9), %%xmm5;mulpd 224(%%r9), %%xmm6;addpd 240(%%r9), %%xmm7;"
                "addpd %%xmm4,%%xmm12;mulpd %%xmm5,%%xmm13;addpd %%xmm6,%%xmm14;mulpd %%xmm7,%%xmm15;"
                "mulpd 256(%%r9), %%xmm0;addpd 272(%%r9), %%xmm1;mulpd 288(%%r9), %%xmm2;addpd 304(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm8;mulpd %%xmm1,%%xmm9;addpd %%xmm2,%%xmm10;mulpd %%xmm3,%%xmm11;"
                "mulpd 320(%%r9), %%xmm4;addpd 336(%%r9), %%xmm5;mulpd 352(%%r9), %%xmm6;addpd 368(%%r9), %%xmm7;"
                "addpd %%xmm4,%%xmm12;mulpd %%xmm5,%%xmm13;addpd %%xmm6,%%xmm14;mulpd %%xmm7,%%xmm15;"
                "mulpd 384(%%r9), %%xmm0;addpd 400(%%r9), %%xmm1;mulpd 416(%%r9), %%xmm2;addpd 432(%%r9), %%xmm3;"
                "addpd %%xmm0,%%xmm8;mulpd %%xmm1,%%xmm9;addpd %%xmm2,%%xmm10;mulpd %%xmm3,%%xmm11;"
                "mulpd 448(%%r9), %%xmm4;addpd 464(%%r9), %%xmm5;mulpd 480(%%r9), %%xmm6;addpd 496(%%r9), %%xmm7;"
                "addpd %%xmm4,%%xmm12;mulpd %%xmm5,%%xmm13;addpd %%xmm6,%%xmm14;mulpd %%xmm7,%%xmm15;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_mul_plus_add_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_mul_plus_add_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_mul_plus_add_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
		);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_div_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_div_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_pd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_pd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_pd_1;"       //|<
                "_sync1_div_pd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_pd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_pd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_pd_1;"       //|<
                "_wait_div_pd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_pd_1;"         //|<
                "_sync2_div_pd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_pd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_pd_1:"
                "divpd (%%r9), %%xmm0;"
                "divpd 16(%%r9), %%xmm0;"
                "divpd 32(%%r9), %%xmm0;"
                "divpd 48(%%r9), %%xmm0;"
                "divpd 64(%%r9), %%xmm0;"
                "divpd 80(%%r9), %%xmm0;"
                "divpd 96(%%r9), %%xmm0;"
                "divpd 112(%%r9), %%xmm0;"
                "divpd 128(%%r9), %%xmm0;"
                "divpd 144(%%r9), %%xmm0;"
                "divpd 160(%%r9), %%xmm0;"
                "divpd 176(%%r9), %%xmm0;"
                "divpd 192(%%r9), %%xmm0;"
                "divpd 208(%%r9), %%xmm0;"
                "divpd 224(%%r9), %%xmm0;"
                "divpd 240(%%r9), %%xmm0;"
                "divpd 256(%%r9), %%xmm0;"
                "divpd 272(%%r9), %%xmm0;"
                "divpd 288(%%r9), %%xmm0;"
                "divpd 304(%%r9), %%xmm0;"
                "divpd 320(%%r9), %%xmm0;"
                "divpd 336(%%r9), %%xmm0;"
                "divpd 352(%%r9), %%xmm0;"
                "divpd 368(%%r9), %%xmm0;"
                "divpd 384(%%r9), %%xmm0;"
                "divpd 400(%%r9), %%xmm0;"
                "divpd 416(%%r9), %%xmm0;"
                "divpd 432(%%r9), %%xmm0;"
                "divpd 448(%%r9), %%xmm0;"
                "divpd 464(%%r9), %%xmm0;"
                "divpd 480(%%r9), %%xmm0;"
                "divpd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_pd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_pd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_pd_2;"       //|<
                "_sync1_div_pd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_pd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_pd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_pd_2;"       //|<
                "_wait_div_pd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_pd_2;"         //|<
                "_sync2_div_pd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_pd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_pd_2:"
                "divpd (%%r9), %%xmm0;divpd 16(%%r9), %%xmm1;"
                "divpd 32(%%r9), %%xmm0;divpd 48(%%r9), %%xmm1;"
                "divpd 64(%%r9), %%xmm0;divpd 80(%%r9), %%xmm1;"
                "divpd 96(%%r9), %%xmm0;divpd 112(%%r9), %%xmm1;"
                "divpd 128(%%r9), %%xmm0;divpd 144(%%r9), %%xmm1;"
                "divpd 160(%%r9), %%xmm0;divpd 176(%%r9), %%xmm1;"
                "divpd 192(%%r9), %%xmm0;divpd 208(%%r9), %%xmm1;"
                "divpd 224(%%r9), %%xmm0;divpd 240(%%r9), %%xmm1;"
                "divpd 256(%%r9), %%xmm0;divpd 272(%%r9), %%xmm1;"
                "divpd 288(%%r9), %%xmm0;divpd 304(%%r9), %%xmm1;"
                "divpd 320(%%r9), %%xmm0;divpd 336(%%r9), %%xmm1;"
                "divpd 352(%%r9), %%xmm0;divpd 368(%%r9), %%xmm1;"
                "divpd 384(%%r9), %%xmm0;divpd 400(%%r9), %%xmm1;"
                "divpd 416(%%r9), %%xmm0;divpd 432(%%r9), %%xmm1;"
                "divpd 448(%%r9), %%xmm0;divpd 464(%%r9), %%xmm1;"
                "divpd 480(%%r9), %%xmm0;divpd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_pd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_pd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_pd_3;"       //|<
                "_sync1_div_pd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_pd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_pd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_pd_3;"       //|<
                "_wait_div_pd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_pd_3;"         //|<
                "_sync2_div_pd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_pd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_pd_3:"
                "divpd (%%r9), %%xmm0;divpd 16(%%r9), %%xmm1;divpd 32(%%r9), %%xmm2;"
                "divpd 48(%%r9), %%xmm0;divpd 64(%%r9), %%xmm1;divpd 80(%%r9), %%xmm2;"
                "divpd 96(%%r9), %%xmm0;divpd 112(%%r9), %%xmm1;divpd 128(%%r9), %%xmm2;"
                "divpd 144(%%r9), %%xmm0;divpd 160(%%r9), %%xmm1;divpd 176(%%r9), %%xmm2;"
                "divpd 192(%%r9), %%xmm0;divpd 208(%%r9), %%xmm1;divpd 224(%%r9), %%xmm2;"
                "divpd 240(%%r9), %%xmm0;divpd 256(%%r9), %%xmm1;divpd 272(%%r9), %%xmm2;"
                "divpd 288(%%r9), %%xmm0;divpd 304(%%r9), %%xmm1;divpd 320(%%r9), %%xmm2;"
                "divpd 336(%%r9), %%xmm0;divpd 352(%%r9), %%xmm1;divpd 368(%%r9), %%xmm2;"
                "divpd 384(%%r9), %%xmm0;divpd 400(%%r9), %%xmm1;divpd 416(%%r9), %%xmm2;"
                "divpd 432(%%r9), %%xmm0;divpd 448(%%r9), %%xmm1;divpd 464(%%r9), %%xmm2;"
                "divpd 480(%%r9), %%xmm0;divpd 496(%%r9), %%xmm1;divpd 512(%%r9), %%xmm2;"
                "divpd 528(%%r9), %%xmm0;divpd 544(%%r9), %%xmm1;divpd 560(%%r9), %%xmm2;"
                "divpd 576(%%r9), %%xmm0;divpd 592(%%r9), %%xmm1;divpd 608(%%r9), %%xmm2;"
                "divpd 624(%%r9), %%xmm0;divpd 640(%%r9), %%xmm1;divpd 656(%%r9), %%xmm2;"
                "divpd 672(%%r9), %%xmm0;divpd 688(%%r9), %%xmm1;divpd 704(%%r9), %%xmm2;"
                "divpd 720(%%r9), %%xmm0;divpd 736(%%r9), %%xmm1;divpd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_div_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_pd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_pd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_pd_4;"       //|<
                "_sync1_div_pd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_pd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_pd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_pd_4;"       //|<
                "_wait_div_pd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_pd_4;"         //|<
                "_sync2_div_pd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_pd_4;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_pd_4:"
                "divpd (%%r9), %%xmm0;divpd 16(%%r9), %%xmm1;divpd 32(%%r9), %%xmm2;divpd 48(%%r9), %%xmm3;"
                "divpd 64(%%r9), %%xmm0;divpd 80(%%r9), %%xmm1;divpd 96(%%r9), %%xmm2;divpd 112(%%r9), %%xmm3;"
                "divpd 128(%%r9), %%xmm0;divpd 144(%%r9), %%xmm1;divpd 160(%%r9), %%xmm2;divpd 176(%%r9), %%xmm3;"
                "divpd 192(%%r9), %%xmm0;divpd 208(%%r9), %%xmm1;divpd 224(%%r9), %%xmm2;divpd 240(%%r9), %%xmm3;"
                "divpd 256(%%r9), %%xmm0;divpd 272(%%r9), %%xmm1;divpd 288(%%r9), %%xmm2;divpd 304(%%r9), %%xmm3;"
                "divpd 320(%%r9), %%xmm0;divpd 336(%%r9), %%xmm1;divpd 352(%%r9), %%xmm2;divpd 368(%%r9), %%xmm3;"
                "divpd 384(%%r9), %%xmm0;divpd 400(%%r9), %%xmm1;divpd 416(%%r9), %%xmm2;divpd 432(%%r9), %%xmm3;"
                "divpd 448(%%r9), %%xmm0;divpd 464(%%r9), %%xmm1;divpd 480(%%r9), %%xmm2;divpd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_pd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_pd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_pd_8;"       //|<
                "_sync1_div_pd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_pd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_pd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_pd_8;"       //|<
                "_wait_div_pd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_pd_8;"         //|<
                "_sync2_div_pd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_pd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_pd_8:"
                "divpd (%%r9), %%xmm0;divpd 16(%%r9), %%xmm1;divpd 32(%%r9), %%xmm2;divpd 48(%%r9), %%xmm3;"
                "divpd 64(%%r9), %%xmm4;divpd 80(%%r9), %%xmm5;divpd 96(%%r9), %%xmm6;divpd 112(%%r9), %%xmm7;"
                "divpd 128(%%r9), %%xmm0;divpd 144(%%r9), %%xmm1;divpd 160(%%r9), %%xmm2;divpd 176(%%r9), %%xmm3;"
                "divpd 192(%%r9), %%xmm4;divpd 208(%%r9), %%xmm5;divpd 224(%%r9), %%xmm6;divpd 240(%%r9), %%xmm7;"
                "divpd 256(%%r9), %%xmm0;divpd 272(%%r9), %%xmm1;divpd 288(%%r9), %%xmm2;divpd 304(%%r9), %%xmm3;"
                "divpd 320(%%r9), %%xmm4;divpd 336(%%r9), %%xmm5;divpd 352(%%r9), %%xmm6;divpd 368(%%r9), %%xmm7;"
                "divpd 384(%%r9), %%xmm0;divpd 400(%%r9), %%xmm1;divpd 416(%%r9), %%xmm2;divpd 432(%%r9), %%xmm3;"
                "divpd 448(%%r9), %%xmm4;divpd 464(%%r9), %%xmm5;divpd 480(%%r9), %%xmm6;divpd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_div_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_div_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ps_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ps_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ps_1;"       //|<
                "_sync1_div_ps_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ps_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ps_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ps_1;"       //|<
                "_wait_div_ps_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ps_1;"         //|<
                "_sync2_div_ps_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ps_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ps_1:"
                "divps (%%r9), %%xmm0;"
                "divps 16(%%r9), %%xmm0;"
                "divps 32(%%r9), %%xmm0;"
                "divps 48(%%r9), %%xmm0;"
                "divps 64(%%r9), %%xmm0;"
                "divps 80(%%r9), %%xmm0;"
                "divps 96(%%r9), %%xmm0;"
                "divps 112(%%r9), %%xmm0;"
                "divps 128(%%r9), %%xmm0;"
                "divps 144(%%r9), %%xmm0;"
                "divps 160(%%r9), %%xmm0;"
                "divps 176(%%r9), %%xmm0;"
                "divps 192(%%r9), %%xmm0;"
                "divps 208(%%r9), %%xmm0;"
                "divps 224(%%r9), %%xmm0;"
                "divps 240(%%r9), %%xmm0;"
                "divps 256(%%r9), %%xmm0;"
                "divps 272(%%r9), %%xmm0;"
                "divps 288(%%r9), %%xmm0;"
                "divps 304(%%r9), %%xmm0;"
                "divps 320(%%r9), %%xmm0;"
                "divps 336(%%r9), %%xmm0;"
                "divps 352(%%r9), %%xmm0;"
                "divps 368(%%r9), %%xmm0;"
                "divps 384(%%r9), %%xmm0;"
                "divps 400(%%r9), %%xmm0;"
                "divps 416(%%r9), %%xmm0;"
                "divps 432(%%r9), %%xmm0;"
                "divps 448(%%r9), %%xmm0;"
                "divps 464(%%r9), %%xmm0;"
                "divps 480(%%r9), %%xmm0;"
                "divps 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ps_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ps_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ps_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ps_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ps_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ps_2;"       //|<
                "_sync1_div_ps_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ps_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ps_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ps_2;"       //|<
                "_wait_div_ps_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ps_2;"         //|<
                "_sync2_div_ps_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ps_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ps_2:"
                "divps (%%r9), %%xmm0;divps 16(%%r9), %%xmm1;"
                "divps 32(%%r9), %%xmm0;divps 48(%%r9), %%xmm1;"
                "divps 64(%%r9), %%xmm0;divps 80(%%r9), %%xmm1;"
                "divps 96(%%r9), %%xmm0;divps 112(%%r9), %%xmm1;"
                "divps 128(%%r9), %%xmm0;divps 144(%%r9), %%xmm1;"
                "divps 160(%%r9), %%xmm0;divps 176(%%r9), %%xmm1;"
                "divps 192(%%r9), %%xmm0;divps 208(%%r9), %%xmm1;"
                "divps 224(%%r9), %%xmm0;divps 240(%%r9), %%xmm1;"
                "divps 256(%%r9), %%xmm0;divps 272(%%r9), %%xmm1;"
                "divps 288(%%r9), %%xmm0;divps 304(%%r9), %%xmm1;"
                "divps 320(%%r9), %%xmm0;divps 336(%%r9), %%xmm1;"
                "divps 352(%%r9), %%xmm0;divps 368(%%r9), %%xmm1;"
                "divps 384(%%r9), %%xmm0;divps 400(%%r9), %%xmm1;"
                "divps 416(%%r9), %%xmm0;divps 432(%%r9), %%xmm1;"
                "divps 448(%%r9), %%xmm0;divps 464(%%r9), %%xmm1;"
                "divps 480(%%r9), %%xmm0;divps 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ps_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ps_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ps_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ps_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ps_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ps_3;"       //|<
                "_sync1_div_ps_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ps_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ps_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ps_3;"       //|<
                "_wait_div_ps_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ps_3;"         //|<
                "_sync2_div_ps_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ps_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ps_3:"
                "divps (%%r9), %%xmm0;divps 16(%%r9), %%xmm1;divps 32(%%r9), %%xmm2;"
                "divps 48(%%r9), %%xmm0;divps 64(%%r9), %%xmm1;divps 80(%%r9), %%xmm2;"
                "divps 96(%%r9), %%xmm0;divps 112(%%r9), %%xmm1;divps 128(%%r9), %%xmm2;"
                "divps 144(%%r9), %%xmm0;divps 160(%%r9), %%xmm1;divps 176(%%r9), %%xmm2;"
                "divps 192(%%r9), %%xmm0;divps 208(%%r9), %%xmm1;divps 224(%%r9), %%xmm2;"
                "divps 240(%%r9), %%xmm0;divps 256(%%r9), %%xmm1;divps 272(%%r9), %%xmm2;"
                "divps 288(%%r9), %%xmm0;divps 304(%%r9), %%xmm1;divps 320(%%r9), %%xmm2;"
                "divps 336(%%r9), %%xmm0;divps 352(%%r9), %%xmm1;divps 368(%%r9), %%xmm2;"
                "divps 384(%%r9), %%xmm0;divps 400(%%r9), %%xmm1;divps 416(%%r9), %%xmm2;"
                "divps 432(%%r9), %%xmm0;divps 448(%%r9), %%xmm1;divps 464(%%r9), %%xmm2;"
                "divps 480(%%r9), %%xmm0;divps 496(%%r9), %%xmm1;divps 512(%%r9), %%xmm2;"
                "divps 528(%%r9), %%xmm0;divps 544(%%r9), %%xmm1;divps 560(%%r9), %%xmm2;"
                "divps 576(%%r9), %%xmm0;divps 592(%%r9), %%xmm1;divps 608(%%r9), %%xmm2;"
                "divps 624(%%r9), %%xmm0;divps 640(%%r9), %%xmm1;divps 656(%%r9), %%xmm2;"
                "divps 672(%%r9), %%xmm0;divps 688(%%r9), %%xmm1;divps 704(%%r9), %%xmm2;"
                "divps 720(%%r9), %%xmm0;divps 736(%%r9), %%xmm1;divps 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ps_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ps_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_div_ps_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ps_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ps_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ps_4;"       //|<
                "_sync1_div_ps_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ps_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ps_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ps_4;"       //|<
                "_wait_div_ps_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ps_4;"         //|<
                "_sync2_div_ps_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ps_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ps_4:"
                "divps (%%r9), %%xmm0;divps 16(%%r9), %%xmm1;divps 32(%%r9), %%xmm2;divps 48(%%r9), %%xmm3;"
                "divps 64(%%r9), %%xmm0;divps 80(%%r9), %%xmm1;divps 96(%%r9), %%xmm2;divps 112(%%r9), %%xmm3;"
                "divps 128(%%r9), %%xmm0;divps 144(%%r9), %%xmm1;divps 160(%%r9), %%xmm2;divps 176(%%r9), %%xmm3;"
                "divps 192(%%r9), %%xmm0;divps 208(%%r9), %%xmm1;divps 224(%%r9), %%xmm2;divps 240(%%r9), %%xmm3;"
                "divps 256(%%r9), %%xmm0;divps 272(%%r9), %%xmm1;divps 288(%%r9), %%xmm2;divps 304(%%r9), %%xmm3;"
                "divps 320(%%r9), %%xmm0;divps 336(%%r9), %%xmm1;divps 352(%%r9), %%xmm2;divps 368(%%r9), %%xmm3;"
                "divps 384(%%r9), %%xmm0;divps 400(%%r9), %%xmm1;divps 416(%%r9), %%xmm2;divps 432(%%r9), %%xmm3;"
                "divps 448(%%r9), %%xmm0;divps 464(%%r9), %%xmm1;divps 480(%%r9), %%xmm2;divps 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ps_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ps_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ps_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ps_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ps_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ps_8;"       //|<
                "_sync1_div_ps_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ps_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ps_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ps_8;"       //|<
                "_wait_div_ps_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ps_8;"         //|<
                "_sync2_div_ps_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ps_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ps_8:"
                "divps (%%r9), %%xmm0;divps 16(%%r9), %%xmm1;divps 32(%%r9), %%xmm2;divps 48(%%r9), %%xmm3;"
                "divps 64(%%r9), %%xmm4;divps 80(%%r9), %%xmm5;divps 96(%%r9), %%xmm6;divps 112(%%r9), %%xmm7;"
                "divps 128(%%r9), %%xmm0;divps 144(%%r9), %%xmm1;divps 160(%%r9), %%xmm2;divps 176(%%r9), %%xmm3;"
                "divps 192(%%r9), %%xmm4;divps 208(%%r9), %%xmm5;divps 224(%%r9), %%xmm6;divps 240(%%r9), %%xmm7;"
                "divps 256(%%r9), %%xmm0;divps 272(%%r9), %%xmm1;divps 288(%%r9), %%xmm2;divps 304(%%r9), %%xmm3;"
                "divps 320(%%r9), %%xmm4;divps 336(%%r9), %%xmm5;divps 352(%%r9), %%xmm6;divps 368(%%r9), %%xmm7;"
                "divps 384(%%r9), %%xmm0;divps 400(%%r9), %%xmm1;divps 416(%%r9), %%xmm2;divps 432(%%r9), %%xmm3;"
                "divps 448(%%r9), %%xmm4;divps 464(%%r9), %%xmm5;divps 480(%%r9), %%xmm6;divps 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ps_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ps_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ps_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_div_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_div_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_sd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_sd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_sd_1;"       //|<
                "_sync1_div_sd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_sd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_sd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_sd_1;"       //|<
                "_wait_div_sd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_sd_1;"         //|<
                "_sync2_div_sd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_sd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_sd_1:"
                "divsd (%%r9), %%xmm0;"
                "divsd 16(%%r9), %%xmm0;"
                "divsd 32(%%r9), %%xmm0;"
                "divsd 48(%%r9), %%xmm0;"
                "divsd 64(%%r9), %%xmm0;"
                "divsd 80(%%r9), %%xmm0;"
                "divsd 96(%%r9), %%xmm0;"
                "divsd 112(%%r9), %%xmm0;"
                "divsd 128(%%r9), %%xmm0;"
                "divsd 144(%%r9), %%xmm0;"
                "divsd 160(%%r9), %%xmm0;"
                "divsd 176(%%r9), %%xmm0;"
                "divsd 192(%%r9), %%xmm0;"
                "divsd 208(%%r9), %%xmm0;"
                "divsd 224(%%r9), %%xmm0;"
                "divsd 240(%%r9), %%xmm0;"
                "divsd 256(%%r9), %%xmm0;"
                "divsd 272(%%r9), %%xmm0;"
                "divsd 288(%%r9), %%xmm0;"
                "divsd 304(%%r9), %%xmm0;"
                "divsd 320(%%r9), %%xmm0;"
                "divsd 336(%%r9), %%xmm0;"
                "divsd 352(%%r9), %%xmm0;"
                "divsd 368(%%r9), %%xmm0;"
                "divsd 384(%%r9), %%xmm0;"
                "divsd 400(%%r9), %%xmm0;"
                "divsd 416(%%r9), %%xmm0;"
                "divsd 432(%%r9), %%xmm0;"
                "divsd 448(%%r9), %%xmm0;"
                "divsd 464(%%r9), %%xmm0;"
                "divsd 480(%%r9), %%xmm0;"
                "divsd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_sd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_sd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_sd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_sd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_sd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_sd_2;"       //|<
                "_sync1_div_sd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_sd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_sd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_sd_2;"       //|<
                "_wait_div_sd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_sd_2;"         //|<
                "_sync2_div_sd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_sd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_sd_2:"
                "divsd (%%r9), %%xmm0;divsd 16(%%r9), %%xmm1;"
                "divsd 32(%%r9), %%xmm0;divsd 48(%%r9), %%xmm1;"
                "divsd 64(%%r9), %%xmm0;divsd 80(%%r9), %%xmm1;"
                "divsd 96(%%r9), %%xmm0;divsd 112(%%r9), %%xmm1;"
                "divsd 128(%%r9), %%xmm0;divsd 144(%%r9), %%xmm1;"
                "divsd 160(%%r9), %%xmm0;divsd 176(%%r9), %%xmm1;"
                "divsd 192(%%r9), %%xmm0;divsd 208(%%r9), %%xmm1;"
                "divsd 224(%%r9), %%xmm0;divsd 240(%%r9), %%xmm1;"
                "divsd 256(%%r9), %%xmm0;divsd 272(%%r9), %%xmm1;"
                "divsd 288(%%r9), %%xmm0;divsd 304(%%r9), %%xmm1;"
                "divsd 320(%%r9), %%xmm0;divsd 336(%%r9), %%xmm1;"
                "divsd 352(%%r9), %%xmm0;divsd 368(%%r9), %%xmm1;"
                "divsd 384(%%r9), %%xmm0;divsd 400(%%r9), %%xmm1;"
                "divsd 416(%%r9), %%xmm0;divsd 432(%%r9), %%xmm1;"
                "divsd 448(%%r9), %%xmm0;divsd 464(%%r9), %%xmm1;"
                "divsd 480(%%r9), %%xmm0;divsd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_sd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_sd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_sd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_sd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_sd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_sd_3;"       //|<
                "_sync1_div_sd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_sd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_sd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_sd_3;"       //|<
                "_wait_div_sd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_sd_3;"         //|<
                "_sync2_div_sd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_sd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_sd_3:"
                "divsd (%%r9), %%xmm0;divsd 16(%%r9), %%xmm1;divsd 32(%%r9), %%xmm2;"
                "divsd 48(%%r9), %%xmm0;divsd 64(%%r9), %%xmm1;divsd 80(%%r9), %%xmm2;"
                "divsd 96(%%r9), %%xmm0;divsd 112(%%r9), %%xmm1;divsd 128(%%r9), %%xmm2;"
                "divsd 144(%%r9), %%xmm0;divsd 160(%%r9), %%xmm1;divsd 176(%%r9), %%xmm2;"
                "divsd 192(%%r9), %%xmm0;divsd 208(%%r9), %%xmm1;divsd 224(%%r9), %%xmm2;"
                "divsd 240(%%r9), %%xmm0;divsd 256(%%r9), %%xmm1;divsd 272(%%r9), %%xmm2;"
                "divsd 288(%%r9), %%xmm0;divsd 304(%%r9), %%xmm1;divsd 320(%%r9), %%xmm2;"
                "divsd 336(%%r9), %%xmm0;divsd 352(%%r9), %%xmm1;divsd 368(%%r9), %%xmm2;"
                "divsd 384(%%r9), %%xmm0;divsd 400(%%r9), %%xmm1;divsd 416(%%r9), %%xmm2;"
                "divsd 432(%%r9), %%xmm0;divsd 448(%%r9), %%xmm1;divsd 464(%%r9), %%xmm2;"
                "divsd 480(%%r9), %%xmm0;divsd 496(%%r9), %%xmm1;divsd 512(%%r9), %%xmm2;"
                "divsd 528(%%r9), %%xmm0;divsd 544(%%r9), %%xmm1;divsd 560(%%r9), %%xmm2;"
                "divsd 576(%%r9), %%xmm0;divsd 592(%%r9), %%xmm1;divsd 608(%%r9), %%xmm2;"
                "divsd 624(%%r9), %%xmm0;divsd 640(%%r9), %%xmm1;divsd 656(%%r9), %%xmm2;"
                "divsd 672(%%r9), %%xmm0;divsd 688(%%r9), %%xmm1;divsd 704(%%r9), %%xmm2;"
                "divsd 720(%%r9), %%xmm0;divsd 736(%%r9), %%xmm1;divsd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_sd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_sd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_div_sd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_sd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_sd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_sd_4;"       //|<
                "_sync1_div_sd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_sd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_sd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_sd_4;"       //|<
                "_wait_div_sd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_sd_4;"         //|<
                "_sync2_div_sd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_sd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_sd_4:"
                "divsd (%%r9), %%xmm0;divsd 16(%%r9), %%xmm1;divsd 32(%%r9), %%xmm2;divsd 48(%%r9), %%xmm3;"
                "divsd 64(%%r9), %%xmm0;divsd 80(%%r9), %%xmm1;divsd 96(%%r9), %%xmm2;divsd 112(%%r9), %%xmm3;"
                "divsd 128(%%r9), %%xmm0;divsd 144(%%r9), %%xmm1;divsd 160(%%r9), %%xmm2;divsd 176(%%r9), %%xmm3;"
                "divsd 192(%%r9), %%xmm0;divsd 208(%%r9), %%xmm1;divsd 224(%%r9), %%xmm2;divsd 240(%%r9), %%xmm3;"
                "divsd 256(%%r9), %%xmm0;divsd 272(%%r9), %%xmm1;divsd 288(%%r9), %%xmm2;divsd 304(%%r9), %%xmm3;"
                "divsd 320(%%r9), %%xmm0;divsd 336(%%r9), %%xmm1;divsd 352(%%r9), %%xmm2;divsd 368(%%r9), %%xmm3;"
                "divsd 384(%%r9), %%xmm0;divsd 400(%%r9), %%xmm1;divsd 416(%%r9), %%xmm2;divsd 432(%%r9), %%xmm3;"
                "divsd 448(%%r9), %%xmm0;divsd 464(%%r9), %%xmm1;divsd 480(%%r9), %%xmm2;divsd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_sd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_sd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_sd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_sd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_sd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_sd_8;"       //|<
                "_sync1_div_sd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_sd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_sd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_sd_8;"       //|<
                "_wait_div_sd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_sd_8;"         //|<
                "_sync2_div_sd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_sd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_sd_8:"
                "divsd (%%r9), %%xmm0;divsd 16(%%r9), %%xmm1;divsd 32(%%r9), %%xmm2;divsd 48(%%r9), %%xmm3;"
                "divsd 64(%%r9), %%xmm4;divsd 80(%%r9), %%xmm5;divsd 96(%%r9), %%xmm6;divsd 112(%%r9), %%xmm7;"
                "divsd 128(%%r9), %%xmm0;divsd 144(%%r9), %%xmm1;divsd 160(%%r9), %%xmm2;divsd 176(%%r9), %%xmm3;"
                "divsd 192(%%r9), %%xmm4;divsd 208(%%r9), %%xmm5;divsd 224(%%r9), %%xmm6;divsd 240(%%r9), %%xmm7;"
                "divsd 256(%%r9), %%xmm0;divsd 272(%%r9), %%xmm1;divsd 288(%%r9), %%xmm2;divsd 304(%%r9), %%xmm3;"
                "divsd 320(%%r9), %%xmm4;divsd 336(%%r9), %%xmm5;divsd 352(%%r9), %%xmm6;divsd 368(%%r9), %%xmm7;"
                "divsd 384(%%r9), %%xmm0;divsd 400(%%r9), %%xmm1;divsd 416(%%r9), %%xmm2;divsd 432(%%r9), %%xmm3;"
                "divsd 448(%%r9), %%xmm4;divsd 464(%%r9), %%xmm5;divsd 480(%%r9), %%xmm6;divsd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_sd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_sd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_sd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_div_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_div_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ss_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ss_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ss_1;"       //|<
                "_sync1_div_ss_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ss_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ss_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ss_1;"       //|<
                "_wait_div_ss_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ss_1;"         //|<
                "_sync2_div_ss_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ss_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ss_1:"
                "divss (%%r9), %%xmm0;"
                "divss 16(%%r9), %%xmm0;"
                "divss 32(%%r9), %%xmm0;"
                "divss 48(%%r9), %%xmm0;"
                "divss 64(%%r9), %%xmm0;"
                "divss 80(%%r9), %%xmm0;"
                "divss 96(%%r9), %%xmm0;"
                "divss 112(%%r9), %%xmm0;"
                "divss 128(%%r9), %%xmm0;"
                "divss 144(%%r9), %%xmm0;"
                "divss 160(%%r9), %%xmm0;"
                "divss 176(%%r9), %%xmm0;"
                "divss 192(%%r9), %%xmm0;"
                "divss 208(%%r9), %%xmm0;"
                "divss 224(%%r9), %%xmm0;"
                "divss 240(%%r9), %%xmm0;"
                "divss 256(%%r9), %%xmm0;"
                "divss 272(%%r9), %%xmm0;"
                "divss 288(%%r9), %%xmm0;"
                "divss 304(%%r9), %%xmm0;"
                "divss 320(%%r9), %%xmm0;"
                "divss 336(%%r9), %%xmm0;"
                "divss 352(%%r9), %%xmm0;"
                "divss 368(%%r9), %%xmm0;"
                "divss 384(%%r9), %%xmm0;"
                "divss 400(%%r9), %%xmm0;"
                "divss 416(%%r9), %%xmm0;"
                "divss 432(%%r9), %%xmm0;"
                "divss 448(%%r9), %%xmm0;"
                "divss 464(%%r9), %%xmm0;"
                "divss 480(%%r9), %%xmm0;"
                "divss 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ss_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ss_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ss_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ss_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ss_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ss_2;"       //|<
                "_sync1_div_ss_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ss_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ss_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ss_2;"       //|<
                "_wait_div_ss_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ss_2;"         //|<
                "_sync2_div_ss_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ss_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ss_2:"
                "divss (%%r9), %%xmm0;divss 16(%%r9), %%xmm1;"
                "divss 32(%%r9), %%xmm0;divss 48(%%r9), %%xmm1;"
                "divss 64(%%r9), %%xmm0;divss 80(%%r9), %%xmm1;"
                "divss 96(%%r9), %%xmm0;divss 112(%%r9), %%xmm1;"
                "divss 128(%%r9), %%xmm0;divss 144(%%r9), %%xmm1;"
                "divss 160(%%r9), %%xmm0;divss 176(%%r9), %%xmm1;"
                "divss 192(%%r9), %%xmm0;divss 208(%%r9), %%xmm1;"
                "divss 224(%%r9), %%xmm0;divss 240(%%r9), %%xmm1;"
                "divss 256(%%r9), %%xmm0;divss 272(%%r9), %%xmm1;"
                "divss 288(%%r9), %%xmm0;divss 304(%%r9), %%xmm1;"
                "divss 320(%%r9), %%xmm0;divss 336(%%r9), %%xmm1;"
                "divss 352(%%r9), %%xmm0;divss 368(%%r9), %%xmm1;"
                "divss 384(%%r9), %%xmm0;divss 400(%%r9), %%xmm1;"
                "divss 416(%%r9), %%xmm0;divss 432(%%r9), %%xmm1;"
                "divss 448(%%r9), %%xmm0;divss 464(%%r9), %%xmm1;"
                "divss 480(%%r9), %%xmm0;divss 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ss_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ss_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ss_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ss_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ss_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ss_3;"       //|<
                "_sync1_div_ss_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ss_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ss_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ss_3;"       //|<
                "_wait_div_ss_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ss_3;"         //|<
                "_sync2_div_ss_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ss_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ss_3:"
                "divss (%%r9), %%xmm0;divss 16(%%r9), %%xmm1;divss 32(%%r9), %%xmm2;"
                "divss 48(%%r9), %%xmm0;divss 64(%%r9), %%xmm1;divss 80(%%r9), %%xmm2;"
                "divss 96(%%r9), %%xmm0;divss 112(%%r9), %%xmm1;divss 128(%%r9), %%xmm2;"
                "divss 144(%%r9), %%xmm0;divss 160(%%r9), %%xmm1;divss 176(%%r9), %%xmm2;"
                "divss 192(%%r9), %%xmm0;divss 208(%%r9), %%xmm1;divss 224(%%r9), %%xmm2;"
                "divss 240(%%r9), %%xmm0;divss 256(%%r9), %%xmm1;divss 272(%%r9), %%xmm2;"
                "divss 288(%%r9), %%xmm0;divss 304(%%r9), %%xmm1;divss 320(%%r9), %%xmm2;"
                "divss 336(%%r9), %%xmm0;divss 352(%%r9), %%xmm1;divss 368(%%r9), %%xmm2;"
                "divss 384(%%r9), %%xmm0;divss 400(%%r9), %%xmm1;divss 416(%%r9), %%xmm2;"
                "divss 432(%%r9), %%xmm0;divss 448(%%r9), %%xmm1;divss 464(%%r9), %%xmm2;"
                "divss 480(%%r9), %%xmm0;divss 496(%%r9), %%xmm1;divss 512(%%r9), %%xmm2;"
                "divss 528(%%r9), %%xmm0;divss 544(%%r9), %%xmm1;divss 560(%%r9), %%xmm2;"
                "divss 576(%%r9), %%xmm0;divss 592(%%r9), %%xmm1;divss 608(%%r9), %%xmm2;"
                "divss 624(%%r9), %%xmm0;divss 640(%%r9), %%xmm1;divss 656(%%r9), %%xmm2;"
                "divss 672(%%r9), %%xmm0;divss 688(%%r9), %%xmm1;divss 704(%%r9), %%xmm2;"
                "divss 720(%%r9), %%xmm0;divss 736(%%r9), %%xmm1;divss 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ss_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ss_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_div_ss_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ss_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ss_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ss_4;"       //|<
                "_sync1_div_ss_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ss_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ss_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ss_4;"       //|<
                "_wait_div_ss_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ss_4;"         //|<
                "_sync2_div_ss_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ss_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ss_4:"
                "divss (%%r9), %%xmm0;divss 16(%%r9), %%xmm1;divss 32(%%r9), %%xmm2;divss 48(%%r9), %%xmm3;"
                "divss 64(%%r9), %%xmm0;divss 80(%%r9), %%xmm1;divss 96(%%r9), %%xmm2;divss 112(%%r9), %%xmm3;"
                "divss 128(%%r9), %%xmm0;divss 144(%%r9), %%xmm1;divss 160(%%r9), %%xmm2;divss 176(%%r9), %%xmm3;"
                "divss 192(%%r9), %%xmm0;divss 208(%%r9), %%xmm1;divss 224(%%r9), %%xmm2;divss 240(%%r9), %%xmm3;"
                "divss 256(%%r9), %%xmm0;divss 272(%%r9), %%xmm1;divss 288(%%r9), %%xmm2;divss 304(%%r9), %%xmm3;"
                "divss 320(%%r9), %%xmm0;divss 336(%%r9), %%xmm1;divss 352(%%r9), %%xmm2;divss 368(%%r9), %%xmm3;"
                "divss 384(%%r9), %%xmm0;divss 400(%%r9), %%xmm1;divss 416(%%r9), %%xmm2;divss 432(%%r9), %%xmm3;"
                "divss 448(%%r9), %%xmm0;divss 464(%%r9), %%xmm1;divss 480(%%r9), %%xmm2;divss 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ss_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ss_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ss_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_div_ss_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_div_ss_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_div_ss_8;"       //|<
                "_sync1_div_ss_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_div_ss_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_div_ss_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_div_ss_8;"       //|<
                "_wait_div_ss_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_div_ss_8;"         //|<
                "_sync2_div_ss_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_div_ss_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_div_ss_8:"
                "divss (%%r9), %%xmm0;divss 16(%%r9), %%xmm1;divss 32(%%r9), %%xmm2;divss 48(%%r9), %%xmm3;"
                "divss 64(%%r9), %%xmm4;divss 80(%%r9), %%xmm5;divss 96(%%r9), %%xmm6;divss 112(%%r9), %%xmm7;"
                "divss 128(%%r9), %%xmm0;divss 144(%%r9), %%xmm1;divss 160(%%r9), %%xmm2;divss 176(%%r9), %%xmm3;"
                "divss 192(%%r9), %%xmm4;divss 208(%%r9), %%xmm5;divss 224(%%r9), %%xmm6;divss 240(%%r9), %%xmm7;"
                "divss 256(%%r9), %%xmm0;divss 272(%%r9), %%xmm1;divss 288(%%r9), %%xmm2;divss 304(%%r9), %%xmm3;"
                "divss 320(%%r9), %%xmm4;divss 336(%%r9), %%xmm5;divss 352(%%r9), %%xmm6;divss 368(%%r9), %%xmm7;"
                "divss 384(%%r9), %%xmm0;divss 400(%%r9), %%xmm1;divss 416(%%r9), %%xmm2;divss 432(%%r9), %%xmm3;"
                "divss 448(%%r9), %%xmm4;divss 464(%%r9), %%xmm5;divss 480(%%r9), %%xmm6;divss 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_div_ss_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_div_ss_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_div_ss_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_sqrt_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_sqrt_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_pd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_pd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_pd_1;"       //|<
                "_sync1_sqrt_pd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_pd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_pd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_pd_1;"       //|<
                "_wait_sqrt_pd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_pd_1;"         //|<
                "_sync2_sqrt_pd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_pd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_pd_1:"
                "sqrtpd (%%r9), %%xmm0;"
                "sqrtpd 16(%%r9), %%xmm0;"
                "sqrtpd 32(%%r9), %%xmm0;"
                "sqrtpd 48(%%r9), %%xmm0;"
                "sqrtpd 64(%%r9), %%xmm0;"
                "sqrtpd 80(%%r9), %%xmm0;"
                "sqrtpd 96(%%r9), %%xmm0;"
                "sqrtpd 112(%%r9), %%xmm0;"
                "sqrtpd 128(%%r9), %%xmm0;"
                "sqrtpd 144(%%r9), %%xmm0;"
                "sqrtpd 160(%%r9), %%xmm0;"
                "sqrtpd 176(%%r9), %%xmm0;"
                "sqrtpd 192(%%r9), %%xmm0;"
                "sqrtpd 208(%%r9), %%xmm0;"
                "sqrtpd 224(%%r9), %%xmm0;"
                "sqrtpd 240(%%r9), %%xmm0;"
                "sqrtpd 256(%%r9), %%xmm0;"
                "sqrtpd 272(%%r9), %%xmm0;"
                "sqrtpd 288(%%r9), %%xmm0;"
                "sqrtpd 304(%%r9), %%xmm0;"
                "sqrtpd 320(%%r9), %%xmm0;"
                "sqrtpd 336(%%r9), %%xmm0;"
                "sqrtpd 352(%%r9), %%xmm0;"
                "sqrtpd 368(%%r9), %%xmm0;"
                "sqrtpd 384(%%r9), %%xmm0;"
                "sqrtpd 400(%%r9), %%xmm0;"
                "sqrtpd 416(%%r9), %%xmm0;"
                "sqrtpd 432(%%r9), %%xmm0;"
                "sqrtpd 448(%%r9), %%xmm0;"
                "sqrtpd 464(%%r9), %%xmm0;"
                "sqrtpd 480(%%r9), %%xmm0;"
                "sqrtpd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_pd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_pd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_pd_2;"       //|<
                "_sync1_sqrt_pd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_pd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_pd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_pd_2;"       //|<
                "_wait_sqrt_pd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_pd_2;"         //|<
                "_sync2_sqrt_pd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_pd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_pd_2:"
                "sqrtpd (%%r9), %%xmm0;sqrtpd 16(%%r9), %%xmm1;"
                "sqrtpd 32(%%r9), %%xmm0;sqrtpd 48(%%r9), %%xmm1;"
                "sqrtpd 64(%%r9), %%xmm0;sqrtpd 80(%%r9), %%xmm1;"
                "sqrtpd 96(%%r9), %%xmm0;sqrtpd 112(%%r9), %%xmm1;"
                "sqrtpd 128(%%r9), %%xmm0;sqrtpd 144(%%r9), %%xmm1;"
                "sqrtpd 160(%%r9), %%xmm0;sqrtpd 176(%%r9), %%xmm1;"
                "sqrtpd 192(%%r9), %%xmm0;sqrtpd 208(%%r9), %%xmm1;"
                "sqrtpd 224(%%r9), %%xmm0;sqrtpd 240(%%r9), %%xmm1;"
                "sqrtpd 256(%%r9), %%xmm0;sqrtpd 272(%%r9), %%xmm1;"
                "sqrtpd 288(%%r9), %%xmm0;sqrtpd 304(%%r9), %%xmm1;"
                "sqrtpd 320(%%r9), %%xmm0;sqrtpd 336(%%r9), %%xmm1;"
                "sqrtpd 352(%%r9), %%xmm0;sqrtpd 368(%%r9), %%xmm1;"
                "sqrtpd 384(%%r9), %%xmm0;sqrtpd 400(%%r9), %%xmm1;"
                "sqrtpd 416(%%r9), %%xmm0;sqrtpd 432(%%r9), %%xmm1;"
                "sqrtpd 448(%%r9), %%xmm0;sqrtpd 464(%%r9), %%xmm1;"
                "sqrtpd 480(%%r9), %%xmm0;sqrtpd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_pd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_pd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_pd_3;"       //|<
                "_sync1_sqrt_pd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_pd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_pd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_pd_3;"       //|<
                "_wait_sqrt_pd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_pd_3;"         //|<
                "_sync2_sqrt_pd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_pd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_pd_3:"
                "sqrtpd (%%r9), %%xmm0;sqrtpd 16(%%r9), %%xmm1;sqrtpd 32(%%r9), %%xmm2;"
                "sqrtpd 48(%%r9), %%xmm0;sqrtpd 64(%%r9), %%xmm1;sqrtpd 80(%%r9), %%xmm2;"
                "sqrtpd 96(%%r9), %%xmm0;sqrtpd 112(%%r9), %%xmm1;sqrtpd 128(%%r9), %%xmm2;"
                "sqrtpd 144(%%r9), %%xmm0;sqrtpd 160(%%r9), %%xmm1;sqrtpd 176(%%r9), %%xmm2;"
                "sqrtpd 192(%%r9), %%xmm0;sqrtpd 208(%%r9), %%xmm1;sqrtpd 224(%%r9), %%xmm2;"
                "sqrtpd 240(%%r9), %%xmm0;sqrtpd 256(%%r9), %%xmm1;sqrtpd 272(%%r9), %%xmm2;"
                "sqrtpd 288(%%r9), %%xmm0;sqrtpd 304(%%r9), %%xmm1;sqrtpd 320(%%r9), %%xmm2;"
                "sqrtpd 336(%%r9), %%xmm0;sqrtpd 352(%%r9), %%xmm1;sqrtpd 368(%%r9), %%xmm2;"
                "sqrtpd 384(%%r9), %%xmm0;sqrtpd 400(%%r9), %%xmm1;sqrtpd 416(%%r9), %%xmm2;"
                "sqrtpd 432(%%r9), %%xmm0;sqrtpd 448(%%r9), %%xmm1;sqrtpd 464(%%r9), %%xmm2;"
                "sqrtpd 480(%%r9), %%xmm0;sqrtpd 496(%%r9), %%xmm1;sqrtpd 512(%%r9), %%xmm2;"
                "sqrtpd 528(%%r9), %%xmm0;sqrtpd 544(%%r9), %%xmm1;sqrtpd 560(%%r9), %%xmm2;"
                "sqrtpd 576(%%r9), %%xmm0;sqrtpd 592(%%r9), %%xmm1;sqrtpd 608(%%r9), %%xmm2;"
                "sqrtpd 624(%%r9), %%xmm0;sqrtpd 640(%%r9), %%xmm1;sqrtpd 656(%%r9), %%xmm2;"
                "sqrtpd 672(%%r9), %%xmm0;sqrtpd 688(%%r9), %%xmm1;sqrtpd 704(%%r9), %%xmm2;"
                "sqrtpd 720(%%r9), %%xmm0;sqrtpd 736(%%r9), %%xmm1;sqrtpd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_sqrt_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_pd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_pd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_pd_4;"       //|<
                "_sync1_sqrt_pd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_pd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_pd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_pd_4;"       //|<
                "_wait_sqrt_pd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_pd_4;"         //|<
                "_sync2_sqrt_pd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_pd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_pd_4:"
                "sqrtpd (%%r9), %%xmm0;sqrtpd 16(%%r9), %%xmm1;sqrtpd 32(%%r9), %%xmm2;sqrtpd 48(%%r9), %%xmm3;"
                "sqrtpd 64(%%r9), %%xmm0;sqrtpd 80(%%r9), %%xmm1;sqrtpd 96(%%r9), %%xmm2;sqrtpd 112(%%r9), %%xmm3;"
                "sqrtpd 128(%%r9), %%xmm0;sqrtpd 144(%%r9), %%xmm1;sqrtpd 160(%%r9), %%xmm2;sqrtpd 176(%%r9), %%xmm3;"
                "sqrtpd 192(%%r9), %%xmm0;sqrtpd 208(%%r9), %%xmm1;sqrtpd 224(%%r9), %%xmm2;sqrtpd 240(%%r9), %%xmm3;"
                "sqrtpd 256(%%r9), %%xmm0;sqrtpd 272(%%r9), %%xmm1;sqrtpd 288(%%r9), %%xmm2;sqrtpd 304(%%r9), %%xmm3;"
                "sqrtpd 320(%%r9), %%xmm0;sqrtpd 336(%%r9), %%xmm1;sqrtpd 352(%%r9), %%xmm2;sqrtpd 368(%%r9), %%xmm3;"
                "sqrtpd 384(%%r9), %%xmm0;sqrtpd 400(%%r9), %%xmm1;sqrtpd 416(%%r9), %%xmm2;sqrtpd 432(%%r9), %%xmm3;"
                "sqrtpd 448(%%r9), %%xmm0;sqrtpd 464(%%r9), %%xmm1;sqrtpd 480(%%r9), %%xmm2;sqrtpd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_pd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_pd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_pd_8;"       //|<
                "_sync1_sqrt_pd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_pd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_pd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_pd_8;"       //|<
                "_wait_sqrt_pd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_pd_8;"         //|<
                "_sync2_sqrt_pd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_pd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_pd_8:"
                "sqrtpd (%%r9), %%xmm0;sqrtpd 16(%%r9), %%xmm1;sqrtpd 32(%%r9), %%xmm2;sqrtpd 48(%%r9), %%xmm3;"
                "sqrtpd 64(%%r9), %%xmm4;sqrtpd 80(%%r9), %%xmm5;sqrtpd 96(%%r9), %%xmm6;sqrtpd 112(%%r9), %%xmm7;"
                "sqrtpd 128(%%r9), %%xmm0;sqrtpd 144(%%r9), %%xmm1;sqrtpd 160(%%r9), %%xmm2;sqrtpd 176(%%r9), %%xmm3;"
                "sqrtpd 192(%%r9), %%xmm4;sqrtpd 208(%%r9), %%xmm5;sqrtpd 224(%%r9), %%xmm6;sqrtpd 240(%%r9), %%xmm7;"
                "sqrtpd 256(%%r9), %%xmm0;sqrtpd 272(%%r9), %%xmm1;sqrtpd 288(%%r9), %%xmm2;sqrtpd 304(%%r9), %%xmm3;"
                "sqrtpd 320(%%r9), %%xmm4;sqrtpd 336(%%r9), %%xmm5;sqrtpd 352(%%r9), %%xmm6;sqrtpd 368(%%r9), %%xmm7;"
                "sqrtpd 384(%%r9), %%xmm0;sqrtpd 400(%%r9), %%xmm1;sqrtpd 416(%%r9), %%xmm2;sqrtpd 432(%%r9), %%xmm3;"
                "sqrtpd 448(%%r9), %%xmm4;sqrtpd 464(%%r9), %%xmm5;sqrtpd 480(%%r9), %%xmm6;sqrtpd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_sqrt_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_sqrt_ps(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ps_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ps_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ps_1;"       //|<
                "_sync1_sqrt_ps_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ps_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ps_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ps_1;"       //|<
                "_wait_sqrt_ps_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ps_1;"         //|<
                "_sync2_sqrt_ps_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ps_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ps_1:"
                "sqrtps (%%r9), %%xmm0;"
                "sqrtps 16(%%r9), %%xmm0;"
                "sqrtps 32(%%r9), %%xmm0;"
                "sqrtps 48(%%r9), %%xmm0;"
                "sqrtps 64(%%r9), %%xmm0;"
                "sqrtps 80(%%r9), %%xmm0;"
                "sqrtps 96(%%r9), %%xmm0;"
                "sqrtps 112(%%r9), %%xmm0;"
                "sqrtps 128(%%r9), %%xmm0;"
                "sqrtps 144(%%r9), %%xmm0;"
                "sqrtps 160(%%r9), %%xmm0;"
                "sqrtps 176(%%r9), %%xmm0;"
                "sqrtps 192(%%r9), %%xmm0;"
                "sqrtps 208(%%r9), %%xmm0;"
                "sqrtps 224(%%r9), %%xmm0;"
                "sqrtps 240(%%r9), %%xmm0;"
                "sqrtps 256(%%r9), %%xmm0;"
                "sqrtps 272(%%r9), %%xmm0;"
                "sqrtps 288(%%r9), %%xmm0;"
                "sqrtps 304(%%r9), %%xmm0;"
                "sqrtps 320(%%r9), %%xmm0;"
                "sqrtps 336(%%r9), %%xmm0;"
                "sqrtps 352(%%r9), %%xmm0;"
                "sqrtps 368(%%r9), %%xmm0;"
                "sqrtps 384(%%r9), %%xmm0;"
                "sqrtps 400(%%r9), %%xmm0;"
                "sqrtps 416(%%r9), %%xmm0;"
                "sqrtps 432(%%r9), %%xmm0;"
                "sqrtps 448(%%r9), %%xmm0;"
                "sqrtps 464(%%r9), %%xmm0;"
                "sqrtps 480(%%r9), %%xmm0;"
                "sqrtps 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ps_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ps_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ps_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ps_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ps_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ps_2;"       //|<
                "_sync1_sqrt_ps_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ps_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ps_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ps_2;"       //|<
                "_wait_sqrt_ps_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ps_2;"         //|<
                "_sync2_sqrt_ps_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ps_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ps_2:"
                "sqrtps (%%r9), %%xmm0;sqrtps 16(%%r9), %%xmm1;"
                "sqrtps 32(%%r9), %%xmm0;sqrtps 48(%%r9), %%xmm1;"
                "sqrtps 64(%%r9), %%xmm0;sqrtps 80(%%r9), %%xmm1;"
                "sqrtps 96(%%r9), %%xmm0;sqrtps 112(%%r9), %%xmm1;"
                "sqrtps 128(%%r9), %%xmm0;sqrtps 144(%%r9), %%xmm1;"
                "sqrtps 160(%%r9), %%xmm0;sqrtps 176(%%r9), %%xmm1;"
                "sqrtps 192(%%r9), %%xmm0;sqrtps 208(%%r9), %%xmm1;"
                "sqrtps 224(%%r9), %%xmm0;sqrtps 240(%%r9), %%xmm1;"
                "sqrtps 256(%%r9), %%xmm0;sqrtps 272(%%r9), %%xmm1;"
                "sqrtps 288(%%r9), %%xmm0;sqrtps 304(%%r9), %%xmm1;"
                "sqrtps 320(%%r9), %%xmm0;sqrtps 336(%%r9), %%xmm1;"
                "sqrtps 352(%%r9), %%xmm0;sqrtps 368(%%r9), %%xmm1;"
                "sqrtps 384(%%r9), %%xmm0;sqrtps 400(%%r9), %%xmm1;"
                "sqrtps 416(%%r9), %%xmm0;sqrtps 432(%%r9), %%xmm1;"
                "sqrtps 448(%%r9), %%xmm0;sqrtps 464(%%r9), %%xmm1;"
                "sqrtps 480(%%r9), %%xmm0;sqrtps 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ps_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ps_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ps_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ps_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ps_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ps_3;"       //|<
                "_sync1_sqrt_ps_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ps_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ps_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ps_3;"       //|<
                "_wait_sqrt_ps_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ps_3;"         //|<
                "_sync2_sqrt_ps_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ps_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ps_3:"
                "sqrtps (%%r9), %%xmm0;sqrtps 16(%%r9), %%xmm1;sqrtps 32(%%r9), %%xmm2;"
                "sqrtps 48(%%r9), %%xmm0;sqrtps 64(%%r9), %%xmm1;sqrtps 80(%%r9), %%xmm2;"
                "sqrtps 96(%%r9), %%xmm0;sqrtps 112(%%r9), %%xmm1;sqrtps 128(%%r9), %%xmm2;"
                "sqrtps 144(%%r9), %%xmm0;sqrtps 160(%%r9), %%xmm1;sqrtps 176(%%r9), %%xmm2;"
                "sqrtps 192(%%r9), %%xmm0;sqrtps 208(%%r9), %%xmm1;sqrtps 224(%%r9), %%xmm2;"
                "sqrtps 240(%%r9), %%xmm0;sqrtps 256(%%r9), %%xmm1;sqrtps 272(%%r9), %%xmm2;"
                "sqrtps 288(%%r9), %%xmm0;sqrtps 304(%%r9), %%xmm1;sqrtps 320(%%r9), %%xmm2;"
                "sqrtps 336(%%r9), %%xmm0;sqrtps 352(%%r9), %%xmm1;sqrtps 368(%%r9), %%xmm2;"
                "sqrtps 384(%%r9), %%xmm0;sqrtps 400(%%r9), %%xmm1;sqrtps 416(%%r9), %%xmm2;"
                "sqrtps 432(%%r9), %%xmm0;sqrtps 448(%%r9), %%xmm1;sqrtps 464(%%r9), %%xmm2;"
                "sqrtps 480(%%r9), %%xmm0;sqrtps 496(%%r9), %%xmm1;sqrtps 512(%%r9), %%xmm2;"
                "sqrtps 528(%%r9), %%xmm0;sqrtps 544(%%r9), %%xmm1;sqrtps 560(%%r9), %%xmm2;"
                "sqrtps 576(%%r9), %%xmm0;sqrtps 592(%%r9), %%xmm1;sqrtps 608(%%r9), %%xmm2;"
                "sqrtps 624(%%r9), %%xmm0;sqrtps 640(%%r9), %%xmm1;sqrtps 656(%%r9), %%xmm2;"
                "sqrtps 672(%%r9), %%xmm0;sqrtps 688(%%r9), %%xmm1;sqrtps 704(%%r9), %%xmm2;"
                "sqrtps 720(%%r9), %%xmm0;sqrtps 736(%%r9), %%xmm1;sqrtps 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ps_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ps_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_sqrt_ps_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ps_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ps_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ps_4;"       //|<
                "_sync1_sqrt_ps_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ps_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ps_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ps_4;"       //|<
                "_wait_sqrt_ps_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ps_4;"         //|<
                "_sync2_sqrt_ps_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ps_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ps_4:"
                "sqrtps (%%r9), %%xmm0;sqrtps 16(%%r9), %%xmm1;sqrtps 32(%%r9), %%xmm2;sqrtps 48(%%r9), %%xmm3;"
                "sqrtps 64(%%r9), %%xmm0;sqrtps 80(%%r9), %%xmm1;sqrtps 96(%%r9), %%xmm2;sqrtps 112(%%r9), %%xmm3;"
                "sqrtps 128(%%r9), %%xmm0;sqrtps 144(%%r9), %%xmm1;sqrtps 160(%%r9), %%xmm2;sqrtps 176(%%r9), %%xmm3;"
                "sqrtps 192(%%r9), %%xmm0;sqrtps 208(%%r9), %%xmm1;sqrtps 224(%%r9), %%xmm2;sqrtps 240(%%r9), %%xmm3;"
                "sqrtps 256(%%r9), %%xmm0;sqrtps 272(%%r9), %%xmm1;sqrtps 288(%%r9), %%xmm2;sqrtps 304(%%r9), %%xmm3;"
                "sqrtps 320(%%r9), %%xmm0;sqrtps 336(%%r9), %%xmm1;sqrtps 352(%%r9), %%xmm2;sqrtps 368(%%r9), %%xmm3;"
                "sqrtps 384(%%r9), %%xmm0;sqrtps 400(%%r9), %%xmm1;sqrtps 416(%%r9), %%xmm2;sqrtps 432(%%r9), %%xmm3;"
                "sqrtps 448(%%r9), %%xmm0;sqrtps 464(%%r9), %%xmm1;sqrtps 480(%%r9), %%xmm2;sqrtps 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ps_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ps_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ps_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ps_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ps_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ps_8;"       //|<
                "_sync1_sqrt_ps_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ps_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ps_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ps_8;"       //|<
                "_wait_sqrt_ps_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ps_8;"         //|<
                "_sync2_sqrt_ps_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ps_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ps_8:"
                "sqrtps (%%r9), %%xmm0;sqrtps 16(%%r9), %%xmm1;sqrtps 32(%%r9), %%xmm2;sqrtps 48(%%r9), %%xmm3;"
                "sqrtps 64(%%r9), %%xmm4;sqrtps 80(%%r9), %%xmm5;sqrtps 96(%%r9), %%xmm6;sqrtps 112(%%r9), %%xmm7;"
                "sqrtps 128(%%r9), %%xmm0;sqrtps 144(%%r9), %%xmm1;sqrtps 160(%%r9), %%xmm2;sqrtps 176(%%r9), %%xmm3;"
                "sqrtps 192(%%r9), %%xmm4;sqrtps 208(%%r9), %%xmm5;sqrtps 224(%%r9), %%xmm6;sqrtps 240(%%r9), %%xmm7;"
                "sqrtps 256(%%r9), %%xmm0;sqrtps 272(%%r9), %%xmm1;sqrtps 288(%%r9), %%xmm2;sqrtps 304(%%r9), %%xmm3;"
                "sqrtps 320(%%r9), %%xmm4;sqrtps 336(%%r9), %%xmm5;sqrtps 352(%%r9), %%xmm6;sqrtps 368(%%r9), %%xmm7;"
                "sqrtps 384(%%r9), %%xmm0;sqrtps 400(%%r9), %%xmm1;sqrtps 416(%%r9), %%xmm2;sqrtps 432(%%r9), %%xmm3;"
                "sqrtps 448(%%r9), %%xmm4;sqrtps 464(%%r9), %%xmm5;sqrtps 480(%%r9), %%xmm6;sqrtps 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ps_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ps_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ps_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_sqrt_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_sqrt_sd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_sd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_sd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_sd_1;"       //|<
                "_sync1_sqrt_sd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_sd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_sd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_sd_1;"       //|<
                "_wait_sqrt_sd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_sd_1;"         //|<
                "_sync2_sqrt_sd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_sd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_sd_1:"
                "sqrtsd (%%r9), %%xmm0;"
                "sqrtsd 16(%%r9), %%xmm0;"
                "sqrtsd 32(%%r9), %%xmm0;"
                "sqrtsd 48(%%r9), %%xmm0;"
                "sqrtsd 64(%%r9), %%xmm0;"
                "sqrtsd 80(%%r9), %%xmm0;"
                "sqrtsd 96(%%r9), %%xmm0;"
                "sqrtsd 112(%%r9), %%xmm0;"
                "sqrtsd 128(%%r9), %%xmm0;"
                "sqrtsd 144(%%r9), %%xmm0;"
                "sqrtsd 160(%%r9), %%xmm0;"
                "sqrtsd 176(%%r9), %%xmm0;"
                "sqrtsd 192(%%r9), %%xmm0;"
                "sqrtsd 208(%%r9), %%xmm0;"
                "sqrtsd 224(%%r9), %%xmm0;"
                "sqrtsd 240(%%r9), %%xmm0;"
                "sqrtsd 256(%%r9), %%xmm0;"
                "sqrtsd 272(%%r9), %%xmm0;"
                "sqrtsd 288(%%r9), %%xmm0;"
                "sqrtsd 304(%%r9), %%xmm0;"
                "sqrtsd 320(%%r9), %%xmm0;"
                "sqrtsd 336(%%r9), %%xmm0;"
                "sqrtsd 352(%%r9), %%xmm0;"
                "sqrtsd 368(%%r9), %%xmm0;"
                "sqrtsd 384(%%r9), %%xmm0;"
                "sqrtsd 400(%%r9), %%xmm0;"
                "sqrtsd 416(%%r9), %%xmm0;"
                "sqrtsd 432(%%r9), %%xmm0;"
                "sqrtsd 448(%%r9), %%xmm0;"
                "sqrtsd 464(%%r9), %%xmm0;"
                "sqrtsd 480(%%r9), %%xmm0;"
                "sqrtsd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_sd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_sd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_sd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_sd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_sd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_sd_2;"       //|<
                "_sync1_sqrt_sd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_sd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_sd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_sd_2;"       //|<
                "_wait_sqrt_sd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_sd_2;"         //|<
                "_sync2_sqrt_sd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_sd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_sd_2:"
                "sqrtsd (%%r9), %%xmm0;sqrtsd 16(%%r9), %%xmm1;"
                "sqrtsd 32(%%r9), %%xmm0;sqrtsd 48(%%r9), %%xmm1;"
                "sqrtsd 64(%%r9), %%xmm0;sqrtsd 80(%%r9), %%xmm1;"
                "sqrtsd 96(%%r9), %%xmm0;sqrtsd 112(%%r9), %%xmm1;"
                "sqrtsd 128(%%r9), %%xmm0;sqrtsd 144(%%r9), %%xmm1;"
                "sqrtsd 160(%%r9), %%xmm0;sqrtsd 176(%%r9), %%xmm1;"
                "sqrtsd 192(%%r9), %%xmm0;sqrtsd 208(%%r9), %%xmm1;"
                "sqrtsd 224(%%r9), %%xmm0;sqrtsd 240(%%r9), %%xmm1;"
                "sqrtsd 256(%%r9), %%xmm0;sqrtsd 272(%%r9), %%xmm1;"
                "sqrtsd 288(%%r9), %%xmm0;sqrtsd 304(%%r9), %%xmm1;"
                "sqrtsd 320(%%r9), %%xmm0;sqrtsd 336(%%r9), %%xmm1;"
                "sqrtsd 352(%%r9), %%xmm0;sqrtsd 368(%%r9), %%xmm1;"
                "sqrtsd 384(%%r9), %%xmm0;sqrtsd 400(%%r9), %%xmm1;"
                "sqrtsd 416(%%r9), %%xmm0;sqrtsd 432(%%r9), %%xmm1;"
                "sqrtsd 448(%%r9), %%xmm0;sqrtsd 464(%%r9), %%xmm1;"
                "sqrtsd 480(%%r9), %%xmm0;sqrtsd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_sd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_sd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_sd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_sd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_sd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_sd_3;"       //|<
                "_sync1_sqrt_sd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_sd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_sd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_sd_3;"       //|<
                "_wait_sqrt_sd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_sd_3;"         //|<
                "_sync2_sqrt_sd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_sd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_sd_3:"
                "sqrtsd (%%r9), %%xmm0;sqrtsd 16(%%r9), %%xmm1;sqrtsd 32(%%r9), %%xmm2;"
                "sqrtsd 48(%%r9), %%xmm0;sqrtsd 64(%%r9), %%xmm1;sqrtsd 80(%%r9), %%xmm2;"
                "sqrtsd 96(%%r9), %%xmm0;sqrtsd 112(%%r9), %%xmm1;sqrtsd 128(%%r9), %%xmm2;"
                "sqrtsd 144(%%r9), %%xmm0;sqrtsd 160(%%r9), %%xmm1;sqrtsd 176(%%r9), %%xmm2;"
                "sqrtsd 192(%%r9), %%xmm0;sqrtsd 208(%%r9), %%xmm1;sqrtsd 224(%%r9), %%xmm2;"
                "sqrtsd 240(%%r9), %%xmm0;sqrtsd 256(%%r9), %%xmm1;sqrtsd 272(%%r9), %%xmm2;"
                "sqrtsd 288(%%r9), %%xmm0;sqrtsd 304(%%r9), %%xmm1;sqrtsd 320(%%r9), %%xmm2;"
                "sqrtsd 336(%%r9), %%xmm0;sqrtsd 352(%%r9), %%xmm1;sqrtsd 368(%%r9), %%xmm2;"
                "sqrtsd 384(%%r9), %%xmm0;sqrtsd 400(%%r9), %%xmm1;sqrtsd 416(%%r9), %%xmm2;"
                "sqrtsd 432(%%r9), %%xmm0;sqrtsd 448(%%r9), %%xmm1;sqrtsd 464(%%r9), %%xmm2;"
                "sqrtsd 480(%%r9), %%xmm0;sqrtsd 496(%%r9), %%xmm1;sqrtsd 512(%%r9), %%xmm2;"
                "sqrtsd 528(%%r9), %%xmm0;sqrtsd 544(%%r9), %%xmm1;sqrtsd 560(%%r9), %%xmm2;"
                "sqrtsd 576(%%r9), %%xmm0;sqrtsd 592(%%r9), %%xmm1;sqrtsd 608(%%r9), %%xmm2;"
                "sqrtsd 624(%%r9), %%xmm0;sqrtsd 640(%%r9), %%xmm1;sqrtsd 656(%%r9), %%xmm2;"
                "sqrtsd 672(%%r9), %%xmm0;sqrtsd 688(%%r9), %%xmm1;sqrtsd 704(%%r9), %%xmm2;"
                "sqrtsd 720(%%r9), %%xmm0;sqrtsd 736(%%r9), %%xmm1;sqrtsd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_sd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_sd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_sqrt_sd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_sd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_sd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_sd_4;"       //|<
                "_sync1_sqrt_sd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_sd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_sd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_sd_4;"       //|<
                "_wait_sqrt_sd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_sd_4;"         //|<
                "_sync2_sqrt_sd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_sd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_sd_4:"
                "sqrtsd (%%r9), %%xmm0;sqrtsd 16(%%r9), %%xmm1;sqrtsd 32(%%r9), %%xmm2;sqrtsd 48(%%r9), %%xmm3;"
                "sqrtsd 64(%%r9), %%xmm0;sqrtsd 80(%%r9), %%xmm1;sqrtsd 96(%%r9), %%xmm2;sqrtsd 112(%%r9), %%xmm3;"
                "sqrtsd 128(%%r9), %%xmm0;sqrtsd 144(%%r9), %%xmm1;sqrtsd 160(%%r9), %%xmm2;sqrtsd 176(%%r9), %%xmm3;"
                "sqrtsd 192(%%r9), %%xmm0;sqrtsd 208(%%r9), %%xmm1;sqrtsd 224(%%r9), %%xmm2;sqrtsd 240(%%r9), %%xmm3;"
                "sqrtsd 256(%%r9), %%xmm0;sqrtsd 272(%%r9), %%xmm1;sqrtsd 288(%%r9), %%xmm2;sqrtsd 304(%%r9), %%xmm3;"
                "sqrtsd 320(%%r9), %%xmm0;sqrtsd 336(%%r9), %%xmm1;sqrtsd 352(%%r9), %%xmm2;sqrtsd 368(%%r9), %%xmm3;"
                "sqrtsd 384(%%r9), %%xmm0;sqrtsd 400(%%r9), %%xmm1;sqrtsd 416(%%r9), %%xmm2;sqrtsd 432(%%r9), %%xmm3;"
                "sqrtsd 448(%%r9), %%xmm0;sqrtsd 464(%%r9), %%xmm1;sqrtsd 480(%%r9), %%xmm2;sqrtsd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_sd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_sd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_sd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_sd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_sd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_sd_8;"       //|<
                "_sync1_sqrt_sd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_sd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_sd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_sd_8;"       //|<
                "_wait_sqrt_sd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_sd_8;"         //|<
                "_sync2_sqrt_sd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_sd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_sd_8:"
                "sqrtsd (%%r9), %%xmm0;sqrtsd 16(%%r9), %%xmm1;sqrtsd 32(%%r9), %%xmm2;sqrtsd 48(%%r9), %%xmm3;"
                "sqrtsd 64(%%r9), %%xmm4;sqrtsd 80(%%r9), %%xmm5;sqrtsd 96(%%r9), %%xmm6;sqrtsd 112(%%r9), %%xmm7;"
                "sqrtsd 128(%%r9), %%xmm0;sqrtsd 144(%%r9), %%xmm1;sqrtsd 160(%%r9), %%xmm2;sqrtsd 176(%%r9), %%xmm3;"
                "sqrtsd 192(%%r9), %%xmm4;sqrtsd 208(%%r9), %%xmm5;sqrtsd 224(%%r9), %%xmm6;sqrtsd 240(%%r9), %%xmm7;"
                "sqrtsd 256(%%r9), %%xmm0;sqrtsd 272(%%r9), %%xmm1;sqrtsd 288(%%r9), %%xmm2;sqrtsd 304(%%r9), %%xmm3;"
                "sqrtsd 320(%%r9), %%xmm4;sqrtsd 336(%%r9), %%xmm5;sqrtsd 352(%%r9), %%xmm6;sqrtsd 368(%%r9), %%xmm7;"
                "sqrtsd 384(%%r9), %%xmm0;sqrtsd 400(%%r9), %%xmm1;sqrtsd 416(%%r9), %%xmm2;sqrtsd 432(%%r9), %%xmm3;"
                "sqrtsd 448(%%r9), %%xmm4;sqrtsd 464(%%r9), %%xmm5;sqrtsd 480(%%r9), %%xmm6;sqrtsd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_sd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_sd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_sd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_sqrt_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_sqrt_ss(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ss_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ss_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ss_1;"       //|<
                "_sync1_sqrt_ss_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ss_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ss_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ss_1;"       //|<
                "_wait_sqrt_ss_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ss_1;"         //|<
                "_sync2_sqrt_ss_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ss_1;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ss_1:"
                "sqrtss (%%r9), %%xmm0;"
                "sqrtss 16(%%r9), %%xmm0;"
                "sqrtss 32(%%r9), %%xmm0;"
                "sqrtss 48(%%r9), %%xmm0;"
                "sqrtss 64(%%r9), %%xmm0;"
                "sqrtss 80(%%r9), %%xmm0;"
                "sqrtss 96(%%r9), %%xmm0;"
                "sqrtss 112(%%r9), %%xmm0;"
                "sqrtss 128(%%r9), %%xmm0;"
                "sqrtss 144(%%r9), %%xmm0;"
                "sqrtss 160(%%r9), %%xmm0;"
                "sqrtss 176(%%r9), %%xmm0;"
                "sqrtss 192(%%r9), %%xmm0;"
                "sqrtss 208(%%r9), %%xmm0;"
                "sqrtss 224(%%r9), %%xmm0;"
                "sqrtss 240(%%r9), %%xmm0;"
                "sqrtss 256(%%r9), %%xmm0;"
                "sqrtss 272(%%r9), %%xmm0;"
                "sqrtss 288(%%r9), %%xmm0;"
                "sqrtss 304(%%r9), %%xmm0;"
                "sqrtss 320(%%r9), %%xmm0;"
                "sqrtss 336(%%r9), %%xmm0;"
                "sqrtss 352(%%r9), %%xmm0;"
                "sqrtss 368(%%r9), %%xmm0;"
                "sqrtss 384(%%r9), %%xmm0;"
                "sqrtss 400(%%r9), %%xmm0;"
                "sqrtss 416(%%r9), %%xmm0;"
                "sqrtss 432(%%r9), %%xmm0;"
                "sqrtss 448(%%r9), %%xmm0;"
                "sqrtss 464(%%r9), %%xmm0;"
                "sqrtss 480(%%r9), %%xmm0;"
                "sqrtss 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ss_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ss_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ss_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ss_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ss_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ss_2;"       //|<
                "_sync1_sqrt_ss_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ss_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ss_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ss_2;"       //|<
                "_wait_sqrt_ss_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ss_2;"         //|<
                "_sync2_sqrt_ss_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ss_2;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ss_2:"
                "sqrtss (%%r9), %%xmm0;sqrtss 16(%%r9), %%xmm1;"
                "sqrtss 32(%%r9), %%xmm0;sqrtss 48(%%r9), %%xmm1;"
                "sqrtss 64(%%r9), %%xmm0;sqrtss 80(%%r9), %%xmm1;"
                "sqrtss 96(%%r9), %%xmm0;sqrtss 112(%%r9), %%xmm1;"
                "sqrtss 128(%%r9), %%xmm0;sqrtss 144(%%r9), %%xmm1;"
                "sqrtss 160(%%r9), %%xmm0;sqrtss 176(%%r9), %%xmm1;"
                "sqrtss 192(%%r9), %%xmm0;sqrtss 208(%%r9), %%xmm1;"
                "sqrtss 224(%%r9), %%xmm0;sqrtss 240(%%r9), %%xmm1;"
                "sqrtss 256(%%r9), %%xmm0;sqrtss 272(%%r9), %%xmm1;"
                "sqrtss 288(%%r9), %%xmm0;sqrtss 304(%%r9), %%xmm1;"
                "sqrtss 320(%%r9), %%xmm0;sqrtss 336(%%r9), %%xmm1;"
                "sqrtss 352(%%r9), %%xmm0;sqrtss 368(%%r9), %%xmm1;"
                "sqrtss 384(%%r9), %%xmm0;sqrtss 400(%%r9), %%xmm1;"
                "sqrtss 416(%%r9), %%xmm0;sqrtss 432(%%r9), %%xmm1;"
                "sqrtss 448(%%r9), %%xmm0;sqrtss 464(%%r9), %%xmm1;"
                "sqrtss 480(%%r9), %%xmm0;sqrtss 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ss_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ss_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ss_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ss_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ss_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ss_3;"       //|<
                "_sync1_sqrt_ss_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ss_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ss_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ss_3;"       //|<
                "_wait_sqrt_ss_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ss_3;"         //|<
                "_sync2_sqrt_ss_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ss_3;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ss_3:"
                "sqrtss (%%r9), %%xmm0;sqrtss 16(%%r9), %%xmm1;sqrtss 32(%%r9), %%xmm2;"
                "sqrtss 48(%%r9), %%xmm0;sqrtss 64(%%r9), %%xmm1;sqrtss 80(%%r9), %%xmm2;"
                "sqrtss 96(%%r9), %%xmm0;sqrtss 112(%%r9), %%xmm1;sqrtss 128(%%r9), %%xmm2;"
                "sqrtss 144(%%r9), %%xmm0;sqrtss 160(%%r9), %%xmm1;sqrtss 176(%%r9), %%xmm2;"
                "sqrtss 192(%%r9), %%xmm0;sqrtss 208(%%r9), %%xmm1;sqrtss 224(%%r9), %%xmm2;"
                "sqrtss 240(%%r9), %%xmm0;sqrtss 256(%%r9), %%xmm1;sqrtss 272(%%r9), %%xmm2;"
                "sqrtss 288(%%r9), %%xmm0;sqrtss 304(%%r9), %%xmm1;sqrtss 320(%%r9), %%xmm2;"
                "sqrtss 336(%%r9), %%xmm0;sqrtss 352(%%r9), %%xmm1;sqrtss 368(%%r9), %%xmm2;"
                "sqrtss 384(%%r9), %%xmm0;sqrtss 400(%%r9), %%xmm1;sqrtss 416(%%r9), %%xmm2;"
                "sqrtss 432(%%r9), %%xmm0;sqrtss 448(%%r9), %%xmm1;sqrtss 464(%%r9), %%xmm2;"
                "sqrtss 480(%%r9), %%xmm0;sqrtss 496(%%r9), %%xmm1;sqrtss 512(%%r9), %%xmm2;"
                "sqrtss 528(%%r9), %%xmm0;sqrtss 544(%%r9), %%xmm1;sqrtss 560(%%r9), %%xmm2;"
                "sqrtss 576(%%r9), %%xmm0;sqrtss 592(%%r9), %%xmm1;sqrtss 608(%%r9), %%xmm2;"
                "sqrtss 624(%%r9), %%xmm0;sqrtss 640(%%r9), %%xmm1;sqrtss 656(%%r9), %%xmm2;"
                "sqrtss 672(%%r9), %%xmm0;sqrtss 688(%%r9), %%xmm1;sqrtss 704(%%r9), %%xmm2;"
                "sqrtss 720(%%r9), %%xmm0;sqrtss 736(%%r9), %%xmm1;sqrtss 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ss_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ss_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_sqrt_ss_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ss_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ss_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ss_4;"       //|<
                "_sync1_sqrt_ss_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ss_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ss_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ss_4;"       //|<
                "_wait_sqrt_ss_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ss_4;"         //|<
                "_sync2_sqrt_ss_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ss_4;"       //<<
                //initialize registers
                "movaps (%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ss_4:"
                "sqrtss (%%r9), %%xmm0;sqrtss 16(%%r9), %%xmm1;sqrtss 32(%%r9), %%xmm2;sqrtss 48(%%r9), %%xmm3;"
                "sqrtss 64(%%r9), %%xmm0;sqrtss 80(%%r9), %%xmm1;sqrtss 96(%%r9), %%xmm2;sqrtss 112(%%r9), %%xmm3;"
                "sqrtss 128(%%r9), %%xmm0;sqrtss 144(%%r9), %%xmm1;sqrtss 160(%%r9), %%xmm2;sqrtss 176(%%r9), %%xmm3;"
                "sqrtss 192(%%r9), %%xmm0;sqrtss 208(%%r9), %%xmm1;sqrtss 224(%%r9), %%xmm2;sqrtss 240(%%r9), %%xmm3;"
                "sqrtss 256(%%r9), %%xmm0;sqrtss 272(%%r9), %%xmm1;sqrtss 288(%%r9), %%xmm2;sqrtss 304(%%r9), %%xmm3;"
                "sqrtss 320(%%r9), %%xmm0;sqrtss 336(%%r9), %%xmm1;sqrtss 352(%%r9), %%xmm2;sqrtss 368(%%r9), %%xmm3;"
                "sqrtss 384(%%r9), %%xmm0;sqrtss 400(%%r9), %%xmm1;sqrtss 416(%%r9), %%xmm2;sqrtss 432(%%r9), %%xmm3;"
                "sqrtss 448(%%r9), %%xmm0;sqrtss 464(%%r9), %%xmm1;sqrtss 480(%%r9), %%xmm2;sqrtss 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ss_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ss_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ss_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_sqrt_ss_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_sqrt_ss_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_sqrt_ss_8;"       //|<
                "_sync1_sqrt_ss_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_sqrt_ss_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_sqrt_ss_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_sqrt_ss_8;"       //|<
                "_wait_sqrt_ss_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_sqrt_ss_8;"         //|<
                "_sync2_sqrt_ss_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_sqrt_ss_8;"       //<<
                //initialize registers
                "movaps 0(%%r9), %%xmm0;"
                "movaps 16(%%r9), %%xmm1;"
                "movaps 32(%%r9), %%xmm2;"
                "movaps 48(%%r9), %%xmm3;"
                "movaps 64(%%r9), %%xmm4;"
                "movaps 80(%%r9), %%xmm5;"
                "movaps 96(%%r9), %%xmm6;"
                "movaps 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_sqrt_ss_8:"
                "sqrtss (%%r9), %%xmm0;sqrtss 16(%%r9), %%xmm1;sqrtss 32(%%r9), %%xmm2;sqrtss 48(%%r9), %%xmm3;"
                "sqrtss 64(%%r9), %%xmm4;sqrtss 80(%%r9), %%xmm5;sqrtss 96(%%r9), %%xmm6;sqrtss 112(%%r9), %%xmm7;"
                "sqrtss 128(%%r9), %%xmm0;sqrtss 144(%%r9), %%xmm1;sqrtss 160(%%r9), %%xmm2;sqrtss 176(%%r9), %%xmm3;"
                "sqrtss 192(%%r9), %%xmm4;sqrtss 208(%%r9), %%xmm5;sqrtss 224(%%r9), %%xmm6;sqrtss 240(%%r9), %%xmm7;"
                "sqrtss 256(%%r9), %%xmm0;sqrtss 272(%%r9), %%xmm1;sqrtss 288(%%r9), %%xmm2;sqrtss 304(%%r9), %%xmm3;"
                "sqrtss 320(%%r9), %%xmm4;sqrtss 336(%%r9), %%xmm5;sqrtss 352(%%r9), %%xmm6;sqrtss 368(%%r9), %%xmm7;"
                "sqrtss 384(%%r9), %%xmm0;sqrtss 400(%%r9), %%xmm1;sqrtss 416(%%r9), %%xmm2;sqrtss 432(%%r9), %%xmm3;"
                "sqrtss 448(%%r9), %%xmm4;sqrtss 464(%%r9), %%xmm5;sqrtss 480(%%r9), %%xmm6;sqrtss 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_sqrt_ss_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_sqrt_ss_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_sqrt_ss_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_and_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_and_pd(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pd_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pd_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pd_1;"       //|<
                "_sync1_and_pd_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pd_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pd_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pd_1;"       //|<
                "_wait_and_pd_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pd_1;"         //|<
                "_sync2_and_pd_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pd_1;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pd_1:"
                "andpd (%%r9), %%xmm0;"
                "andpd 16(%%r9), %%xmm0;"
                "andpd 32(%%r9), %%xmm0;"
                "andpd 48(%%r9), %%xmm0;"
                "andpd 64(%%r9), %%xmm0;"
                "andpd 80(%%r9), %%xmm0;"
                "andpd 96(%%r9), %%xmm0;"
                "andpd 112(%%r9), %%xmm0;"
                "andpd 128(%%r9), %%xmm0;"
                "andpd 144(%%r9), %%xmm0;"
                "andpd 160(%%r9), %%xmm0;"
                "andpd 176(%%r9), %%xmm0;"
                "andpd 192(%%r9), %%xmm0;"
                "andpd 208(%%r9), %%xmm0;"
                "andpd 224(%%r9), %%xmm0;"
                "andpd 240(%%r9), %%xmm0;"
                "andpd 256(%%r9), %%xmm0;"
                "andpd 272(%%r9), %%xmm0;"
                "andpd 288(%%r9), %%xmm0;"
                "andpd 304(%%r9), %%xmm0;"
                "andpd 320(%%r9), %%xmm0;"
                "andpd 336(%%r9), %%xmm0;"
                "andpd 352(%%r9), %%xmm0;"
                "andpd 368(%%r9), %%xmm0;"
                "andpd 384(%%r9), %%xmm0;"
                "andpd 400(%%r9), %%xmm0;"
                "andpd 416(%%r9), %%xmm0;"
                "andpd 432(%%r9), %%xmm0;"
                "andpd 448(%%r9), %%xmm0;"
                "andpd 464(%%r9), %%xmm0;"
                "andpd 480(%%r9), %%xmm0;"
                "andpd 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pd_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pd_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pd_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pd_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pd_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pd_2;"       //|<
                "_sync1_and_pd_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pd_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pd_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pd_2;"       //|<
                "_wait_and_pd_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pd_2;"         //|<
                "_sync2_and_pd_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pd_2;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pd_2:"
                "andpd (%%r9), %%xmm0;andpd 16(%%r9), %%xmm1;"
                "andpd 32(%%r9), %%xmm0;andpd 48(%%r9), %%xmm1;"
                "andpd 64(%%r9), %%xmm0;andpd 80(%%r9), %%xmm1;"
                "andpd 96(%%r9), %%xmm0;andpd 112(%%r9), %%xmm1;"
                "andpd 128(%%r9), %%xmm0;andpd 144(%%r9), %%xmm1;"
                "andpd 160(%%r9), %%xmm0;andpd 176(%%r9), %%xmm1;"
                "andpd 192(%%r9), %%xmm0;andpd 208(%%r9), %%xmm1;"
                "andpd 224(%%r9), %%xmm0;andpd 240(%%r9), %%xmm1;"
                "andpd 256(%%r9), %%xmm0;andpd 272(%%r9), %%xmm1;"
                "andpd 288(%%r9), %%xmm0;andpd 304(%%r9), %%xmm1;"
                "andpd 320(%%r9), %%xmm0;andpd 336(%%r9), %%xmm1;"
                "andpd 352(%%r9), %%xmm0;andpd 368(%%r9), %%xmm1;"
                "andpd 384(%%r9), %%xmm0;andpd 400(%%r9), %%xmm1;"
                "andpd 416(%%r9), %%xmm0;andpd 432(%%r9), %%xmm1;"
                "andpd 448(%%r9), %%xmm0;andpd 464(%%r9), %%xmm1;"
                "andpd 480(%%r9), %%xmm0;andpd 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pd_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pd_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pd_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pd_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pd_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pd_3;"       //|<
                "_sync1_and_pd_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pd_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pd_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pd_3;"       //|<
                "_wait_and_pd_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pd_3;"         //|<
                "_sync2_and_pd_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pd_3;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pd_3:"
                "andpd (%%r9), %%xmm0;andpd 16(%%r9), %%xmm1;andpd 32(%%r9), %%xmm2;"
                "andpd 48(%%r9), %%xmm0;andpd 64(%%r9), %%xmm1;andpd 80(%%r9), %%xmm2;"
                "andpd 96(%%r9), %%xmm0;andpd 112(%%r9), %%xmm1;andpd 128(%%r9), %%xmm2;"
                "andpd 144(%%r9), %%xmm0;andpd 160(%%r9), %%xmm1;andpd 176(%%r9), %%xmm2;"
                "andpd 192(%%r9), %%xmm0;andpd 208(%%r9), %%xmm1;andpd 224(%%r9), %%xmm2;"
                "andpd 240(%%r9), %%xmm0;andpd 256(%%r9), %%xmm1;andpd 272(%%r9), %%xmm2;"
                "andpd 288(%%r9), %%xmm0;andpd 304(%%r9), %%xmm1;andpd 320(%%r9), %%xmm2;"
                "andpd 336(%%r9), %%xmm0;andpd 352(%%r9), %%xmm1;andpd 368(%%r9), %%xmm2;"
                "andpd 384(%%r9), %%xmm0;andpd 400(%%r9), %%xmm1;andpd 416(%%r9), %%xmm2;"
                "andpd 432(%%r9), %%xmm0;andpd 448(%%r9), %%xmm1;andpd 464(%%r9), %%xmm2;"
                "andpd 480(%%r9), %%xmm0;andpd 496(%%r9), %%xmm1;andpd 512(%%r9), %%xmm2;"
                "andpd 528(%%r9), %%xmm0;andpd 544(%%r9), %%xmm1;andpd 560(%%r9), %%xmm2;"
                "andpd 576(%%r9), %%xmm0;andpd 592(%%r9), %%xmm1;andpd 608(%%r9), %%xmm2;"
                "andpd 624(%%r9), %%xmm0;andpd 640(%%r9), %%xmm1;andpd 656(%%r9), %%xmm2;"
                "andpd 672(%%r9), %%xmm0;andpd 688(%%r9), %%xmm1;andpd 704(%%r9), %%xmm2;"
                "andpd 720(%%r9), %%xmm0;andpd 736(%%r9), %%xmm1;andpd 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pd_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pd_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_and_pd_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pd_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pd_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pd_4;"       //|<
                "_sync1_and_pd_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pd_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pd_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pd_4;"       //|<
                "_wait_and_pd_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pd_4;"         //|<
                "_sync2_and_pd_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pd_4;"       //<<
                //initialize registers
                "movapd (%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pd_4:"
                "andpd (%%r9), %%xmm0;andpd 16(%%r9), %%xmm1;andpd 32(%%r9), %%xmm2;andpd 48(%%r9), %%xmm3;"
                "andpd 64(%%r9), %%xmm0;andpd 80(%%r9), %%xmm1;andpd 96(%%r9), %%xmm2;andpd 112(%%r9), %%xmm3;"
                "andpd 128(%%r9), %%xmm0;andpd 144(%%r9), %%xmm1;andpd 160(%%r9), %%xmm2;andpd 176(%%r9), %%xmm3;"
                "andpd 192(%%r9), %%xmm0;andpd 208(%%r9), %%xmm1;andpd 224(%%r9), %%xmm2;andpd 240(%%r9), %%xmm3;"
                "andpd 256(%%r9), %%xmm0;andpd 272(%%r9), %%xmm1;andpd 288(%%r9), %%xmm2;andpd 304(%%r9), %%xmm3;"
                "andpd 320(%%r9), %%xmm0;andpd 336(%%r9), %%xmm1;andpd 352(%%r9), %%xmm2;andpd 368(%%r9), %%xmm3;"
                "andpd 384(%%r9), %%xmm0;andpd 400(%%r9), %%xmm1;andpd 416(%%r9), %%xmm2;andpd 432(%%r9), %%xmm3;"
                "andpd 448(%%r9), %%xmm0;andpd 464(%%r9), %%xmm1;andpd 480(%%r9), %%xmm2;andpd 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pd_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pd_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pd_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pd_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pd_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pd_8;"       //|<
                "_sync1_and_pd_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pd_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pd_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pd_8;"       //|<
                "_wait_and_pd_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pd_8;"         //|<
                "_sync2_and_pd_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pd_8;"       //<<
                //initialize registers
                "movapd 0(%%r9), %%xmm0;"
                "movapd 16(%%r9), %%xmm1;"
                "movapd 32(%%r9), %%xmm2;"
                "movapd 48(%%r9), %%xmm3;"
                "movapd 64(%%r9), %%xmm4;"
                "movapd 80(%%r9), %%xmm5;"
                "movapd 96(%%r9), %%xmm6;"
                "movapd 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pd_8:"
                "andpd (%%r9), %%xmm0;andpd 16(%%r9), %%xmm1;andpd 32(%%r9), %%xmm2;andpd 48(%%r9), %%xmm3;"
                "andpd 64(%%r9), %%xmm4;andpd 80(%%r9), %%xmm5;andpd 96(%%r9), %%xmm6;andpd 112(%%r9), %%xmm7;"
                "andpd 128(%%r9), %%xmm0;andpd 144(%%r9), %%xmm1;andpd 160(%%r9), %%xmm2;andpd 176(%%r9), %%xmm3;"
                "andpd 192(%%r9), %%xmm4;andpd 208(%%r9), %%xmm5;andpd 224(%%r9), %%xmm6;andpd 240(%%r9), %%xmm7;"
                "andpd 256(%%r9), %%xmm0;andpd 272(%%r9), %%xmm1;andpd 288(%%r9), %%xmm2;andpd 304(%%r9), %%xmm3;"
                "andpd 320(%%r9), %%xmm4;andpd 336(%%r9), %%xmm5;andpd 352(%%r9), %%xmm6;andpd 368(%%r9), %%xmm7;"
                "andpd 384(%%r9), %%xmm0;andpd 400(%%r9), %%xmm1;andpd 416(%%r9), %%xmm2;andpd 432(%%r9), %%xmm3;"
                "andpd 448(%%r9), %%xmm4;andpd 464(%%r9), %%xmm5;andpd 480(%%r9), %%xmm6;andpd 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pd_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pd_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pd_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_and_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_and_pi(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm %i accesses %i\n",id,accesses);fflush(stdout);
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pi_1;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pi_1:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pi_1;"       //|<
                "_sync1_and_pi_1:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pi_1;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pi_1;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pi_1;"       //|<
                "_wait_and_pi_1:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pi_1;"         //|<
                "_sync2_and_pi_1:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pi_1;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pi_1:"
                "pand (%%r9), %%xmm0;"
                "pand 16(%%r9), %%xmm0;"
                "pand 32(%%r9), %%xmm0;"
                "pand 48(%%r9), %%xmm0;"
                "pand 64(%%r9), %%xmm0;"
                "pand 80(%%r9), %%xmm0;"
                "pand 96(%%r9), %%xmm0;"
                "pand 112(%%r9), %%xmm0;"
                "pand 128(%%r9), %%xmm0;"
                "pand 144(%%r9), %%xmm0;"
                "pand 160(%%r9), %%xmm0;"
                "pand 176(%%r9), %%xmm0;"
                "pand 192(%%r9), %%xmm0;"
                "pand 208(%%r9), %%xmm0;"
                "pand 224(%%r9), %%xmm0;"
                "pand 240(%%r9), %%xmm0;"
                "pand 256(%%r9), %%xmm0;"
                "pand 272(%%r9), %%xmm0;"
                "pand 288(%%r9), %%xmm0;"
                "pand 304(%%r9), %%xmm0;"
                "pand 320(%%r9), %%xmm0;"
                "pand 336(%%r9), %%xmm0;"
                "pand 352(%%r9), %%xmm0;"
                "pand 368(%%r9), %%xmm0;"
                "pand 384(%%r9), %%xmm0;"
                "pand 400(%%r9), %%xmm0;"
                "pand 416(%%r9), %%xmm0;"
                "pand 432(%%r9), %%xmm0;"
                "pand 448(%%r9), %%xmm0;"
                "pand 464(%%r9), %%xmm0;"
                "pand 480(%%r9), %%xmm0;"
                "pand 496(%%r9), %%xmm0;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pi_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pi_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pi_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pi_2;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pi_2:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pi_2;"       //|<
                "_sync1_and_pi_2:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pi_2;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pi_2;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pi_2;"       //|<
                "_wait_and_pi_2:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pi_2;"         //|<
                "_sync2_and_pi_2:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pi_2;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pi_2:"
                "pand (%%r9), %%xmm0;pand 16(%%r9), %%xmm1;"
                "pand 32(%%r9), %%xmm0;pand 48(%%r9), %%xmm1;"
                "pand 64(%%r9), %%xmm0;pand 80(%%r9), %%xmm1;"
                "pand 96(%%r9), %%xmm0;pand 112(%%r9), %%xmm1;"
                "pand 128(%%r9), %%xmm0;pand 144(%%r9), %%xmm1;"
                "pand 160(%%r9), %%xmm0;pand 176(%%r9), %%xmm1;"
                "pand 192(%%r9), %%xmm0;pand 208(%%r9), %%xmm1;"
                "pand 224(%%r9), %%xmm0;pand 240(%%r9), %%xmm1;"
                "pand 256(%%r9), %%xmm0;pand 272(%%r9), %%xmm1;"
                "pand 288(%%r9), %%xmm0;pand 304(%%r9), %%xmm1;"
                "pand 320(%%r9), %%xmm0;pand 336(%%r9), %%xmm1;"
                "pand 352(%%r9), %%xmm0;pand 368(%%r9), %%xmm1;"
                "pand 384(%%r9), %%xmm0;pand 400(%%r9), %%xmm1;"
                "pand 416(%%r9), %%xmm0;pand 432(%%r9), %%xmm1;"
                "pand 448(%%r9), %%xmm0;pand 464(%%r9), %%xmm1;"
                "pand 480(%%r9), %%xmm0;pand 496(%%r9), %%xmm1;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pi_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pi_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pi_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pi_3;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pi_3:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pi_3;"       //|<
                "_sync1_and_pi_3:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pi_3;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pi_3;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pi_3;"       //|<
                "_wait_and_pi_3:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pi_3;"         //|<
                "_sync2_and_pi_3:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pi_3;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pi_3:"
                "pand (%%r9), %%xmm0;pand 16(%%r9), %%xmm1;pand 32(%%r9), %%xmm2;"
                "pand 48(%%r9), %%xmm0;pand 64(%%r9), %%xmm1;pand 80(%%r9), %%xmm2;"
                "pand 96(%%r9), %%xmm0;pand 112(%%r9), %%xmm1;pand 128(%%r9), %%xmm2;"
                "pand 144(%%r9), %%xmm0;pand 160(%%r9), %%xmm1;pand 176(%%r9), %%xmm2;"
                "pand 192(%%r9), %%xmm0;pand 208(%%r9), %%xmm1;pand 224(%%r9), %%xmm2;"
                "pand 240(%%r9), %%xmm0;pand 256(%%r9), %%xmm1;pand 272(%%r9), %%xmm2;"
                "pand 288(%%r9), %%xmm0;pand 304(%%r9), %%xmm1;pand 320(%%r9), %%xmm2;"
                "pand 336(%%r9), %%xmm0;pand 352(%%r9), %%xmm1;pand 368(%%r9), %%xmm2;"
                "pand 384(%%r9), %%xmm0;pand 400(%%r9), %%xmm1;pand 416(%%r9), %%xmm2;"
                "pand 432(%%r9), %%xmm0;pand 448(%%r9), %%xmm1;pand 464(%%r9), %%xmm2;"
                "pand 480(%%r9), %%xmm0;pand 496(%%r9), %%xmm1;pand 512(%%r9), %%xmm2;"
                "pand 528(%%r9), %%xmm0;pand 544(%%r9), %%xmm1;pand 560(%%r9), %%xmm2;"
                "pand 576(%%r9), %%xmm0;pand 592(%%r9), %%xmm1;pand 608(%%r9), %%xmm2;"
                "pand 624(%%r9), %%xmm0;pand 640(%%r9), %%xmm1;pand 656(%%r9), %%xmm2;"
                "pand 672(%%r9), %%xmm0;pand 688(%%r9), %%xmm1;pand 704(%%r9), %%xmm2;"
                "pand 720(%%r9), %%xmm0;pand 736(%%r9), %%xmm1;pand 752(%%r9), %%xmm2;"     
                "add $768,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pi_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pi_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_and_pi_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pi_4;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pi_4:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pi_4;"       //|<
                "_sync1_and_pi_4:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pi_4;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pi_4;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pi_4;"       //|<
                "_wait_and_pi_4:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pi_4;"         //|<
                "_sync2_and_pi_4:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pi_4;"       //<<
                //initialize registers
                "movdqa (%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                "movdqa 48(%%r9), %%xmm3;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pi_4:"
                "pand (%%r9), %%xmm0;pand 16(%%r9), %%xmm1;pand 32(%%r9), %%xmm2;pand 48(%%r9), %%xmm3;"
                "pand 64(%%r9), %%xmm0;pand 80(%%r9), %%xmm1;pand 96(%%r9), %%xmm2;pand 112(%%r9), %%xmm3;"
                "pand 128(%%r9), %%xmm0;pand 144(%%r9), %%xmm1;pand 160(%%r9), %%xmm2;pand 176(%%r9), %%xmm3;"
                "pand 192(%%r9), %%xmm0;pand 208(%%r9), %%xmm1;pand 224(%%r9), %%xmm2;pand 240(%%r9), %%xmm3;"
                "pand 256(%%r9), %%xmm0;pand 272(%%r9), %%xmm1;pand 288(%%r9), %%xmm2;pand 304(%%r9), %%xmm3;"
                "pand 320(%%r9), %%xmm0;pand 336(%%r9), %%xmm1;pand 352(%%r9), %%xmm2;pand 368(%%r9), %%xmm3;"
                "pand 384(%%r9), %%xmm0;pand 400(%%r9), %%xmm1;pand 416(%%r9), %%xmm2;pand 432(%%r9), %%xmm3;"
                "pand 448(%%r9), %%xmm0;pand 464(%%r9), %%xmm1;pand 480(%%r9), %%xmm2;pand 496(%%r9), %%xmm3;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pi_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pi_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pi_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_and_pi_8;"       //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_and_pi_8:"           //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_and_pi_8;"       //|<
                "_sync1_and_pi_8:"           //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_and_pi_8;"       //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_and_pi_8;"        //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_and_pi_8;"       //|<
                "_wait_and_pi_8:"            //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_and_pi_8;"         //|<
                "_sync2_and_pi_8:"           //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_and_pi_8;"       //<<
                //initialize registers
                "movdqa 0(%%r9), %%xmm0;"
                "movdqa 16(%%r9), %%xmm1;"
                "movdqa 32(%%r9), %%xmm2;"
                "movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm4;"
                "movdqa 80(%%r9), %%xmm5;"
                "movdqa 96(%%r9), %%xmm6;"
                "movdqa 112(%%r9), %%xmm7;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_and_pi_8:"
                "pand (%%r9), %%xmm0;pand 16(%%r9), %%xmm1;pand 32(%%r9), %%xmm2;pand 48(%%r9), %%xmm3;"
                "pand 64(%%r9), %%xmm4;pand 80(%%r9), %%xmm5;pand 96(%%r9), %%xmm6;pand 112(%%r9), %%xmm7;"
                "pand 128(%%r9), %%xmm0;pand 144(%%r9), %%xmm1;pand 160(%%r9), %%xmm2;pand 176(%%r9), %%xmm3;"
                "pand 192(%%r9), %%xmm4;pand 208(%%r9), %%xmm5;pand 224(%%r9), %%xmm6;pand 240(%%r9), %%xmm7;"
                "pand 256(%%r9), %%xmm0;pand 272(%%r9), %%xmm1;pand 288(%%r9), %%xmm2;pand 304(%%r9), %%xmm3;"
                "pand 320(%%r9), %%xmm4;pand 336(%%r9), %%xmm5;pand 352(%%r9), %%xmm6;pand 368(%%r9), %%xmm7;"
                "pand 384(%%r9), %%xmm0;pand 400(%%r9), %%xmm1;pand 416(%%r9), %%xmm2;pand 432(%%r9), %%xmm3;"
                "pand 448(%%r9), %%xmm4;pand 464(%%r9), %%xmm5;pand 480(%%r9), %%xmm6;pand 496(%%r9), %%xmm7;"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_and_pi_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_and_pi_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_and_pi_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
        data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif
      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

double asm_work_store(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_store(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif
   
   length=data->length;
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_1;"        //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_1:"            //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_1;"        //|<
                "_sync1_store_1:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_1;"        //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_1;"         //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_1;"        //|<
                "_wait_store_1:"             //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_1;"          //|<
                "_sync2_store_1:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_1;"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_1:"
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
                "jnz _skip_reset_store_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_2;"        //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_2:"            //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_2;"        //|<
                "_sync1_store_2:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_2;"        //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_2;"         //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_2;"        //|<
                "_wait_store_2:"             //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_2;"          //|<
                "_sync2_store_2:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_2;"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_2:"
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
                "jnz _skip_reset_store_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_3;"        //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_3:"            //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_3;"        //|<
                "_sync1_store_3:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_3;"        //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_3;"         //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_3;"        //|<
                "_wait_store_3:"             //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_3;"          //|<
                "_sync2_store_3:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_3;"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_3:"
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
                "jnz _skip_reset_store_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_store_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_4;"        //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_4:"            //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_4;"        //|<
                "_sync1_store_4:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_4;"        //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_4;"         //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_4;"        //|<
                "_wait_store_4:"             //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_4;"          //|<
                "_sync2_store_4:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_4;"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_4:"
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
                "jnz _skip_reset_store_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_8;"        //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_8:"            //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_8;"        //|<
                "_sync1_store_8:"            //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_8;"        //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_8;"         //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_8;"        //|<
                "_wait_store_8:"             //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_8;"          //|<
                "_sync2_store_8:"            //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_8;"        //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_8:"
                "movdqa %%xmm0,(%%r9);movdqa %%xmm1,16(%%r9);movdqa %%xmm2,32(%%r9);movdqa %%xmm3,48(%%r9);"
                "movdqa %%xmm4,64(%%r9);movdqa %%xmm5,80(%%r9);movdqa %%xmm6,96(%%r9);movdqa %%xmm7,112(%%r9);"
                "movdqa %%xmm0,128(%%r9);movdqa %%xmm1,144(%%r9);movdqa %%xmm2,160(%%r9);movdqa %%xmm3,176(%%r9);"
                "movdqa %%xmm4,192(%%r9);movdqa %%xmm5,208(%%r9);movdqa %%xmm6,224(%%r9);movdqa %%xmm7,240(%%r9);"
                "movdqa %%xmm0,256(%%r9);movdqa %%xmm1,272(%%r9);movdqa %%xmm2,288(%%r9);movdqa %%xmm3,304(%%r9);"
                "movdqa %%xmm4,320(%%r9);movdqa %%xmm5,336(%%r9);movdqa %%xmm6,352(%%r9);movdqa %%xmm7,368(%%r9);"
                "movdqa %%xmm0,384(%%r9);movdqa %%xmm1,400(%%r9);movdqa %%xmm2,416(%%r9);movdqa %%xmm3,432(%%r9);"
                "movdqa %%xmm4,448(%%r9);movdqa %%xmm5,464(%%r9);movdqa %%xmm6,480(%%r9);movdqa %%xmm7,496(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_store_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
         data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_store_nt(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_store_nt(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_nt_1;"     //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_nt_1:"         //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_nt_1;"     //|<
                "_sync1_store_nt_1:"         //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_nt_1;"     //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_nt_1;"      //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_nt_1;"     //|<
                "_wait_store_nt_1:"          //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_nt_1;"       //|<
                "_sync2_store_nt_1:"         //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_nt_1;"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_nt_1:"
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
                "jnz _skip_reset_store_nt_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_nt_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_nt_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_nt_2;"     //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_nt_2:"         //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_nt_2;"     //|<
                "_sync1_store_nt_2:"         //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_nt_2;"     //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_nt_2;"      //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_nt_2;"     //|<
                "_wait_store_nt_2:"          //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_nt_2;"       //|<
                "_sync2_store_nt_2:"         //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_nt_2;"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_nt_2:"
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
                "jnz _skip_reset_store_nt_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_nt_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_nt_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_nt_3;"     //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_nt_3:"         //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_nt_3;"     //|<
                "_sync1_store_nt_3:"         //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_nt_3;"     //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_nt_3;"      //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_nt_3;"     //|<
                "_wait_store_nt_3:"          //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_nt_3;"       //|<
                "_sync2_store_nt_3:"         //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_nt_3;"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_nt_3:"
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
                "jnz _skip_reset_store_nt_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_nt_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_store_nt_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_nt_4;"     //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_nt_4:"         //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_nt_4;"     //|<
                "_sync1_store_nt_4:"         //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_nt_4;"     //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_nt_4;"      //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_nt_4;"     //|<
                "_wait_store_nt_4:"          //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_nt_4;"       //|<
                "_sync2_store_nt_4:"         //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_nt_4;"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_nt_4:"
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
                "jnz _skip_reset_store_nt_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_nt_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_nt_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_store_nt_8;"     //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_store_nt_8:"         //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_store_nt_8;"     //|<
                "_sync1_store_nt_8:"         //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_store_nt_8;"     //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_store_nt_8;"      //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_store_nt_8;"     //|<
                "_wait_store_nt_8:"          //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_store_nt_8;"       //|<
                "_sync2_store_nt_8:"         //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_store_nt_8;"     //<<
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_store_nt_8:"
                "movntdq %%xmm0,(%%r9);movntdq %%xmm1,16(%%r9);movntdq %%xmm2,32(%%r9);movntdq %%xmm3,48(%%r9);"
                "movntdq %%xmm4,64(%%r9);movntdq %%xmm5,80(%%r9);movntdq %%xmm6,96(%%r9);movntdq %%xmm7,112(%%r9);"
                "movntdq %%xmm0,128(%%r9);movntdq %%xmm1,144(%%r9);movntdq %%xmm2,160(%%r9);movntdq %%xmm3,176(%%r9);"
                "movntdq %%xmm4,192(%%r9);movntdq %%xmm5,208(%%r9);movntdq %%xmm6,224(%%r9);movntdq %%xmm7,240(%%r9);"
                "movntdq %%xmm0,256(%%r9);movntdq %%xmm1,272(%%r9);movntdq %%xmm2,288(%%r9);movntdq %%xmm3,304(%%r9);"
                "movntdq %%xmm4,320(%%r9);movntdq %%xmm5,336(%%r9);movntdq %%xmm6,352(%%r9);movntdq %%xmm7,368(%%r9);"
                "movntdq %%xmm0,384(%%r9);movntdq %%xmm1,400(%%r9);movntdq %%xmm2,416(%%r9);movntdq %%xmm3,432(%%r9);"
                "movntdq %%xmm4,448(%%r9);movntdq %%xmm5,464(%%r9);movntdq %%xmm6,480(%%r9);movntdq %%xmm7,496(%%r9);"
                "add $512,%%r9;"
                "sub $1,%%r10;"
                "jnz _skip_reset_store_nt_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "_skip_reset_store_nt_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_store_nt_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
         data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_copy(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_copy(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_1;"         //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_1:"             //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_1;"         //|<
                "_sync1_copy_1:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_1;"         //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_1;"          //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_1;"          //|<
                "_wait_copy_1:"              //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_1;"           //|<
                "_sync2_copy_1:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_1;"         //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_1:"
                "movdqa (%%r9), %%xmm0;movdqa %%xmm0,(%%r11);"
                "movdqa 16(%%r9), %%xmm0;movdqa %%xmm0,16(%%r11);"
                "movdqa 32(%%r9), %%xmm0;movdqa %%xmm0,32(%%r11);"
                "movdqa 48(%%r9), %%xmm0;movdqa %%xmm0,48(%%r11);"
                "movdqa 64(%%r9), %%xmm0;movdqa %%xmm0,64(%%r11);"
                "movdqa 80(%%r9), %%xmm0;movdqa %%xmm0,80(%%r11);"
                "movdqa 96(%%r9), %%xmm0;movdqa %%xmm0,96(%%r11);"
                "movdqa 112(%%r9), %%xmm0;movdqa %%xmm0,112(%%r11);"
                "movdqa 128(%%r9), %%xmm0;movdqa %%xmm0,128(%%r11);"
                "movdqa 144(%%r9), %%xmm0;movdqa %%xmm0,144(%%r11);"
                "movdqa 160(%%r9), %%xmm0;movdqa %%xmm0,160(%%r11);"
                "movdqa 176(%%r9), %%xmm0;movdqa %%xmm0,176(%%r11);"
                "movdqa 192(%%r9), %%xmm0;movdqa %%xmm0,192(%%r11);"
                "movdqa 208(%%r9), %%xmm0;movdqa %%xmm0,208(%%r11);"
                "movdqa 224(%%r9), %%xmm0;movdqa %%xmm0,224(%%r11);"
                "movdqa 240(%%r9), %%xmm0;movdqa %%xmm0,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_2;"         //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_2:"             //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_2;"         //|<
                "_sync1_copy_2:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_2;"         //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_2;"          //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_2;"         //|<
                "_wait_copy_2:"              //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_2;"           //|<
                "_sync2_copy_2:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_2;"         //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_2:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;"
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);"
                
                "movdqa 32(%%r9), %%xmm0;movdqa 48(%%r9), %%xmm1;"
                "movdqa %%xmm0,32(%%r11);movdqa %%xmm1,48(%%r11);"
                
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;"
                "movdqa %%xmm0,64(%%r11);movdqa %%xmm1,80(%%r11);"
                
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;"
                "movdqa %%xmm0,96(%%r11);movdqa %%xmm1,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;"
                "movdqa %%xmm0,128(%%r11);movdqa %%xmm1,144(%%r11);"
                
                "movdqa 160(%%r9), %%xmm0;movdqa 176(%%r9), %%xmm1;"
                "movdqa %%xmm0,160(%%r11);movdqa %%xmm1,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;"
                "movdqa %%xmm0,192(%%r11);movdqa %%xmm1,208(%%r11);"
                
                "movdqa 224(%%r9), %%xmm0;movdqa 240(%%r9), %%xmm1;"
                "movdqa %%xmm0,224(%%r11);movdqa %%xmm1,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_3;"         //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_3:"             //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_3;"         //|<
                "_sync1_copy_3:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_3;"         //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_3;"          //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_3;"         //|<
                "_wait_copy_3:"              //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_3;"           //|<
                "_sync2_copy_3:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_3;"         //<<
                "mov $384,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_3:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;"
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);movdqa %%xmm2,32(%%r11);"
                
                "movdqa 48(%%r9), %%xmm0;movdqa 64(%%r9), %%xmm1;movdqa 80(%%r9), %%xmm2;"
                "movdqa %%xmm0,48(%%r11);movdqa %%xmm1,64(%%r11);movdqa %%xmm2,80(%%r11);"
                
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;movdqa 128(%%r9), %%xmm2;"
                "movdqa %%xmm0,96(%%r11);movdqa %%xmm1,112(%%r11);movdqa %%xmm2,128(%%r11);"
                
                "movdqa 144(%%r9), %%xmm0;movdqa 160(%%r9), %%xmm1;movdqa 176(%%r9), %%xmm2;"
                "movdqa %%xmm0,144(%%r11);movdqa %%xmm1,160(%%r11);movdqa %%xmm2,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;"
                "movdqa %%xmm0,192(%%r11);movdqa %%xmm1,208(%%r11);movdqa %%xmm2,224(%%r11);"
                
                "movdqa 240(%%r9), %%xmm0;movdqa 256(%%r9), %%xmm1;movdqa 272(%%r9), %%xmm2;"
                "movdqa %%xmm0,240(%%r11);movdqa %%xmm1,256(%%r11);movdqa %%xmm2,272(%%r11);"
                                
                "movdqa 288(%%r9), %%xmm0;movdqa 304(%%r9), %%xmm1;movdqa 320(%%r9), %%xmm2;"
                "movdqa %%xmm0,288(%%r11);movdqa %%xmm1,304(%%r11);movdqa %%xmm2,320(%%r11);"
                
                "movdqa 336(%%r9), %%xmm0;movdqa 352(%%r9), %%xmm1;movdqa 368(%%r9), %%xmm2;"
                "movdqa %%xmm0,336(%%r11);movdqa %%xmm1,352(%%r11);movdqa %%xmm2,368(%%r11);"     
                "add $384,%%r9;"
                "add $384,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $384,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_copy_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_4;"         //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_4:"             //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_4;"         //|<
                "_sync1_copy_4:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_4;"         //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_4;"          //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_4;"         //|<
                "_wait_copy_4:"              //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_4;"           //|<
                "_sync2_copy_4:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_4;"         //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_4:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);movdqa %%xmm2,32(%%r11);movdqa %%xmm3,48(%%r11);"
                
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;"
                "movdqa %%xmm0,64(%%r11);movdqa %%xmm1,80(%%r11);movdqa %%xmm2,96(%%r11);movdqa %%xmm3,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movdqa %%xmm0,128(%%r11);movdqa %%xmm1,144(%%r11);movdqa %%xmm2,160(%%r11);movdqa %%xmm3,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;"
                "movdqa %%xmm0,192(%%r11);movdqa %%xmm1,208(%%r11);movdqa %%xmm2,224(%%r11);movdqa %%xmm3,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_8;"         //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_8:"             //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_8;"         //|<
                "_sync1_copy_8:"             //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_8;"         //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_8;"          //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_8;"         //|<
                "_wait_copy_8:"              //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_8;"           //|<
                "_sync2_copy_8:"             //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_8;"         //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_8:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm4;movdqa 80(%%r9), %%xmm5;movdqa 96(%%r9), %%xmm6;movdqa 112(%%r9), %%xmm7;"
                
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);movdqa %%xmm2,32(%%r11);movdqa %%xmm3,48(%%r11);"
                "movdqa %%xmm4,64(%%r11);movdqa %%xmm5,80(%%r11);movdqa %%xmm6,96(%%r11);movdqa %%xmm7,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movdqa 192(%%r9), %%xmm4;movdqa 208(%%r9), %%xmm5;movdqa 224(%%r9), %%xmm6;movdqa 240(%%r9), %%xmm7;"
                
                "movdqa %%xmm0,128(%%r11);movdqa %%xmm1,144(%%r11);movdqa %%xmm2,160(%%r11);movdqa %%xmm3,176(%%r11);"
                "movdqa %%xmm4,192(%%r11);movdqa %%xmm5,208(%%r11);movdqa %%xmm6,224(%%r11);movdqa %%xmm7,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
         data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_copy_nt(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_copy_nt(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_nt_1;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_nt_1:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_nt_1;"      //|<
                "_sync1_copy_nt_1:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_nt_1;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_nt_1;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_nt_1;"      //|<
                "_wait_copy_nt_1:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_nt_1;"        //|<
                "_sync2_copy_nt_1:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_nt_1;"      //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_nt_1:"
                "movdqa (%%r9), %%xmm0;movntdq %%xmm0,(%%r11);"
                "movdqa 16(%%r9), %%xmm0;movntdq %%xmm0,16(%%r11);"
                "movdqa 32(%%r9), %%xmm0;movntdq %%xmm0,32(%%r11);"
                "movdqa 48(%%r9), %%xmm0;movntdq %%xmm0,48(%%r11);"
                "movdqa 64(%%r9), %%xmm0;movntdq %%xmm0,64(%%r11);"
                "movdqa 80(%%r9), %%xmm0;movntdq %%xmm0,80(%%r11);"
                "movdqa 96(%%r9), %%xmm0;movntdq %%xmm0,96(%%r11);"
                "movdqa 112(%%r9), %%xmm0;movntdq %%xmm0,112(%%r11);"
                "movdqa 128(%%r9), %%xmm0;movntdq %%xmm0,128(%%r11);"
                "movdqa 144(%%r9), %%xmm0;movntdq %%xmm0,144(%%r11);"
                "movdqa 160(%%r9), %%xmm0;movntdq %%xmm0,160(%%r11);"
                "movdqa 176(%%r9), %%xmm0;movntdq %%xmm0,176(%%r11);"
                "movdqa 192(%%r9), %%xmm0;movntdq %%xmm0,192(%%r11);"
                "movdqa 208(%%r9), %%xmm0;movntdq %%xmm0,208(%%r11);"
                "movdqa 224(%%r9), %%xmm0;movntdq %%xmm0,224(%%r11);"
                "movdqa 240(%%r9), %%xmm0;movntdq %%xmm0,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_nt_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_nt_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_nt_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_nt_2;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_nt_2:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_nt_2;"      //|<
                "_sync1_copy_nt_2:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_nt_2;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_nt_2;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_nt_2;"      //|<
                "_wait_copy_nt_2:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_nt_2;"        //|<
                "_sync2_copy_nt_2:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_nt_2;"      //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_nt_2:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;"
                "movntdq %%xmm0,(%%r11);movntdq %%xmm1,16(%%r11);"
                
                "movdqa 32(%%r9), %%xmm0;movdqa 48(%%r9), %%xmm1;"
                "movntdq %%xmm0,32(%%r11);movntdq %%xmm1,48(%%r11);"
                
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;"
                "movntdq %%xmm0,64(%%r11);movntdq %%xmm1,80(%%r11);"
                
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;"
                "movntdq %%xmm0,96(%%r11);movntdq %%xmm1,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;"
                "movntdq %%xmm0,128(%%r11);movntdq %%xmm1,144(%%r11);"
                
                "movdqa 160(%%r9), %%xmm0;movdqa 176(%%r9), %%xmm1;"
                "movntdq %%xmm0,160(%%r11);movntdq %%xmm1,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;"
                "movntdq %%xmm0,192(%%r11);movntdq %%xmm1,208(%%r11);"
                
                "movdqa 224(%%r9), %%xmm0;movdqa 240(%%r9), %%xmm1;"
                "movntdq %%xmm0,224(%%r11);movntdq %%xmm1,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_nt_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_nt_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_nt_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                             
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_nt_3;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_nt_3:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_nt_3;"      //|<
                "_sync1_copy_nt_3:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_nt_3;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_nt_3;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_nt_3;"      //|<
                "_wait_copy_nt_3:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_nt_3;"        //|<
                "_sync2_copy_nt_3:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_nt_3;"      //<<
                "mov $384,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_nt_3:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;"
                "movntdq %%xmm0,(%%r11);movntdq %%xmm1,16(%%r11);movntdq %%xmm2,32(%%r11);"
                
                "movdqa 48(%%r9), %%xmm0;movdqa 64(%%r9), %%xmm1;movdqa 80(%%r9), %%xmm2;"
                "movntdq %%xmm0,48(%%r11);movntdq %%xmm1,64(%%r11);movntdq %%xmm2,80(%%r11);"
                
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;movdqa 128(%%r9), %%xmm2;"
                "movntdq %%xmm0,96(%%r11);movntdq %%xmm1,112(%%r11);movntdq %%xmm2,128(%%r11);"
                
                "movdqa 144(%%r9), %%xmm0;movdqa 160(%%r9), %%xmm1;movdqa 176(%%r9), %%xmm2;"
                "movntdq %%xmm0,144(%%r11);movntdq %%xmm1,160(%%r11);movntdq %%xmm2,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;"
                "movntdq %%xmm0,192(%%r11);movntdq %%xmm1,208(%%r11);movntdq %%xmm2,224(%%r11);"
                
                "movdqa 240(%%r9), %%xmm0;movdqa 256(%%r9), %%xmm1;movdqa 272(%%r9), %%xmm2;"
                "movntdq %%xmm0,240(%%r11);movntdq %%xmm1,256(%%r11);movntdq %%xmm2,272(%%r11);"
                                
                "movdqa 288(%%r9), %%xmm0;movdqa 304(%%r9), %%xmm1;movdqa 320(%%r9), %%xmm2;"
                "movntdq %%xmm0,288(%%r11);movntdq %%xmm1,304(%%r11);movntdq %%xmm2,320(%%r11);"
                
                "movdqa 336(%%r9), %%xmm0;movdqa 352(%%r9), %%xmm1;movdqa 368(%%r9), %%xmm2;"
                "movntdq %%xmm0,336(%%r11);movntdq %%xmm1,352(%%r11);movntdq %%xmm2,368(%%r11);"     
                "add $384,%%r9;"
                "add $384,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_nt_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $384,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_nt_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_copy_nt_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_nt_4;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_nt_4:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_nt_4;"      //|<
                "_sync1_copy_nt_4:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_nt_4;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_nt_4;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_nt_4;"      //|<
                "_wait_copy_nt_4:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_nt_4;"        //|<
                "_sync2_copy_nt_4:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_nt_4;"      //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_nt_4:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movntdq %%xmm0,(%%r11);movntdq %%xmm1,16(%%r11);movntdq %%xmm2,32(%%r11);movntdq %%xmm3,48(%%r11);"
                
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;"
                "movntdq %%xmm0,64(%%r11);movntdq %%xmm1,80(%%r11);movntdq %%xmm2,96(%%r11);movntdq %%xmm3,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movntdq %%xmm0,128(%%r11);movntdq %%xmm1,144(%%r11);movntdq %%xmm2,160(%%r11);movntdq %%xmm3,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;"
                "movntdq %%xmm0,192(%%r11);movntdq %%xmm1,208(%%r11);movntdq %%xmm2,224(%%r11);movntdq %%xmm3,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_nt_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_nt_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_nt_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_copy_nt_8;"      //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_copy_nt_8:"          //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_copy_nt_8;"      //|<
                "_sync1_copy_nt_8:"          //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_copy_nt_8;"      //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_copy_nt_8;"       //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_copy_nt_8;"      //|<
                "_wait_copy_nt_8:"           //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_copy_nt_8;"        //|<
                "_sync2_copy_nt_8:"          //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_copy_nt_8;"      //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_copy_nt_8:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm4;movdqa 80(%%r9), %%xmm5;movdqa 96(%%r9), %%xmm6;movdqa 112(%%r9), %%xmm7;"
                
                "movntdq %%xmm0,(%%r11);movntdq %%xmm1,16(%%r11);movntdq %%xmm2,32(%%r11);movntdq %%xmm3,48(%%r11);"
                "movntdq %%xmm4,64(%%r11);movntdq %%xmm5,80(%%r11);movntdq %%xmm6,96(%%r11);movntdq %%xmm7,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movdqa 192(%%r9), %%xmm4;movdqa 208(%%r9), %%xmm5;movdqa 224(%%r9), %%xmm6;movdqa 240(%%r9), %%xmm7;"
                
                "movntdq %%xmm0,128(%%r11);movntdq %%xmm1,144(%%r11);movntdq %%xmm2,160(%%r11);movntdq %%xmm3,176(%%r11);"
                "movntdq %%xmm4,192(%%r11);movntdq %%xmm5,208(%%r11);movntdq %%xmm6,224(%%r11);movntdq %%xmm7,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_copy_nt_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_copy_nt_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_copy_nt_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
         data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * assembler implementation of bandwidth measurement
 * TODO: implement additional synchronisation for unsynchronized TSCs
 */
double asm_work_scale_int(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data) __attribute__((noinline));
double asm_work_scale_int(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length,unsigned long long call_latency,unsigned long long freq,unsigned long long runing_threads,unsigned long long id, volatile void * sync_ptr,threaddata_t* data)
{
   unsigned long long passes;
   double ret;
   unsigned long long a,b,c,d;
   unsigned long long length;
   int i;
   
   #ifdef USE_PAPI
    if ((!id) && (data->num_events)) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   
   #ifdef UNCORE
    if (data->monitor_uncore)
    {
     for(i=0; i < data->data->outp.pfp_pmd_count; i++) {
        data->pd[i].reg_num   = data->data->outp.pfp_pmds[i].reg_num;
        data->pd[i].reg_value = 0;
     }  
     pfm_write_pmds(data->fd, data->pd, data->data->outp.pfp_pmd_count); 
     pfm_start(data->fd, NULL);
    }
   #endif
   
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_START("L1");
     if (data->region==REGION_L2) VT_USER_START("L2");
     if (data->region==REGION_L3) VT_USER_START("L3");
     if (data->region==REGION_RAM) VT_USER_START("RAM");
    #endif

   length=data->length;
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_scale_int_1;"    //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_scale_int_1:"        //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_scale_int_1;"    //|<
                "_sync1_scale_int_1:"        //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_scale_int_1;"    //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_scale_int_1;"     //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_scale_int_1;"    //|<
                "_wait_scale_int_1:"         //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_scale_int_1;"      //|<
                "_sync2_scale_int_1:"        //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_scale_int_1;"    //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //initialize registers
                "movdqa (%%r9), %%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_scale_int_1:"
                "movdqa (%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,(%%r11);"
                "movdqa 16(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,16(%%r11);"
                "movdqa 32(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,32(%%r11);"
                "movdqa 48(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,48(%%r11);"
                "movdqa 64(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,64(%%r11);"
                "movdqa 80(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,80(%%r11);"
                "movdqa 96(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,96(%%r11);"
                "movdqa 112(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,112(%%r11);"
                "movdqa 128(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,128(%%r11);"
                "movdqa 144(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,144(%%r11);"
                "movdqa 160(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,160(%%r11);"
                "movdqa 176(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,176(%%r11);"
                "movdqa 192(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,192(%%r11);"
                "movdqa 208(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,208(%%r11);"
                "movdqa 224(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,224(%%r11);"
                "movdqa 240(%%r9), %%xmm0;pmuldq %%xmm15,%%xmm0;movdqa %%xmm0,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_scale_int_1;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_scale_int_1:"
                "sub $32,%%r15;"
                "jnz _work_loop_scale_int_1;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm15"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 2:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                                
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_scale_int_2;"    //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_scale_int_2:"        //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_scale_int_2;"    //|<
                "_sync1_scale_int_2:"        //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_scale_int_2;"    //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_scale_int_2;"     //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_scale_int_2;"    //|<
                "_wait_scale_int_2:"         //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_scale_int_2;"      //|<
                "_sync2_scale_int_2:"        //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_scale_int_2;"    //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //initialize registers
                "movdqa (%%r9), %%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_scale_int_2:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);"
                
                "movdqa 32(%%r9), %%xmm0;movdqa 48(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,32(%%r11);movdqa %%xmm1,48(%%r11);"
                
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,64(%%r11);movdqa %%xmm1,80(%%r11);"
                
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,96(%%r11);movdqa %%xmm1,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,128(%%r11);movdqa %%xmm1,144(%%r11);"
                
                "movdqa 160(%%r9), %%xmm0;movdqa 176(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,160(%%r11);movdqa %%xmm1,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,192(%%r11);movdqa %%xmm1,208(%%r11);"
                
                "movdqa 224(%%r9), %%xmm0;movdqa 240(%%r9), %%xmm1;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;"
                "movdqa %%xmm0,224(%%r11);movdqa %%xmm1,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_scale_int_2;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_scale_int_2:"
                "sub $32,%%r15;"
                "jnz _work_loop_scale_int_2;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm15"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 3:
      passes=accesses/48;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_scale_int_3;"    //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_scale_int_3:"        //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_scale_int_3;"    //|<
                "_sync1_scale_int_3:"        //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_scale_int_3;"    //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_scale_int_3;"     //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $10000,%%rax;"          //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_scale_int_3;"    //|<
                "_wait_scale_int_3:"         //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_scale_int_3;"      //|<
                "_sync2_scale_int_3:"        //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_scale_int_3;"    //<<
                "mov $384,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //initialize registers
                "movdqa (%%r9), %%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_scale_int_3:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);movdqa %%xmm2,32(%%r11);"
                
                "movdqa 48(%%r9), %%xmm0;movdqa 64(%%r9), %%xmm1;movdqa 80(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,48(%%r11);movdqa %%xmm1,64(%%r11);movdqa %%xmm2,80(%%r11);"
                
                "movdqa 96(%%r9), %%xmm0;movdqa 112(%%r9), %%xmm1;movdqa 128(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,96(%%r11);movdqa %%xmm1,112(%%r11);movdqa %%xmm2,128(%%r11);"
                
                "movdqa 144(%%r9), %%xmm0;movdqa 160(%%r9), %%xmm1;movdqa 176(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,144(%%r11);movdqa %%xmm1,160(%%r11);movdqa %%xmm2,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,192(%%r11);movdqa %%xmm1,208(%%r11);movdqa %%xmm2,224(%%r11);"
                
                "movdqa 240(%%r9), %%xmm0;movdqa 256(%%r9), %%xmm1;movdqa 272(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,240(%%r11);movdqa %%xmm1,256(%%r11);movdqa %%xmm2,272(%%r11);"
                                
                "movdqa 288(%%r9), %%xmm0;movdqa 304(%%r9), %%xmm1;movdqa 320(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,288(%%r11);movdqa %%xmm1,304(%%r11);movdqa %%xmm2,320(%%r11);"
                
                "movdqa 336(%%r9), %%xmm0;movdqa 352(%%r9), %%xmm1;movdqa 368(%%r9), %%xmm2;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;"
                "movdqa %%xmm0,336(%%r11);movdqa %%xmm1,352(%%r11);movdqa %%xmm2,368(%%r11);"     
                "add $384,%%r9;"
                "add $384,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_scale_int_3;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $384,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_scale_int_3:"
                "sub $48,%%r15;"
                "jnz _work_loop_scale_int_3;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm15"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 4:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                              
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_scale_int_4;"    //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_scale_int_4:"        //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_scale_int_4;"    //|<
                "_sync1_scale_int_4:"        //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_scale_int_4;"    //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_scale_int_4;"     //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_scale_int_4;"    //|<
                "_wait_scale_int_4:"         //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_scale_int_4;"      //|<
                "_sync2_scale_int_4:"        //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_scale_int_4;"    //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //initialize registers
                "movdqa (%%r9), %%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_scale_int_4:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;pmuldq %%xmm15,%%xmm3;"
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);movdqa %%xmm2,32(%%r11);movdqa %%xmm3,48(%%r11);"
                
                "movdqa 64(%%r9), %%xmm0;movdqa 80(%%r9), %%xmm1;movdqa 96(%%r9), %%xmm2;movdqa 112(%%r9), %%xmm3;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;pmuldq %%xmm15,%%xmm3;"
                "movdqa %%xmm0,64(%%r11);movdqa %%xmm1,80(%%r11);movdqa %%xmm2,96(%%r11);movdqa %%xmm3,112(%%r11);"
                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;pmuldq %%xmm15,%%xmm3;"
                "movdqa %%xmm0,128(%%r11);movdqa %%xmm1,144(%%r11);movdqa %%xmm2,160(%%r11);movdqa %%xmm3,176(%%r11);"
                
                "movdqa 192(%%r9), %%xmm0;movdqa 208(%%r9), %%xmm1;movdqa 224(%%r9), %%xmm2;movdqa 240(%%r9), %%xmm3;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;pmuldq %%xmm15,%%xmm3;"
                "movdqa %%xmm0,192(%%r11);movdqa %%xmm1,208(%%r11);movdqa %%xmm2,224(%%r11);movdqa %%xmm3,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_scale_int_4;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_scale_int_4:"
                "sub $32,%%r15;"
                "jnz _work_loop_scale_int_4;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm15"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
    case 8:
    default:
      passes=accesses/32;
      assert(accesses<length);
      if (!passes) return 0;   
      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                "mfence;"
                "push %9;"
                "push %8;"
                "pop %%r8;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "mov %%r9,%%r14;"
                "pop %%r15;"                                               
                 //sync
                "mov %%r12,%%rbx;"
                "add $1,%%rbx;"              //Synchronisation >>
                "cmp $0,%%r12;"              //|master thread resets start time >
                "jne _sync0_scale_int_8;"    //|
                "mov $0,%%r13;"              //|
                "mov %%r13,8(%%r8);"         //|
                "mfence;"                    //|<
                "_sync0_scale_int_8:"        //|atomically replace thread_id with thread_id+1 >
                  "mov %%r12,%%rax;"         //|
                  "lock cmpxchg %%bl,(%%r8);"//|
                "jnz _sync0_scale_int_8;"    //|<
                "_sync1_scale_int_8:"        //|wait untill all threads completed their cmpxchg >
                  "cmp %%r11,(%%r8);"        //|
                "jne _sync1_scale_int_8;"    //|<
                "cmp $0,%%r12;"              //|master thread selects start time >
                "jne _wait_scale_int_8;"     //|
                "rdtsc;"                     //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                "add $100,%%rax;"            //|
                "mov %%rax,8(%%r8);"         //|
                "mov %%rax,%%r13;"           //|
                "mfence;"                    //|
                "jmp _sync2_scale_int_8;"    //|<
                "_wait_scale_int_8:"         //|other threads wait until start time is selected  >
                  "mov 8(%%r8),%%r13;"       //|
                  "cmp $0,%%r13;"            //|
                "je _wait_scale_int_8;"      //|<
                "_sync2_scale_int_8:"        //|all threads wait until starting time is reached >
                  "rdtsc;"                   //|
                "shl $32,%%rdx;"             //|
                "add %%rdx,%%rax;"           //|
                  "cmp %%rax,%%r13;"         //|<
                "jge _sync2_scale_int_8;"    //<<
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                //initialize registers
                "movdqa (%%r9), %%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "push %%rax;"
                "mov %%r10,%%r8;"
                "mfence;"
                ".align 64;"
                "_work_loop_scale_int_8:"
                "movdqa (%%r9), %%xmm0;movdqa 16(%%r9), %%xmm1;movdqa 32(%%r9), %%xmm2;movdqa 48(%%r9), %%xmm3;"
                "movdqa 64(%%r9), %%xmm4;movdqa 80(%%r9), %%xmm5;movdqa 96(%%r9), %%xmm6;movdqa 112(%%r9), %%xmm7;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;pmuldq %%xmm15,%%xmm3;"
                "pmuldq %%xmm15,%%xmm4;pmuldq %%xmm15,%%xmm5;pmuldq %%xmm15,%%xmm6;pmuldq %%xmm15,%%xmm7;"
                "movdqa %%xmm0,(%%r11);movdqa %%xmm1,16(%%r11);movdqa %%xmm2,32(%%r11);movdqa %%xmm3,48(%%r11);"
                "movdqa %%xmm4,64(%%r11);movdqa %%xmm5,80(%%r11);movdqa %%xmm6,96(%%r11);movdqa %%xmm7,112(%%r11);"
                                
                "movdqa 128(%%r9), %%xmm0;movdqa 144(%%r9), %%xmm1;movdqa 160(%%r9), %%xmm2;movdqa 176(%%r9), %%xmm3;"
                "movdqa 192(%%r9), %%xmm4;movdqa 208(%%r9), %%xmm5;movdqa 224(%%r9), %%xmm6;movdqa 240(%%r9), %%xmm7;"
                "pmuldq %%xmm15,%%xmm0;pmuldq %%xmm15,%%xmm1;pmuldq %%xmm15,%%xmm2;pmuldq %%xmm15,%%xmm3;"
                "pmuldq %%xmm15,%%xmm4;pmuldq %%xmm15,%%xmm5;pmuldq %%xmm15,%%xmm6;pmuldq %%xmm15,%%xmm7;"
                "movdqa %%xmm0,128(%%r11);movdqa %%xmm1,144(%%r11);movdqa %%xmm2,160(%%r11);movdqa %%xmm3,176(%%r11);"                
                "movdqa %%xmm4,192(%%r11);movdqa %%xmm5,208(%%r11);movdqa %%xmm6,224(%%r11);movdqa %%xmm7,240(%%r11);"
                "add $256,%%r9;"
                "add $256,%%r11;"
                "sub $1,%%r10;"
                "jnz _skip_reset_scale_int_8;"
                "mov %%r14,%%r9;"
                "mov %%r8,%%r10;"
                "mov $256,%%rax;"
                "mul %%r10;"
                "mov %%rax,%%r11;"
                "add %%r9,%%r11;"
                "_skip_reset_scale_int_8:"
                "sub $32,%%r15;"
                "jnz _work_loop_scale_int_8;"
                "mfence;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "pop %%rbx;"
                : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
                : "a"(addr), "b" (passes), "c" (runing_threads), "d" (id), "r" (sync_ptr), "r" (length)
                : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm15"
								);
        data->start_ts=b;
        data->end_ts=a;
        ret=(((double)(length*16))/((double)(((a-b)-call_latency))/freq))*0.000000001;
        break;
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
      
         data->papi_results[i]=(double)data->values[i]/(double)(length);
          #ifdef USE_VTRACE
             VT_COUNT_DOUBLE_VAL(data->data->cid_papi[i], data->papi_results[i]);
          #endif

      }
      __asm__ __volatile__("mfence;");
    }
    else for (i=0;i<data->num_events;i++) data->papi_results[i]==(double)0;
  #endif	
	
   #ifdef UNCORE
   if (data->monitor_uncore)
   {
    pfm_stop(data->fd);
    if (pfm_read_pmds(data->fd, data->pd, data->data->inp.pfp_event_count) == -1) {
      fprintf(stderr, "Thread %i: pfm_read_pmds failed\n",data->cpu_id);
      perror("");
      exit(1);
    }
    else
    {
      //printf("Thread %i: %i\n",data->cpu_id,data->pd[0].reg_value);
      #ifdef USE_VTRACE
      for (i=0;i<data->data->pfmon_num_events;i++)
      {
        VT_COUNT_DOUBLE_VAL(data->data->cid_pfm[i], (double) data->pd[i].reg_value);
      }
      #endif
    }
   }
   #endif	
	
    #ifdef USE_VTRACE
     if (data->region==REGION_L1) VT_USER_END("L1");
     if (data->region==REGION_L2) VT_USER_END("L2");
     if (data->region==REGION_L3) VT_USER_END("L3");
     if (data->region==REGION_RAM) VT_USER_END("RAM");
    #endif	
	
	return ret;
}

/*
 * function that does the measurement
 */
void inline _work( unsigned long long memsize, int offset, int function, int burst_length, volatile mydata_t* data, double **results)
{
  int latency,i,j,t;
  double tmax;
  double tmp=(double)0;
  unsigned long long tmp2,tmp3;
  unsigned long long total_cyc;
 
  #ifdef USE_VTRACE
  unsigned int cid_bw, gid_bw,cid_ms, gid_ms;
  gid_bw = VT_COUNT_GROUP_DEF("bandwidth");
  cid_bw = VT_COUNT_DEF("accumulated bandwidth", "MB/s", VT_COUNT_TYPE_DOUBLE, gid_bw);
  gid_ms = VT_COUNT_GROUP_DEF("memsize");
  cid_ms = VT_COUNT_DEF("total memsize", "MB", VT_COUNT_TYPE_DOUBLE, gid_ms);
  #endif
  
	/* aligned address */
	unsigned long long aligned_addr,accesses;
	
  aligned_addr=(unsigned long long)(data->buffer) + offset;
 
  data->ack=0;

  accesses=memsize/(2*sizeof(unsigned long long));
  accesses=(accesses>>5)*32;

 // printf("starting measurment %i accesses in %i Bytes of memory\n",accesses,memsize);
   t=data->num_threads-1;
   tmax=0;

  
   if (((accesses/32))>2*(t+1)) 
   {
    data->running_threads=t+1;
    #ifdef USE_VTRACE
    VT_USER_START("COM");
    #endif
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

     data->synch[0]=0;
     data->synch[1]=0;
    
     //tell other threads to start
     if (t)
     {
       for (j=0;j<t;j++)
       {
         //printf("start %i\n",j+1);fflush(stdout);
         data->thread_comm[j+1]=THREAD_WORK;
         while (!data->ack);
         //printf("started %i\n",data->ack);fflush(stdout);
         data->ack=0;
       }
     }      

      data->threaddata[0].region=REGION_RAM;
      if ((data->cpuinfo->Cachelevels>=1)&&((memsize/(t+1))<=((data->cpuinfo->D_Cache_Size[0]+data->cpuinfo->U_Cache_Size[0])/min(data->cpuinfo->Cache_shared[0],data->threads_per_package[data->threaddata[0].package])))) data->threaddata[0].region=REGION_L1;
      else if ((data->cpuinfo->Cachelevels>=2)&&((memsize/(t+1))<=((data->cpuinfo->D_Cache_Size[1]+data->cpuinfo->U_Cache_Size[1])/min(data->cpuinfo->Cache_shared[1],data->threads_per_package[data->threaddata[0].package])))) data->threaddata[0].region=REGION_L2;
      else if ((data->cpuinfo->Cachelevels>=3)&&((memsize/(t+1))<=((data->cpuinfo->D_Cache_Size[2]+data->cpuinfo->U_Cache_Size[2])/min(data->cpuinfo->Cache_shared[2],data->threads_per_package[data->threaddata[0].package])))) data->threaddata[0].region=REGION_L3;

     #ifdef USE_VTRACE
     VT_USER_END("COM");
     #endif
      //printf("Thread 0, address: %lu\n",aligned_addr);
      
      /* call ASM implementation */
      //printf("call asm impl. latency: %i cycles\n",latency);
      
      //#ifdef UNCORE
      //pfm_start(data->threaddata[0].fd,NULL);
      //#endif
      
      //printf("start 0 \n",j+1);fflush(stdout);
      switch(function)
      {
        case USE_LOAD_PI: tmp=asm_work_load_pi(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_LOAD_PD: tmp=asm_work_load_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_LOAD_PS: tmp=asm_work_load_ps(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_STORE: tmp=asm_work_store(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_STORE_NT: tmp=asm_work_store_nt(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_COPY: tmp=asm_work_copy(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_COPY_NT: tmp=asm_work_copy_nt(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_SCALE_INT: tmp=asm_work_scale_int(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;        
        case USE_MUL_PI: tmp=asm_work_mul_pi(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_ADD_PI: tmp=asm_work_add_pi(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MUL_PD: tmp=asm_work_mul_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_ADD_PD: tmp=asm_work_add_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MUL_PS: tmp=asm_work_mul_ps(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_ADD_PS: tmp=asm_work_add_ps(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MUL_SD: tmp=asm_work_mul_sd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_ADD_SD: tmp=asm_work_add_sd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MUL_SS: tmp=asm_work_mul_ss(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_ADD_SS: tmp=asm_work_add_ss(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_DIV_PD: tmp=asm_work_div_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_DIV_PS: tmp=asm_work_div_ps(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_DIV_SD: tmp=asm_work_div_sd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_DIV_SS: tmp=asm_work_div_ss(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_SQRT_PD: tmp=asm_work_sqrt_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_SQRT_PS: tmp=asm_work_sqrt_ps(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_SQRT_SD: tmp=asm_work_sqrt_sd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_SQRT_SS: tmp=asm_work_sqrt_ss(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_AND_PD: tmp=asm_work_and_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_AND_PI: tmp=asm_work_and_pi(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MUL_PLUS_ADD_PD: tmp=asm_work_mul_plus_add_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        case USE_MUL_ADD_PD: tmp=asm_work_mul_add_pd(aligned_addr,((accesses/(t+1))/32)*32,burst_length,latency,data->cpuinfo->clockrate,data->running_threads,0,(&(data->synch[0])),&(data->threaddata[0]));break;
        default:
         fprintf(stderr,"Error: unknown function %i\n",function);
         exit(1);
      }
      //printf("stop 0 \n",j+1);fflush(stdout);
      
      tmp2=data->threaddata[0].start_ts;
      tmp3=data->threaddata[0].end_ts;
      
     // printf (":id %i,%llu - %llu : %llu\n",0,tmp2,tmp3,tmp3-tmp2);
     #ifdef USE_VTRACE
     VT_USER_START("WAIT"); 
     #endif
     //wait for other THREADS
     if (t)
     {
       for (j=0;j<t;j++)
       {
         //printf("wait %i\n",j+1);fflush(stdout);
         data->thread_comm[j+1]=THREAD_WAIT;       
         while (!data->ack);
         data->ack=0;
         //printf("arrived\n");fflush(stdout);

         if (data->threaddata[j+1].start_ts<tmp2) tmp2=data->threaddata[j+1].start_ts;       
         if (data->threaddata[j+1].end_ts>tmp3) tmp3=data->threaddata[j+1].end_ts; 
         //printf (":id %i,%llu - %llu : %llu\n",j+1,data->threaddata[j+1].start_ts,data->threaddata[j+1].end_ts,data->threaddata[j+1].end_ts-data->threaddata[j+1].start_ts);      
       }
     }
     #ifdef USE_VTRACE
     VT_USER_END("WAIT");
     #endif
     
     //#ifdef UNCORE
     //pfm_stop(data->threaddata[0].fd);
     //#endif

     //printf ("%llu - %llu : %llu\n\n",tmp2,tmp3,tmp3-tmp2);
     tmp2=(tmp3-tmp2)-latency;

      
     if ((int)tmp!=-1)
      {
       tmp=((((double)(data->threaddata[0].length*(t+1)*16)))/ ((double)(tmp2)/data->cpuinfo->clockrate)) *0.000000001;
      
       if (tmax==0)  tmax=tmp;
       if (tmp>tmax) tmax=tmp;
      }
     
    //printf("%llu - %llu\n",total_cyc,data->cpuinfo->clockrate);  
    #ifdef USE_VTRACE
    VT_COUNT_DOUBLE_VAL(cid_ms, (double) memsize);
    VT_COUNT_DOUBLE_VAL(cid_bw, (double) tmax );
    #endif
   }
   else tmax=0;
  
   if (tmax)
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

/*
 * loop for additional worker threads
 * communicating with master thread using shared variables
 */
 void *thread(void *threaddata)
{
  int id= ((threaddata_t *) threaddata)->thread_id;
  volatile mydata_t* global_data = ((threaddata_t *) threaddata)->data; //communication
  threaddata_t* mydata = (threaddata_t*)threaddata;
  char* filename=NULL;

  struct timespec wait_ns;
  int i,j,k,fd;
  double tmp=(double)0;
  unsigned long long tmp2,tmp3,old=THREAD_STOP;
  
  wait_ns.tv_sec=0;
  wait_ns.tv_nsec=100000;
  
  do
  {
   old=global_data->thread_comm[id];
  }
  while (old!=THREAD_INIT);
  global_data->ack=id;

  cpu_set(((threaddata_t *) threaddata)->cpu_id);

  #ifdef USE_VTRACE
   VT_USER_START("INIT");
  #endif  

  
  if(mydata->buffersize)
  {
    if (global_data->hugepages==HUGEPAGES_OFF) mydata->buffer = _mm_malloc( mydata->buffersize,mydata->alignment);
    if (global_data->hugepages==HUGEPAGES_ON)
    {
      char *dir;
      dir=bi_getenv("BENCHIT_KERNEL_HUGEPAGE_DIR",0);
      filename=(char*)malloc((strlen(dir)+20)*sizeof(char));
      sprintf(filename,"%s/thread_data_%i",dir,id);
      mydata->buffer=NULL;
      fd=open(filename,O_CREAT|O_RDWR,0664);
      if (fd == -1)
      {
        fprintf( stderr, "Allocation of buffer failed\n" ); fflush( stderr );
        perror("open");
        exit( 127 );
      } 
      mydata->buffer=(char*) mmap(NULL,mydata->buffersize,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
      close(fd);unlink(filename);
    } 
    
  //fill buffer
  switch (mydata->FUNCTION)
  {
   /*case USE_MUL_PI:
   case USE_ADD_PI:
   tmp2=8*mydata->BURST_LENGTH*sizeof(int);
   for (i=0;i<=mydata->buffersize-tmp2;i+=tmp2)
   {
      for(j=0;j<4*mydata->BURST_LENGTH;j++)
        *((int*)((unsigned long long)mydata->buffer+i+j*sizeof(int)))=(int)global_data->INT_INIT[0];
      for(j=4*mydata->BURST_LENGTH;j<4*mydata->BURST_LENGTH;j++)
        *((int*)((unsigned long long)mydata->buffer+i+j*sizeof(int)))=(int)global_data->INT_INIT[1];
   }
   break;*/
     
     
   case USE_LOAD_PI:
   case USE_STORE:
   case USE_STORE_NT:
   case USE_COPY:
   case USE_COPY_NT:
   case USE_SCALE_INT:
   case USE_MUL_PI:
   case USE_ADD_PI:
   case USE_MUL_SI:
   case USE_ADD_SI:
   case USE_AND_PI:

   tmp2=4*mydata->BURST_LENGTH*sizeof(long long);
   for (i=0;i<=mydata->buffersize-tmp2;i+=tmp2)
   {
      for(j=0;j<2*mydata->BURST_LENGTH;j++)
        *((long long*)((unsigned long long)mydata->buffer+i+j*sizeof(long long)))=global_data->INT_INIT[0];
      for(j=2*mydata->BURST_LENGTH;j<4*mydata->BURST_LENGTH;j++)
        *((long long*)((unsigned long long)mydata->buffer+i+j*sizeof(long long)))=global_data->INT_INIT[1];
   }
   break;
   
   case USE_LOAD_PD:
   case USE_MUL_PD:
   case USE_ADD_PD:
   case USE_MUL_SD:
   case USE_ADD_SD:
   case USE_DIV_PD:
   case USE_DIV_SD:
   case USE_SQRT_PD:
   case USE_SQRT_SD:
   case USE_AND_PD:
   case USE_MUL_ADD_PD:
   case USE_MUL_PLUS_ADD_PD:
   
   /* avoid FP overflows:
      create x, -1/x, x, -1/x, -x, 1/x, -x, 1/x pattern to guarantee stable register values for add, mul, and mul+add 
      (i.e sum = 0, product = 1, and sum of all partial products = 0) */
   tmp2=16*mydata->BURST_LENGTH*sizeof(double);
   for (i=0;i<=mydata->buffersize-tmp2;i+=tmp2)
   {
      for(j=0;j<2*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=global_data->FP_INIT[0];
      for(j=2*mydata->BURST_LENGTH;j<4*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=-1.0*global_data->FP_INIT[1];
      for(j=4*mydata->BURST_LENGTH;j<6*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=global_data->FP_INIT[0];
      for(j=6*mydata->BURST_LENGTH;j<8*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=-1.0*global_data->FP_INIT[1];
      for(j=8*mydata->BURST_LENGTH;j<10*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=-1.0*global_data->FP_INIT[0];
      for(j=10*mydata->BURST_LENGTH;j<12*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=global_data->FP_INIT[1];
      for(j=12*mydata->BURST_LENGTH;j<14*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=-1.0*global_data->FP_INIT[0];
      for(j=14*mydata->BURST_LENGTH;j<16*mydata->BURST_LENGTH;j++)
        *((double*)((unsigned long long)mydata->buffer+i+j*sizeof(double)))=global_data->FP_INIT[1];
   }
   break;
   
   case USE_LOAD_PS:
   case USE_MUL_PS:
   case USE_ADD_PS:
   case USE_MUL_SS:
   case USE_ADD_SS:
   case USE_DIV_PS:
   case USE_DIV_SS:
   case USE_SQRT_PS:
   case USE_SQRT_SS:
   
   /* avoid FP overflows:
      create x, -1/x, x, -1/x, -x, 1/x, -x, 1/x pattern to guarantee stable register values for add, mul, and mul+add 
      (i.e sum = 0, product = 1, and sum of all partial products = 0) */
   tmp2=32*mydata->BURST_LENGTH*sizeof(float);
   for (i=0;i<=mydata->buffersize-tmp2;i+=tmp2)
   {
      for(j=0;j<4*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=(float)global_data->FP_INIT[0];
      for(j=4*mydata->BURST_LENGTH;j<8*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=-1.0*(float)global_data->FP_INIT[1];
      for(j=8*mydata->BURST_LENGTH;j<12*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=(float)global_data->FP_INIT[0];
      for(j=12*mydata->BURST_LENGTH;j<16*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=-1.0*(float)global_data->FP_INIT[1];
      for(j=16*mydata->BURST_LENGTH;j<20*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=-1.0*(float)global_data->FP_INIT[0];
      for(j=20*mydata->BURST_LENGTH;j<24*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=(float)global_data->FP_INIT[1];
      for(j=24*mydata->BURST_LENGTH;j<28*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=-1.0*(float)global_data->FP_INIT[0];
      for(j=28*mydata->BURST_LENGTH;j<32*mydata->BURST_LENGTH;j++)
        *((float*)((unsigned long long)mydata->buffer+i+j*sizeof(float)))=(float)global_data->FP_INIT[1];
   }
   break;
   
   default:
    fprintf( stderr, "Error: initialization failed: unknown mode:%i \n",mydata->FUNCTION );      
    pthread_exit( NULL );
  }
    //clflush(mydata->buffer,mydata->buffersize,*(mydata->cpuinfo));
    mydata->aligned_addr=(unsigned long long)(mydata->buffer) + mydata->offset+ id*mydata->thread_offset;
  }  
  
  #ifdef UNCORE
  mydata->fd=global_data->threaddata[0].fd;
  if (mydata->monitor_uncore)
  {
     mydata->ctx.ctx_flags=PFM_FL_SYSTEM_WIDE;
     mydata->fd = pfm_create_context(&mydata->ctx, NULL, NULL, 0);
     if (mydata->fd == -1) {
        printf("Thread %i: pfm_create_context failed\n",mydata->cpu_id);
        fflush(stdout);perror("");fflush(stderr);
        exit(-1);
     }
     /*if (pfm_dispatch_events(&global_data->inp, &global_data->mod_inp, &global_data->outp, NULL) != PFMLIB_SUCCESS) {
        printf("Thread %i: cannot dispatch events\n",mydata->cpu_id);
        fflush(stdout);perror("");fflush(stderr);
        exit(-1);
     }*/
     
     for(i=0; i < global_data->outp.pfp_pmc_count; i++) {
        mydata->pc[i].reg_num   = global_data->outp.pfp_pmcs[i].reg_num;
        mydata->pc[i].reg_value = global_data->outp.pfp_pmcs[i].reg_value;
     }
     for(i=0; i < global_data->outp.pfp_pmd_count; i++) {
        mydata->pd[i].reg_num   = global_data->outp.pfp_pmds[i].reg_num;
        mydata->pd[i].reg_value = 0;
     }   

     if (pfm_write_pmcs(mydata->fd,mydata->pc, global_data->outp.pfp_pmc_count) == -1) {
        printf("Thread %i: pfm_write_pmcs failed\n",mydata->cpu_id);
        fflush(stdout);perror("");fflush(stderr);
        exit(1); 
     }

     if (pfm_write_pmds(mydata->fd, mydata->pd, global_data->outp.pfp_pmd_count) == -1) {
        printf("Thread %i: pfm_write_pmds failed\n",mydata->cpu_id);
        fflush(stdout);perror("");fflush(stderr);
        exit(1);
     }

     mydata->load_arg.load_pid = mydata->cpu_id;
     if (pfm_load_context(mydata->fd, &(mydata->load_arg)) == -1) {
        printf("Thread %i: pfm_load_context failed\n",mydata->cpu_id);
        fflush(stdout);perror("");fflush(stderr);
        exit(1);
     }
   }
   #endif
  #ifdef USE_VTRACE
   VT_USER_END("INIT");
  #endif
  #ifdef USE_VTRACE
   VT_USER_START("IDLE");
  #endif
  while(1)
  {
     //printf("Thread %i: comm= %i\n",id+1,data->thread_comm[id]);
     switch (global_data->thread_comm[id]){
       case THREAD_USE_MEMORY: 
         if (old!=THREAD_USE_MEMORY)
         {
           if ((old==THREAD_WAIT)||(old==THREAD_WORK)||(old==THREAD_INIT))
           {
             #ifdef USE_VTRACE
             VT_USER_END("IDLE");
             #endif
           }
           old=THREAD_USE_MEMORY;
           global_data->ack=id;
           
           if (!mydata->buffersize)
           {
             mydata->buffer = (char*) (((unsigned long long)global_data->buffer+((id)*mydata->memsize)+mydata->alignment)&((mydata->alignment-1)^0xffffffffffffffffULL));
             mydata->aligned_addr = (unsigned long long)(mydata->buffer) + mydata->offset;
           }
           
           // use memory
           use_memory((void*)mydata->aligned_addr,mydata->memsize,mydata->USE_MODE,mydata->USE_DIRECTION,mydata->NUM_USES,*(mydata->cpuinfo));
 
           global_data->done=id;
         }
         else 
         {
           tmp=100;while(tmp>0) tmp--; 
         }        
         break;
       case THREAD_WAIT: // waiting
          if (old!=THREAD_WAIT)
          {
            old=THREAD_WAIT;
            global_data->ack=id;
            #ifdef USE_VTRACE
            if (old!=THREAD_INIT) VT_USER_START("IDLE");
            #endif
          }
          tmp=100;while(tmp) tmp--;
          break;
       case THREAD_WORK:
          if (old!=THREAD_WORK) 
          {
           if ((old==THREAD_WAIT)||(old==THREAD_INIT))
           {
             #ifdef USE_VTRACE
             VT_USER_END("IDLE");
             #endif
           }
            old=THREAD_WORK;
            global_data->ack=id;
            
            if (!mydata->buffersize)
            {
             mydata->buffer = (char*) (((unsigned long long)global_data->buffer+((id)*mydata->memsize)+mydata->alignment)&((mydata->alignment-1)^0xffffffffffffffffULL));
             mydata->aligned_addr = (unsigned long long)(mydata->buffer) + mydata->offset;
            }
            //printf("Thread %i, address: %lu\n",id,mydata->aligned_addr);

            mydata->region=REGION_RAM;
            if ((global_data->cpuinfo->Cachelevels>=1)&&((mydata->memsize)<=((global_data->cpuinfo->D_Cache_Size[0]+global_data->cpuinfo->U_Cache_Size[0])/min(global_data->cpuinfo->Cache_shared[0],global_data->threads_per_package[mydata->package])))) mydata->region=REGION_L1;            
            else if ((global_data->cpuinfo->Cachelevels>=2)&&((mydata->memsize)<=((global_data->cpuinfo->D_Cache_Size[1]+global_data->cpuinfo->U_Cache_Size[1])/min(global_data->cpuinfo->Cache_shared[1],global_data->threads_per_package[mydata->package])))) mydata->region=REGION_L2;
            else if ((global_data->cpuinfo->Cachelevels>=3)&&((mydata->memsize)<=((global_data->cpuinfo->D_Cache_Size[2]+global_data->cpuinfo->U_Cache_Size[2])/min(global_data->cpuinfo->Cache_shared[2],global_data->threads_per_package[mydata->package])))) mydata->region=REGION_L3;

            /* call ASM implementation */
            switch (mydata->FUNCTION)
            {
               case USE_LOAD_PI: tmp=asm_work_load_pi(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_LOAD_PD: tmp=asm_work_load_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_LOAD_PS: tmp=asm_work_load_ps(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_STORE: tmp=asm_work_store(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                             
               case USE_STORE_NT: tmp=asm_work_store_nt(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;               
               case USE_COPY: tmp=asm_work_copy(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                             
               case USE_COPY_NT: tmp=asm_work_copy_nt(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_SCALE_INT: tmp=asm_work_scale_int(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                             
               case USE_ADD_PI: tmp=asm_work_add_pi(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_MUL_PI: tmp=asm_work_mul_pi(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_ADD_PD: tmp=asm_work_add_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_MUL_PD: tmp=asm_work_mul_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_ADD_PS: tmp=asm_work_add_ps(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_MUL_PS: tmp=asm_work_mul_ps(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_ADD_SD: tmp=asm_work_add_sd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_MUL_SD: tmp=asm_work_mul_sd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_ADD_SS: tmp=asm_work_add_ss(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;
               case USE_MUL_SS: tmp=asm_work_mul_ss(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_DIV_PD: tmp=asm_work_div_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_DIV_PS: tmp=asm_work_div_ps(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_DIV_SD: tmp=asm_work_div_sd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_DIV_SS: tmp=asm_work_div_ss(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_SQRT_PD: tmp=asm_work_sqrt_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_SQRT_PS: tmp=asm_work_sqrt_ps(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_SQRT_SD: tmp=asm_work_sqrt_sd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_SQRT_SS: tmp=asm_work_sqrt_ss(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_AND_PD: tmp=asm_work_and_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_AND_PI: tmp=asm_work_and_pi(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_MUL_ADD_PD: tmp=asm_work_mul_add_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               case USE_MUL_PLUS_ADD_PD: tmp=asm_work_mul_plus_add_pd(mydata->aligned_addr,mydata->accesses,mydata->BURST_LENGTH,0,global_data->cpuinfo->clockrate,global_data->running_threads,id,(&(global_data->synch[0])),mydata);break;                                                            
               
               default:
                 fprintf(stderr,"Error: unknown function %i\n",mydata->FUNCTION);
                 pthread_exit(NULL);
            }
            global_data->done=id;
            #ifdef USE_VTRACE
            VT_USER_START("IDLE");
            #endif
          }
          else 
          {
            tmp=100;while(tmp>0) tmp--;
          }  
          break;
       case THREAD_INIT: // used for parallel initialisation only
          tmp=100;while(tmp) tmp--;
          break;
       case THREAD_STOP: // exit
       default:
         if ((old==THREAD_WAIT)||(old==THREAD_WORK))
           {
             #ifdef USE_VTRACE
             VT_USER_END("IDLE");
             #endif
           }
         if (global_data->hugepages==HUGEPAGES_ON)
         {
           if(mydata->buffer!=NULL) munmap((void*)mydata->buffer,mydata->buffersize);
         }
         pthread_exit(NULL);
    }
  }
}
