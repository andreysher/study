/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/** Kernel: measures write bandwidth of data located in different cache
 *         levels or memory of certain CPUs.
 *******************************************************************/
 
/*
 * TODO - adopt cache and TLB parameters to refer to identifiers returned by 
 *        the hardware detection
 *      - AVX and Larrabee support
 *      - support low level Performance Counter APIs to get access to uncore/NB events#
 *      - local alloc of flush buffer
 *      - memory layout improvements (as for single-r1w1)
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
void inline use_memory(void* buffer,unsigned long long memsize,int mode, int direction,int repeat,cpu_info_t cpuinfo)
{
   int i,j,tmp=0xd08a721b;
   unsigned long long stride = 128;

   for (i=cpuinfo.Cachelevels;i>0;i--)
   {
     if (cpuinfo.Cacheline_size[i-1]<stride) stride=cpuinfo.Cacheline_size[i-1];
   }

   if ((mode==MODE_MODIFIED)||(mode==MODE_EXCLUSIVE)||(mode==MODE_INVALID))
   {
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
   }
   //now buffer is invalid in other caches, modified in local cache
   if ((mode==MODE_EXCLUSIVE)||(mode==MODE_SHARED)||(mode==MODE_OWNED)||(mode==MODE_FORWARD)) 
   {
     if (mode==MODE_EXCLUSIVE) 
     {
      clflush(buffer,memsize,cpuinfo);
      //now buffer is invalid in local cache
     }
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
static double asm_work_movdqa(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data) __attribute__((noinline)); 
static double asm_work_movdqa(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data)
{
   unsigned long long passes;
   double ret;
   int i;
   
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   
   #ifdef USE_PAPI
    if (data->num_events) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                //"mov $0, %%rax;"
                //"cpuid;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_1:"
                "movdqa %%xmm0,(%%r10);"
                "movdqa %%xmm0,16(%%r10);"
                "movdqa %%xmm0,32(%%r10);"
                "movdqa %%xmm0,48(%%r10);"
                "movdqa %%xmm0,64(%%r10);"
                "movdqa %%xmm0,80(%%r10);"
                "movdqa %%xmm0,96(%%r10);"
                "movdqa %%xmm0,112(%%r10);"
                "movdqa %%xmm0,128(%%r10);"
                "movdqa %%xmm0,144(%%r10);"
                "movdqa %%xmm0,160(%%r10);"
                "movdqa %%xmm0,176(%%r10);"
                "movdqa %%xmm0,192(%%r10);"
                "movdqa %%xmm0,208(%%r10);"
                "movdqa %%xmm0,224(%%r10);"
                "movdqa %%xmm0,240(%%r10);"
                "movdqa %%xmm0,256(%%r10);"
                "movdqa %%xmm0,272(%%r10);"
                "movdqa %%xmm0,288(%%r10);"
                "movdqa %%xmm0,304(%%r10);"
                "movdqa %%xmm0,320(%%r10);"
                "movdqa %%xmm0,336(%%r10);"
                "movdqa %%xmm0,352(%%r10);"
                "movdqa %%xmm0,368(%%r10);"
                "movdqa %%xmm0,384(%%r10);"
                "movdqa %%xmm0,400(%%r10);"
                "movdqa %%xmm0,416(%%r10);"
                "movdqa %%xmm0,432(%%r10);"
                "movdqa %%xmm0,448(%%r10);"
                "movdqa %%xmm0,464(%%r10);"
                "movdqa %%xmm0,480(%%r10);"
                "movdqa %%xmm0,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
                "jnz _work_loop_movdqa_1;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                //"mov $0, %%rax;"
                //"cpuid;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "memory"
     );                
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_2:"
                "movdqa %%xmm0,(%%r10);movdqa %%xmm1,16(%%r10);"
                "movdqa %%xmm0,32(%%r10);movdqa %%xmm1,48(%%r10);"
                "movdqa %%xmm0,64(%%r10);movdqa %%xmm1,80(%%r10);"
                "movdqa %%xmm0,96(%%r10);movdqa %%xmm1,112(%%r10);"
                "movdqa %%xmm0,128(%%r10);movdqa %%xmm1,144(%%r10);"
                "movdqa %%xmm0,160(%%r10);movdqa %%xmm1,176(%%r10);"
                "movdqa %%xmm0,192(%%r10);movdqa %%xmm1,208(%%r10);"
                "movdqa %%xmm0,224(%%r10);movdqa %%xmm1,240(%%r10);"
                "movdqa %%xmm0,256(%%r10);movdqa %%xmm1,272(%%r10);"
                "movdqa %%xmm0,288(%%r10);movdqa %%xmm1,304(%%r10);"
                "movdqa %%xmm0,320(%%r10);movdqa %%xmm1,336(%%r10);"
                "movdqa %%xmm0,352(%%r10);movdqa %%xmm1,368(%%r10);"
                "movdqa %%xmm0,384(%%r10);movdqa %%xmm1,400(%%r10);"
                "movdqa %%xmm0,416(%%r10);movdqa %%xmm1,432(%%r10);"
                "movdqa %%xmm0,448(%%r10);movdqa %%xmm1,464(%%r10);"
                "movdqa %%xmm0,480(%%r10);movdqa %%xmm1,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "memory"
     );
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 3:
      passes=accesses/48;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_3:"
                "movdqa %%xmm0,(%%r10);movdqa %%xmm1,16(%%r10);movdqa %%xmm2,32(%%r10);"
                "movdqa %%xmm0,48(%%r10);movdqa %%xmm1,64(%%r10);movdqa %%xmm2,80(%%r10);"
                "movdqa %%xmm0,96(%%r10);movdqa %%xmm1,112(%%r10);movdqa %%xmm2,128(%%r10);"
                "movdqa %%xmm0,144(%%r10);movdqa %%xmm1,160(%%r10);movdqa %%xmm2,176(%%r10);"
                "movdqa %%xmm0,192(%%r10);movdqa %%xmm1,208(%%r10);movdqa %%xmm2,224(%%r10);"
                "movdqa %%xmm0,240(%%r10);movdqa %%xmm1,256(%%r10);movdqa %%xmm2,272(%%r10);"
                "movdqa %%xmm0,288(%%r10);movdqa %%xmm1,304(%%r10);movdqa %%xmm2,320(%%r10);"
                "movdqa %%xmm0,336(%%r10);movdqa %%xmm1,352(%%r10);movdqa %%xmm2,368(%%r10);"
                "movdqa %%xmm0,384(%%r10);movdqa %%xmm1,400(%%r10);movdqa %%xmm2,416(%%r10);"
                "movdqa %%xmm0,432(%%r10);movdqa %%xmm1,448(%%r10);movdqa %%xmm2,464(%%r10);"
                "movdqa %%xmm0,480(%%r10);movdqa %%xmm1,496(%%r10);movdqa %%xmm2,512(%%r10);"
                "movdqa %%xmm0,528(%%r10);movdqa %%xmm1,544(%%r10);movdqa %%xmm2,560(%%r10);"
                "movdqa %%xmm0,576(%%r10);movdqa %%xmm1,592(%%r10);movdqa %%xmm2,608(%%r10);"
                "movdqa %%xmm0,624(%%r10);movdqa %%xmm1,640(%%r10);movdqa %%xmm2,656(%%r10);"
                "movdqa %%xmm0,672(%%r10);movdqa %%xmm1,688(%%r10);movdqa %%xmm2,704(%%r10);"
                "movdqa %%xmm0,720(%%r10);movdqa %%xmm1,736(%%r10);movdqa %%xmm2,752(%%r10);"                               
                "add $768,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "%xmm2", "memory"
     );
     ret=(((double)(passes*48*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_4:"
                "movdqa %%xmm0,(%%r10);movdqa %%xmm1,16(%%r10);movdqa %%xmm2,32(%%r10);movdqa %%xmm3,48(%%r10);"
                "movdqa %%xmm0,64(%%r10);movdqa %%xmm1,80(%%r10);movdqa %%xmm2,96(%%r10);movdqa %%xmm3,112(%%r10);"
                "movdqa %%xmm0,128(%%r10);movdqa %%xmm1,144(%%r10);movdqa %%xmm2,160(%%r10);movdqa %%xmm3,176(%%r10);"
                "movdqa %%xmm0,192(%%r10);movdqa %%xmm1,208(%%r10);movdqa %%xmm2,224(%%r10);movdqa %%xmm3,240(%%r10);"
                "movdqa %%xmm0,256(%%r10);movdqa %%xmm1,272(%%r10);movdqa %%xmm2,288(%%r10);movdqa %%xmm3,304(%%r10);"
                "movdqa %%xmm0,320(%%r10);movdqa %%xmm1,336(%%r10);movdqa %%xmm2,352(%%r10);movdqa %%xmm3,368(%%r10);"
                "movdqa %%xmm0,384(%%r10);movdqa %%xmm1,400(%%r10);movdqa %%xmm2,416(%%r10);movdqa %%xmm3,432(%%r10);"
                "movdqa %%xmm0,448(%%r10);movdqa %%xmm1,464(%%r10);movdqa %%xmm2,480(%%r10);movdqa %%xmm3,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
     );                
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    default: ret=0.0;break;
   }
  //printf("end asm\n");fflush(stdout);
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (data->num_events) PAPI_read(data->Eventset,data->values);
    for (i=0;i<data->num_events;i++)
    {
       data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
    }
  #endif
  return ret;
}

/*
 * assembler implementation of bandwidth measurement
 */
static double asm_work_movdqu(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data) __attribute__((noinline)); 
static double asm_work_movdqu(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data)
{
   unsigned long long passes;
   double ret;
   int i;
   
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   #ifdef USE_PAPI
    if (data->num_events) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif  
   switch (burst_length)
   {
     case 1:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_1:"
                "movdqu %%xmm0,(%%r10);"
                "movdqu %%xmm0,16(%%r10);"
                "movdqu %%xmm0,32(%%r10);"
                "movdqu %%xmm0,48(%%r10);"
                "movdqu %%xmm0,64(%%r10);"
                "movdqu %%xmm0,80(%%r10);"
                "movdqu %%xmm0,96(%%r10);"
                "movdqu %%xmm0,112(%%r10);"
                "movdqu %%xmm0,128(%%r10);"
                "movdqu %%xmm0,144(%%r10);"
                "movdqu %%xmm0,160(%%r10);"
                "movdqu %%xmm0,176(%%r10);"
                "movdqu %%xmm0,192(%%r10);"
                "movdqu %%xmm0,208(%%r10);"
                "movdqu %%xmm0,224(%%r10);"
                "movdqu %%xmm0,240(%%r10);"
                "movdqu %%xmm0,256(%%r10);"
                "movdqu %%xmm0,272(%%r10);"
                "movdqu %%xmm0,288(%%r10);"
                "movdqu %%xmm0,304(%%r10);"
                "movdqu %%xmm0,320(%%r10);"
                "movdqu %%xmm0,336(%%r10);"
                "movdqu %%xmm0,352(%%r10);"
                "movdqu %%xmm0,368(%%r10);"
                "movdqu %%xmm0,384(%%r10);"
                "movdqu %%xmm0,400(%%r10);"
                "movdqu %%xmm0,416(%%r10);"
                "movdqu %%xmm0,432(%%r10);"
                "movdqu %%xmm0,448(%%r10);"
                "movdqu %%xmm0,464(%%r10);"
                "movdqu %%xmm0,480(%%r10);"
                "movdqu %%xmm0,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "memory"
     ); 
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_2:"
                "movdqu %%xmm0,(%%r10);movdqu %%xmm1,16(%%r10);"
                "movdqu %%xmm0,32(%%r10);movdqu %%xmm1,48(%%r10);"
                "movdqu %%xmm0,64(%%r10);movdqu %%xmm1,80(%%r10);"
                "movdqu %%xmm0,96(%%r10);movdqu %%xmm1,112(%%r10);"
                "movdqu %%xmm0,128(%%r10);movdqu %%xmm1,144(%%r10);"
                "movdqu %%xmm0,160(%%r10);movdqu %%xmm1,176(%%r10);"
                "movdqu %%xmm0,192(%%r10);movdqu %%xmm1,208(%%r10);"
                "movdqu %%xmm0,224(%%r10);movdqu %%xmm1,240(%%r10);"
                "movdqu %%xmm0,256(%%r10);movdqu %%xmm1,272(%%r10);"
                "movdqu %%xmm0,288(%%r10);movdqu %%xmm1,304(%%r10);"
                "movdqu %%xmm0,320(%%r10);movdqu %%xmm1,336(%%r10);"
                "movdqu %%xmm0,352(%%r10);movdqu %%xmm1,368(%%r10);"
                "movdqu %%xmm0,384(%%r10);movdqu %%xmm1,400(%%r10);"
                "movdqu %%xmm0,416(%%r10);movdqu %%xmm1,432(%%r10);"
                "movdqu %%xmm0,448(%%r10);movdqu %%xmm1,464(%%r10);"
                "movdqu %%xmm0,480(%%r10);movdqu %%xmm1,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "memory"
     );
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 3:
      passes=accesses/48;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_3:"
                "movdqu %%xmm0,(%%r10);movdqu %%xmm1,16(%%r10);movdqu %%xmm2,32(%%r10);"
                "movdqu %%xmm0,48(%%r10);movdqu %%xmm1,64(%%r10);movdqu %%xmm2,80(%%r10);"
                "movdqu %%xmm0,96(%%r10);movdqu %%xmm1,112(%%r10);movdqu %%xmm2,128(%%r10);"
                "movdqu %%xmm0,144(%%r10);movdqu %%xmm1,160(%%r10);movdqu %%xmm2,176(%%r10);"
                "movdqu %%xmm0,192(%%r10);movdqu %%xmm1,208(%%r10);movdqu %%xmm2,224(%%r10);"
                "movdqu %%xmm0,240(%%r10);movdqu %%xmm1,256(%%r10);movdqu %%xmm2,272(%%r10);"
                "movdqu %%xmm0,288(%%r10);movdqu %%xmm1,304(%%r10);movdqu %%xmm2,320(%%r10);"
                "movdqu %%xmm0,336(%%r10);movdqu %%xmm1,352(%%r10);movdqu %%xmm2,368(%%r10);"
                "movdqu %%xmm0,384(%%r10);movdqu %%xmm1,400(%%r10);movdqu %%xmm2,416(%%r10);"
                "movdqu %%xmm0,432(%%r10);movdqu %%xmm1,448(%%r10);movdqu %%xmm2,464(%%r10);"
                "movdqu %%xmm0,480(%%r10);movdqu %%xmm1,496(%%r10);movdqu %%xmm2,512(%%r10);"
                "movdqu %%xmm0,528(%%r10);movdqu %%xmm1,544(%%r10);movdqu %%xmm2,560(%%r10);"
                "movdqu %%xmm0,576(%%r10);movdqu %%xmm1,592(%%r10);movdqu %%xmm2,608(%%r10);"
                "movdqu %%xmm0,624(%%r10);movdqu %%xmm1,640(%%r10);movdqu %%xmm2,656(%%r10);"
                "movdqu %%xmm0,672(%%r10);movdqu %%xmm1,688(%%r10);movdqu %%xmm2,704(%%r10);"
                "movdqu %%xmm0,720(%%r10);movdqu %%xmm1,736(%%r10);movdqu %%xmm2,752(%%r10);"                               
                "add $768,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "%xmm2", "memory"
     );
     ret=(((double)(passes*48*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_4:"
                "movdqu %%xmm0,(%%r10);movdqu %%xmm1,16(%%r10);movdqu %%xmm2,32(%%r10);movdqu %%xmm3,48(%%r10);"
                "movdqu %%xmm0,64(%%r10);movdqu %%xmm1,80(%%r10);movdqu %%xmm2,96(%%r10);movdqu %%xmm3,112(%%r10);"
                "movdqu %%xmm0,128(%%r10);movdqu %%xmm1,144(%%r10);movdqu %%xmm2,160(%%r10);movdqu %%xmm3,176(%%r10);"
                "movdqu %%xmm0,192(%%r10);movdqu %%xmm1,208(%%r10);movdqu %%xmm2,224(%%r10);movdqu %%xmm3,240(%%r10);"
                "movdqu %%xmm0,256(%%r10);movdqu %%xmm1,272(%%r10);movdqu %%xmm2,288(%%r10);movdqu %%xmm3,304(%%r10);"
                "movdqu %%xmm0,320(%%r10);movdqu %%xmm1,336(%%r10);movdqu %%xmm2,352(%%r10);movdqu %%xmm3,368(%%r10);"
                "movdqu %%xmm0,384(%%r10);movdqu %%xmm1,400(%%r10);movdqu %%xmm2,416(%%r10);movdqu %%xmm3,432(%%r10);"
                "movdqu %%xmm0,448(%%r10);movdqu %%xmm1,464(%%r10);movdqu %%xmm2,480(%%r10);movdqu %%xmm3,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
     );                
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    default: ret=0.0;break;
   }
  //printf("end asm\n");fflush(stdout);
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (data->num_events) PAPI_read(data->Eventset,data->values);
    for (i=0;i<data->num_events;i++)
    {
       data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
    }
  #endif
  return ret;
}

/*
 * assembler implementation of bandwidth measurement
 */
static double asm_work_movntdq(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data) __attribute__((noinline)); 
static double asm_work_movntdq(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data)
{
   unsigned long long passes;
   double ret;
   int i;
   
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   
   #ifdef USE_PAPI
    if (data->num_events) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movntdq_1:"
                "movntdq %%xmm0,(%%r10);"
                "movntdq %%xmm0,16(%%r10);"
                "movntdq %%xmm0,32(%%r10);"
                "movntdq %%xmm0,48(%%r10);"
                "movntdq %%xmm0,64(%%r10);"
                "movntdq %%xmm0,80(%%r10);"
                "movntdq %%xmm0,96(%%r10);"
                "movntdq %%xmm0,112(%%r10);"
                "movntdq %%xmm0,128(%%r10);"
                "movntdq %%xmm0,144(%%r10);"
                "movntdq %%xmm0,160(%%r10);"
                "movntdq %%xmm0,176(%%r10);"
                "movntdq %%xmm0,192(%%r10);"
                "movntdq %%xmm0,208(%%r10);"
                "movntdq %%xmm0,224(%%r10);"
                "movntdq %%xmm0,240(%%r10);"
                "movntdq %%xmm0,256(%%r10);"
                "movntdq %%xmm0,272(%%r10);"
                "movntdq %%xmm0,288(%%r10);"
                "movntdq %%xmm0,304(%%r10);"
                "movntdq %%xmm0,320(%%r10);"
                "movntdq %%xmm0,336(%%r10);"
                "movntdq %%xmm0,352(%%r10);"
                "movntdq %%xmm0,368(%%r10);"
                "movntdq %%xmm0,384(%%r10);"
                "movntdq %%xmm0,400(%%r10);"
                "movntdq %%xmm0,416(%%r10);"
                "movntdq %%xmm0,432(%%r10);"
                "movntdq %%xmm0,448(%%r10);"
                "movntdq %%xmm0,464(%%r10);"
                "movntdq %%xmm0,480(%%r10);"
                "movntdq %%xmm0,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "memory"
     ); 
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movntdq_2:"
                "movntdq %%xmm0,(%%r10);movntdq %%xmm1,16(%%r10);"
                "movntdq %%xmm0,32(%%r10);movntdq %%xmm1,48(%%r10);"
                "movntdq %%xmm0,64(%%r10);movntdq %%xmm1,80(%%r10);"
                "movntdq %%xmm0,96(%%r10);movntdq %%xmm1,112(%%r10);"
                "movntdq %%xmm0,128(%%r10);movntdq %%xmm1,144(%%r10);"
                "movntdq %%xmm0,160(%%r10);movntdq %%xmm1,176(%%r10);"
                "movntdq %%xmm0,192(%%r10);movntdq %%xmm1,208(%%r10);"
                "movntdq %%xmm0,224(%%r10);movntdq %%xmm1,240(%%r10);"
                "movntdq %%xmm0,256(%%r10);movntdq %%xmm1,272(%%r10);"
                "movntdq %%xmm0,288(%%r10);movntdq %%xmm1,304(%%r10);"
                "movntdq %%xmm0,320(%%r10);movntdq %%xmm1,336(%%r10);"
                "movntdq %%xmm0,352(%%r10);movntdq %%xmm1,368(%%r10);"
                "movntdq %%xmm0,384(%%r10);movntdq %%xmm1,400(%%r10);"
                "movntdq %%xmm0,416(%%r10);movntdq %%xmm1,432(%%r10);"
                "movntdq %%xmm0,448(%%r10);movntdq %%xmm1,464(%%r10);"
                "movntdq %%xmm0,480(%%r10);movntdq %%xmm1,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "memory"
     );
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 3:
      passes=accesses/48;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movntdq_3:"
                "movntdq %%xmm0,(%%r10);movntdq %%xmm1,16(%%r10);movntdq %%xmm2,32(%%r10);"
                "movntdq %%xmm0,48(%%r10);movntdq %%xmm1,64(%%r10);movntdq %%xmm2,80(%%r10);"
                "movntdq %%xmm0,96(%%r10);movntdq %%xmm1,112(%%r10);movntdq %%xmm2,128(%%r10);"
                "movntdq %%xmm0,144(%%r10);movntdq %%xmm1,160(%%r10);movntdq %%xmm2,176(%%r10);"
                "movntdq %%xmm0,192(%%r10);movntdq %%xmm1,208(%%r10);movntdq %%xmm2,224(%%r10);"
                "movntdq %%xmm0,240(%%r10);movntdq %%xmm1,256(%%r10);movntdq %%xmm2,272(%%r10);"
                "movntdq %%xmm0,288(%%r10);movntdq %%xmm1,304(%%r10);movntdq %%xmm2,320(%%r10);"
                "movntdq %%xmm0,336(%%r10);movntdq %%xmm1,352(%%r10);movntdq %%xmm2,368(%%r10);"
                "movntdq %%xmm0,384(%%r10);movntdq %%xmm1,400(%%r10);movntdq %%xmm2,416(%%r10);"
                "movntdq %%xmm0,432(%%r10);movntdq %%xmm1,448(%%r10);movntdq %%xmm2,464(%%r10);"
                "movntdq %%xmm0,480(%%r10);movntdq %%xmm1,496(%%r10);movntdq %%xmm2,512(%%r10);"
                "movntdq %%xmm0,528(%%r10);movntdq %%xmm1,544(%%r10);movntdq %%xmm2,560(%%r10);"
                "movntdq %%xmm0,576(%%r10);movntdq %%xmm1,592(%%r10);movntdq %%xmm2,608(%%r10);"
                "movntdq %%xmm0,624(%%r10);movntdq %%xmm1,640(%%r10);movntdq %%xmm2,656(%%r10);"
                "movntdq %%xmm0,672(%%r10);movntdq %%xmm1,688(%%r10);movntdq %%xmm2,704(%%r10);"
                "movntdq %%xmm0,720(%%r10);movntdq %%xmm1,736(%%r10);movntdq %%xmm2,752(%%r10);"                               
                "add $768,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "%xmm2", "memory"
     );
     ret=(((double)(passes*48*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;
   
     /*
      * Input: addr -> RBX (pointer to the buffer)
      *        passes -> RCX
      * Output : RAX stop timestamp - start timestamp
      */
     __asm__ __volatile__(
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movntdq_4:"
                "movntdq %%xmm0,(%%r10);movntdq %%xmm1,16(%%r10);movntdq %%xmm2,32(%%r10);movntdq %%xmm3,48(%%r10);"
                "movntdq %%xmm0,64(%%r10);movntdq %%xmm1,80(%%r10);movntdq %%xmm2,96(%%r10);movntdq %%xmm3,112(%%r10);"
                "movntdq %%xmm0,128(%%r10);movntdq %%xmm1,144(%%r10);movntdq %%xmm2,160(%%r10);movntdq %%xmm3,176(%%r10);"
                "movntdq %%xmm0,192(%%r10);movntdq %%xmm1,208(%%r10);movntdq %%xmm2,224(%%r10);movntdq %%xmm3,240(%%r10);"
                "movntdq %%xmm0,256(%%r10);movntdq %%xmm1,272(%%r10);movntdq %%xmm2,288(%%r10);movntdq %%xmm3,304(%%r10);"
                "movntdq %%xmm0,320(%%r10);movntdq %%xmm1,336(%%r10);movntdq %%xmm2,352(%%r10);movntdq %%xmm3,368(%%r10);"
                "movntdq %%xmm0,384(%%r10);movntdq %%xmm1,400(%%r10);movntdq %%xmm2,416(%%r10);movntdq %%xmm3,432(%%r10);"
                "movntdq %%xmm0,448(%%r10);movntdq %%xmm1,464(%%r10);movntdq %%xmm2,480(%%r10);movntdq %%xmm3,496(%%r10);"
                "add $512,%%r10;"
                "sub $1,%%r11;"
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
                "sub %%r9,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx", "%r9", "%r10","%r11", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
     );                
     ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
     break;
    default: ret=0.0;break;
   }
  //printf("end asm\n");fflush(stdout);
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (data->num_events) PAPI_read(data->Eventset,data->values);
    for (i=0;i<data->num_events;i++)
    {
       data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
    }
  #endif
  return ret;
}

/*
 * assembler implementation of bandwidth measurement using mov instruction
 */
static double asm_work_movnti(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data) __attribute__((noinline)); 
static double asm_work_movnti(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data)
{
   unsigned long long passes;
   double ret;
   int i;
 
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);

   #ifdef USE_PAPI
    if (data->num_events) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movnti_1:"
                "movnti %%r8, (%%rbx);"
                "movnti %%r8, 8(%%rbx);"
                "movnti %%r8, 16(%%rbx);"
                "movnti %%r8, 24(%%rbx);"
                "movnti %%r8, 32(%%rbx);"
                "movnti %%r8, 40(%%rbx);"
                "movnti %%r8, 48(%%rbx);"
                "movnti %%r8, 56(%%rbx);"
                "movnti %%r8, 64(%%rbx);"
                "movnti %%r8, 72(%%rbx);"
                "movnti %%r8, 80(%%rbx);"
                "movnti %%r8, 88(%%rbx);"
                "movnti %%r8, 96(%%rbx);"
                "movnti %%r8, 104(%%rbx);"
                "movnti %%r8, 112(%%rbx);"
                "movnti %%r8, 120(%%rbx);"
                "movnti %%r8, 128(%%rbx);"
                "movnti %%r8, 136(%%rbx);"
                "movnti %%r8, 144(%%rbx);"
                "movnti %%r8, 152(%%rbx);"
                "movnti %%r8, 160(%%rbx);"
                "movnti %%r8, 168(%%rbx);"
                "movnti %%r8, 176(%%rbx);"
                "movnti %%r8, 184(%%rbx);"
                "movnti %%r8, 192(%%rbx);"
                "movnti %%r8, 200(%%rbx);"
                "movnti %%r8, 208(%%rbx);"
                "movnti %%r8, 216(%%rbx);"
                "movnti %%r8, 224(%%rbx);"
                "movnti %%r8, 232(%%rbx);"
                "movnti %%r8, 240(%%rbx);"
                "movnti %%r8, 248(%%rbx);"
                "movnti %%r8, 256(%%rbx);"
                "movnti %%r8, 264(%%rbx);"
                "movnti %%r8, 272(%%rbx);"
                "movnti %%r8, 280(%%rbx);"
                "movnti %%r8, 288(%%rbx);"
                "movnti %%r8, 296(%%rbx);"
                "movnti %%r8, 304(%%rbx);"
                "movnti %%r8, 312(%%rbx);"
                "movnti %%r8, 320(%%rbx);"
                "movnti %%r8, 328(%%rbx);"
                "movnti %%r8, 336(%%rbx);"
                "movnti %%r8, 344(%%rbx);"
                "movnti %%r8, 352(%%rbx);"
                "movnti %%r8, 360(%%rbx);"
                "movnti %%r8, 368(%%rbx);"
                "movnti %%r8, 376(%%rbx);"
                "movnti %%r8, 384(%%rbx);"
                "movnti %%r8, 392(%%rbx);"
                "movnti %%r8, 400(%%rbx);"
                "movnti %%r8, 408(%%rbx);"
                "movnti %%r8, 416(%%rbx);"
                "movnti %%r8, 424(%%rbx);"
                "movnti %%r8, 432(%%rbx);"
                "movnti %%r8, 440(%%rbx);"
                "movnti %%r8, 448(%%rbx);"
                "movnti %%r8, 456(%%rbx);"
                "movnti %%r8, 464(%%rbx);"
                "movnti %%r8, 472(%%rbx);"
                "movnti %%r8, 480(%%rbx);"
                "movnti %%r8, 488(%%rbx);"
                "movnti %%r8, 496(%%rbx);"
                "movnti %%r8, 504(%%rbx);"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movnti_1;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "memory"
								);
								
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movnti_2:"
                "movnti %%r8, (%%rbx);movnti %%r9, 8(%%rbx);"
                "movnti %%r8, 16(%%rbx);movnti %%r9, 24(%%rbx);"
                "movnti %%r8, 32(%%rbx);movnti %%r9, 40(%%rbx);"
                "movnti %%r8, 48(%%rbx);movnti %%r9, 56(%%rbx);"
                "movnti %%r8, 64(%%rbx);movnti %%r9, 72(%%rbx);"
                "movnti %%r8, 80(%%rbx);movnti %%r9, 88(%%rbx);"
                "movnti %%r8, 96(%%rbx);movnti %%r9, 104(%%rbx);"
                "movnti %%r8, 112(%%rbx);movnti %%r9, 120(%%rbx);"
                "movnti %%r8, 128(%%rbx);movnti %%r9, 136(%%rbx);"
                "movnti %%r8, 144(%%rbx);movnti %%r9, 152(%%rbx);"
                "movnti %%r8, 160(%%rbx);movnti %%r9, 168(%%rbx);"
                "movnti %%r8, 176(%%rbx);movnti %%r9, 184(%%rbx);"
                "movnti %%r8, 192(%%rbx);movnti %%r9, 200(%%rbx);"
                "movnti %%r8, 208(%%rbx);movnti %%r9, 216(%%rbx);"
                "movnti %%r8, 224(%%rbx);movnti %%r9, 232(%%rbx);"
                "movnti %%r8, 240(%%rbx);movnti %%r9, 248(%%rbx);"
                "movnti %%r8, 256(%%rbx);movnti %%r9, 264(%%rbx);"
                "movnti %%r8, 272(%%rbx);movnti %%r9, 280(%%rbx);"
                "movnti %%r8, 288(%%rbx);movnti %%r9, 296(%%rbx);"
                "movnti %%r8, 304(%%rbx);movnti %%r9, 312(%%rbx);"
                "movnti %%r8, 320(%%rbx);movnti %%r9, 328(%%rbx);"
                "movnti %%r8, 336(%%rbx);movnti %%r9, 344(%%rbx);"
                "movnti %%r8, 352(%%rbx);movnti %%r9, 360(%%rbx);"
                "movnti %%r8, 368(%%rbx);movnti %%r9, 376(%%rbx);"
                "movnti %%r8, 384(%%rbx);movnti %%r9, 392(%%rbx);"
                "movnti %%r8, 400(%%rbx);movnti %%r9, 408(%%rbx);"
                "movnti %%r8, 416(%%rbx);movnti %%r9, 424(%%rbx);"
                "movnti %%r8, 432(%%rbx);movnti %%r9, 440(%%rbx);"
                "movnti %%r8, 448(%%rbx);movnti %%r9, 456(%%rbx);"
                "movnti %%r8, 464(%%rbx);movnti %%r9, 472(%%rbx);"
                "movnti %%r8, 480(%%rbx);movnti %%r9, 488(%%rbx);"
                "movnti %%r8, 496(%%rbx);movnti %%r9, 504(%%rbx);"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movnti_2;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "%r9", "memory"
								);
								
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
	  case 3:
      passes=accesses/48;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movnti_3:"
                "movnti %%r8, (%%rbx);movnti %%r9, 8(%%rbx);movnti %%r10, 16(%%rbx);"
                "movnti %%r8, 24(%%rbx);movnti %%r9, 32(%%rbx);movnti %%r10, 40(%%rbx);"
                "movnti %%r8, 48(%%rbx);movnti %%r9, 56(%%rbx);movnti %%r10, 64(%%rbx);"
                "movnti %%r8, 72(%%rbx);movnti %%r9, 80(%%rbx);movnti %%r10, 88(%%rbx);"
                "movnti %%r8, 96(%%rbx);movnti %%r9, 104(%%rbx);movnti %%r10, 112(%%rbx);"
                "movnti %%r8, 120(%%rbx);movnti %%r9, 128(%%rbx);movnti %%r10, 136(%%rbx);"
                "movnti %%r8, 144(%%rbx);movnti %%r9, 152(%%rbx);movnti %%r10, 160(%%rbx);"
                "movnti %%r8, 168(%%rbx);movnti %%r9, 176(%%rbx);movnti %%r10, 184(%%rbx);"
                "movnti %%r8, 192(%%rbx);movnti %%r9, 200(%%rbx);movnti %%r10, 208(%%rbx);"
                "movnti %%r8, 216(%%rbx);movnti %%r9, 224(%%rbx);movnti %%r10, 232(%%rbx);"
                "movnti %%r8, 240(%%rbx);movnti %%r9, 248(%%rbx);movnti %%r10, 256(%%rbx);"
                "movnti %%r8, 264(%%rbx);movnti %%r9, 272(%%rbx);movnti %%r10, 280(%%rbx);"
                "movnti %%r8, 288(%%rbx);movnti %%r9, 296(%%rbx);movnti %%r10, 304(%%rbx);"
                "movnti %%r8, 312(%%rbx);movnti %%r9, 320(%%rbx);movnti %%r10, 328(%%rbx);"
                "movnti %%r8, 336(%%rbx);movnti %%r9, 344(%%rbx);movnti %%r10, 352(%%rbx);"
                "movnti %%r8, 360(%%rbx);movnti %%r9, 368(%%rbx);movnti %%r10, 376(%%rbx);"
                "movnti %%r8, 384(%%rbx);movnti %%r9, 392(%%rbx);movnti %%r10, 400(%%rbx);"
                "movnti %%r8, 408(%%rbx);movnti %%r9, 416(%%rbx);movnti %%r10, 424(%%rbx);"
                "movnti %%r8, 432(%%rbx);movnti %%r9, 440(%%rbx);movnti %%r10, 448(%%rbx);"
                "movnti %%r8, 456(%%rbx);movnti %%r9, 464(%%rbx);movnti %%r10, 472(%%rbx);"
                "movnti %%r8, 480(%%rbx);movnti %%r9, 488(%%rbx);movnti %%r10, 496(%%rbx);"
                "movnti %%r8, 504(%%rbx);movnti %%r9, 512(%%rbx);movnti %%r10, 520(%%rbx);"
                "movnti %%r8, 528(%%rbx);movnti %%r9, 536(%%rbx);movnti %%r10, 544(%%rbx);"
                "movnti %%r8, 552(%%rbx);movnti %%r9, 560(%%rbx);movnti %%r10, 568(%%rbx);"
                "movnti %%r8, 576(%%rbx);movnti %%r9, 584(%%rbx);movnti %%r10, 592(%%rbx);"
                "movnti %%r8, 600(%%rbx);movnti %%r9, 608(%%rbx);movnti %%r10, 616(%%rbx);"
                "movnti %%r8, 624(%%rbx);movnti %%r9, 632(%%rbx);movnti %%r10, 640(%%rbx);"
                "movnti %%r8, 648(%%rbx);movnti %%r9, 656(%%rbx);movnti %%r10, 664(%%rbx);"
                "movnti %%r8, 672(%%rbx);movnti %%r9, 680(%%rbx);movnti %%r10, 688(%%rbx);"
                "movnti %%r8, 696(%%rbx);movnti %%r9, 704(%%rbx);movnti %%r10, 712(%%rbx);"
                "movnti %%r8, 720(%%rbx);movnti %%r9, 728(%%rbx);movnti %%r10, 736(%%rbx);"
                "movnti %%r8, 744(%%rbx);movnti %%r9, 752(%%rbx);movnti %%r10, 760(%%rbx);"                
                "add $768,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movnti_3;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "%r9", "%r10", "memory"
								);
								
			 ret=(((double)(passes*48*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movnti_4:"
                "movnti %%r8, (%%rbx);movnti %%r9, 8(%%rbx);movnti %%r10, 16(%%rbx);movnti %%r11, 24(%%rbx);"
                "movnti %%r8, 32(%%rbx);movnti %%r9, 40(%%rbx);movnti %%r10, 48(%%rbx);movnti %%r11, 56(%%rbx);"
                "movnti %%r8, 64(%%rbx);movnti %%r9, 72(%%rbx);movnti %%r10, 80(%%rbx);movnti %%r11, 88(%%rbx);"
                "movnti %%r8, 96(%%rbx);movnti %%r9, 104(%%rbx);movnti %%r10, 112(%%rbx);movnti %%r11, 120(%%rbx);"
                "movnti %%r8, 128(%%rbx);movnti %%r9, 136(%%rbx);movnti %%r10, 144(%%rbx);movnti %%r11, 152(%%rbx);"
                "movnti %%r8, 160(%%rbx);movnti %%r9, 168(%%rbx);movnti %%r10, 176(%%rbx);movnti %%r11, 184(%%rbx);"
                "movnti %%r8, 192(%%rbx);movnti %%r9, 200(%%rbx);movnti %%r10, 208(%%rbx);movnti %%r11, 216(%%rbx);"
                "movnti %%r8, 224(%%rbx);movnti %%r9, 232(%%rbx);movnti %%r10, 240(%%rbx);movnti %%r11, 248(%%rbx);"
                "movnti %%r8, 256(%%rbx);movnti %%r9, 264(%%rbx);movnti %%r10, 272(%%rbx);movnti %%r11, 280(%%rbx);"
                "movnti %%r8, 288(%%rbx);movnti %%r9, 296(%%rbx);movnti %%r10, 304(%%rbx);movnti %%r11, 312(%%rbx);"
                "movnti %%r8, 320(%%rbx);movnti %%r9, 328(%%rbx);movnti %%r10, 336(%%rbx);movnti %%r11, 344(%%rbx);"
                "movnti %%r8, 352(%%rbx);movnti %%r9, 360(%%rbx);movnti %%r10, 368(%%rbx);movnti %%r11, 376(%%rbx);"
                "movnti %%r8, 384(%%rbx);movnti %%r9, 392(%%rbx);movnti %%r10, 400(%%rbx);movnti %%r11, 408(%%rbx);"
                "movnti %%r8, 416(%%rbx);movnti %%r9, 424(%%rbx);movnti %%r10, 432(%%rbx);movnti %%r11, 440(%%rbx);"
                "movnti %%r8, 448(%%rbx);movnti %%r9, 456(%%rbx);movnti %%r10, 464(%%rbx);movnti %%r11, 472(%%rbx);"
                "movnti %%r8, 480(%%rbx);movnti %%r9, 488(%%rbx);movnti %%r10, 496(%%rbx);movnti %%r11, 504(%%rbx);"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movnti_4;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "%r9", "%r10", "%r11", "memory"
								);
								
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
			 default: ret=0.0;break;
   }

  //printf("end asm\n");fflush(stdout);
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (data->num_events) PAPI_read(data->Eventset,data->values);
    for (i=0;i<data->num_events;i++)
    {
       data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
    }
  #endif
	 return ret;
}

/*
 * assembler implementation of bandwidth measurement using mov instruction
 */
static double asm_work_mov(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data) __attribute__((noinline)); 
static double asm_work_mov(unsigned long long addr, unsigned long long accesses, unsigned long long burst_length, unsigned long long call_latency,unsigned long long freq,volatile mydata_t *data)
{
   unsigned long long passes;
   double ret;
   int i;
 
   //printf("start asm : 0x%016lx\n",*(unsigned long long*)addr);fflush(stdout);
   
   #ifdef USE_PAPI
    if (data->num_events) PAPI_reset(data->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif
   switch (burst_length)
   {
    case 1:
      passes=accesses/32;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_1:"
                "mov %%r8, (%%rbx);"
                "mov %%r8, 8(%%rbx);"
                "mov %%r8, 16(%%rbx);"
                "mov %%r8, 24(%%rbx);"
                "mov %%r8, 32(%%rbx);"
                "mov %%r8, 40(%%rbx);"
                "mov %%r8, 48(%%rbx);"
                "mov %%r8, 56(%%rbx);"
                "mov %%r8, 64(%%rbx);"
                "mov %%r8, 72(%%rbx);"
                "mov %%r8, 80(%%rbx);"
                "mov %%r8, 88(%%rbx);"
                "mov %%r8, 96(%%rbx);"
                "mov %%r8, 104(%%rbx);"
                "mov %%r8, 112(%%rbx);"
                "mov %%r8, 120(%%rbx);"
                "mov %%r8, 128(%%rbx);"
                "mov %%r8, 136(%%rbx);"
                "mov %%r8, 144(%%rbx);"
                "mov %%r8, 152(%%rbx);"
                "mov %%r8, 160(%%rbx);"
                "mov %%r8, 168(%%rbx);"
                "mov %%r8, 176(%%rbx);"
                "mov %%r8, 184(%%rbx);"
                "mov %%r8, 192(%%rbx);"
                "mov %%r8, 200(%%rbx);"
                "mov %%r8, 208(%%rbx);"
                "mov %%r8, 216(%%rbx);"
                "mov %%r8, 224(%%rbx);"
                "mov %%r8, 232(%%rbx);"
                "mov %%r8, 240(%%rbx);"
                "mov %%r8, 248(%%rbx);"
                "mov %%r8, 256(%%rbx);"
                "mov %%r8, 264(%%rbx);"
                "mov %%r8, 272(%%rbx);"
                "mov %%r8, 280(%%rbx);"
                "mov %%r8, 288(%%rbx);"
                "mov %%r8, 296(%%rbx);"
                "mov %%r8, 304(%%rbx);"
                "mov %%r8, 312(%%rbx);"
                "mov %%r8, 320(%%rbx);"
                "mov %%r8, 328(%%rbx);"
                "mov %%r8, 336(%%rbx);"
                "mov %%r8, 344(%%rbx);"
                "mov %%r8, 352(%%rbx);"
                "mov %%r8, 360(%%rbx);"
                "mov %%r8, 368(%%rbx);"
                "mov %%r8, 376(%%rbx);"
                "mov %%r8, 384(%%rbx);"
                "mov %%r8, 392(%%rbx);"
                "mov %%r8, 400(%%rbx);"
                "mov %%r8, 408(%%rbx);"
                "mov %%r8, 416(%%rbx);"
                "mov %%r8, 424(%%rbx);"
                "mov %%r8, 432(%%rbx);"
                "mov %%r8, 440(%%rbx);"
                "mov %%r8, 448(%%rbx);"
                "mov %%r8, 456(%%rbx);"
                "mov %%r8, 464(%%rbx);"
                "mov %%r8, 472(%%rbx);"
                "mov %%r8, 480(%%rbx);"
                "mov %%r8, 488(%%rbx);"
                "mov %%r8, 496(%%rbx);"
                "mov %%r8, 504(%%rbx);"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_mov_1;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "memory"
								);
								
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
    case 2:
      passes=accesses/32;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_2:"
                "mov %%r8, (%%rbx);mov %%r9, 8(%%rbx);"
                "mov %%r8, 16(%%rbx);mov %%r9, 24(%%rbx);"
                "mov %%r8, 32(%%rbx);mov %%r9, 40(%%rbx);"
                "mov %%r8, 48(%%rbx);mov %%r9, 56(%%rbx);"
                "mov %%r8, 64(%%rbx);mov %%r9, 72(%%rbx);"
                "mov %%r8, 80(%%rbx);mov %%r9, 88(%%rbx);"
                "mov %%r8, 96(%%rbx);mov %%r9, 104(%%rbx);"
                "mov %%r8, 112(%%rbx);mov %%r9, 120(%%rbx);"
                "mov %%r8, 128(%%rbx);mov %%r9, 136(%%rbx);"
                "mov %%r8, 144(%%rbx);mov %%r9, 152(%%rbx);"
                "mov %%r8, 160(%%rbx);mov %%r9, 168(%%rbx);"
                "mov %%r8, 176(%%rbx);mov %%r9, 184(%%rbx);"
                "mov %%r8, 192(%%rbx);mov %%r9, 200(%%rbx);"
                "mov %%r8, 208(%%rbx);mov %%r9, 216(%%rbx);"
                "mov %%r8, 224(%%rbx);mov %%r9, 232(%%rbx);"
                "mov %%r8, 240(%%rbx);mov %%r9, 248(%%rbx);"
                "mov %%r8, 256(%%rbx);mov %%r9, 264(%%rbx);"
                "mov %%r8, 272(%%rbx);mov %%r9, 280(%%rbx);"
                "mov %%r8, 288(%%rbx);mov %%r9, 296(%%rbx);"
                "mov %%r8, 304(%%rbx);mov %%r9, 312(%%rbx);"
                "mov %%r8, 320(%%rbx);mov %%r9, 328(%%rbx);"
                "mov %%r8, 336(%%rbx);mov %%r9, 344(%%rbx);"
                "mov %%r8, 352(%%rbx);mov %%r9, 360(%%rbx);"
                "mov %%r8, 368(%%rbx);mov %%r9, 376(%%rbx);"
                "mov %%r8, 384(%%rbx);mov %%r9, 392(%%rbx);"
                "mov %%r8, 400(%%rbx);mov %%r9, 408(%%rbx);"
                "mov %%r8, 416(%%rbx);mov %%r9, 424(%%rbx);"
                "mov %%r8, 432(%%rbx);mov %%r9, 440(%%rbx);"
                "mov %%r8, 448(%%rbx);mov %%r9, 456(%%rbx);"
                "mov %%r8, 464(%%rbx);mov %%r9, 472(%%rbx);"
                "mov %%r8, 480(%%rbx);mov %%r9, 488(%%rbx);"
                "mov %%r8, 496(%%rbx);mov %%r9, 504(%%rbx);"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_mov_2;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "%r9", "memory"
								);
								
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
	  case 3:
      passes=accesses/48;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_3:"
                "mov %%r8, (%%rbx);mov %%r9, 8(%%rbx);mov %%r10, 16(%%rbx);"
                "mov %%r8, 24(%%rbx);mov %%r9, 32(%%rbx);mov %%r10, 40(%%rbx);"
                "mov %%r8, 48(%%rbx);mov %%r9, 56(%%rbx);mov %%r10, 64(%%rbx);"
                "mov %%r8, 72(%%rbx);mov %%r9, 80(%%rbx);mov %%r10, 88(%%rbx);"
                "mov %%r8, 96(%%rbx);mov %%r9, 104(%%rbx);mov %%r10, 112(%%rbx);"
                "mov %%r8, 120(%%rbx);mov %%r9, 128(%%rbx);mov %%r10, 136(%%rbx);"
                "mov %%r8, 144(%%rbx);mov %%r9, 152(%%rbx);mov %%r10, 160(%%rbx);"
                "mov %%r8, 168(%%rbx);mov %%r9, 176(%%rbx);mov %%r10, 184(%%rbx);"
                "mov %%r8, 192(%%rbx);mov %%r9, 200(%%rbx);mov %%r10, 208(%%rbx);"
                "mov %%r8, 216(%%rbx);mov %%r9, 224(%%rbx);mov %%r10, 232(%%rbx);"
                "mov %%r8, 240(%%rbx);mov %%r9, 248(%%rbx);mov %%r10, 256(%%rbx);"
                "mov %%r8, 264(%%rbx);mov %%r9, 272(%%rbx);mov %%r10, 280(%%rbx);"
                "mov %%r8, 288(%%rbx);mov %%r9, 296(%%rbx);mov %%r10, 304(%%rbx);"
                "mov %%r8, 312(%%rbx);mov %%r9, 320(%%rbx);mov %%r10, 328(%%rbx);"
                "mov %%r8, 336(%%rbx);mov %%r9, 344(%%rbx);mov %%r10, 352(%%rbx);"
                "mov %%r8, 360(%%rbx);mov %%r9, 368(%%rbx);mov %%r10, 376(%%rbx);"
                "mov %%r8, 384(%%rbx);mov %%r9, 392(%%rbx);mov %%r10, 400(%%rbx);"
                "mov %%r8, 408(%%rbx);mov %%r9, 416(%%rbx);mov %%r10, 424(%%rbx);"
                "mov %%r8, 432(%%rbx);mov %%r9, 440(%%rbx);mov %%r10, 448(%%rbx);"
                "mov %%r8, 456(%%rbx);mov %%r9, 464(%%rbx);mov %%r10, 472(%%rbx);"
                "mov %%r8, 480(%%rbx);mov %%r9, 488(%%rbx);mov %%r10, 496(%%rbx);"
                "mov %%r8, 504(%%rbx);mov %%r9, 512(%%rbx);mov %%r10, 520(%%rbx);"
                "mov %%r8, 528(%%rbx);mov %%r9, 536(%%rbx);mov %%r10, 544(%%rbx);"
                "mov %%r8, 552(%%rbx);mov %%r9, 560(%%rbx);mov %%r10, 568(%%rbx);"
                "mov %%r8, 576(%%rbx);mov %%r9, 584(%%rbx);mov %%r10, 592(%%rbx);"
                "mov %%r8, 600(%%rbx);mov %%r9, 608(%%rbx);mov %%r10, 616(%%rbx);"
                "mov %%r8, 624(%%rbx);mov %%r9, 632(%%rbx);mov %%r10, 640(%%rbx);"
                "mov %%r8, 648(%%rbx);mov %%r9, 656(%%rbx);mov %%r10, 664(%%rbx);"
                "mov %%r8, 672(%%rbx);mov %%r9, 680(%%rbx);mov %%r10, 688(%%rbx);"
                "mov %%r8, 696(%%rbx);mov %%r9, 704(%%rbx);mov %%r10, 712(%%rbx);"
                "mov %%r8, 720(%%rbx);mov %%r9, 728(%%rbx);mov %%r10, 736(%%rbx);"
                "mov %%r8, 744(%%rbx);mov %%r9, 752(%%rbx);mov %%r10, 760(%%rbx);"                
                "add $768,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_mov_3;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "%r9", "%r10", "memory"
								);
								
			 ret=(((double)(passes*48*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
    case 4:
      passes=accesses/32;
      if (!passes) return 0;

      /*
       * Input: addr -> RBX (pointer to the buffer)
       *        passes -> RCX
       * Output : RAX stop timestamp - start timestamp
       */
       __asm__ __volatile__(
                 //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_mov_4:"
                "mov %%r8, (%%rbx);mov %%r9, 8(%%rbx);mov %%r10, 16(%%rbx);mov %%r11, 24(%%rbx);"
                "mov %%r8, 32(%%rbx);mov %%r9, 40(%%rbx);mov %%r10, 48(%%rbx);mov %%r11, 56(%%rbx);"
                "mov %%r8, 64(%%rbx);mov %%r9, 72(%%rbx);mov %%r10, 80(%%rbx);mov %%r11, 88(%%rbx);"
                "mov %%r8, 96(%%rbx);mov %%r9, 104(%%rbx);mov %%r10, 112(%%rbx);mov %%r11, 120(%%rbx);"
                "mov %%r8, 128(%%rbx);mov %%r9, 136(%%rbx);mov %%r10, 144(%%rbx);mov %%r11, 152(%%rbx);"
                "mov %%r8, 160(%%rbx);mov %%r9, 168(%%rbx);mov %%r10, 176(%%rbx);mov %%r11, 184(%%rbx);"
                "mov %%r8, 192(%%rbx);mov %%r9, 200(%%rbx);mov %%r10, 208(%%rbx);mov %%r11, 216(%%rbx);"
                "mov %%r8, 224(%%rbx);mov %%r9, 232(%%rbx);mov %%r10, 240(%%rbx);mov %%r11, 248(%%rbx);"
                "mov %%r8, 256(%%rbx);mov %%r9, 264(%%rbx);mov %%r10, 272(%%rbx);mov %%r11, 280(%%rbx);"
                "mov %%r8, 288(%%rbx);mov %%r9, 296(%%rbx);mov %%r10, 304(%%rbx);mov %%r11, 312(%%rbx);"
                "mov %%r8, 320(%%rbx);mov %%r9, 328(%%rbx);mov %%r10, 336(%%rbx);mov %%r11, 344(%%rbx);"
                "mov %%r8, 352(%%rbx);mov %%r9, 360(%%rbx);mov %%r10, 368(%%rbx);mov %%r11, 376(%%rbx);"
                "mov %%r8, 384(%%rbx);mov %%r9, 392(%%rbx);mov %%r10, 400(%%rbx);mov %%r11, 408(%%rbx);"
                "mov %%r8, 416(%%rbx);mov %%r9, 424(%%rbx);mov %%r10, 432(%%rbx);mov %%r11, 440(%%rbx);"
                "mov %%r8, 448(%%rbx);mov %%r9, 456(%%rbx);mov %%r10, 464(%%rbx);mov %%r11, 472(%%rbx);"
                "mov %%r8, 480(%%rbx);mov %%r9, 488(%%rbx);mov %%r10, 496(%%rbx);mov %%r11, 504(%%rbx);"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_mov_4;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                "mov %%rdx,%%rbx;"
                //second timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "sub %%rbx,%%rax;"
                : "=a" (addr)
                : "b"(addr), "c" (passes)
                : "%rdx","%r8", "%r9", "%r10", "%r11", "memory"
								);
								
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/(((double)freq)*0.000000001)));
			 break;
			 default: ret=0.0;break;
   }

  //printf("end asm\n");fflush(stdout);
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (data->num_events) PAPI_read(data->Eventset,data->values);
    for (i=0;i<data->num_events;i++)
    {
       data->papi_results[i]=(double)data->values[i]/(double)(passes*32);
    }
  #endif
	 return ret;
}

/*
 * function that does the measurement
 */
void inline _work( unsigned long long memsize, int offset, int function, int burst_length, int runs, volatile mydata_t* data, double **results)
{
  int latency,i,t;
  double tmax;
  double tmp=(double)0;
  unsigned long long tmp2,tmp3;

	/* aligned address */
	unsigned long long aligned_addr,accesses;
	
  aligned_addr=(unsigned long long)(data->buffer) + offset;

  accesses=memsize/(2*sizeof(unsigned long long));

  latency=data->cpuinfo->rdtsc_latency;

  //printf("starting measurment %i accesses in %i Bytes of memory\n",accesses,memsize);
  for (t=0;t<data->num_threads;t++)
  {
   tmax=0;
  
   if(!t) aligned_addr=(unsigned long long)(data->buffer) + offset;
   else aligned_addr=data->threaddata[t].aligned_addr;
  
   if (accesses) 
   {
    for (i=0;i<runs;i++)
    {
     //access whole buffer to warm up cache and tlb
     use_memory((void*)aligned_addr,memsize,MODE_MODIFIED,data->USE_DIRECTION,data->NUM_USES,*(data->cpuinfo));

     if ((data->USE_MODE==MODE_FORWARD))
      {
        //tell another thread to use memory
        unsigned long long tmp;
        tmp=data->threaddata[data->SHARE_CPU].aligned_addr;
        if (t) data->threaddata[data->SHARE_CPU].aligned_addr=data->threaddata[t].aligned_addr;
        else data->threaddata[data->SHARE_CPU].aligned_addr=aligned_addr;
        data->threaddata[data->SHARE_CPU].memsize=memsize;
        data->threaddata[data->SHARE_CPU].accesses=accesses;
        data->threaddata[data->SHARE_CPU].USE_MODE=MODE_EXCLUSIVE;
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[data->SHARE_CPU]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[data->SHARE_CPU]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 3\n");
        data->ack=0;
        while (!data->done); //printf("wait for done 3\n");
        data->done=0;
        data->threaddata[data->SHARE_CPU].aligned_addr=tmp;
      }
      
       if (!t) 
      {
      
        //access whole buffer to warm up cache and tlb
        if ((data->USE_MODE==MODE_SHARED)) use_memory((void*)aligned_addr,memsize,MODE_EXCLUSIVE,data->USE_DIRECTION,data->NUM_USES,*(data->cpuinfo));
        else if ((data->USE_MODE==MODE_OWNED)) use_memory((void*)aligned_addr,memsize,MODE_MODIFIED,data->USE_DIRECTION,data->NUM_USES,*(data->cpuinfo));
        else use_memory((void*)aligned_addr,memsize,data->USE_MODE,data->USE_DIRECTION,data->NUM_USES,*(data->cpuinfo));
      }

      if (t)
      {
        //tell other thread to use memory
        data->threaddata[t].memsize=memsize;
        data->threaddata[t].accesses=accesses;
        if ((data->USE_MODE==MODE_SHARED)) data->threaddata[t].USE_MODE=MODE_EXCLUSIVE;
        if ((data->USE_MODE==MODE_OWNED)) data->threaddata[t].USE_MODE=MODE_MODIFIED;
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[t]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[t]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 2\n");
        data->ack=0;
        while (!data->done);// printf("wait for done 2\n");
        data->done=0;     
        
      }
     
      //flush cachelevels as specified in PARAMETERS
      flush_caches((void*) data->threaddata[t].aligned_addr,memsize,data->settings,data->NUM_FLUSHES,data->FLUSH_MODE,data->cache_flush_area,data->cpuinfo);
          
      if ((data->USE_MODE==MODE_SHARED)||(data->USE_MODE==MODE_OWNED))
      {
        //tell another thread to use memory
        unsigned long long tmp;
        tmp=data->threaddata[data->SHARE_CPU].aligned_addr;
        if (t) data->threaddata[data->SHARE_CPU].aligned_addr=data->threaddata[t].aligned_addr;
        else data->threaddata[data->SHARE_CPU].aligned_addr=aligned_addr;
        data->threaddata[data->SHARE_CPU].memsize=memsize;
        data->threaddata[data->SHARE_CPU].accesses=accesses;
        data->threaddata[data->SHARE_CPU].USE_MODE=data->USE_MODE;
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[data->SHARE_CPU]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[data->SHARE_CPU]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 3\n");
        data->ack=0;
        while (!data->done); //printf("wait for done 3\n");
        data->done=0;
        data->threaddata[data->SHARE_CPU].aligned_addr=tmp;
      }
           
      /* call ASM implementation */
      //printf("call asm impl.\n");
      switch(function)
      {
        case USE_MOVNTDQ: tmp=asm_work_movntdq(aligned_addr,accesses,burst_length,latency,data->cpuinfo->clockrate,data);break;
        case USE_MOVDQA: tmp=asm_work_movdqa(aligned_addr,accesses,burst_length,latency,data->cpuinfo->clockrate,data);break;
        case USE_MOVDQU: tmp=asm_work_movdqu(aligned_addr,accesses,burst_length,latency,data->cpuinfo->clockrate,data);break;
        case USE_MOVNTI: tmp=asm_work_movnti(aligned_addr,accesses,burst_length,latency,data->cpuinfo->clockrate,data);break;
        case USE_MOV: tmp=asm_work_mov(aligned_addr,accesses,burst_length,latency,data->cpuinfo->clockrate,data);break;
        default: break;
      }
      if ((int)tmp!=-1)
      {
       if (tmp>tmax) tmax=tmp;
      }
    }
   }
   else tmax=0;
  
   if (tmax) (*results)[t]=tmax;
   else (*results)[t]=INVALID_MEASUREMENT;
  }
}
