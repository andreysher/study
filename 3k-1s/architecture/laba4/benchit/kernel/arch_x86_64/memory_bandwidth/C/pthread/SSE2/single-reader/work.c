/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures read bandwidth of data located in different cache
 *         levels or memory of certain CPUs.
 *******************************************************************/

/*
 * TODO - adopt cache and TLB parameters to refer to identifiers returned by 
 *        the hardware detection
 *      - AVX and Larrabee support
 *      - support low level Performance Counter APIs to get access to uncore/NB events
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
 * assembler implementation of bandwidth measurement using movdqa instruction
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
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqa_1:"
                "movdqa (%%rbx), %%xmm0;"
                "movdqa 16(%%rbx), %%xmm0;"
                "movdqa 32(%%rbx), %%xmm0;"
                "movdqa 48(%%rbx), %%xmm0;"
                "movdqa 64(%%rbx), %%xmm0;"
                "movdqa 80(%%rbx), %%xmm0;"
                "movdqa 96(%%rbx), %%xmm0;"
                "movdqa 112(%%rbx), %%xmm0;"
                "movdqa 128(%%rbx), %%xmm0;"
                "movdqa 144(%%rbx), %%xmm0;"
                "movdqa 160(%%rbx), %%xmm0;"
                "movdqa 176(%%rbx), %%xmm0;"
                "movdqa 192(%%rbx), %%xmm0;"
                "movdqa 208(%%rbx), %%xmm0;"
                "movdqa 224(%%rbx), %%xmm0;"
                "movdqa 240(%%rbx), %%xmm0;"
                "movdqa 256(%%rbx), %%xmm0;"
                "movdqa 272(%%rbx), %%xmm0;"
                "movdqa 288(%%rbx), %%xmm0;"
                "movdqa 304(%%rbx), %%xmm0;"
                "movdqa 320(%%rbx), %%xmm0;"
                "movdqa 336(%%rbx), %%xmm0;"
                "movdqa 352(%%rbx), %%xmm0;"
                "movdqa 368(%%rbx), %%xmm0;"
                "movdqa 384(%%rbx), %%xmm0;"
                "movdqa 400(%%rbx), %%xmm0;"
                "movdqa 416(%%rbx), %%xmm0;"
                "movdqa 432(%%rbx), %%xmm0;"
                "movdqa 448(%%rbx), %%xmm0;"
                "movdqa 464(%%rbx), %%xmm0;"
                "movdqa 480(%%rbx), %%xmm0;"
                "movdqa 496(%%rbx), %%xmm0;"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqa_1;"
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
                : "%rdx","%xmm0"
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
                "_work_loop_movdqa_2:"
                "movdqa (%%rbx), %%xmm0;movdqa 16(%%rbx), %%xmm1;"
                "movdqa 32(%%rbx), %%xmm0;movdqa 48(%%rbx), %%xmm1;"
                "movdqa 64(%%rbx), %%xmm0;movdqa 80(%%rbx), %%xmm1;"
                "movdqa 96(%%rbx), %%xmm0;movdqa 112(%%rbx), %%xmm1;"
                "movdqa 128(%%rbx), %%xmm0;movdqa 144(%%rbx), %%xmm1;"
                "movdqa 160(%%rbx), %%xmm0;movdqa 176(%%rbx), %%xmm1;"
                "movdqa 192(%%rbx), %%xmm0;movdqa 208(%%rbx), %%xmm1;"
                "movdqa 224(%%rbx), %%xmm0;movdqa 240(%%rbx), %%xmm1;"
                "movdqa 256(%%rbx), %%xmm0;movdqa 272(%%rbx), %%xmm1;"
                "movdqa 288(%%rbx), %%xmm0;movdqa 304(%%rbx), %%xmm1;"
                "movdqa 320(%%rbx), %%xmm0;movdqa 336(%%rbx), %%xmm1;"
                "movdqa 352(%%rbx), %%xmm0;movdqa 368(%%rbx), %%xmm1;"
                "movdqa 384(%%rbx), %%xmm0;movdqa 400(%%rbx), %%xmm1;"
                "movdqa 416(%%rbx), %%xmm0;movdqa 432(%%rbx), %%xmm1;"
                "movdqa 448(%%rbx), %%xmm0;movdqa 464(%%rbx), %%xmm1;"
                "movdqa 480(%%rbx), %%xmm0;movdqa 496(%%rbx), %%xmm1;"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqa_2;"
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
                : "%rdx","%xmm0", "%xmm1"
								);
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/freq))*0.000000001;
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
                "_work_loop_movdqa_3:"
                "movdqa (%%rbx), %%xmm0;movdqa 16(%%rbx), %%xmm1;movdqa 32(%%rbx), %%xmm2;"
                "movdqa 48(%%rbx), %%xmm0;movdqa 64(%%rbx), %%xmm1;movdqa 80(%%rbx), %%xmm2;"
                "movdqa 96(%%rbx), %%xmm0;movdqa 112(%%rbx), %%xmm1;movdqa 128(%%rbx), %%xmm2;"
                "movdqa 144(%%rbx), %%xmm0;movdqa 160(%%rbx), %%xmm1;movdqa 176(%%rbx), %%xmm2;"
                "movdqa 192(%%rbx), %%xmm0;movdqa 208(%%rbx), %%xmm1;movdqa 224(%%rbx), %%xmm2;"
                "movdqa 240(%%rbx), %%xmm0;movdqa 256(%%rbx), %%xmm1;movdqa 272(%%rbx), %%xmm2;"
                "movdqa 288(%%rbx), %%xmm0;movdqa 304(%%rbx), %%xmm1;movdqa 320(%%rbx), %%xmm2;"
                "movdqa 336(%%rbx), %%xmm0;movdqa 352(%%rbx), %%xmm1;movdqa 368(%%rbx), %%xmm2;"
                "movdqa 384(%%rbx), %%xmm0;movdqa 400(%%rbx), %%xmm1;movdqa 416(%%rbx), %%xmm2;"
                "movdqa 432(%%rbx), %%xmm0;movdqa 448(%%rbx), %%xmm1;movdqa 464(%%rbx), %%xmm2;"
                "movdqa 480(%%rbx), %%xmm0;movdqa 496(%%rbx), %%xmm1;movdqa 512(%%rbx), %%xmm2;"
                "movdqa 528(%%rbx), %%xmm0;movdqa 544(%%rbx), %%xmm1;movdqa 560(%%rbx), %%xmm2;"
                "movdqa 576(%%rbx), %%xmm0;movdqa 592(%%rbx), %%xmm1;movdqa 608(%%rbx), %%xmm2;"
                "movdqa 624(%%rbx), %%xmm0;movdqa 640(%%rbx), %%xmm1;movdqa 656(%%rbx), %%xmm2;"
                "movdqa 672(%%rbx), %%xmm0;movdqa 688(%%rbx), %%xmm1;movdqa 704(%%rbx), %%xmm2;"
                "movdqa 720(%%rbx), %%xmm0;movdqa 736(%%rbx), %%xmm1;movdqa 752(%%rbx), %%xmm2;"                
                "add $768,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqa_3;"
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
                : "%rdx","%xmm0", "%xmm1", "%xmm2"
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
                "_work_loop_movdqa_4:"
                "movdqa (%%rbx), %%xmm0;movdqa 16(%%rbx), %%xmm1;movdqa 32(%%rbx), %%xmm2;movdqa 48(%%rbx), %%xmm3;"
                "movdqa 64(%%rbx), %%xmm0;movdqa 80(%%rbx), %%xmm1;movdqa 96(%%rbx), %%xmm2;movdqa 112(%%rbx), %%xmm3;"
                "movdqa 128(%%rbx), %%xmm0;movdqa 144(%%rbx), %%xmm1;movdqa 160(%%rbx), %%xmm2;movdqa 176(%%rbx), %%xmm3;"
                "movdqa 192(%%rbx), %%xmm0;movdqa 208(%%rbx), %%xmm1;movdqa 224(%%rbx), %%xmm2;movdqa 240(%%rbx), %%xmm3;"
                "movdqa 256(%%rbx), %%xmm0;movdqa 272(%%rbx), %%xmm1;movdqa 288(%%rbx), %%xmm2;movdqa 304(%%rbx), %%xmm3;"
                "movdqa 320(%%rbx), %%xmm0;movdqa 336(%%rbx), %%xmm1;movdqa 352(%%rbx), %%xmm2;movdqa 368(%%rbx), %%xmm3;"
                "movdqa 384(%%rbx), %%xmm0;movdqa 400(%%rbx), %%xmm1;movdqa 416(%%rbx), %%xmm2;movdqa 432(%%rbx), %%xmm3;"
                "movdqa 448(%%rbx), %%xmm0;movdqa 464(%%rbx), %%xmm1;movdqa 480(%%rbx), %%xmm2;movdqa 496(%%rbx), %%xmm3;"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqa_4;"
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
                : "%rdx","%xmm0", "%xmm1", "%xmm2", "%xmm3"
								);
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/freq))*0.000000001;
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
 * assembler implementation of bandwidth measurement using movdqu instruction
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
                "mov %%rax,%%rdx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_work_loop_movdqu_1:"
                "movdqu (%%rbx), %%xmm0;"
                "movdqu 16(%%rbx), %%xmm0;"
                "movdqu 32(%%rbx), %%xmm0;"
                "movdqu 48(%%rbx), %%xmm0;"
                "movdqu 64(%%rbx), %%xmm0;"
                "movdqu 80(%%rbx), %%xmm0;"
                "movdqu 96(%%rbx), %%xmm0;"
                "movdqu 112(%%rbx), %%xmm0;"
                "movdqu 128(%%rbx), %%xmm0;"
                "movdqu 144(%%rbx), %%xmm0;"
                "movdqu 160(%%rbx), %%xmm0;"
                "movdqu 176(%%rbx), %%xmm0;"
                "movdqu 192(%%rbx), %%xmm0;"
                "movdqu 208(%%rbx), %%xmm0;"
                "movdqu 224(%%rbx), %%xmm0;"
                "movdqu 240(%%rbx), %%xmm0;"
                "movdqu 256(%%rbx), %%xmm0;"
                "movdqu 272(%%rbx), %%xmm0;"
                "movdqu 288(%%rbx), %%xmm0;"
                "movdqu 304(%%rbx), %%xmm0;"
                "movdqu 320(%%rbx), %%xmm0;"
                "movdqu 336(%%rbx), %%xmm0;"
                "movdqu 352(%%rbx), %%xmm0;"
                "movdqu 368(%%rbx), %%xmm0;"
                "movdqu 384(%%rbx), %%xmm0;"
                "movdqu 400(%%rbx), %%xmm0;"
                "movdqu 416(%%rbx), %%xmm0;"
                "movdqu 432(%%rbx), %%xmm0;"
                "movdqu 448(%%rbx), %%xmm0;"
                "movdqu 464(%%rbx), %%xmm0;"
                "movdqu 480(%%rbx), %%xmm0;"
                "movdqu 496(%%rbx), %%xmm0;"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqu_1;"
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
                : "%rdx","%xmm0"
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
                "_work_loop_movdqu_2:"
                "movdqu (%%rbx), %%xmm0;movdqu 16(%%rbx), %%xmm1;"
                "movdqu 32(%%rbx), %%xmm0;movdqu 48(%%rbx), %%xmm1;"
                "movdqu 64(%%rbx), %%xmm0;movdqu 80(%%rbx), %%xmm1;"
                "movdqu 96(%%rbx), %%xmm0;movdqu 112(%%rbx), %%xmm1;"
                "movdqu 128(%%rbx), %%xmm0;movdqu 144(%%rbx), %%xmm1;"
                "movdqu 160(%%rbx), %%xmm0;movdqu 176(%%rbx), %%xmm1;"
                "movdqu 192(%%rbx), %%xmm0;movdqu 208(%%rbx), %%xmm1;"
                "movdqu 224(%%rbx), %%xmm0;movdqu 240(%%rbx), %%xmm1;"
                "movdqu 256(%%rbx), %%xmm0;movdqu 272(%%rbx), %%xmm1;"
                "movdqu 288(%%rbx), %%xmm0;movdqu 304(%%rbx), %%xmm1;"
                "movdqu 320(%%rbx), %%xmm0;movdqu 336(%%rbx), %%xmm1;"
                "movdqu 352(%%rbx), %%xmm0;movdqu 368(%%rbx), %%xmm1;"
                "movdqu 384(%%rbx), %%xmm0;movdqu 400(%%rbx), %%xmm1;"
                "movdqu 416(%%rbx), %%xmm0;movdqu 432(%%rbx), %%xmm1;"
                "movdqu 448(%%rbx), %%xmm0;movdqu 464(%%rbx), %%xmm1;"
                "movdqu 480(%%rbx), %%xmm0;movdqu 496(%%rbx), %%xmm1;"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqu_2;"
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
                : "%rdx","%xmm0", "%xmm1"
								);
			 ret=(((double)(passes*32*16))/((double)(((addr)-call_latency))/freq))*0.000000001;
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
                "_work_loop_movdqu_3:"
                "movdqu (%%rbx), %%xmm0;movdqu 16(%%rbx), %%xmm1;movdqu 32(%%rbx), %%xmm2;"
                "movdqu 48(%%rbx), %%xmm0;movdqu 64(%%rbx), %%xmm1;movdqu 80(%%rbx), %%xmm2;"
                "movdqu 96(%%rbx), %%xmm0;movdqu 112(%%rbx), %%xmm1;movdqu 128(%%rbx), %%xmm2;"
                "movdqu 144(%%rbx), %%xmm0;movdqu 160(%%rbx), %%xmm1;movdqu 176(%%rbx), %%xmm2;"
                "movdqu 192(%%rbx), %%xmm0;movdqu 208(%%rbx), %%xmm1;movdqu 224(%%rbx), %%xmm2;"
                "movdqu 240(%%rbx), %%xmm0;movdqu 256(%%rbx), %%xmm1;movdqu 272(%%rbx), %%xmm2;"
                "movdqu 288(%%rbx), %%xmm0;movdqu 304(%%rbx), %%xmm1;movdqu 320(%%rbx), %%xmm2;"
                "movdqu 336(%%rbx), %%xmm0;movdqu 352(%%rbx), %%xmm1;movdqu 368(%%rbx), %%xmm2;"
                "movdqu 384(%%rbx), %%xmm0;movdqu 400(%%rbx), %%xmm1;movdqu 416(%%rbx), %%xmm2;"
                "movdqu 432(%%rbx), %%xmm0;movdqu 448(%%rbx), %%xmm1;movdqu 464(%%rbx), %%xmm2;"
                "movdqu 480(%%rbx), %%xmm0;movdqu 496(%%rbx), %%xmm1;movdqu 512(%%rbx), %%xmm2;"
                "movdqu 528(%%rbx), %%xmm0;movdqu 544(%%rbx), %%xmm1;movdqu 560(%%rbx), %%xmm2;"
                "movdqu 576(%%rbx), %%xmm0;movdqu 592(%%rbx), %%xmm1;movdqu 608(%%rbx), %%xmm2;"
                "movdqu 624(%%rbx), %%xmm0;movdqu 640(%%rbx), %%xmm1;movdqu 656(%%rbx), %%xmm2;"
                "movdqu 672(%%rbx), %%xmm0;movdqu 688(%%rbx), %%xmm1;movdqu 704(%%rbx), %%xmm2;"
                "movdqu 720(%%rbx), %%xmm0;movdqu 736(%%rbx), %%xmm1;movdqu 752(%%rbx), %%xmm2;"                
                "add $768,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqu_3;"
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
                : "%rdx","%xmm0", "%xmm1", "%xmm2"
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
                "_work_loop_movdqu_4:"
                "movdqu (%%rbx), %%xmm0;movdqu 16(%%rbx), %%xmm1;movdqu 32(%%rbx), %%xmm2;movdqu 48(%%rbx), %%xmm3;"
                "movdqu 64(%%rbx), %%xmm0;movdqu 80(%%rbx), %%xmm1;movdqu 96(%%rbx), %%xmm2;movdqu 112(%%rbx), %%xmm3;"
                "movdqu 128(%%rbx), %%xmm0;movdqu 144(%%rbx), %%xmm1;movdqu 160(%%rbx), %%xmm2;movdqu 176(%%rbx), %%xmm3;"
                "movdqu 192(%%rbx), %%xmm0;movdqu 208(%%rbx), %%xmm1;movdqu 224(%%rbx), %%xmm2;movdqu 240(%%rbx), %%xmm3;"
                "movdqu 256(%%rbx), %%xmm0;movdqu 272(%%rbx), %%xmm1;movdqu 288(%%rbx), %%xmm2;movdqu 304(%%rbx), %%xmm3;"
                "movdqu 320(%%rbx), %%xmm0;movdqu 336(%%rbx), %%xmm1;movdqu 352(%%rbx), %%xmm2;movdqu 368(%%rbx), %%xmm3;"
                "movdqu 384(%%rbx), %%xmm0;movdqu 400(%%rbx), %%xmm1;movdqu 416(%%rbx), %%xmm2;movdqu 432(%%rbx), %%xmm3;"
                "movdqu 448(%%rbx), %%xmm0;movdqu 464(%%rbx), %%xmm1;movdqu 480(%%rbx), %%xmm2;movdqu 496(%%rbx), %%xmm3;"
                "add $512,%%rbx;"
                "sub $1,%%rcx;"
                "jnz _work_loop_movdqu_4;"
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
                : "%rdx","%xmm0", "%xmm1", "%xmm2", "%xmm3"
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
                "mov (%%rbx), %%r8;"
                "mov 8(%%rbx), %%r8;"
                "mov 16(%%rbx), %%r8;"
                "mov 24(%%rbx), %%r8;"
                "mov 32(%%rbx), %%r8;"
                "mov 40(%%rbx), %%r8;"
                "mov 48(%%rbx), %%r8;"
                "mov 56(%%rbx), %%r8;"
                "mov 64(%%rbx), %%r8;"
                "mov 72(%%rbx), %%r8;"
                "mov 80(%%rbx), %%r8;"
                "mov 88(%%rbx), %%r8;"
                "mov 96(%%rbx), %%r8;"
                "mov 104(%%rbx), %%r8;"
                "mov 112(%%rbx), %%r8;"
                "mov 120(%%rbx), %%r8;"
                "mov 128(%%rbx), %%r8;"
                "mov 136(%%rbx), %%r8;"
                "mov 144(%%rbx), %%r8;"
                "mov 152(%%rbx), %%r8;"
                "mov 160(%%rbx), %%r8;"
                "mov 168(%%rbx), %%r8;"
                "mov 176(%%rbx), %%r8;"
                "mov 184(%%rbx), %%r8;"
                "mov 192(%%rbx), %%r8;"
                "mov 200(%%rbx), %%r8;"
                "mov 208(%%rbx), %%r8;"
                "mov 216(%%rbx), %%r8;"
                "mov 224(%%rbx), %%r8;"
                "mov 232(%%rbx), %%r8;"
                "mov 240(%%rbx), %%r8;"
                "mov 248(%%rbx), %%r8;"
                "mov 256(%%rbx), %%r8;"
                "mov 264(%%rbx), %%r8;"
                "mov 272(%%rbx), %%r8;"
                "mov 280(%%rbx), %%r8;"
                "mov 288(%%rbx), %%r8;"
                "mov 296(%%rbx), %%r8;"
                "mov 304(%%rbx), %%r8;"
                "mov 312(%%rbx), %%r8;"
                "mov 320(%%rbx), %%r8;"
                "mov 328(%%rbx), %%r8;"
                "mov 336(%%rbx), %%r8;"
                "mov 344(%%rbx), %%r8;"
                "mov 352(%%rbx), %%r8;"
                "mov 360(%%rbx), %%r8;"
                "mov 368(%%rbx), %%r8;"
                "mov 376(%%rbx), %%r8;"
                "mov 384(%%rbx), %%r8;"
                "mov 392(%%rbx), %%r8;"
                "mov 400(%%rbx), %%r8;"
                "mov 408(%%rbx), %%r8;"
                "mov 416(%%rbx), %%r8;"
                "mov 424(%%rbx), %%r8;"
                "mov 432(%%rbx), %%r8;"
                "mov 440(%%rbx), %%r8;"
                "mov 448(%%rbx), %%r8;"
                "mov 456(%%rbx), %%r8;"
                "mov 464(%%rbx), %%r8;"
                "mov 472(%%rbx), %%r8;"
                "mov 480(%%rbx), %%r8;"
                "mov 488(%%rbx), %%r8;"
                "mov 496(%%rbx), %%r8;"
                "mov 504(%%rbx), %%r8;"
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
                : "%rdx","%r8"
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
                "mov (%%rbx), %%r8;mov 8(%%rbx), %%r9;"
                "mov 16(%%rbx), %%r8;mov 24(%%rbx), %%r9;"
                "mov 32(%%rbx), %%r8;mov 40(%%rbx), %%r9;"
                "mov 48(%%rbx), %%r8;mov 56(%%rbx), %%r9;"
                "mov 64(%%rbx), %%r8;mov 72(%%rbx), %%r9;"
                "mov 80(%%rbx), %%r8;mov 88(%%rbx), %%r9;"
                "mov 96(%%rbx), %%r8;mov 104(%%rbx), %%r9;"
                "mov 112(%%rbx), %%r8;mov 120(%%rbx), %%r9;"
                "mov 128(%%rbx), %%r8;mov 136(%%rbx), %%r9;"
                "mov 144(%%rbx), %%r8;mov 152(%%rbx), %%r9;"
                "mov 160(%%rbx), %%r8;mov 168(%%rbx), %%r9;"
                "mov 176(%%rbx), %%r8;mov 184(%%rbx), %%r9;"
                "mov 192(%%rbx), %%r8;mov 200(%%rbx), %%r9;"
                "mov 208(%%rbx), %%r8;mov 216(%%rbx), %%r9;"
                "mov 224(%%rbx), %%r8;mov 232(%%rbx), %%r9;"
                "mov 240(%%rbx), %%r8;mov 248(%%rbx), %%r9;"
                "mov 256(%%rbx), %%r8;mov 264(%%rbx), %%r9;"
                "mov 272(%%rbx), %%r8;mov 280(%%rbx), %%r9;"
                "mov 288(%%rbx), %%r8;mov 296(%%rbx), %%r9;"
                "mov 304(%%rbx), %%r8;mov 312(%%rbx), %%r9;"
                "mov 320(%%rbx), %%r8;mov 328(%%rbx), %%r9;"
                "mov 336(%%rbx), %%r8;mov 344(%%rbx), %%r9;"
                "mov 352(%%rbx), %%r8;mov 360(%%rbx), %%r9;"
                "mov 368(%%rbx), %%r8;mov 376(%%rbx), %%r9;"
                "mov 384(%%rbx), %%r8;mov 392(%%rbx), %%r9;"
                "mov 400(%%rbx), %%r8;mov 408(%%rbx), %%r9;"
                "mov 416(%%rbx), %%r8;mov 424(%%rbx), %%r9;"
                "mov 432(%%rbx), %%r8;mov 440(%%rbx), %%r9;"
                "mov 448(%%rbx), %%r8;mov 456(%%rbx), %%r9;"
                "mov 464(%%rbx), %%r8;mov 472(%%rbx), %%r9;"
                "mov 480(%%rbx), %%r8;mov 488(%%rbx), %%r9;"
                "mov 496(%%rbx), %%r8;mov 504(%%rbx), %%r9;"
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
                : "%rdx","%r8", "%r9"
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
                "mov (%%rbx), %%r8;mov 8(%%rbx), %%r9;mov 16(%%rbx), %%r10;"
                "mov 24(%%rbx), %%r8;mov 32(%%rbx), %%r9;mov 40(%%rbx), %%r10;"
                "mov 48(%%rbx), %%r8;mov 56(%%rbx), %%r9;mov 64(%%rbx), %%r10;"
                "mov 72(%%rbx), %%r8;mov 80(%%rbx), %%r9;mov 88(%%rbx), %%r10;"
                "mov 96(%%rbx), %%r8;mov 104(%%rbx), %%r9;mov 112(%%rbx), %%r10;"
                "mov 120(%%rbx), %%r8;mov 128(%%rbx), %%r9;mov 136(%%rbx), %%r10;"
                "mov 144(%%rbx), %%r8;mov 152(%%rbx), %%r9;mov 160(%%rbx), %%r10;"
                "mov 168(%%rbx), %%r8;mov 176(%%rbx), %%r9;mov 184(%%rbx), %%r10;"
                "mov 192(%%rbx), %%r8;mov 200(%%rbx), %%r9;mov 208(%%rbx), %%r10;"
                "mov 216(%%rbx), %%r8;mov 224(%%rbx), %%r9;mov 232(%%rbx), %%r10;"
                "mov 240(%%rbx), %%r8;mov 248(%%rbx), %%r9;mov 256(%%rbx), %%r10;"
                "mov 264(%%rbx), %%r8;mov 272(%%rbx), %%r9;mov 280(%%rbx), %%r10;"
                "mov 288(%%rbx), %%r8;mov 296(%%rbx), %%r9;mov 304(%%rbx), %%r10;"
                "mov 312(%%rbx), %%r8;mov 320(%%rbx), %%r9;mov 328(%%rbx), %%r10;"
                "mov 336(%%rbx), %%r8;mov 344(%%rbx), %%r9;mov 352(%%rbx), %%r10;"
                "mov 360(%%rbx), %%r8;mov 368(%%rbx), %%r9;mov 376(%%rbx), %%r10;"
                "mov 384(%%rbx), %%r8;mov 392(%%rbx), %%r9;mov 400(%%rbx), %%r10;"
                "mov 408(%%rbx), %%r8;mov 416(%%rbx), %%r9;mov 424(%%rbx), %%r10;"
                "mov 432(%%rbx), %%r8;mov 440(%%rbx), %%r9;mov 448(%%rbx), %%r10;"
                "mov 456(%%rbx), %%r8;mov 464(%%rbx), %%r9;mov 472(%%rbx), %%r10;"
                "mov 480(%%rbx), %%r8;mov 488(%%rbx), %%r9;mov 496(%%rbx), %%r10;"
                "mov 504(%%rbx), %%r8;mov 512(%%rbx), %%r9;mov 520(%%rbx), %%r10;"
                "mov 528(%%rbx), %%r8;mov 536(%%rbx), %%r9;mov 544(%%rbx), %%r10;"
                "mov 552(%%rbx), %%r8;mov 560(%%rbx), %%r9;mov 568(%%rbx), %%r10;"
                "mov 576(%%rbx), %%r8;mov 584(%%rbx), %%r9;mov 592(%%rbx), %%r10;"
                "mov 600(%%rbx), %%r8;mov 608(%%rbx), %%r9;mov 616(%%rbx), %%r10;"
                "mov 624(%%rbx), %%r8;mov 632(%%rbx), %%r9;mov 640(%%rbx), %%r10;"
                "mov 648(%%rbx), %%r8;mov 656(%%rbx), %%r9;mov 664(%%rbx), %%r10;"
                "mov 672(%%rbx), %%r8;mov 680(%%rbx), %%r9;mov 688(%%rbx), %%r10;"
                "mov 696(%%rbx), %%r8;mov 704(%%rbx), %%r9;mov 712(%%rbx), %%r10;"
                "mov 720(%%rbx), %%r8;mov 728(%%rbx), %%r9;mov 736(%%rbx), %%r10;"
                "mov 744(%%rbx), %%r8;mov 752(%%rbx), %%r9;mov 760(%%rbx), %%r10;"                
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
                : "%rdx","%r8", "%r9", "%r10"
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
                "mov (%%rbx), %%r8;mov 8(%%rbx), %%r9;mov 16(%%rbx), %%r10;mov 24(%%rbx), %%r11;"
                "mov 32(%%rbx), %%r8;mov 40(%%rbx), %%r9;mov 48(%%rbx), %%r10;mov 56(%%rbx), %%r11;"
                "mov 64(%%rbx), %%r8;mov 72(%%rbx), %%r9;mov 80(%%rbx), %%r10;mov 88(%%rbx), %%r11;"
                "mov 96(%%rbx), %%r8;mov 104(%%rbx), %%r9;mov 112(%%rbx), %%r10;mov 120(%%rbx), %%r11;"
                "mov 128(%%rbx), %%r8;mov 136(%%rbx), %%r9;mov 144(%%rbx), %%r10;mov 152(%%rbx), %%r11;"
                "mov 160(%%rbx), %%r8;mov 168(%%rbx), %%r9;mov 176(%%rbx), %%r10;mov 184(%%rbx), %%r11;"
                "mov 192(%%rbx), %%r8;mov 200(%%rbx), %%r9;mov 208(%%rbx), %%r10;mov 216(%%rbx), %%r11;"
                "mov 224(%%rbx), %%r8;mov 232(%%rbx), %%r9;mov 240(%%rbx), %%r10;mov 248(%%rbx), %%r11;"
                "mov 256(%%rbx), %%r8;mov 264(%%rbx), %%r9;mov 272(%%rbx), %%r10;mov 280(%%rbx), %%r11;"
                "mov 288(%%rbx), %%r8;mov 296(%%rbx), %%r9;mov 304(%%rbx), %%r10;mov 312(%%rbx), %%r11;"
                "mov 320(%%rbx), %%r8;mov 328(%%rbx), %%r9;mov 336(%%rbx), %%r10;mov 344(%%rbx), %%r11;"
                "mov 352(%%rbx), %%r8;mov 360(%%rbx), %%r9;mov 368(%%rbx), %%r10;mov 376(%%rbx), %%r11;"
                "mov 384(%%rbx), %%r8;mov 392(%%rbx), %%r9;mov 400(%%rbx), %%r10;mov 408(%%rbx), %%r11;"
                "mov 416(%%rbx), %%r8;mov 424(%%rbx), %%r9;mov 432(%%rbx), %%r10;mov 440(%%rbx), %%r11;"
                "mov 448(%%rbx), %%r8;mov 456(%%rbx), %%r9;mov 464(%%rbx), %%r10;mov 472(%%rbx), %%r11;"
                "mov 480(%%rbx), %%r8;mov 488(%%rbx), %%r9;mov 496(%%rbx), %%r10;mov 504(%%rbx), %%r11;"
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
                : "%rdx","%r8", "%r9", "%r10", "%r11"
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
        case USE_MOVDQA: tmp=asm_work_movdqa(aligned_addr,accesses,burst_length,latency,data->cpuinfo->clockrate,data);break;
        case USE_MOVDQU: tmp=asm_work_movdqu(aligned_addr,accesses,burst_length,latency,data->cpuinfo->clockrate,data);break;
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
