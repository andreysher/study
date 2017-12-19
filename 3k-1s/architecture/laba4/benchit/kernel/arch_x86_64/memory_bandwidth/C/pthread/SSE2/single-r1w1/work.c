/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id$
 * $URL$
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: measures combined bandwidth of one read and one write stream located
 *         in different cache levels or memory of certain CPUs.
 *******************************************************************/
 
/*
 * TODO - adopt cache and TLB parameters to refer to identifiers returned by 
 *        the hardware detection
 *      - AVX and Larrabee support
 *      - support low level Performance Counter APIs to get access to uncore/NB events
 *      - optional local alloc of flush buffer
 */
 
#include "interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "work.h"
#include "shared.h"

#ifdef USE_PAPI
#include <papi.h>
#endif

/* some defines to make code a little more readable */
#define PARAMS ((param_t*)aligned_addr)
#define THREAD_PARAMS ((param_t*)aligned_addr)->thread_params
#define SHARE_CPU_PARAMS ((param_t*)aligned_addr)->share_cpu_params

/*
 * use a block of memory to ensure it is in the caches afterwards
 * MODE_EXCLUSIVE: - cache line will be exclusive in cache of calling CPU
 * MODE_MODIFIED:  - cache line will be modified in cache of calling CPU
 * MODE_INVALID:   - cache line will be invalid in all caches
 * MODE_SHARED/MODE_OWNED/MODE_FORWARD:
 *   - these modes perform a read-only access (on SHARE_CPU)
 *   - together with accesses on another CPU with MODE_{EXCLUSIVE|MODIFIED} cache line will be in the
 *     desired coherency state in the cache of the OTHER CPU, and SHARED in the cache of SHARE_CPU
 *     (see USE MODA ADAPTION in file work.c)
 */
void inline use_memory(param_t *params)
{
  /* create TLB entries for the whole area */
  /* deactivated as it seems to make things worse rather than improve anything */
  /* params->j = params->num_uses; 
  while(params->j--) 
  {
    if (params->use_mode_1!=MODE_DISABLED) for (params->i=0;params->i<params->memsize/2;params->i+=1024)
     __asm__ __volatile__("movdqa (%%rax),%%xmm0;":: "a" (params->addr_1+params->i): "%xmm0");
    if (params->use_mode_2!=MODE_DISABLED) for (params->i=0;params->i<params->memsize/2;params->i+=1024)
     __asm__ __volatile__("movdqa (%%rax),%%xmm0;":: "a" (params->addr_2+params->i): "%xmm0");      
  } */
  /* now cached page table entries can be evicted without penalty */
  
  if (params->use_direction==FIFO)
  {
     if ((params->use_mode_1&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID))&&(params->use_mode_2&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID))){
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_1+params->i))=(double)params->value;
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_2+params->i))=(double)params->value;
      }
     }
     else if (params->use_mode_1&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID)) {
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_1+params->i))=(double)params->value;
      }
     }
     else if (params->use_mode_2&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID)) {
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_2+params->i))=(double)params->value;
      } 
     } /* -> modified */
     params->j = 1;
     if (params->use_mode_1&(MODE_EXCLUSIVE|MODE_INVALID)) while(params->j--) {
      __asm__ __volatile__("mfence;"::);      
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
         __asm__ __volatile__("clflush (%%rax);":: "a" (params->addr_1+params->i));     
      }
     }
      __asm__ __volatile__("mfence;"::);
     params->j = 1;
     if (params->use_mode_2&(MODE_EXCLUSIVE|MODE_INVALID)) while(params->j--) {
      __asm__ __volatile__("mfence;"::);      
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
         __asm__ __volatile__("clflush (%%rax);":: "a" (params->addr_2+params->i));       
      }
      __asm__ __volatile__("mfence;"::);
     } /* -> invalid */
     if ((params->use_mode_1&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD)) && (params->use_mode_2&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD))) {
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_1+params->i));
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_2+params->i));
      }
     } 
     else if (params->use_mode_1&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD)) {
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_1+params->i));
         if (params->use_mode_2&MODE_MODIFIED) {params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_2+params->i));}
      }
     }      
     else if (params->use_mode_2&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD)) {
      for (params->i=0;params->i<params->memsize/2;params->i+=STRIDE) {
         if (params->use_mode_1&MODE_MODIFIED) {params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_1+params->i));}
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_2+params->i));
      }            
     }  /* -> exclusive / shared 
         * modified stream is read again which does not change coherency state*/
  }
  else
  {
  if ((params->use_mode_1&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID))&&(params->use_mode_2&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID))){
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE) {
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_1+params->i))=(double)params->value;
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_2+params->i))=(double)params->value;
      }
     }
     else if (params->use_mode_1&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID)) {
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE) {
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_1+params->i))=(double)params->value;
      }
     }
     else if (params->use_mode_2&(MODE_EXCLUSIVE|MODE_MODIFIED|MODE_INVALID)) {
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE) {
        params->j = params->num_uses; while(params->j--) *((double*)(params->addr_2+params->i))=(double)params->value;
      } 
     } /* -> modified */
     params->j = 1;
     if (params->use_mode_1&(MODE_EXCLUSIVE|MODE_INVALID)) while(params->j--) {
      __asm__ __volatile__("mfence;"::);      
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE) {
         __asm__ __volatile__("clflush (%%rax);":: "a" (params->addr_1+params->i));     
      }
     }
      __asm__ __volatile__("mfence;"::);
     params->j = 1;
     if (params->use_mode_2&(MODE_EXCLUSIVE|MODE_INVALID)) while(params->j--) {
      __asm__ __volatile__("mfence;"::);      
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE) {
         __asm__ __volatile__("clflush (%%rax);":: "a" (params->addr_2+params->i));       
      }
      __asm__ __volatile__("mfence;"::);
     } /* -> invalid */
     if ((params->use_mode_1&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD)) && (params->use_mode_2&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD))) {
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE) {
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_1+params->i));
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_2+params->i));
      }
     } 
     else if (params->use_mode_1&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD)) {
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE) {
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_1+params->i));
         if (params->use_mode_2&MODE_MODIFIED) {params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_2+params->i));}
      }
     }      
     else if (params->use_mode_2&(MODE_EXCLUSIVE|MODE_SHARED|MODE_OWNED|MODE_FORWARD)) {
      for (params->i=params->memsize/2-STRIDE;params->i>=0;params->i-=STRIDE)       {
         if (params->use_mode_1&MODE_MODIFIED) {params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_1+params->i));}
         params->j = params->num_uses; while(params->j--) params->value= *((double*)(params->addr_2+params->i));
      }            
     }  /* -> exclusive / shared 
         * modified stream is read again which does not change coherency state*/
  }
  /* rebuild TLB entries for the whole area */
  /* deactivated as it seems to make things worse rather than improve anything */
  /* params->j = params->num_uses; 
  while(params->j--)
  {
    if (params->use_mode_1!=MODE_DISABLED) for (params->i=0;params->i<params->memsize/2;params->i+=4096)
     __asm__ __volatile__("movdqa (%%rax),%%xmm0;":: "a" (params->addr_1+params->i): "%xmm0");
    if (params->use_mode_2!=MODE_DISABLED) for (params->i=0;params->i<params->memsize/2;params->i+=4096)
     __asm__ __volatile__("movdqa (%%rax),%%xmm0;":: "a" (params->addr_2+params->i): "%xmm0");      
  }*/ 
}

static void asm_copy_mov_1(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_1:"
            
                "mov (%%rsi),%%r8;mov %%r8,(%%rdi);"
                "mov 8(%%rsi),%%r8;mov %%r8,8(%%rdi);"
                "mov 16(%%rsi),%%r8;mov %%r8,16(%%rdi);"
                "mov 24(%%rsi),%%r8;mov %%r8,24(%%rdi);"
                "mov 32(%%rsi),%%r8;mov %%r8,32(%%rdi);"
                "mov 40(%%rsi),%%r8;mov %%r8,40(%%rdi);"
                "mov 48(%%rsi),%%r8;mov %%r8,48(%%rdi);"
                "mov 56(%%rsi),%%r8;mov %%r8,56(%%rdi);"
                "mov 64(%%rsi),%%r8;mov %%r8,64(%%rdi);"
                "mov 72(%%rsi),%%r8;mov %%r8,72(%%rdi);"
                "mov 80(%%rsi),%%r8;mov %%r8,80(%%rdi);"
                "mov 88(%%rsi),%%r8;mov %%r8,88(%%rdi);"
                "mov 96(%%rsi),%%r8;mov %%r8,96(%%rdi);"
                "mov 104(%%rsi),%%r8;mov %%r8,104(%%rdi);"
                "mov 112(%%rsi),%%r8;mov %%r8,112(%%rdi);"
                "mov 120(%%rsi),%%r8;mov %%r8,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "memory"
   );             
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_mov_2(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_2:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov %%r8,(%%rdi);"
                "mov %%r9,8(%%rdi);"
                
                "mov 16(%%rsi),%%r8;"
                "mov 24(%%rsi),%%r9;"
                "mov %%r8,16(%%rdi);"
                "mov %%r9,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov %%r8,32(%%rdi);"
                "mov %%r9,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov %%r8,48(%%rdi);"
                "mov %%r9,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov %%r8,64(%%rdi);"
                "mov %%r9,72(%%rdi);"
                
                "mov 80(%%rsi),%%r8;"
                "mov 88(%%rsi),%%r9;"
                "mov %%r8,80(%%rdi);"
                "mov %%r9,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov %%r8,96(%%rdi);"
                "mov %%r9,104(%%rdi);"
                
                "mov 112(%%rsi),%%r8;"
                "mov 120(%%rsi),%%r9;"
                "mov %%r8,112(%%rdi);"
                "mov %%r9,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "memory"
   );               
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_mov_3(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_3:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov %%r8,(%%rdi);"
                "mov %%r9,8(%%rdi);"
                "mov %%r10,16(%%rdi);"
                
                "mov 24(%%rsi),%%r8;"
                "mov 32(%%rsi),%%r9;"
                "mov 40(%%rsi),%%r10;"
                "mov %%r8,24(%%rdi);"
                "mov %%r9,32(%%rdi);"
                "mov %%r10,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov 64(%%rsi),%%r10;"
                "mov %%r8,48(%%rdi);"
                "mov %%r9,56(%%rdi);"
                "mov %%r10,64(%%rdi);" 
                
                "mov 72(%%rsi),%%r8;"
                "mov 80(%%rsi),%%r9;"
                "mov 88(%%rsi),%%r10;"
                "mov %%r8,72(%%rdi);"
                "mov %%r9,80(%%rdi);"
                "mov %%r10,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov %%r8,96(%%rdi);"
                "mov %%r9,104(%%rdi);"
                "mov %%r10,112(%%rdi);"
                
                "mov 120(%%rsi),%%r8;"
                "mov 128(%%rsi),%%r9;"
                "mov 136(%%rsi),%%r10;"                
                "mov %%r8,120(%%rdi);"
                "mov %%r9,128(%%rdi);"
                "mov %%r10,136(%%rdi);"
                
                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r10", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_mov_4(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_4:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov 24(%%rsi),%%r11;"
                "mov %%r8,(%%rdi);"
                "mov %%r9,8(%%rdi);"                
                "mov %%r10,16(%%rdi);"
                "mov %%r11,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov 48(%%rsi),%%r10;"
                "mov 56(%%rsi),%%r11;"
                "mov %%r8,32(%%rdi);"
                "mov %%r9,40(%%rdi);"
                "mov %%r10,48(%%rdi);"
                "mov %%r11,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov 80(%%rsi),%%r10;"
                "mov 88(%%rsi),%%r11;"
                "mov %%r8,64(%%rdi);"
                "mov %%r9,72(%%rdi);" 
                "mov %%r10,80(%%rdi);"
                "mov %%r11,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov 120(%%rsi),%%r11;"
                "mov %%r8,96(%%rdi);"
                "mov %%r9,104(%%rdi);"
                "mov %%r10,112(%%rdi);"
                "mov %%r11,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_1(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_1:"
            
                "mov (%%rsi),%%r8;mov %%r12,(%%rdi);"
                "mov 8(%%rsi),%%r8;mov %%r12,8(%%rdi);"
                "mov 16(%%rsi),%%r8;mov %%r12,16(%%rdi);"
                "mov 24(%%rsi),%%r8;mov %%r12,24(%%rdi);"
                "mov 32(%%rsi),%%r8;mov %%r12,32(%%rdi);"
                "mov 40(%%rsi),%%r8;mov %%r12,40(%%rdi);"
                "mov 48(%%rsi),%%r8;mov %%r12,48(%%rdi);"
                "mov 56(%%rsi),%%r8;mov %%r12,56(%%rdi);"
                "mov 64(%%rsi),%%r8;mov %%r12,64(%%rdi);"
                "mov 72(%%rsi),%%r8;mov %%r12,72(%%rdi);"
                "mov 80(%%rsi),%%r8;mov %%r12,80(%%rdi);"
                "mov 88(%%rsi),%%r8;mov %%r12,88(%%rdi);"
                "mov 96(%%rsi),%%r8;mov %%r12,96(%%rdi);"
                "mov 104(%%rsi),%%r8;mov %%r12,104(%%rdi);"
                "mov 112(%%rsi),%%r8;mov %%r12,112(%%rdi);"
                "mov 120(%%rsi),%%r8;mov %%r12,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r12", "memory"
   );             
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_2(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_2:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov %%r12,(%%rdi);"
                "mov %%r13,8(%%rdi);"
                
                "mov 16(%%rsi),%%r8;"
                "mov 24(%%rsi),%%r9;"
                "mov %%r12,16(%%rdi);"
                "mov %%r13,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov %%r12,32(%%rdi);"
                "mov %%r13,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov %%r12,48(%%rdi);"
                "mov %%r13,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov %%r12,64(%%rdi);"
                "mov %%r13,72(%%rdi);"
                
                "mov 80(%%rsi),%%r8;"
                "mov 88(%%rsi),%%r9;"
                "mov %%r12,80(%%rdi);"
                "mov %%r13,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov %%r12,96(%%rdi);"
                "mov %%r13,104(%%rdi);"
                
                "mov 112(%%rsi),%%r8;"
                "mov 120(%%rsi),%%r9;"
                "mov %%r12,112(%%rdi);"
                "mov %%r13,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r12", "%r13", "memory"
   );               
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_3(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_3:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov %%r12,(%%rdi);"
                "mov %%r13,8(%%rdi);"
                "mov %%r14,16(%%rdi);"
                
                "mov 24(%%rsi),%%r8;"
                "mov 32(%%rsi),%%r9;"
                "mov 40(%%rsi),%%r10;"
                "mov %%r12,24(%%rdi);"
                "mov %%r13,32(%%rdi);"
                "mov %%r14,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov 64(%%rsi),%%r10;"
                "mov %%r12,48(%%rdi);"
                "mov %%r13,56(%%rdi);"
                "mov %%r14,64(%%rdi);" 
                
                "mov 72(%%rsi),%%r8;"
                "mov 80(%%rsi),%%r9;"
                "mov 88(%%rsi),%%r10;"
                "mov %%r12,72(%%rdi);"
                "mov %%r13,80(%%rdi);"
                "mov %%r14,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov %%r12,96(%%rdi);"
                "mov %%r13,104(%%rdi);"
                "mov %%r14,112(%%rdi);"
                
                "mov 120(%%rsi),%%r8;"
                "mov 128(%%rsi),%%r9;"
                "mov 136(%%rsi),%%r10;"                
                "mov %%r12,120(%%rdi);"
                "mov %%r13,128(%%rdi);"
                "mov %%r14,136(%%rdi);"
                
                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_4(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_4:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov 24(%%rsi),%%r11;"
                "mov %%r12,(%%rdi);"
                "mov %%r13,8(%%rdi);"                
                "mov %%r14,16(%%rdi);"
                "mov %%r15,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov 48(%%rsi),%%r10;"
                "mov 56(%%rsi),%%r11;"
                "mov %%r12,32(%%rdi);"
                "mov %%r13,40(%%rdi);"
                "mov %%r14,48(%%rdi);"
                "mov %%r15,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov 80(%%rsi),%%r10;"
                "mov 88(%%rsi),%%r11;"
                "mov %%r12,64(%%rdi);"
                "mov %%r13,72(%%rdi);" 
                "mov %%r14,80(%%rdi);"
                "mov %%r15,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov 120(%%rsi),%%r11;"
                "mov %%r12,96(%%rdi);"
                "mov %%r13,104(%%rdi);"
                "mov %%r14,112(%%rdi);"
                "mov %%r15,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                :  "%rsi", "%rdi","%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_1(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_1:"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,(%%rdi);"
                "movq 8(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,8(%%rdi);"
                "movq 16(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,16(%%rdi);"
                "movq 24(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,24(%%rdi);"
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,32(%%rdi);"
                "movq 40(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,40(%%rdi);"
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,48(%%rdi);"
                "movq 56(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,56(%%rdi);"
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,64(%%rdi);"
                "movq 72(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,72(%%rdi);"
                "movq 80(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,80(%%rdi);"
                "movq 88(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,88(%%rdi);"
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,96(%%rdi);"
                "movq 104(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,104(%%rdi);"
                "movq 112(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,112(%%rdi);"
                "movq 120(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%xmm0", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_2(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_2:"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,(%%rdi);"
                "movq %%xmm1,8(%%rdi);"
                
                "movq 16(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 24(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,16(%%rdi);"
                "movq %%xmm1,24(%%rdi);"
                
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 40(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,32(%%rdi);"
                "movq %%xmm1,40(%%rdi);"
                
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 56(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,48(%%rdi);"
                "movq %%xmm1,56(%%rdi);"
                
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 72(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,64(%%rdi);"
                "movq %%xmm1,72(%%rdi);"
                
                "movq 80(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 88(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,80(%%rdi);"
                "movq %%xmm1,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,96(%%rdi);"
                "movq %%xmm1,104(%%rdi);"
                
                "movq 112(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 120(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,112(%%rdi);"
                "movq %%xmm1,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%xmm0", "%xmm1", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_3(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_3:"               
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 16(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,(%%rdi);"
                "movq %%xmm1,8(%%rdi);"
                "movq %%xmm2,16(%%rdi);"
                
                "movq 24(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 32(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 40(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,24(%%rdi);"
                "movq %%xmm1,32(%%rdi);"
                "movq %%xmm2,40(%%rdi);"
                
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 56(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 64(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,48(%%rdi);"
                "movq %%xmm1,56(%%rdi);"
                "movq %%xmm2,64(%%rdi);"
                
                "movq 72(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 80(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 88(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,72(%%rdi);"
                "movq %%xmm1,80(%%rdi);"
                "movq %%xmm2,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 112(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,96(%%rdi);"
                "movq %%xmm1,104(%%rdi);"
                "movq %%xmm2,112(%%rdi);"
                
                "movq 120(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 128(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 136(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,120(%%rdi);"
                "movq %%xmm1,128(%%rdi);"
                "movq %%xmm2,136(%%rdi);"

                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%xmm0", "%xmm1", "%xmm2", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_4(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_4:"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 16(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 24(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,(%%rdi);"
                "movq %%xmm1,8(%%rdi);"
                "movq %%xmm2,16(%%rdi);"
                "movq %%xmm3,24(%%rdi);"
                
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 40(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 48(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 56(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,32(%%rdi);"
                "movq %%xmm1,40(%%rdi);"
                "movq %%xmm2,48(%%rdi);"
                "movq %%xmm3,56(%%rdi);"
                
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 72(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 80(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 88(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,64(%%rdi);"
                "movq %%xmm1,72(%%rdi);"
                "movq %%xmm2,80(%%rdi);"
                "movq %%xmm3,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 112(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 120(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,96(%%rdi);"
                "movq %%xmm1,104(%%rdi);"
                "movq %%xmm2,112(%%rdi);"
                "movq %%xmm3,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_mov_clflush_1(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_clflush_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_clflush_1:"
                
                 /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "mov (%%rsi),%%r8;mov %%r8,(%%rdi);"
                "mov 8(%%rsi),%%r8;mov %%r8,8(%%rdi);"
                "mov 16(%%rsi),%%r8;mov %%r8,16(%%rdi);"
                "mov 24(%%rsi),%%r8;mov %%r8,24(%%rdi);"
                "mov 32(%%rsi),%%r8;mov %%r8,32(%%rdi);"
                "mov 40(%%rsi),%%r8;mov %%r8,40(%%rdi);"
                "mov 48(%%rsi),%%r8;mov %%r8,48(%%rdi);"
                "mov 56(%%rsi),%%r8;mov %%r8,56(%%rdi);"
                "mov 64(%%rsi),%%r8;mov %%r8,64(%%rdi);"
                "mov 72(%%rsi),%%r8;mov %%r8,72(%%rdi);"
                "mov 80(%%rsi),%%r8;mov %%r8,80(%%rdi);"
                "mov 88(%%rsi),%%r8;mov %%r8,88(%%rdi);"
                "mov 96(%%rsi),%%r8;mov %%r8,96(%%rdi);"
                "mov 104(%%rsi),%%r8;mov %%r8,104(%%rdi);"
                "mov 112(%%rsi),%%r8;mov %%r8,112(%%rdi);"
                "mov 120(%%rsi),%%r8;mov %%r8,120(%%rdi);"
                
                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_clflush_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "memory"
   );             
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_mov_clflush_2(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_clflush_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_clflush_2:"
                
                 /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"    
                       
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov %%r8,(%%rdi);"
                "mov %%r9,8(%%rdi);"
                
                "mov 16(%%rsi),%%r8;"
                "mov 24(%%rsi),%%r9;"
                "mov %%r8,16(%%rdi);"
                "mov %%r9,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov %%r8,32(%%rdi);"
                "mov %%r9,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov %%r8,48(%%rdi);"
                "mov %%r9,56(%%rdi);"

                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov %%r8,64(%%rdi);"
                "mov %%r9,72(%%rdi);"
                
                "mov 80(%%rsi),%%r8;"
                "mov 88(%%rsi),%%r9;"
                "mov %%r8,80(%%rdi);"
                "mov %%r9,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov %%r8,96(%%rdi);"
                "mov %%r9,104(%%rdi);"
                
                "mov 112(%%rsi),%%r8;"
                "mov 120(%%rsi),%%r9;"
                "mov %%r8,112(%%rdi);"
                "mov %%r9,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_clflush_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "memory"
   );               
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_mov_clflush_3(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_clflush_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_clflush_3:"
                
                /* remove old data from caches */
                "clflush -448(%%rdi);clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -448(%%rsi);clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov %%r8,(%%rdi);"
                "mov %%r9,8(%%rdi);"
                "mov %%r10,16(%%rdi);"
                
                "mov 24(%%rsi),%%r8;"
                "mov 32(%%rsi),%%r9;"
                "mov 40(%%rsi),%%r10;"
                "mov %%r8,24(%%rdi);"
                "mov %%r9,32(%%rdi);"
                "mov %%r10,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov 64(%%rsi),%%r10;"
                "mov %%r8,48(%%rdi);"
                "mov %%r9,56(%%rdi);"
                "mov %%r10,64(%%rdi);" 
                
                "mov 72(%%rsi),%%r8;"
                "mov 80(%%rsi),%%r9;"
                "mov 88(%%rsi),%%r10;"
                "mov %%r8,72(%%rdi);"
                "mov %%r9,80(%%rdi);"
                "mov %%r10,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov %%r8,96(%%rdi);"
                "mov %%r9,104(%%rdi);"
                "mov %%r10,112(%%rdi);"
                
                "mov 120(%%rsi),%%r8;"
                "mov 128(%%rsi),%%r9;"
                "mov 136(%%rsi),%%r10;"                
                "mov %%r8,120(%%rdi);"
                "mov %%r9,128(%%rdi);"
                "mov %%r10,136(%%rdi);"
                
                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_clflush_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r10", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_mov_clflush_4(param_t *params) __attribute__((noinline)); 
static void asm_copy_mov_clflush_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_mov_clflush_4:"
            
                /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov 24(%%rsi),%%r11;"
                "mov %%r8,(%%rdi);"
                "mov %%r9,8(%%rdi);"                
                "mov %%r10,16(%%rdi);"
                "mov %%r11,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov 48(%%rsi),%%r10;"
                "mov 56(%%rsi),%%r11;"
                "mov %%r8,32(%%rdi);"
                "mov %%r9,40(%%rdi);"
                "mov %%r10,48(%%rdi);"
                "mov %%r11,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov 80(%%rsi),%%r10;"
                "mov 88(%%rsi),%%r11;"
                "mov %%r8,64(%%rdi);"
                "mov %%r9,72(%%rdi);" 
                "mov %%r10,80(%%rdi);"
                "mov %%r11,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov 120(%%rsi),%%r11;"
                "mov %%r8,96(%%rdi);"
                "mov %%r9,104(%%rdi);"
                "mov %%r10,112(%%rdi);"
                "mov %%r11,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_mov_clflush_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_clflush_1(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_clflush_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_clflush_1:"
                
                /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "mov (%%rsi),%%r8;mov %%r12,(%%rdi);"
                "mov 8(%%rsi),%%r8;mov %%r12,8(%%rdi);"
                "mov 16(%%rsi),%%r8;mov %%r12,16(%%rdi);"
                "mov 24(%%rsi),%%r8;mov %%r12,24(%%rdi);"
                "mov 32(%%rsi),%%r8;mov %%r12,32(%%rdi);"
                "mov 40(%%rsi),%%r8;mov %%r12,40(%%rdi);"
                "mov 48(%%rsi),%%r8;mov %%r12,48(%%rdi);"
                "mov 56(%%rsi),%%r8;mov %%r12,56(%%rdi);"
                "mov 64(%%rsi),%%r8;mov %%r12,64(%%rdi);"
                "mov 72(%%rsi),%%r8;mov %%r12,72(%%rdi);"
                "mov 80(%%rsi),%%r8;mov %%r12,80(%%rdi);"
                "mov 88(%%rsi),%%r8;mov %%r12,88(%%rdi);"
                "mov 96(%%rsi),%%r8;mov %%r12,96(%%rdi);"
                "mov 104(%%rsi),%%r8;mov %%r12,104(%%rdi);"
                "mov 112(%%rsi),%%r8;mov %%r12,112(%%rdi);"
                "mov 120(%%rsi),%%r8;mov %%r12,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_clflush_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r12", "memory"
   );             
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_clflush_2(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_clflush_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_clflush_2:"
                
                /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov %%r12,(%%rdi);"
                "mov %%r13,8(%%rdi);"
                
                "mov 16(%%rsi),%%r8;"
                "mov 24(%%rsi),%%r9;"
                "mov %%r12,16(%%rdi);"
                "mov %%r13,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov %%r12,32(%%rdi);"
                "mov %%r13,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov %%r12,48(%%rdi);"
                "mov %%r13,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov %%r12,64(%%rdi);"
                "mov %%r13,72(%%rdi);"
                
                "mov 80(%%rsi),%%r8;"
                "mov 88(%%rsi),%%r9;"
                "mov %%r12,80(%%rdi);"
                "mov %%r13,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov %%r12,96(%%rdi);"
                "mov %%r13,104(%%rdi);"
                
                "mov 112(%%rsi),%%r8;"
                "mov 120(%%rsi),%%r9;"
                "mov %%r12,112(%%rdi);"
                "mov %%r13,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_clflush_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r12", "%r13", "memory"
   );               
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_clflush_3(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_clflush_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_clflush_3:"
                
                /* remove old data from caches */
                "clflush -448(%%rdi);clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -448(%%rsi);clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov %%r12,(%%rdi);"
                "mov %%r13,8(%%rdi);"
                "mov %%r14,16(%%rdi);"
                
                "mov 24(%%rsi),%%r8;"
                "mov 32(%%rsi),%%r9;"
                "mov 40(%%rsi),%%r10;"
                "mov %%r12,24(%%rdi);"
                "mov %%r13,32(%%rdi);"
                "mov %%r14,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov 64(%%rsi),%%r10;"
                "mov %%r12,48(%%rdi);"
                "mov %%r13,56(%%rdi);"
                "mov %%r14,64(%%rdi);" 
                
                "mov 72(%%rsi),%%r8;"
                "mov 80(%%rsi),%%r9;"
                "mov 88(%%rsi),%%r10;"
                "mov %%r12,72(%%rdi);"
                "mov %%r13,80(%%rdi);"
                "mov %%r14,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov %%r12,96(%%rdi);"
                "mov %%r13,104(%%rdi);"
                "mov %%r14,112(%%rdi);"
                
                "mov 120(%%rsi),%%r8;"
                "mov 128(%%rsi),%%r9;"
                "mov 136(%%rsi),%%r10;"                
                "mov %%r12,120(%%rdi);"
                "mov %%r13,128(%%rdi);"
                "mov %%r14,136(%%rdi);"

                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_clflush_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_mov_clflush_4(param_t *params) __attribute__((noinline)); 
static void asm_indep_mov_clflush_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_mov_clflush_4:"

                /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
                            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov 24(%%rsi),%%r11;"
                "mov %%r12,(%%rdi);"
                "mov %%r13,8(%%rdi);"                
                "mov %%r14,16(%%rdi);"
                "mov %%r15,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov 48(%%rsi),%%r10;"
                "mov 56(%%rsi),%%r11;"
                "mov %%r12,32(%%rdi);"
                "mov %%r13,40(%%rdi);"
                "mov %%r14,48(%%rdi);"
                "mov %%r15,56(%%rdi);"

                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov 80(%%rsi),%%r10;"
                "mov 88(%%rsi),%%r11;"
                "mov %%r12,64(%%rdi);"
                "mov %%r13,72(%%rdi);" 
                "mov %%r14,80(%%rdi);"
                "mov %%r15,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov 120(%%rsi),%%r11;"
                "mov %%r12,96(%%rdi);"
                "mov %%r13,104(%%rdi);"
                "mov %%r14,112(%%rdi);"
                "mov %%r15,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_mov_clflush_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                :  "%rsi", "%rdi","%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_clflush_1(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_clflush_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_clflush_1:"
                
                /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,(%%rdi);"
                "movq 8(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,8(%%rdi);"
                "movq 16(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,16(%%rdi);"
                "movq 24(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,24(%%rdi);"
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,32(%%rdi);"
                "movq 40(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,40(%%rdi);"
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,48(%%rdi);"
                "movq 56(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,56(%%rdi);"
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,64(%%rdi);"
                "movq 72(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,72(%%rdi);"
                "movq 80(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,80(%%rdi);"
                "movq 88(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,88(%%rdi);"
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,96(%%rdi);"
                "movq 104(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,104(%%rdi);"
                "movq 112(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,112(%%rdi);"
                "movq 120(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_clflush_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%xmm0", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_clflush_2(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_clflush_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_clflush_2:"
                
                /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,(%%rdi);"
                "movq %%xmm1,8(%%rdi);"
                
                "movq 16(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 24(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,16(%%rdi);"
                "movq %%xmm1,24(%%rdi);"
                
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 40(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,32(%%rdi);"
                "movq %%xmm1,40(%%rdi);"
                
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 56(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,48(%%rdi);"
                "movq %%xmm1,56(%%rdi);"
                
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 72(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,64(%%rdi);"
                "movq %%xmm1,72(%%rdi);"
                
                "movq 80(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 88(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,80(%%rdi);"
                "movq %%xmm1,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,96(%%rdi);"
                "movq %%xmm1,104(%%rdi);"
                
                "movq 112(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 120(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0,112(%%rdi);"
                "movq %%xmm1,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_clflush_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%xmm0", "%xmm1", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_clflush_3(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_clflush_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_clflush_3:"
                
                /* remove old data from caches */
                "clflush -448(%%rdi);clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -448(%%rsi);clflush -384(%%rsi);""clflush -320(%%rsi);"          
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 16(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,(%%rdi);"
                "movq %%xmm1,8(%%rdi);"
                "movq %%xmm2,16(%%rdi);"
                
                "movq 24(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 32(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 40(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,24(%%rdi);"
                "movq %%xmm1,32(%%rdi);"
                "movq %%xmm2,40(%%rdi);"
                
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 56(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 64(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,48(%%rdi);"
                "movq %%xmm1,56(%%rdi);"
                "movq %%xmm2,64(%%rdi);"
                
                "movq 72(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 80(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 88(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,72(%%rdi);"
                "movq %%xmm1,80(%%rdi);"
                "movq %%xmm2,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 112(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,96(%%rdi);"
                "movq %%xmm1,104(%%rdi);"
                "movq %%xmm2,112(%%rdi);"
                
                "movq 120(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 128(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 136(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0,120(%%rdi);"
                "movq %%xmm1,128(%%rdi);"
                "movq %%xmm2,136(%%rdi);"

                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_clflush_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%xmm0", "%xmm1", "%xmm2", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_mov_clflush_4(param_t *params) __attribute__((noinline)); 
static void asm_scale_mov_clflush_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_mov_clflush_4:"
                
                /* remove old data from caches */
                "clflush -384(%%rdi);""clflush -320(%%rdi);"
                //"clflush -384(%%rsi);""clflush -320(%%rsi);"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 16(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 24(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,(%%rdi);"
                "movq %%xmm1,8(%%rdi);"
                "movq %%xmm2,16(%%rdi);"
                "movq %%xmm3,24(%%rdi);"
                
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 40(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 48(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 56(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,32(%%rdi);"
                "movq %%xmm1,40(%%rdi);"
                "movq %%xmm2,48(%%rdi);"
                "movq %%xmm3,56(%%rdi);"
                
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 72(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 80(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 88(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,64(%%rdi);"
                "movq %%xmm1,72(%%rdi);"
                "movq %%xmm2,80(%%rdi);"
                "movq %%xmm3,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 112(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 120(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0,96(%%rdi);"
                "movq %%xmm1,104(%%rdi);"
                "movq %%xmm2,112(%%rdi);"
                "movq %%xmm3,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_mov_clflush_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}


static void asm_copy_movnti_1(param_t *params) __attribute__((noinline)); 
static void asm_copy_movnti_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movnti_1:"
            
                "mov (%%rsi),%%r8;movnti %%r8,(%%rdi);"
                "mov 8(%%rsi),%%r8;movnti %%r8,8(%%rdi);"
                "mov 16(%%rsi),%%r8;movnti %%r8,16(%%rdi);"
                "mov 24(%%rsi),%%r8;movnti %%r8,24(%%rdi);"
                "mov 32(%%rsi),%%r8;movnti %%r8,32(%%rdi);"
                "mov 40(%%rsi),%%r8;movnti %%r8,40(%%rdi);"
                "mov 48(%%rsi),%%r8;movnti %%r8,48(%%rdi);"
                "mov 56(%%rsi),%%r8;movnti %%r8,56(%%rdi);"
                "mov 64(%%rsi),%%r8;movnti %%r8,64(%%rdi);"
                "mov 72(%%rsi),%%r8;movnti %%r8,72(%%rdi);"
                "mov 80(%%rsi),%%r8;movnti %%r8,80(%%rdi);"
                "mov 88(%%rsi),%%r8;movnti %%r8,88(%%rdi);"
                "mov 96(%%rsi),%%r8;movnti %%r8,96(%%rdi);"
                "mov 104(%%rsi),%%r8;movnti %%r8,104(%%rdi);"
                "mov 112(%%rsi),%%r8;movnti %%r8,112(%%rdi);"
                "mov 120(%%rsi),%%r8;movnti %%r8,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_movnti_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "memory"
   );             
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movnti_2(param_t *params) __attribute__((noinline)); 
static void asm_copy_movnti_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movnti_2:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "movnti %%r8,(%%rdi);"
                "movnti %%r9,8(%%rdi);"
                
                "mov 16(%%rsi),%%r8;"
                "mov 24(%%rsi),%%r9;"
                "movnti %%r8,16(%%rdi);"
                "movnti %%r9,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "movnti %%r8,32(%%rdi);"
                "movnti %%r9,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "movnti %%r8,48(%%rdi);"
                "movnti %%r9,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "movnti %%r8,64(%%rdi);"
                "movnti %%r9,72(%%rdi);"
                
                "mov 80(%%rsi),%%r8;"
                "mov 88(%%rsi),%%r9;"
                "movnti %%r8,80(%%rdi);"
                "movnti %%r9,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "movnti %%r8,96(%%rdi);"
                "movnti %%r9,104(%%rdi);"
                
                "mov 112(%%rsi),%%r8;"
                "mov 120(%%rsi),%%r9;"
                "movnti %%r8,112(%%rdi);"
                "movnti %%r9,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_movnti_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "memory"
   );               
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movnti_3(param_t *params) __attribute__((noinline)); 
static void asm_copy_movnti_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movnti_3:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "movnti %%r8,(%%rdi);"
                "movnti %%r9,8(%%rdi);"
                "movnti %%r10,16(%%rdi);"
                
                "mov 24(%%rsi),%%r8;"
                "mov 32(%%rsi),%%r9;"
                "mov 40(%%rsi),%%r10;"
                "movnti %%r8,24(%%rdi);"
                "movnti %%r9,32(%%rdi);"
                "movnti %%r10,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov 64(%%rsi),%%r10;"
                "movnti %%r8,48(%%rdi);"
                "movnti %%r9,56(%%rdi);"
                "movnti %%r10,64(%%rdi);" 
                
                "mov 72(%%rsi),%%r8;"
                "mov 80(%%rsi),%%r9;"
                "mov 88(%%rsi),%%r10;"
                "movnti %%r8,72(%%rdi);"
                "movnti %%r9,80(%%rdi);"
                "movnti %%r10,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "movnti %%r8,96(%%rdi);"
                "movnti %%r9,104(%%rdi);"
                "movnti %%r10,112(%%rdi);"
                
                "mov 120(%%rsi),%%r8;"
                "mov 128(%%rsi),%%r9;"
                "mov 136(%%rsi),%%r10;"                
                "movnti %%r8,120(%%rdi);"
                "movnti %%r9,128(%%rdi);"
                "movnti %%r10,136(%%rdi);"
                
                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_movnti_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r10", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movnti_4(param_t *params) __attribute__((noinline)); 
static void asm_copy_movnti_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movnti_4:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov 24(%%rsi),%%r11;"
                "movnti %%r8,(%%rdi);"
                "movnti %%r9,8(%%rdi);"                
                "movnti %%r10,16(%%rdi);"
                "movnti %%r11,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov 48(%%rsi),%%r10;"
                "mov 56(%%rsi),%%r11;"
                "movnti %%r8,32(%%rdi);"
                "movnti %%r9,40(%%rdi);"
                "movnti %%r10,48(%%rdi);"
                "movnti %%r11,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov 80(%%rsi),%%r10;"
                "mov 88(%%rsi),%%r11;"
                "movnti %%r8,64(%%rdi);"
                "movnti %%r9,72(%%rdi);" 
                "movnti %%r10,80(%%rdi);"
                "movnti %%r11,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov 120(%%rsi),%%r11;"
                "movnti %%r8,96(%%rdi);"
                "movnti %%r9,104(%%rdi);"
                "movnti %%r10,112(%%rdi);"
                "movnti %%r11,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _copy_loop_movnti_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movnti_1(param_t *params) __attribute__((noinline)); 
static void asm_indep_movnti_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movnti_1:"
            
                "mov (%%rsi),%%r8;movnti %%r12,(%%rdi);"
                "mov 8(%%rsi),%%r8;movnti %%r12,8(%%rdi);"
                "mov 16(%%rsi),%%r8;movnti %%r12,16(%%rdi);"
                "mov 24(%%rsi),%%r8;movnti %%r12,24(%%rdi);"
                "mov 32(%%rsi),%%r8;movnti %%r12,32(%%rdi);"
                "mov 40(%%rsi),%%r8;movnti %%r12,40(%%rdi);"
                "mov 48(%%rsi),%%r8;movnti %%r12,48(%%rdi);"
                "mov 56(%%rsi),%%r8;movnti %%r12,56(%%rdi);"
                "mov 64(%%rsi),%%r8;movnti %%r12,64(%%rdi);"
                "mov 72(%%rsi),%%r8;movnti %%r12,72(%%rdi);"
                "mov 80(%%rsi),%%r8;movnti %%r12,80(%%rdi);"
                "mov 88(%%rsi),%%r8;movnti %%r12,88(%%rdi);"
                "mov 96(%%rsi),%%r8;movnti %%r12,96(%%rdi);"
                "mov 104(%%rsi),%%r8;movnti %%r12,104(%%rdi);"
                "mov 112(%%rsi),%%r8;movnti %%r12,112(%%rdi);"
                "mov 120(%%rsi),%%r8;movnti %%r12,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_movnti_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r12", "memory"
   );             
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movnti_2(param_t *params) __attribute__((noinline)); 
static void asm_indep_movnti_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movnti_2:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "movnti %%r12,(%%rdi);"
                "movnti %%r13,8(%%rdi);"
                
                "mov 16(%%rsi),%%r8;"
                "mov 24(%%rsi),%%r9;"
                "movnti %%r12,16(%%rdi);"
                "movnti %%r13,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "movnti %%r12,32(%%rdi);"
                "movnti %%r13,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "movnti %%r12,48(%%rdi);"
                "movnti %%r13,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "movnti %%r12,64(%%rdi);"
                "movnti %%r13,72(%%rdi);"
                
                "mov 80(%%rsi),%%r8;"
                "mov 88(%%rsi),%%r9;"
                "movnti %%r12,80(%%rdi);"
                "movnti %%r13,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "movnti %%r12,96(%%rdi);"
                "movnti %%r13,104(%%rdi);"
                
                "mov 112(%%rsi),%%r8;"
                "mov 120(%%rsi),%%r9;"
                "movnti %%r12,112(%%rdi);"
                "movnti %%r13,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_movnti_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r12", "%r13", "memory"
   );               
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movnti_3(param_t *params) __attribute__((noinline)); 
static void asm_indep_movnti_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movnti_3:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "movnti %%r12,(%%rdi);"
                "movnti %%r13,8(%%rdi);"
                "movnti %%r14,16(%%rdi);"
                
                "mov 24(%%rsi),%%r8;"
                "mov 32(%%rsi),%%r9;"
                "mov 40(%%rsi),%%r10;"
                "movnti %%r12,24(%%rdi);"
                "movnti %%r13,32(%%rdi);"
                "movnti %%r14,40(%%rdi);"
                
                "mov 48(%%rsi),%%r8;"
                "mov 56(%%rsi),%%r9;"
                "mov 64(%%rsi),%%r10;"
                "movnti %%r12,48(%%rdi);"
                "movnti %%r13,56(%%rdi);"
                "movnti %%r14,64(%%rdi);" 
                
                "mov 72(%%rsi),%%r8;"
                "mov 80(%%rsi),%%r9;"
                "mov 88(%%rsi),%%r10;"
                "movnti %%r12,72(%%rdi);"
                "movnti %%r13,80(%%rdi);"
                "movnti %%r14,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "movnti %%r12,96(%%rdi);"
                "movnti %%r13,104(%%rdi);"
                "movnti %%r14,112(%%rdi);"
                
                "mov 120(%%rsi),%%r8;"
                "mov 128(%%rsi),%%r9;"
                "mov 136(%%rsi),%%r10;"                
                "movnti %%r12,120(%%rdi);"
                "movnti %%r13,128(%%rdi);"
                "movnti %%r14,136(%%rdi);"
                
                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_movnti_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r10", "%r12", "%r13", "%r14", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movnti_4(param_t *params) __attribute__((noinline)); 
static void asm_indep_movnti_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
     __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movnti_4:"
            
                "mov (%%rsi),%%r8;"
                "mov 8(%%rsi),%%r9;"
                "mov 16(%%rsi),%%r10;"
                "mov 24(%%rsi),%%r11;"
                "movnti %%r12,(%%rdi);"
                "movnti %%r13,8(%%rdi);"                
                "movnti %%r14,16(%%rdi);"
                "movnti %%r15,24(%%rdi);"
                
                "mov 32(%%rsi),%%r8;"
                "mov 40(%%rsi),%%r9;"
                "mov 48(%%rsi),%%r10;"
                "mov 56(%%rsi),%%r11;"
                "movnti %%r12,32(%%rdi);"
                "movnti %%r13,40(%%rdi);"
                "movnti %%r14,48(%%rdi);"
                "movnti %%r15,56(%%rdi);"
                
                "mov 64(%%rsi),%%r8;"
                "mov 72(%%rsi),%%r9;"
                "mov 80(%%rsi),%%r10;"
                "mov 88(%%rsi),%%r11;"
                "movnti %%r12,64(%%rdi);"
                "movnti %%r13,72(%%rdi);" 
                "movnti %%r14,80(%%rdi);"
                "movnti %%r15,88(%%rdi);"
                
                "mov 96(%%rsi),%%r8;"
                "mov 104(%%rsi),%%r9;"
                "mov 112(%%rsi),%%r10;"
                "mov 120(%%rsi),%%r11;"
                "movnti %%r12,96(%%rdi);"
                "movnti %%r13,104(%%rdi);"
                "movnti %%r14,112(%%rdi);"
                "movnti %%r15,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _indep_loop_movnti_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                :  "%rsi", "%rdi","%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movnti_1(param_t *params) __attribute__((noinline)); 
static void asm_scale_movnti_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movnti_1:"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,(%%rdi);"
                "movq 8(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,8(%%rdi);"
                "movq 16(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,16(%%rdi);"
                "movq 24(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,24(%%rdi);"
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,32(%%rdi);"
                "movq 40(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,40(%%rdi);"
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,48(%%rdi);"
                "movq 56(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,56(%%rdi);"
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,64(%%rdi);"
                "movq 72(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,72(%%rdi);"
                "movq 80(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,80(%%rdi);"
                "movq 88(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,88(%%rdi);"
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,96(%%rdi);"
                "movq 104(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,104(%%rdi);"
                "movq 112(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,112(%%rdi);"
                "movq 120(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;movq %%xmm0, %%r8; movnti %%r8,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_movnti_1;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%r8", "%xmm0", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movnti_2(param_t *params) __attribute__((noinline)); 
static void asm_scale_movnti_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movnti_2:"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,8(%%rdi);"
                
                "movq 16(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 24(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,16(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,24(%%rdi);"
                
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 40(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,32(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,40(%%rdi);"
                
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 56(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,48(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,56(%%rdi);"
                
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 72(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,64(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,72(%%rdi);"
                
                "movq 80(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 88(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,80(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,96(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,104(%%rdi);"
                
                "movq 112(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 120(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq %%xmm0, %%r8; movnti %%r8,112(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_movnti_2;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%r8", "%r9", "%xmm0", "%xmm1", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movnti_3(param_t *params) __attribute__((noinline)); 
static void asm_scale_movnti_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movnti_3:"               
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 16(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0, %%r8; movnti %%r8,(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,8(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,16(%%rdi);"
                
                "movq 24(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 32(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 40(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0, %%r8; movnti %%r8,24(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,32(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,40(%%rdi);"
                
                "movq 48(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 56(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 64(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0, %%r8; movnti %%r8,48(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,56(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,64(%%rdi);"
                
                "movq 72(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 80(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 88(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %% xmm0, %%r8; movnti %%r8,72(%%rdi);"
                "movq %% xmm1, %%r9; movnti %%r9,80(%%rdi);"
                "movq %% xmm2, %%r10; movnti %%r10,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 112(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0, %%r8; movnti %%r8,96(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,104(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,112(%%rdi);"
                
                "movq 120(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 128(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 136(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq %%xmm0, %%r8; movnti %%r8,120(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,128(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,136(%%rdi);"

                "add $144,%%rsi;"
                "add $144,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_movnti_3;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi","%r8", "%r9", "%r10",  "%xmm0", "%xmm1", "%xmm2", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movnti_4(param_t *params) __attribute__((noinline)); 
static void asm_scale_movnti_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%rsi;"
                "mov %%rdx,%%rdi;"
                "movq (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%rbx;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movnti_4:"
            
                "movq (%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 8(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 16(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 24(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0, %%r8; movnti %%r8,(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,8(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,16(%%rdi);"
                "movq %%xmm3, %%r11; movnti %%r11,24(%%rdi);"
                
                "movq 32(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 40(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 48(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 56(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0, %%r8; movnti %%r8,32(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,40(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,48(%%rdi);"
                "movq %%xmm3, %%r11; movnti %%r11,56(%%rdi);"
                
                "movq 64(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 72(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 80(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 88(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0, %%r8; movnti %%r8,64(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,72(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,80(%%rdi);"
                "movq %%xmm3, %%r11; movnti %%r11,88(%%rdi);"
                
                "movq 96(%%rsi),%%xmm0;mulsd %%xmm15,%%xmm0;"
                "movq 104(%%rsi),%%xmm1;mulsd %%xmm15,%%xmm1;"
                "movq 112(%%rsi),%%xmm2;mulsd %%xmm15,%%xmm2;"
                "movq 120(%%rsi),%%xmm3;mulsd %%xmm15,%%xmm3;"
                "movq %%xmm0, %%r8; movnti %%r8,96(%%rdi);"
                "movq %%xmm1, %%r9; movnti %%r9,104(%%rdi);"
                "movq %%xmm2, %%r10; movnti %%r10,112(%%rdi);"
                "movq %%xmm3, %%r11; movnti %%r11,120(%%rdi);"

                "add $128,%%rsi;"
                "add $128,%%rdi;"
                "sub $1,%%rcx;"
                "jnz _scale_loop_movnti_4;"
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
                "sub %%rbx,%%rax;"
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%rsi", "%rdi", "%r8", "%r9", "%r10", "%r11", "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movapd_1(param_t *params) __attribute__((noinline)); 
static void asm_copy_movapd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movapd_1:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;movapd %%xmm0,(%%r12);"
                "movapd 16(%%r10),%%xmm0;movapd %%xmm0,16(%%r12);"
                "movapd 32(%%r10),%%xmm0;movapd %%xmm0,32(%%r12);"
                "movapd 48(%%r10),%%xmm0;movapd %%xmm0,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;movapd %%xmm0,64(%%r12);"
                "movapd 80(%%r13),%%xmm0;movapd %%xmm0,80(%%r12);"
                "movapd 96(%%r13),%%xmm0;movapd %%xmm0,96(%%r12);"
                "movapd 112(%%r13),%%xmm0;movapd %%xmm0,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;movapd %%xmm0,128(%%r14);"
                "movapd 144(%%r13),%%xmm0;movapd %%xmm0,144(%%r14);"
                "movapd 160(%%r13),%%xmm0;movapd %%xmm0,160(%%r14);"
                "movapd 176(%%r13),%%xmm0;movapd %%xmm0,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;movapd %%xmm0,192(%%r14);"
                "movapd 208(%%r13),%%xmm0;movapd %%xmm0,208(%%r14);"
                "movapd 224(%%r13),%%xmm0;movapd %%xmm0,224(%%r14);"
                "movapd 240(%%r13),%%xmm0;movapd %%xmm0,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _copy_loop_movapd_1;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movapd_2(param_t *params) __attribute__((noinline)); 
static void asm_copy_movapd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movapd_2:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                
                "movapd 32(%%r10),%%xmm0;"
                "movapd 48(%%r10),%%xmm1;"                
                "movapd %%xmm0,32(%%r12);"
                "movapd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movapd %%xmm0,64(%%r12);"
                "movapd %%xmm1,80(%%r12);"
                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movapd %%xmm0,96(%%r12);"
                "movapd %%xmm1,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd %%xmm0,128(%%r14);"
                "movapd %%xmm1,144(%%r14);"
                
                "movapd 160(%%r13),%%xmm0;"
                "movapd 176(%%r13),%%xmm1;"
                "movapd %%xmm0,160(%%r14);"
                "movapd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd %%xmm0,192(%%r14);"
                "movapd %%xmm1,208(%%r14);"                
                
                "movapd 224(%%r13),%%xmm0;"
                "movapd 240(%%r13),%%xmm1;"
                "movapd %%xmm0,224(%%r14);"
                "movapd %%xmm1,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movapd_2;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movapd_3(param_t *params) __attribute__((noinline)); 
static void asm_copy_movapd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movapd_3:"               
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                "movapd %%xmm2,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movapd 48(%%r10),%%xmm0;"                
                "movapd 64(%%r10),%%xmm1;"
                "movapd 80(%%r10),%%xmm2;"                
                "movapd %%xmm0,48(%%r12);"  
                "movapd %%xmm1,64(%%r12);"
                "movapd %%xmm2,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movapd 128(%%r13),%%xmm2;"
                "movapd %%xmm0,96(%%r12);"
                "movapd %%xmm1,112(%%r12);"
                "movapd %%xmm2,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movapd 144(%%r13),%%xmm0;"
                "movapd 160(%%r13),%%xmm1;"
                "movapd 176(%%r13),%%xmm2;"
                "movapd %%xmm0,144(%%r14);"
                "movapd %%xmm1,160(%%r14);"
                "movapd %%xmm2,176(%%r14);"

                "add $288,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movapd %%xmm0,192(%%r14);"
                "movapd %%xmm1,208(%%r14);"                
                "movapd %%xmm2,224(%%r14);"
                
                "movapd 240(%%r13),%%xmm0;"
                "movapd 256(%%r13),%%xmm1;"
                "movapd 272(%%r13),%%xmm2;"                
                "movapd %%xmm0,240(%%r14);"
                "movapd %%xmm1,256(%%r14);"                
                "movapd %%xmm2,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _copy_loop_movapd_3;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movapd_4(param_t *params) __attribute__((noinline)); 
static void asm_copy_movapd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movapd_4:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                "movapd %%xmm2,32(%%r12);"
                "movapd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movapd 96(%%r13),%%xmm2;"
                "movapd 112(%%r13),%%xmm3;"
                "movapd %%xmm0,64(%%r12);"
                "movapd %%xmm1,80(%%r12);"
                "movapd %%xmm2,96(%%r12);"
                "movapd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movapd %%xmm0,128(%%r14);"
                "movapd %%xmm1,144(%%r14);"
                "movapd %%xmm2,160(%%r14);"
                "movapd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movapd 240(%%r13),%%xmm3;"                                               
                "movapd %%xmm0,192(%%r14);"
                "movapd %%xmm1,208(%%r14);"
                "movapd %%xmm2,224(%%r14);"
                "movapd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movapd_4;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movapd_8(param_t *params) __attribute__((noinline)); 
static void asm_copy_movapd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movapd_8:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movapd 64(%%r13),%%xmm4;"
                "movapd 80(%%r13),%%xmm5;"
                "movapd 96(%%r13),%%xmm6;"
                "movapd 112(%%r13),%%xmm7;"  
                
                "mov %%r12, %%r14;"
                
                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                "movapd %%xmm2,32(%%r12);"
                "movapd %%xmm3,48(%%r12);" 
                "movapd %%xmm4,64(%%r12);"
                "movapd %%xmm5,80(%%r12);"
                "movapd %%xmm6,96(%%r12);"
                "movapd %%xmm7,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movapd 192(%%r13),%%xmm4;"
                "movapd 208(%%r13),%%xmm5;"
                "movapd 224(%%r13),%%xmm6;"
                "movapd 240(%%r13),%%xmm7;"

                "add $256,%%r12;"
                
                "movapd %%xmm0,128(%%r14);"
                "movapd %%xmm1,144(%%r14);"
                "movapd %%xmm2,160(%%r14);"
                "movapd %%xmm3,176(%%r14);"
                "movapd %%xmm4,192(%%r14);"
                "movapd %%xmm5,208(%%r14);"
                "movapd %%xmm6,224(%%r14);"
                "movapd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movapd_8;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movapd_1(param_t *params) __attribute__((noinline)); 
static void asm_indep_movapd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movapd_1:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;movapd %%xmm1,(%%r12);"
                "movapd 16(%%r10),%%xmm0;movapd %%xmm1,16(%%r12);"
                "movapd 32(%%r10),%%xmm0;movapd %%xmm1,32(%%r12);"
                "movapd 48(%%r10),%%xmm0;movapd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;movapd %%xmm1,64(%%r12);"
                "movapd 80(%%r13),%%xmm0;movapd %%xmm1,80(%%r12);"
                "movapd 96(%%r13),%%xmm0;movapd %%xmm1,96(%%r12);"
                "movapd 112(%%r13),%%xmm0;movapd %%xmm1,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;movapd %%xmm1,128(%%r14);"
                "movapd 144(%%r13),%%xmm0;movapd %%xmm1,144(%%r14);"
                "movapd 160(%%r13),%%xmm0;movapd %%xmm1,160(%%r14);"
                "movapd 176(%%r13),%%xmm0;movapd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;movapd %%xmm1,192(%%r14);"
                "movapd 208(%%r13),%%xmm0;movapd %%xmm1,208(%%r14);"
                "movapd 224(%%r13),%%xmm0;movapd %%xmm1,224(%%r14);"
                "movapd 240(%%r13),%%xmm0;movapd %%xmm1,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _indep_loop_movapd_1;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0","%xmm1", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movapd_2(param_t *params) __attribute__((noinline)); 
static void asm_indep_movapd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movapd_2:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd %%xmm2,(%%r12);"
                "movapd %%xmm3,16(%%r12);"
                
                "movapd 32(%%r10),%%xmm0;"
                "movapd 48(%%r10),%%xmm1;"                
                "movapd %%xmm2,32(%%r12);"
                "movapd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movapd %%xmm2,64(%%r12);"
                "movapd %%xmm3,80(%%r12);"
                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movapd %%xmm2,96(%%r12);"
                "movapd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd %%xmm2,128(%%r14);"
                "movapd %%xmm3,144(%%r14);"
                
                "movapd 160(%%r13),%%xmm0;"
                "movapd 176(%%r13),%%xmm1;"
                "movapd %%xmm2,160(%%r14);"
                "movapd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd %%xmm2,192(%%r14);"
                "movapd %%xmm3,208(%%r14);"                
                
                "movapd 224(%%r13),%%xmm0;"
                "movapd 240(%%r13),%%xmm1;"
                "movapd %%xmm2,224(%%r14);"
                "movapd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movapd_2;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movapd_3(param_t *params) __attribute__((noinline)); 
static void asm_indep_movapd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movapd_3:"               
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd %%xmm3,(%%r12);"
                "movapd %%xmm4,16(%%r12);"
                "movapd %%xmm5,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movapd 48(%%r10),%%xmm0;"                
                "movapd 64(%%r10),%%xmm1;"
                "movapd 80(%%r10),%%xmm2;"                
                "movapd %%xmm3,48(%%r12);"  
                "movapd %%xmm4,64(%%r12);"
                "movapd %%xmm5,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movapd 128(%%r13),%%xmm2;"
                "movapd %%xmm3,96(%%r12);"
                "movapd %%xmm4,112(%%r12);"
                "movapd %%xmm5,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movapd 144(%%r13),%%xmm0;"
                "movapd 160(%%r13),%%xmm1;"
                "movapd 176(%%r13),%%xmm2;"
                "movapd %%xmm3,144(%%r14);"
                "movapd %%xmm4,160(%%r14);"
                "movapd %%xmm5,176(%%r14);"

                "add $288,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movapd %%xmm3,192(%%r14);"
                "movapd %%xmm4,208(%%r14);"                
                "movapd %%xmm5,224(%%r14);"
                
                "movapd 240(%%r13),%%xmm0;"
                "movapd 256(%%r13),%%xmm1;"
                "movapd 272(%%r13),%%xmm2;"                
                "movapd %%xmm3,240(%%r14);"
                "movapd %%xmm4,256(%%r14);"                
                "movapd %%xmm5,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _indep_loop_movapd_3;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movapd_4(param_t *params) __attribute__((noinline)); 
static void asm_indep_movapd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movapd_4:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movapd %%xmm4,(%%r12);"
                "movapd %%xmm5,16(%%r12);"
                "movapd %%xmm6,32(%%r12);"
                "movapd %%xmm7,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movapd 96(%%r13),%%xmm2;"
                "movapd 112(%%r13),%%xmm3;"
                "movapd %%xmm4,64(%%r12);"
                "movapd %%xmm5,80(%%r12);"
                "movapd %%xmm6,96(%%r12);"
                "movapd %%xmm7,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movapd %%xmm4,128(%%r14);"
                "movapd %%xmm5,144(%%r14);"
                "movapd %%xmm6,160(%%r14);"
                "movapd %%xmm7,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movapd 240(%%r13),%%xmm3;"                                               
                "movapd %%xmm4,192(%%r14);"
                "movapd %%xmm5,208(%%r14);"
                "movapd %%xmm6,224(%%r14);"
                "movapd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movapd_4;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movapd_8(param_t *params) __attribute__((noinline)); 
static void asm_indep_movapd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movapd_8:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movapd 64(%%r13),%%xmm4;"
                "movapd 80(%%r13),%%xmm5;"
                "movapd 96(%%r13),%%xmm6;"
                "movapd 112(%%r13),%%xmm7;"

                "mov %%r12, %%r14;"
                
                "movapd %%xmm8,(%%r12);"
                "movapd %%xmm9,16(%%r12);"
                "movapd %%xmm10,32(%%r12);"
                "movapd %%xmm11,48(%%r12);"
                "movapd %%xmm12,64(%%r12);"
                "movapd %%xmm13,80(%%r12);"
                "movapd %%xmm14,96(%%r12);"
                "movapd %%xmm15,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movapd 192(%%r13),%%xmm4;"
                "movapd 208(%%r13),%%xmm5;"
                "movapd 224(%%r13),%%xmm6;"
                "movapd 240(%%r13),%%xmm7;"

                "add $256,%%r12;"
                
                "movapd %%xmm8,128(%%r14);"
                "movapd %%xmm9,144(%%r14);"
                "movapd %%xmm10,160(%%r14);"
                "movapd %%xmm11,176(%%r14);"                                               
                "movapd %%xmm12,192(%%r14);"
                "movapd %%xmm13,208(%%r14);"
                "movapd %%xmm14,224(%%r14);"
                "movapd %%xmm15,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movapd_8;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11","%xmm12", "%xmm13", "%xmm14", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movapd_1(param_t *params) __attribute__((noinline)); 
static void asm_scale_movapd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm5;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movapd_1:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,(%%r12);"
                "movapd 16(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,16(%%r12);"
                "movapd 32(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,32(%%r12);"
                "movapd 48(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,64(%%r12);"
                "movapd 80(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,80(%%r12);"
                "movapd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,96(%%r12);"
                "movapd 112(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,128(%%r14);"
                "movapd 144(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,144(%%r14);"
                "movapd 160(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,160(%%r14);"
                "movapd 176(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,192(%%r14);"
                "movapd 208(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,208(%%r14);"
                "movapd 224(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,224(%%r14);"
                "movapd 240(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movapd %%xmm0,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _scale_loop_movapd_1;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movapd_2(param_t *params) __attribute__((noinline)); 
static void asm_scale_movapd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movapd_2:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                
                "movapd 32(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 48(%%r10),%%xmm1;""mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,32(%%r12);"
                "movapd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 80(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,64(%%r12);"
                "movapd %%xmm1,80(%%r12);"
                
                "movapd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 112(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,96(%%r12);"
                "movapd %%xmm1,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,128(%%r14);"
                "movapd %%xmm1,144(%%r14);"
                
                "movapd 160(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 176(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,160(%%r14);"
                "movapd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,192(%%r14);"
                "movapd %%xmm1,208(%%r14);"                
                
                "movapd 224(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 240(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd %%xmm0,224(%%r14);"
                "movapd %%xmm1,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movapd_2;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movapd_3(param_t *params) __attribute__((noinline)); 
static void asm_scale_movapd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movapd_3:"               
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                "movapd %%xmm2,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movapd 48(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"                
                "movapd 64(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 80(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd %%xmm0,48(%%r12);"  
                "movapd %%xmm1,64(%%r12);"
                "movapd %%xmm2,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movapd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 112(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 128(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd %%xmm0,96(%%r12);"
                "movapd %%xmm1,112(%%r12);"
                "movapd %%xmm2,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movapd 144(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 160(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 176(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd %%xmm0,144(%%r14);"
                "movapd %%xmm1,160(%%r14);"
                "movapd %%xmm2,176(%%r14);"

                "add $288,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 224(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd %%xmm0,192(%%r14);"
                "movapd %%xmm1,208(%%r14);"                
                "movapd %%xmm2,224(%%r14);"
                
                "movapd 240(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 256(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 272(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd %%xmm0,240(%%r14);"
                "movapd %%xmm1,256(%%r14);"                
                "movapd %%xmm2,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _scale_loop_movapd_3;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movapd_4(param_t *params) __attribute__((noinline)); 
static void asm_scale_movapd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movapd_4:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 48(%%r10),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                "movapd %%xmm2,32(%%r12);"
                "movapd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 80(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 96(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 112(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd %%xmm0,64(%%r12);"
                "movapd %%xmm1,80(%%r12);"
                "movapd %%xmm2,96(%%r12);"
                "movapd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 160(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 176(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd %%xmm0,128(%%r14);"
                "movapd %%xmm1,144(%%r14);"
                "movapd %%xmm2,160(%%r14);"
                "movapd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 224(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 240(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd %%xmm0,192(%%r14);"
                "movapd %%xmm1,208(%%r14);"
                "movapd %%xmm2,224(%%r14);"
                "movapd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movapd_4;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movapd_8(param_t *params) __attribute__((noinline)); 
static void asm_scale_movapd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movapd_8:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 48(%%r10),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd 64(%%r10),%%xmm4;mulpd %%xmm15,%%xmm4;"
                "movapd 80(%%r10),%%xmm5;mulpd %%xmm15,%%xmm5;"
                "movapd 96(%%r10),%%xmm6;mulpd %%xmm15,%%xmm6;"
                "movapd 112(%%r10),%%xmm7;mulpd %%xmm15,%%xmm7;"

                "mov %%r12, %%r14;"

                "movapd %%xmm0,(%%r12);"
                "movapd %%xmm1,16(%%r12);"
                "movapd %%xmm2,32(%%r12);"
                "movapd %%xmm3,48(%%r12);"
                "movapd %%xmm4,64(%%r12);"
                "movapd %%xmm5,80(%%r12);"
                "movapd %%xmm6,96(%%r12);"
                "movapd %%xmm7,112(%%r12);" 
                               
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 160(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 176(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd 192(%%r13),%%xmm4;mulpd %%xmm15,%%xmm4;"
                "movapd 208(%%r13),%%xmm5;mulpd %%xmm15,%%xmm5;"
                "movapd 224(%%r13),%%xmm6;mulpd %%xmm15,%%xmm6;"
                "movapd 240(%%r13),%%xmm7;mulpd %%xmm15,%%xmm7;"

                "add $256,%%r12;"

                "movapd %%xmm0,128(%%r14);"
                "movapd %%xmm1,144(%%r14);"
                "movapd %%xmm2,160(%%r14);"
                "movapd %%xmm3,176(%%r14);"
                "movapd %%xmm4,192(%%r14);"
                "movapd %%xmm5,208(%%r14);"
                "movapd %%xmm6,224(%%r14);"
                "movapd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movapd_8;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movupd_1(param_t *params) __attribute__((noinline)); 
static void asm_copy_movupd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movupd_1:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;movupd %%xmm0,(%%r12);"
                "movupd 16(%%r10),%%xmm0;movupd %%xmm0,16(%%r12);"
                "movupd 32(%%r10),%%xmm0;movupd %%xmm0,32(%%r12);"
                "movupd 48(%%r10),%%xmm0;movupd %%xmm0,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;movupd %%xmm0,64(%%r12);"
                "movupd 80(%%r13),%%xmm0;movupd %%xmm0,80(%%r12);"
                "movupd 96(%%r13),%%xmm0;movupd %%xmm0,96(%%r12);"
                "movupd 112(%%r13),%%xmm0;movupd %%xmm0,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;movupd %%xmm0,128(%%r14);"
                "movupd 144(%%r13),%%xmm0;movupd %%xmm0,144(%%r14);"
                "movupd 160(%%r13),%%xmm0;movupd %%xmm0,160(%%r14);"
                "movupd 176(%%r13),%%xmm0;movupd %%xmm0,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;movupd %%xmm0,192(%%r14);"
                "movupd 208(%%r13),%%xmm0;movupd %%xmm0,208(%%r14);"
                "movupd 224(%%r13),%%xmm0;movupd %%xmm0,224(%%r14);"
                "movupd 240(%%r13),%%xmm0;movupd %%xmm0,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _copy_loop_movupd_1;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movupd_2(param_t *params) __attribute__((noinline)); 
static void asm_copy_movupd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movupd_2:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                
                "movupd 32(%%r10),%%xmm0;"
                "movupd 48(%%r10),%%xmm1;"                
                "movupd %%xmm0,32(%%r12);"
                "movupd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;"
                "movupd 80(%%r13),%%xmm1;"
                "movupd %%xmm0,64(%%r12);"
                "movupd %%xmm1,80(%%r12);"
                
                "movupd 96(%%r13),%%xmm0;"
                "movupd 112(%%r13),%%xmm1;"
                "movupd %%xmm0,96(%%r12);"
                "movupd %%xmm1,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;"
                "movupd 144(%%r13),%%xmm1;"
                "movupd %%xmm0,128(%%r14);"
                "movupd %%xmm1,144(%%r14);"
                
                "movupd 160(%%r13),%%xmm0;"
                "movupd 176(%%r13),%%xmm1;"
                "movupd %%xmm0,160(%%r14);"
                "movupd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;"
                "movupd 208(%%r13),%%xmm1;"
                "movupd %%xmm0,192(%%r14);"
                "movupd %%xmm1,208(%%r14);"                
                
                "movupd 224(%%r13),%%xmm0;"
                "movupd 240(%%r13),%%xmm1;"
                "movupd %%xmm0,224(%%r14);"
                "movupd %%xmm1,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movupd_2;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movupd_3(param_t *params) __attribute__((noinline)); 
static void asm_copy_movupd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movupd_3:"               
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd 32(%%r10),%%xmm2;"
                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                "movupd %%xmm2,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movupd 48(%%r10),%%xmm0;"                
                "movupd 64(%%r10),%%xmm1;"
                "movupd 80(%%r10),%%xmm2;"                
                "movupd %%xmm0,48(%%r12);"  
                "movupd %%xmm1,64(%%r12);"
                "movupd %%xmm2,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movupd 96(%%r13),%%xmm0;"
                "movupd 112(%%r13),%%xmm1;"
                "movupd 128(%%r13),%%xmm2;"
                "movupd %%xmm0,96(%%r12);"
                "movupd %%xmm1,112(%%r12);"
                "movupd %%xmm2,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movupd 144(%%r13),%%xmm0;"
                "movupd 160(%%r13),%%xmm1;"
                "movupd 176(%%r13),%%xmm2;"
                "movupd %%xmm0,144(%%r14);"
                "movupd %%xmm1,160(%%r14);"
                "movupd %%xmm2,176(%%r14);"

                "add $288,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;"
                "movupd 208(%%r13),%%xmm1;"
                "movupd 224(%%r13),%%xmm2;"
                "movupd %%xmm0,192(%%r14);"
                "movupd %%xmm1,208(%%r14);"                
                "movupd %%xmm2,224(%%r14);"
                
                "movupd 240(%%r13),%%xmm0;"
                "movupd 256(%%r13),%%xmm1;"
                "movupd 272(%%r13),%%xmm2;"                
                "movupd %%xmm0,240(%%r14);"
                "movupd %%xmm1,256(%%r14);"                
                "movupd %%xmm2,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _copy_loop_movupd_3;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movupd_4(param_t *params) __attribute__((noinline)); 
static void asm_copy_movupd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movupd_4:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd 32(%%r10),%%xmm2;"
                "movupd 48(%%r10),%%xmm3;"
                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                "movupd %%xmm2,32(%%r12);"
                "movupd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;"
                "movupd 80(%%r13),%%xmm1;"
                "movupd 96(%%r13),%%xmm2;"
                "movupd 112(%%r13),%%xmm3;"
                "movupd %%xmm0,64(%%r12);"
                "movupd %%xmm1,80(%%r12);"
                "movupd %%xmm2,96(%%r12);"
                "movupd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;"
                "movupd 144(%%r13),%%xmm1;"
                "movupd 160(%%r13),%%xmm2;"
                "movupd 176(%%r13),%%xmm3;"
                "movupd %%xmm0,128(%%r14);"
                "movupd %%xmm1,144(%%r14);"
                "movupd %%xmm2,160(%%r14);"
                "movupd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;"
                "movupd 208(%%r13),%%xmm1;"
                "movupd 224(%%r13),%%xmm2;"
                "movupd 240(%%r13),%%xmm3;"                                               
                "movupd %%xmm0,192(%%r14);"
                "movupd %%xmm1,208(%%r14);"
                "movupd %%xmm2,224(%%r14);"
                "movupd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movupd_4;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movupd_8(param_t *params) __attribute__((noinline)); 
static void asm_copy_movupd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movupd_8:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd 32(%%r10),%%xmm2;"
                "movupd 48(%%r10),%%xmm3;"
                "movupd 64(%%r13),%%xmm4;"
                "movupd 80(%%r13),%%xmm5;"
                "movupd 96(%%r13),%%xmm6;"
                "movupd 112(%%r13),%%xmm7;"  
                
                "mov %%r12, %%r14;"
                
                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                "movupd %%xmm2,32(%%r12);"
                "movupd %%xmm3,48(%%r12);" 
                "movupd %%xmm4,64(%%r12);"
                "movupd %%xmm5,80(%%r12);"
                "movupd %%xmm6,96(%%r12);"
                "movupd %%xmm7,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;"
                "movupd 144(%%r13),%%xmm1;"
                "movupd 160(%%r13),%%xmm2;"
                "movupd 176(%%r13),%%xmm3;"
                "movupd 192(%%r13),%%xmm4;"
                "movupd 208(%%r13),%%xmm5;"
                "movupd 224(%%r13),%%xmm6;"
                "movupd 240(%%r13),%%xmm7;"

                "add $256,%%r12;"
                
                "movupd %%xmm0,128(%%r14);"
                "movupd %%xmm1,144(%%r14);"
                "movupd %%xmm2,160(%%r14);"
                "movupd %%xmm3,176(%%r14);"
                "movupd %%xmm4,192(%%r14);"
                "movupd %%xmm5,208(%%r14);"
                "movupd %%xmm6,224(%%r14);"
                "movupd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movupd_8;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movupd_1(param_t *params) __attribute__((noinline)); 
static void asm_indep_movupd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movupd_1:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;movupd %%xmm1,(%%r12);"
                "movupd 16(%%r10),%%xmm0;movupd %%xmm1,16(%%r12);"
                "movupd 32(%%r10),%%xmm0;movupd %%xmm1,32(%%r12);"
                "movupd 48(%%r10),%%xmm0;movupd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;movupd %%xmm1,64(%%r12);"
                "movupd 80(%%r13),%%xmm0;movupd %%xmm1,80(%%r12);"
                "movupd 96(%%r13),%%xmm0;movupd %%xmm1,96(%%r12);"
                "movupd 112(%%r13),%%xmm0;movupd %%xmm1,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;movupd %%xmm1,128(%%r14);"
                "movupd 144(%%r13),%%xmm0;movupd %%xmm1,144(%%r14);"
                "movupd 160(%%r13),%%xmm0;movupd %%xmm1,160(%%r14);"
                "movupd 176(%%r13),%%xmm0;movupd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;movupd %%xmm1,192(%%r14);"
                "movupd 208(%%r13),%%xmm0;movupd %%xmm1,208(%%r14);"
                "movupd 224(%%r13),%%xmm0;movupd %%xmm1,224(%%r14);"
                "movupd 240(%%r13),%%xmm0;movupd %%xmm1,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _indep_loop_movupd_1;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0","%xmm1", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movupd_2(param_t *params) __attribute__((noinline)); 
static void asm_indep_movupd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movupd_2:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd %%xmm2,(%%r12);"
                "movupd %%xmm3,16(%%r12);"
                
                "movupd 32(%%r10),%%xmm0;"
                "movupd 48(%%r10),%%xmm1;"                
                "movupd %%xmm2,32(%%r12);"
                "movupd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;"
                "movupd 80(%%r13),%%xmm1;"
                "movupd %%xmm2,64(%%r12);"
                "movupd %%xmm3,80(%%r12);"
                
                "movupd 96(%%r13),%%xmm0;"
                "movupd 112(%%r13),%%xmm1;"
                "movupd %%xmm2,96(%%r12);"
                "movupd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;"
                "movupd 144(%%r13),%%xmm1;"
                "movupd %%xmm2,128(%%r14);"
                "movupd %%xmm3,144(%%r14);"
                
                "movupd 160(%%r13),%%xmm0;"
                "movupd 176(%%r13),%%xmm1;"
                "movupd %%xmm2,160(%%r14);"
                "movupd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;"
                "movupd 208(%%r13),%%xmm1;"
                "movupd %%xmm2,192(%%r14);"
                "movupd %%xmm3,208(%%r14);"                
                
                "movupd 224(%%r13),%%xmm0;"
                "movupd 240(%%r13),%%xmm1;"
                "movupd %%xmm2,224(%%r14);"
                "movupd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movupd_2;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movupd_3(param_t *params) __attribute__((noinline)); 
static void asm_indep_movupd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movupd_3:"               
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd 32(%%r10),%%xmm2;"
                "movupd %%xmm3,(%%r12);"
                "movupd %%xmm4,16(%%r12);"
                "movupd %%xmm5,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movupd 48(%%r10),%%xmm0;"                
                "movupd 64(%%r10),%%xmm1;"
                "movupd 80(%%r10),%%xmm2;"                
                "movupd %%xmm3,48(%%r12);"  
                "movupd %%xmm4,64(%%r12);"
                "movupd %%xmm5,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movupd 96(%%r13),%%xmm0;"
                "movupd 112(%%r13),%%xmm1;"
                "movupd 128(%%r13),%%xmm2;"
                "movupd %%xmm3,96(%%r12);"
                "movupd %%xmm4,112(%%r12);"
                "movupd %%xmm5,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movupd 144(%%r13),%%xmm0;"
                "movupd 160(%%r13),%%xmm1;"
                "movupd 176(%%r13),%%xmm2;"
                "movupd %%xmm3,144(%%r14);"
                "movupd %%xmm4,160(%%r14);"
                "movupd %%xmm5,176(%%r14);"

                "add $288,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;"
                "movupd 208(%%r13),%%xmm1;"
                "movupd 224(%%r13),%%xmm2;"
                "movupd %%xmm3,192(%%r14);"
                "movupd %%xmm4,208(%%r14);"                
                "movupd %%xmm5,224(%%r14);"
                
                "movupd 240(%%r13),%%xmm0;"
                "movupd 256(%%r13),%%xmm1;"
                "movupd 272(%%r13),%%xmm2;"                
                "movupd %%xmm3,240(%%r14);"
                "movupd %%xmm4,256(%%r14);"                
                "movupd %%xmm5,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _indep_loop_movupd_3;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movupd_4(param_t *params) __attribute__((noinline)); 
static void asm_indep_movupd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movupd_4:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd 32(%%r10),%%xmm2;"
                "movupd 48(%%r10),%%xmm3;"
                "movupd %%xmm4,(%%r12);"
                "movupd %%xmm5,16(%%r12);"
                "movupd %%xmm6,32(%%r12);"
                "movupd %%xmm7,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;"
                "movupd 80(%%r13),%%xmm1;"
                "movupd 96(%%r13),%%xmm2;"
                "movupd 112(%%r13),%%xmm3;"
                "movupd %%xmm4,64(%%r12);"
                "movupd %%xmm5,80(%%r12);"
                "movupd %%xmm6,96(%%r12);"
                "movupd %%xmm7,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;"
                "movupd 144(%%r13),%%xmm1;"
                "movupd 160(%%r13),%%xmm2;"
                "movupd 176(%%r13),%%xmm3;"
                "movupd %%xmm4,128(%%r14);"
                "movupd %%xmm5,144(%%r14);"
                "movupd %%xmm6,160(%%r14);"
                "movupd %%xmm7,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;"
                "movupd 208(%%r13),%%xmm1;"
                "movupd 224(%%r13),%%xmm2;"
                "movupd 240(%%r13),%%xmm3;"                                               
                "movupd %%xmm4,192(%%r14);"
                "movupd %%xmm5,208(%%r14);"
                "movupd %%xmm6,224(%%r14);"
                "movupd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movupd_4;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movupd_8(param_t *params) __attribute__((noinline)); 
static void asm_indep_movupd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movupd_8:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;"
                "movupd 16(%%r10),%%xmm1;"
                "movupd 32(%%r10),%%xmm2;"
                "movupd 48(%%r10),%%xmm3;"
                "movupd 64(%%r13),%%xmm4;"
                "movupd 80(%%r13),%%xmm5;"
                "movupd 96(%%r13),%%xmm6;"
                "movupd 112(%%r13),%%xmm7;"

                "mov %%r12, %%r14;"
                
                "movupd %%xmm8,(%%r12);"
                "movupd %%xmm9,16(%%r12);"
                "movupd %%xmm10,32(%%r12);"
                "movupd %%xmm11,48(%%r12);"
                "movupd %%xmm12,64(%%r12);"
                "movupd %%xmm13,80(%%r12);"
                "movupd %%xmm14,96(%%r12);"
                "movupd %%xmm15,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;"
                "movupd 144(%%r13),%%xmm1;"
                "movupd 160(%%r13),%%xmm2;"
                "movupd 176(%%r13),%%xmm3;"
                "movupd 192(%%r13),%%xmm4;"
                "movupd 208(%%r13),%%xmm5;"
                "movupd 224(%%r13),%%xmm6;"
                "movupd 240(%%r13),%%xmm7;"

                "add $256,%%r12;"
                
                "movupd %%xmm8,128(%%r14);"
                "movupd %%xmm9,144(%%r14);"
                "movupd %%xmm10,160(%%r14);"
                "movupd %%xmm11,176(%%r14);"                                               
                "movupd %%xmm12,192(%%r14);"
                "movupd %%xmm13,208(%%r14);"
                "movupd %%xmm14,224(%%r14);"
                "movupd %%xmm15,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movupd_8;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11","%xmm12", "%xmm13", "%xmm14", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movupd_1(param_t *params) __attribute__((noinline)); 
static void asm_scale_movupd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm5;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movupd_1:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,(%%r12);"
                "movupd 16(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,16(%%r12);"
                "movupd 32(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,32(%%r12);"
                "movupd 48(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,64(%%r12);"
                "movupd 80(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,80(%%r12);"
                "movupd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,96(%%r12);"
                "movupd 112(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,128(%%r14);"
                "movupd 144(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,144(%%r14);"
                "movupd 160(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,160(%%r14);"
                "movupd 176(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,192(%%r14);"
                "movupd 208(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,208(%%r14);"
                "movupd 224(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,224(%%r14);"
                "movupd 240(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movupd %%xmm0,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _scale_loop_movupd_1;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movupd_2(param_t *params) __attribute__((noinline)); 
static void asm_scale_movupd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movupd_2:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                
                "movupd 32(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 48(%%r10),%%xmm1;""mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,32(%%r12);"
                "movupd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 80(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,64(%%r12);"
                "movupd %%xmm1,80(%%r12);"
                
                "movupd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 112(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,96(%%r12);"
                "movupd %%xmm1,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,128(%%r14);"
                "movupd %%xmm1,144(%%r14);"
                
                "movupd 160(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 176(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,160(%%r14);"
                "movupd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,192(%%r14);"
                "movupd %%xmm1,208(%%r14);"                
                
                "movupd 224(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 240(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd %%xmm0,224(%%r14);"
                "movupd %%xmm1,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movupd_2;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movupd_3(param_t *params) __attribute__((noinline)); 
static void asm_scale_movupd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movupd_3:"               
                
                "movupd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                "movupd %%xmm2,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movupd 48(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"                
                "movupd 64(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 80(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd %%xmm0,48(%%r12);"  
                "movupd %%xmm1,64(%%r12);"
                "movupd %%xmm2,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movupd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 112(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 128(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd %%xmm0,96(%%r12);"
                "movupd %%xmm1,112(%%r12);"
                "movupd %%xmm2,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movupd 144(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 160(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 176(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd %%xmm0,144(%%r14);"
                "movupd %%xmm1,160(%%r14);"
                "movupd %%xmm2,176(%%r14);"

                "add $288,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 224(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd %%xmm0,192(%%r14);"
                "movupd %%xmm1,208(%%r14);"                
                "movupd %%xmm2,224(%%r14);"
                
                "movupd 240(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 256(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 272(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd %%xmm0,240(%%r14);"
                "movupd %%xmm1,256(%%r14);"                
                "movupd %%xmm2,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _scale_loop_movupd_3;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movupd_4(param_t *params) __attribute__((noinline)); 
static void asm_scale_movupd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movupd_4:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd 48(%%r10),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                "movupd %%xmm2,32(%%r12);"
                "movupd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movupd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 80(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 96(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd 112(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movupd %%xmm0,64(%%r12);"
                "movupd %%xmm1,80(%%r12);"
                "movupd %%xmm2,96(%%r12);"
                "movupd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 160(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd 176(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movupd %%xmm0,128(%%r14);"
                "movupd %%xmm1,144(%%r14);"
                "movupd %%xmm2,160(%%r14);"
                "movupd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movupd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 224(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd 240(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movupd %%xmm0,192(%%r14);"
                "movupd %%xmm1,208(%%r14);"
                "movupd %%xmm2,224(%%r14);"
                "movupd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movupd_4;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movupd_8(param_t *params) __attribute__((noinline)); 
static void asm_scale_movupd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movupd_8:"
                
                "mov %%r10, %%r13;"
                
                "movupd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd 48(%%r10),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movupd 64(%%r10),%%xmm4;mulpd %%xmm15,%%xmm4;"
                "movupd 80(%%r10),%%xmm5;mulpd %%xmm15,%%xmm5;"
                "movupd 96(%%r10),%%xmm6;mulpd %%xmm15,%%xmm6;"
                "movupd 112(%%r10),%%xmm7;mulpd %%xmm15,%%xmm7;"

                "mov %%r12, %%r14;"

                "movupd %%xmm0,(%%r12);"
                "movupd %%xmm1,16(%%r12);"
                "movupd %%xmm2,32(%%r12);"
                "movupd %%xmm3,48(%%r12);"
                "movupd %%xmm4,64(%%r12);"
                "movupd %%xmm5,80(%%r12);"
                "movupd %%xmm6,96(%%r12);"
                "movupd %%xmm7,112(%%r12);" 
                               
                "add $256,%%r10;"
                
                "movupd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movupd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movupd 160(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movupd 176(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movupd 192(%%r13),%%xmm4;mulpd %%xmm15,%%xmm4;"
                "movupd 208(%%r13),%%xmm5;mulpd %%xmm15,%%xmm5;"
                "movupd 224(%%r13),%%xmm6;mulpd %%xmm15,%%xmm6;"
                "movupd 240(%%r13),%%xmm7;mulpd %%xmm15,%%xmm7;"

                "add $256,%%r12;"

                "movupd %%xmm0,128(%%r14);"
                "movupd %%xmm1,144(%%r14);"
                "movupd %%xmm2,160(%%r14);"
                "movupd %%xmm3,176(%%r14);"
                "movupd %%xmm4,192(%%r14);"
                "movupd %%xmm5,208(%%r14);"
                "movupd %%xmm6,224(%%r14);"
                "movupd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movupd_8;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}


static void asm_copy_movntpd_1(param_t *params) __attribute__((noinline)); 
static void asm_copy_movntpd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movntpd_1:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;movntpd %%xmm0,(%%r12);"
                "movapd 16(%%r10),%%xmm0;movntpd %%xmm0,16(%%r12);"
                "movapd 32(%%r10),%%xmm0;movntpd %%xmm0,32(%%r12);"
                "movapd 48(%%r10),%%xmm0;movntpd %%xmm0,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;movntpd %%xmm0,64(%%r12);"
                "movapd 80(%%r13),%%xmm0;movntpd %%xmm0,80(%%r12);"
                "movapd 96(%%r13),%%xmm0;movntpd %%xmm0,96(%%r12);"
                "movapd 112(%%r13),%%xmm0;movntpd %%xmm0,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;movntpd %%xmm0,128(%%r14);"
                "movapd 144(%%r13),%%xmm0;movntpd %%xmm0,144(%%r14);"
                "movapd 160(%%r13),%%xmm0;movntpd %%xmm0,160(%%r14);"
                "movapd 176(%%r13),%%xmm0;movntpd %%xmm0,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;movntpd %%xmm0,192(%%r14);"
                "movapd 208(%%r13),%%xmm0;movntpd %%xmm0,208(%%r14);"
                "movapd 224(%%r13),%%xmm0;movntpd %%xmm0,224(%%r14);"
                "movapd 240(%%r13),%%xmm0;movntpd %%xmm0,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _copy_loop_movntpd_1;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movntpd_2(param_t *params) __attribute__((noinline)); 
static void asm_copy_movntpd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movntpd_2:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                
                "movapd 32(%%r10),%%xmm0;"
                "movapd 48(%%r10),%%xmm1;"                
                "movntpd %%xmm0,32(%%r12);"
                "movntpd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movntpd %%xmm0,64(%%r12);"
                "movntpd %%xmm1,80(%%r12);"
                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movntpd %%xmm0,96(%%r12);"
                "movntpd %%xmm1,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movntpd %%xmm0,128(%%r14);"
                "movntpd %%xmm1,144(%%r14);"
                
                "movapd 160(%%r13),%%xmm0;"
                "movapd 176(%%r13),%%xmm1;"
                "movntpd %%xmm0,160(%%r14);"
                "movntpd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movntpd %%xmm0,192(%%r14);"
                "movntpd %%xmm1,208(%%r14);"                
                
                "movapd 224(%%r13),%%xmm0;"
                "movapd 240(%%r13),%%xmm1;"
                "movntpd %%xmm0,224(%%r14);"
                "movntpd %%xmm1,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movntpd_2;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movntpd_3(param_t *params) __attribute__((noinline)); 
static void asm_copy_movntpd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movntpd_3:"               
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                "movntpd %%xmm2,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movapd 48(%%r10),%%xmm0;"                
                "movapd 64(%%r10),%%xmm1;"
                "movapd 80(%%r10),%%xmm2;"                
                "movntpd %%xmm0,48(%%r12);"  
                "movntpd %%xmm1,64(%%r12);"
                "movntpd %%xmm2,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movapd 128(%%r13),%%xmm2;"
                "movntpd %%xmm0,96(%%r12);"
                "movntpd %%xmm1,112(%%r12);"
                "movntpd %%xmm2,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movapd 144(%%r13),%%xmm0;"
                "movapd 160(%%r13),%%xmm1;"
                "movapd 176(%%r13),%%xmm2;"
                "movntpd %%xmm0,144(%%r14);"
                "movntpd %%xmm1,160(%%r14);"
                "movntpd %%xmm2,176(%%r14);"

                "add $288,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movntpd %%xmm0,192(%%r14);"
                "movntpd %%xmm1,208(%%r14);"                
                "movntpd %%xmm2,224(%%r14);"
                
                "movapd 240(%%r13),%%xmm0;"
                "movapd 256(%%r13),%%xmm1;"
                "movapd 272(%%r13),%%xmm2;"                
                "movntpd %%xmm0,240(%%r14);"
                "movntpd %%xmm1,256(%%r14);"                
                "movntpd %%xmm2,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _copy_loop_movntpd_3;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movntpd_4(param_t *params) __attribute__((noinline)); 
static void asm_copy_movntpd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movntpd_4:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                "movntpd %%xmm2,32(%%r12);"
                "movntpd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movapd 96(%%r13),%%xmm2;"
                "movapd 112(%%r13),%%xmm3;"
                "movntpd %%xmm0,64(%%r12);"
                "movntpd %%xmm1,80(%%r12);"
                "movntpd %%xmm2,96(%%r12);"
                "movntpd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movntpd %%xmm0,128(%%r14);"
                "movntpd %%xmm1,144(%%r14);"
                "movntpd %%xmm2,160(%%r14);"
                "movntpd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movapd 240(%%r13),%%xmm3;"                                               
                "movntpd %%xmm0,192(%%r14);"
                "movntpd %%xmm1,208(%%r14);"
                "movntpd %%xmm2,224(%%r14);"
                "movntpd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movntpd_4;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_copy_movntpd_8(param_t *params) __attribute__((noinline)); 
static void asm_copy_movntpd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_copy_loop_movntpd_8:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movapd 64(%%r13),%%xmm4;"
                "movapd 80(%%r13),%%xmm5;"
                "movapd 96(%%r13),%%xmm6;"
                "movapd 112(%%r13),%%xmm7;"  
                
                "mov %%r12, %%r14;"
                
                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                "movntpd %%xmm2,32(%%r12);"
                "movntpd %%xmm3,48(%%r12);" 
                "movntpd %%xmm4,64(%%r12);"
                "movntpd %%xmm5,80(%%r12);"
                "movntpd %%xmm6,96(%%r12);"
                "movntpd %%xmm7,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movapd 192(%%r13),%%xmm4;"
                "movapd 208(%%r13),%%xmm5;"
                "movapd 224(%%r13),%%xmm6;"
                "movapd 240(%%r13),%%xmm7;"

                "add $256,%%r12;"
                
                "movntpd %%xmm0,128(%%r14);"
                "movntpd %%xmm1,144(%%r14);"
                "movntpd %%xmm2,160(%%r14);"
                "movntpd %%xmm3,176(%%r14);"
                "movntpd %%xmm4,192(%%r14);"
                "movntpd %%xmm5,208(%%r14);"
                "movntpd %%xmm6,224(%%r14);"
                "movntpd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _copy_loop_movntpd_8;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movntpd_1(param_t *params) __attribute__((noinline)); 
static void asm_indep_movntpd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movntpd_1:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;movntpd %%xmm1,(%%r12);"
                "movapd 16(%%r10),%%xmm0;movntpd %%xmm1,16(%%r12);"
                "movapd 32(%%r10),%%xmm0;movntpd %%xmm1,32(%%r12);"
                "movapd 48(%%r10),%%xmm0;movntpd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;movntpd %%xmm1,64(%%r12);"
                "movapd 80(%%r13),%%xmm0;movntpd %%xmm1,80(%%r12);"
                "movapd 96(%%r13),%%xmm0;movntpd %%xmm1,96(%%r12);"
                "movapd 112(%%r13),%%xmm0;movntpd %%xmm1,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;movntpd %%xmm1,128(%%r14);"
                "movapd 144(%%r13),%%xmm0;movntpd %%xmm1,144(%%r14);"
                "movapd 160(%%r13),%%xmm0;movntpd %%xmm1,160(%%r14);"
                "movapd 176(%%r13),%%xmm0;movntpd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;movntpd %%xmm1,192(%%r14);"
                "movapd 208(%%r13),%%xmm0;movntpd %%xmm1,208(%%r14);"
                "movapd 224(%%r13),%%xmm0;movntpd %%xmm1,224(%%r14);"
                "movapd 240(%%r13),%%xmm0;movntpd %%xmm1,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _indep_loop_movntpd_1;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0","%xmm1", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movntpd_2(param_t *params) __attribute__((noinline)); 
static void asm_indep_movntpd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movntpd_2:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movntpd %%xmm2,(%%r12);"
                "movntpd %%xmm3,16(%%r12);"
                
                "movapd 32(%%r10),%%xmm0;"
                "movapd 48(%%r10),%%xmm1;"                
                "movntpd %%xmm2,32(%%r12);"
                "movntpd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movntpd %%xmm2,64(%%r12);"
                "movntpd %%xmm3,80(%%r12);"
                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movntpd %%xmm2,96(%%r12);"
                "movntpd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movntpd %%xmm2,128(%%r14);"
                "movntpd %%xmm3,144(%%r14);"
                
                "movapd 160(%%r13),%%xmm0;"
                "movapd 176(%%r13),%%xmm1;"
                "movntpd %%xmm2,160(%%r14);"
                "movntpd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movntpd %%xmm2,192(%%r14);"
                "movntpd %%xmm3,208(%%r14);"                
                
                "movapd 224(%%r13),%%xmm0;"
                "movapd 240(%%r13),%%xmm1;"
                "movntpd %%xmm2,224(%%r14);"
                "movntpd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movntpd_2;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movntpd_3(param_t *params) __attribute__((noinline)); 
static void asm_indep_movntpd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movntpd_3:"               
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movntpd %%xmm3,(%%r12);"
                "movntpd %%xmm4,16(%%r12);"
                "movntpd %%xmm5,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movapd 48(%%r10),%%xmm0;"                
                "movapd 64(%%r10),%%xmm1;"
                "movapd 80(%%r10),%%xmm2;"                
                "movntpd %%xmm3,48(%%r12);"  
                "movntpd %%xmm4,64(%%r12);"
                "movntpd %%xmm5,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movapd 96(%%r13),%%xmm0;"
                "movapd 112(%%r13),%%xmm1;"
                "movapd 128(%%r13),%%xmm2;"
                "movntpd %%xmm3,96(%%r12);"
                "movntpd %%xmm4,112(%%r12);"
                "movntpd %%xmm5,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movapd 144(%%r13),%%xmm0;"
                "movapd 160(%%r13),%%xmm1;"
                "movapd 176(%%r13),%%xmm2;"
                "movntpd %%xmm3,144(%%r14);"
                "movntpd %%xmm4,160(%%r14);"
                "movntpd %%xmm5,176(%%r14);"

                "add $288,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movntpd %%xmm3,192(%%r14);"
                "movntpd %%xmm4,208(%%r14);"                
                "movntpd %%xmm5,224(%%r14);"
                
                "movapd 240(%%r13),%%xmm0;"
                "movapd 256(%%r13),%%xmm1;"
                "movapd 272(%%r13),%%xmm2;"                
                "movntpd %%xmm3,240(%%r14);"
                "movntpd %%xmm4,256(%%r14);"                
                "movntpd %%xmm5,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _indep_loop_movntpd_3;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movntpd_4(param_t *params) __attribute__((noinline)); 
static void asm_indep_movntpd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movntpd_4:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movntpd %%xmm4,(%%r12);"
                "movntpd %%xmm5,16(%%r12);"
                "movntpd %%xmm6,32(%%r12);"
                "movntpd %%xmm7,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;"
                "movapd 80(%%r13),%%xmm1;"
                "movapd 96(%%r13),%%xmm2;"
                "movapd 112(%%r13),%%xmm3;"
                "movntpd %%xmm4,64(%%r12);"
                "movntpd %%xmm5,80(%%r12);"
                "movntpd %%xmm6,96(%%r12);"
                "movntpd %%xmm7,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movntpd %%xmm4,128(%%r14);"
                "movntpd %%xmm5,144(%%r14);"
                "movntpd %%xmm6,160(%%r14);"
                "movntpd %%xmm7,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;"
                "movapd 208(%%r13),%%xmm1;"
                "movapd 224(%%r13),%%xmm2;"
                "movapd 240(%%r13),%%xmm3;"                                               
                "movntpd %%xmm4,192(%%r14);"
                "movntpd %%xmm5,208(%%r14);"
                "movntpd %%xmm6,224(%%r14);"
                "movntpd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movntpd_4;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_indep_movntpd_8(param_t *params) __attribute__((noinline)); 
static void asm_indep_movntpd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_indep_loop_movntpd_8:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;"
                "movapd 16(%%r10),%%xmm1;"
                "movapd 32(%%r10),%%xmm2;"
                "movapd 48(%%r10),%%xmm3;"
                "movapd 64(%%r13),%%xmm4;"
                "movapd 80(%%r13),%%xmm5;"
                "movapd 96(%%r13),%%xmm6;"
                "movapd 112(%%r13),%%xmm7;"

                "mov %%r12, %%r14;"
                
                "movntpd %%xmm8,(%%r12);"
                "movntpd %%xmm9,16(%%r12);"
                "movntpd %%xmm10,32(%%r12);"
                "movntpd %%xmm11,48(%%r12);"
                "movntpd %%xmm12,64(%%r12);"
                "movntpd %%xmm13,80(%%r12);"
                "movntpd %%xmm14,96(%%r12);"
                "movntpd %%xmm15,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;"
                "movapd 144(%%r13),%%xmm1;"
                "movapd 160(%%r13),%%xmm2;"
                "movapd 176(%%r13),%%xmm3;"
                "movapd 192(%%r13),%%xmm4;"
                "movapd 208(%%r13),%%xmm5;"
                "movapd 224(%%r13),%%xmm6;"
                "movapd 240(%%r13),%%xmm7;"

                "add $256,%%r12;"
                
                "movntpd %%xmm8,128(%%r14);"
                "movntpd %%xmm9,144(%%r14);"
                "movntpd %%xmm10,160(%%r14);"
                "movntpd %%xmm11,176(%%r14);"                                               
                "movntpd %%xmm12,192(%%r14);"
                "movntpd %%xmm13,208(%%r14);"
                "movntpd %%xmm14,224(%%r14);"
                "movntpd %%xmm15,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _indep_loop_movntpd_8;"
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
                : "=a" (params->rax)
                : "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11","%xmm12", "%xmm13", "%xmm14", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movntpd_1(param_t *params) __attribute__((noinline)); 
static void asm_scale_movntpd_1(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm5;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movntpd_1:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,(%%r12);"
                "movapd 16(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,16(%%r12);"
                "movapd 32(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,32(%%r12);"
                "movapd 48(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,64(%%r12);"
                "movapd 80(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,80(%%r12);"
                "movapd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,96(%%r12);"
                "movapd 112(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,112(%%r12);"              
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,128(%%r14);"
                "movapd 144(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,144(%%r14);"
                "movapd 160(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,160(%%r14);"
                "movapd 176(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,192(%%r14);"
                "movapd 208(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,208(%%r14);"
                "movapd 224(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,224(%%r14);"
                "movapd 240(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;movntpd %%xmm0,240(%%r14);"                                              

                "sub $1,%%r11;"
                "jnz _scale_loop_movntpd_1;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movntpd_2(param_t *params) __attribute__((noinline)); 
static void asm_scale_movntpd_2(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movntpd_2:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                
                "movapd 32(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 48(%%r10),%%xmm1;""mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,32(%%r12);"
                "movntpd %%xmm1,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 80(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,64(%%r12);"
                "movntpd %%xmm1,80(%%r12);"
                
                "movapd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 112(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,96(%%r12);"
                "movntpd %%xmm1,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,128(%%r14);"
                "movntpd %%xmm1,144(%%r14);"
                
                "movapd 160(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 176(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,160(%%r14);"
                "movntpd %%xmm1,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,192(%%r14);"
                "movntpd %%xmm1,208(%%r14);"                
                
                "movapd 224(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 240(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movntpd %%xmm0,224(%%r14);"
                "movntpd %%xmm1,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movntpd_2;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movntpd_3(param_t *params) __attribute__((noinline)); 
static void asm_scale_movntpd_3(param_t *params)
{
  #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movntpd_3:"               
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                "movntpd %%xmm2,32(%%r12);"
                
                "mov %%r10, %%r13;"
                
                "movapd 48(%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"                
                "movapd 64(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 80(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movntpd %%xmm0,48(%%r12);"  
                "movntpd %%xmm1,64(%%r12);"
                "movntpd %%xmm2,80(%%r12);"
                
                "mov %%r12, %%r14;"
                                
                "movapd 96(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 112(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 128(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movntpd %%xmm0,96(%%r12);"
                "movntpd %%xmm1,112(%%r12);"
                "movntpd %%xmm2,128(%%r12);"              
                
                "add $288,%%r10;"
                                
                "movapd 144(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 160(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 176(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movntpd %%xmm0,144(%%r14);"
                "movntpd %%xmm1,160(%%r14);"
                "movntpd %%xmm2,176(%%r14);"

                "add $288,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 224(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movntpd %%xmm0,192(%%r14);"
                "movntpd %%xmm1,208(%%r14);"                
                "movntpd %%xmm2,224(%%r14);"
                
                "movapd 240(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 256(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 272(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movntpd %%xmm0,240(%%r14);"
                "movntpd %%xmm1,256(%%r14);"                
                "movntpd %%xmm2,272(%%r14);"
                
                "sub $1,%%r11;"
                "jnz _scale_loop_movntpd_3;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movntpd_4(param_t *params) __attribute__((noinline)); 
static void asm_scale_movntpd_4(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movntpd_4:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 48(%%r10),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                "movntpd %%xmm2,32(%%r12);"
                "movntpd %%xmm3,48(%%r12);"  
                
                "mov %%r12, %%r14;"
                
                "movapd 64(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 80(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 96(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 112(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movntpd %%xmm0,64(%%r12);"
                "movntpd %%xmm1,80(%%r12);"
                "movntpd %%xmm2,96(%%r12);"
                "movntpd %%xmm3,112(%%r12);"               
                
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 160(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 176(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movntpd %%xmm0,128(%%r14);"
                "movntpd %%xmm1,144(%%r14);"
                "movntpd %%xmm2,160(%%r14);"
                "movntpd %%xmm3,176(%%r14);"

                "add $256,%%r12;"
                
                "movapd 192(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 208(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 224(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 240(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movntpd %%xmm0,192(%%r14);"
                "movntpd %%xmm1,208(%%r14);"
                "movntpd %%xmm2,224(%%r14);"
                "movntpd %%xmm3,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movntpd_4;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

static void asm_scale_movntpd_8(param_t *params) __attribute__((noinline)); 
static void asm_scale_movntpd_8(param_t *params)
{
   #ifdef USE_PAPI
    if (params->num_events) PAPI_reset(params->Eventset);
    //__asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
   #endif   
   /*
    * Input: RAX: scaling factor
    *        RBX: addr1 (pointer to the source buffer)
    *        RCX: passes (number of loop iterations)
    *        RDX: addr2 (pointer to destination buffer)
    * Output : RAX stop timestamp - start timestamp
    */
    __asm__ __volatile__(
                "mov %%rbx,%%r10;"
                "mov %%rcx,%%r11;"
                "mov %%rdx,%%r12;"
                "movddup (%%rax),%%xmm15;"
                //first timestamp
                "rdtsc;"
                "shl $32,%%rdx;"
                "add %%rdx,%%rax;"
                "mov %%rax,%%r9;"
                #ifdef FORCE_CPUID
                "mov $0, %%rax;"
                "cpuid;"
                #else
                "mfence;"
                #endif
                ".align 64;"
                "_scale_loop_movntpd_8:"
                
                "mov %%r10, %%r13;"
                
                "movapd (%%r10),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 16(%%r10),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 32(%%r10),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 48(%%r10),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd 64(%%r10),%%xmm4;mulpd %%xmm15,%%xmm4;"
                "movapd 80(%%r10),%%xmm5;mulpd %%xmm15,%%xmm5;"
                "movapd 96(%%r10),%%xmm6;mulpd %%xmm15,%%xmm6;"
                "movapd 112(%%r10),%%xmm7;mulpd %%xmm15,%%xmm7;"

                "mov %%r12, %%r14;"

                "movntpd %%xmm0,(%%r12);"
                "movntpd %%xmm1,16(%%r12);"
                "movntpd %%xmm2,32(%%r12);"
                "movntpd %%xmm3,48(%%r12);"
                "movntpd %%xmm4,64(%%r12);"
                "movntpd %%xmm5,80(%%r12);"
                "movntpd %%xmm6,96(%%r12);"
                "movntpd %%xmm7,112(%%r12);" 
                               
                "add $256,%%r10;"
                
                "movapd 128(%%r13),%%xmm0;mulpd %%xmm15,%%xmm0;"
                "movapd 144(%%r13),%%xmm1;mulpd %%xmm15,%%xmm1;"
                "movapd 160(%%r13),%%xmm2;mulpd %%xmm15,%%xmm2;"
                "movapd 176(%%r13),%%xmm3;mulpd %%xmm15,%%xmm3;"
                "movapd 192(%%r13),%%xmm4;mulpd %%xmm15,%%xmm4;"
                "movapd 208(%%r13),%%xmm5;mulpd %%xmm15,%%xmm5;"
                "movapd 224(%%r13),%%xmm6;mulpd %%xmm15,%%xmm6;"
                "movapd 240(%%r13),%%xmm7;mulpd %%xmm15,%%xmm7;"

                "add $256,%%r12;"

                "movntpd %%xmm0,128(%%r14);"
                "movntpd %%xmm1,144(%%r14);"
                "movntpd %%xmm2,160(%%r14);"
                "movntpd %%xmm3,176(%%r14);"
                "movntpd %%xmm4,192(%%r14);"
                "movntpd %%xmm5,208(%%r14);"
                "movntpd %%xmm6,224(%%r14);"
                "movntpd %%xmm7,240(%%r14);"

                "sub $1,%%r11;"
                "jnz _scale_loop_movntpd_8;"
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
                : "=a" (params->rax)
                : "a" (&(params->factor)), "b"(params->addr_1), "c" (params->passes), "d" (params->addr_2)
                : "%r9", "%r10","%r11", "%r12", "%r13", "%r14","%xmm0", "%xmm1", "%xmm2", "%xmm3","%xmm4", "%xmm5", "%xmm6", "%xmm7", "%xmm15", "memory"
   );                
  #ifdef USE_PAPI
    // __asm__ __volatile__("cpuid;"::: "%rax" ,"%rbx","%rcx", "%rdx");
    if (params->num_events) PAPI_read(params->Eventset,params->values);
  #endif
}

/*
 * function that does the measurement
 */
void inline _work(unsigned long long memsize, volatile mydata_t* data, double **results)
{
	register unsigned long long aligned_addr;	  /* used as pointer to param_t structure (PARAMS) */
	void (*asm_work)(param_t*)=NULL;            /* pointer to selected measurement routine */

 /* select asm routine according to selected function and method and burst length */
  switch(data->function)
  {
   case USE_MOV:
     switch (data->method)
     {
      case METHOD_INDEP:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_indep_mov_1; break;
          case 2: asm_work=&asm_indep_mov_2; break;
          case 3: asm_work=&asm_indep_mov_3; break;
          case 4: asm_work=&asm_indep_mov_4; break;
          default: break;
        }
        break;
      case METHOD_COPY: 
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_copy_mov_1; break;
          case 2: asm_work=&asm_copy_mov_2; break;
          case 3: asm_work=&asm_copy_mov_3; break;
          case 4: asm_work=&asm_copy_mov_4; break; 
          default: break;
        }
        break;
      case METHOD_SCALE:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_scale_mov_1; break;
          case 2: asm_work=&asm_scale_mov_2; break;
          case 3: asm_work=&asm_scale_mov_3; break;
          case 4: asm_work=&asm_scale_mov_4; break; 
          default: break;
        }
        break;
      default: break;
     }
     break;
   case USE_MOVNTI:
     switch (data->method)
     {
      case METHOD_INDEP:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_indep_movnti_1; break;
          case 2: asm_work=&asm_indep_movnti_2; break;
          case 3: asm_work=&asm_indep_movnti_3; break;
          case 4: asm_work=&asm_indep_movnti_4; break;
          default: break;
        }
        break;
      case METHOD_COPY: 
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_copy_movnti_1; break;
          case 2: asm_work=&asm_copy_movnti_2; break;
          case 3: asm_work=&asm_copy_movnti_3; break;
          case 4: asm_work=&asm_copy_movnti_4; break; 
          default: break;
        }
        break;
      case METHOD_SCALE:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_scale_movnti_1; break;
          case 2: asm_work=&asm_scale_movnti_2; break;
          case 3: asm_work=&asm_scale_movnti_3; break;
          case 4: asm_work=&asm_scale_movnti_4; break; 
          default: break;
        }
        break;
      default: break;
     }
     break;
   case USE_MOV_CLFLUSH:
     switch (data->method)
     {
      case METHOD_INDEP:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_indep_mov_clflush_1; break;
          case 2: asm_work=&asm_indep_mov_clflush_2; break;
          case 3: asm_work=&asm_indep_mov_clflush_3; break;
          case 4: asm_work=&asm_indep_mov_clflush_4; break;
          default: break;
        }
        break;
      case METHOD_COPY: 
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_copy_mov_clflush_1; break;
          case 2: asm_work=&asm_copy_mov_clflush_2; break;
          case 3: asm_work=&asm_copy_mov_clflush_3; break;
          case 4: asm_work=&asm_copy_mov_clflush_4; break; 
          default: break;
        }
        break;
      case METHOD_SCALE:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_scale_mov_clflush_1; break;
          case 2: asm_work=&asm_scale_mov_clflush_2; break;
          case 3: asm_work=&asm_scale_mov_clflush_3; break;
          case 4: asm_work=&asm_scale_mov_clflush_4; break; 
          default: break;
        }
        break;
      default: break;
     }
     break;
   case USE_MOVUPD:
     switch (data->method)
     {
      case METHOD_INDEP:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_indep_movupd_1; break;
          case 2: asm_work=&asm_indep_movupd_2; break;
          case 3: asm_work=&asm_indep_movupd_3; break;
          case 4: asm_work=&asm_indep_movupd_4; break;
          case 8: asm_work=&asm_indep_movupd_8; break;
          default: break;
        }
        break;
      case METHOD_COPY: 
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_copy_movupd_1; break;
          case 2: asm_work=&asm_copy_movupd_2; break;
          case 3: asm_work=&asm_copy_movupd_3; break;
          case 4: asm_work=&asm_copy_movupd_4; break;
          case 8: asm_work=&asm_copy_movupd_8; break;
          default: break;
        }
        break;
      case METHOD_SCALE:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_scale_movupd_1; break;
          case 2: asm_work=&asm_scale_movupd_2; break;
          case 3: asm_work=&asm_scale_movupd_3; break;
          case 4: asm_work=&asm_scale_movupd_4; break;
          case 8: asm_work=&asm_scale_movupd_8; break;
          default: break;
        }
        break;
      default: break;
     }
     break;
   case USE_MOVAPD: 
     switch (data->method)
     {
      case METHOD_INDEP:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_indep_movapd_1; break;
          case 2: asm_work=&asm_indep_movapd_2; break;
          case 3: asm_work=&asm_indep_movapd_3; break;
          case 4: asm_work=&asm_indep_movapd_4; break;
          case 8: asm_work=&asm_indep_movapd_8; break;
          default: break;
        }
        break;
      case METHOD_COPY: 
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_copy_movapd_1; break;
          case 2: asm_work=&asm_copy_movapd_2; break;
          case 3: asm_work=&asm_copy_movapd_3; break;
          case 4: asm_work=&asm_copy_movapd_4; break;
          case 8: asm_work=&asm_copy_movapd_8; break;
          default: break;
        }
        break;
      case METHOD_SCALE:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_scale_movapd_1; break;
          case 2: asm_work=&asm_scale_movapd_2; break;
          case 3: asm_work=&asm_scale_movapd_3; break;
          case 4: asm_work=&asm_scale_movapd_4; break;
          case 8: asm_work=&asm_scale_movapd_8; break;
          default: break;
        }
        break;
      default: break;
     }
     break;
   case USE_MOVNTPD:
     switch (data->method)
     {
      case METHOD_INDEP:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_indep_movntpd_1; break;
          case 2: asm_work=&asm_indep_movntpd_2; break;
          case 3: asm_work=&asm_indep_movntpd_3; break;
          case 4: asm_work=&asm_indep_movntpd_4; break;
          case 8: asm_work=&asm_indep_movntpd_8; break;
          default: break;
        }
        break;
      case METHOD_COPY: 
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_copy_movntpd_1; break;
          case 2: asm_work=&asm_copy_movntpd_2; break;
          case 3: asm_work=&asm_copy_movntpd_3; break;
          case 4: asm_work=&asm_copy_movntpd_4; break;
          case 8: asm_work=&asm_copy_movntpd_8; break;
          default: break;
        }
        break;
      case METHOD_SCALE:
        switch (data->burst_length) 
        {
          case 1: asm_work=&asm_scale_movntpd_1; break;
          case 2: asm_work=&asm_scale_movntpd_2; break;
          case 3: asm_work=&asm_scale_movntpd_3; break;
          case 4: asm_work=&asm_scale_movntpd_4; break;
          case 8: asm_work=&asm_scale_movntpd_8; break;
          default: break;
        }
        break;
      default: break;
     }
     break;
   default: break;
  }

  /* MEMORY LAYOUT
   * - aligned_addr points to the middle of the buffer
   * - an area in the middle of the buffer is used for the param_t structure, that contains all information that is
   *   needeed duringthe measurement ( PARAMS is defined as ((param_t*)aligned_addr) )
   * - the read buffer is mapped to the area before the parameter structure
   * - the write buffer starts behind the parameters structure
   * - parameter structure and write buffer are always contiguous memory 
   *   - a fixed read buffer (BENCHIT_KERNEL_LAYOUT="F") is not placed cache index aware with respect to the write buffer
   *     thus interference is likely (requires at least 2-way set associative cache)
   *   - contiguous memory can be enforced (BENCHIT_KERNEL_LAYOUT="C") to reduce interference, in this case the read buffer
   *     grows backwards starting from the parameter structure , thus read buffer, parameter structure, and write buffer
   *     are contiguous memory (with some gaps to avoid unwanted prefetching) that would even fit a direct mapped cache.
   *     However, this method is likely to cause bank conflicts 
   *   - BENCHIT_KERNEL_LAYOUT="A{1|2}" combine both approaches to find optimum */
  aligned_addr=(unsigned long long)(data->buffer) + data->buffersize/2 + data->offset; /* used as pointer to parameter structure (PARAMS) */

	/* perform measurement for each selected core (BENCHIT_KERNEL_CPU_LIST) */
  /* thread_id: currently selected CPU
   * thread_id = 0: bandwidth of local access on first selected CPU
   *                - thread on first selected CPU accesses data
   *                - afterwards the same thread measures bandwidth for accesses to this data
   * thread_id > 0: bandwidth of remote accesses
   *                - thread on another CPU accesses data to bring it in its caches
   *                - afterwards thread on first selected CPU measures bandwidth for that data
   *                - (BENCHIT_KERNEL_READ_LOCAL/BENCHIT_KERNEL_WRITE_LOCAL) can be used to keep
   *                  one of the streams local in all cases*/
  for (PARAMS->thread_id=0;PARAMS->thread_id<data->num_threads;PARAMS->thread_id++)
  {
    /* USE MODE ADAPTION (for BENCHIT_KERNEL_*_USE_MODE={S|F|O})
     * enforcing data to be in one of the shared coherency states (SHARED/OWNED/FORWARD), is implemented by adapting the target state for
     * individual accesses (a specific core (BENCHIT_KERNEL_SHARE_CPU) is used to share cachelines with the currently selected CPU (thread_id))
     * Forward: - Thread on SHARE_CPU accesses data with use mode EXCLUSIVE
     *          - Thread on selected CPU accesses data with use mode FORWARD (read only)
     *          Note: Forward seems to be a per-package state
     *                - Cores will have the line in shared state
     *                - L3 will have it in shared state (and 2 core valid bits set) if both cores share a package (die)
     *                - only if cores are in different packacges (dies), one L3 (last accessing core determines which one) will mark the line with state Forward
     *          Note: only usefull if coherency protocol is MESIF !!!
     * Shared:  - Thread on selected CPU accesses data with use mode EXCLUSIVE
     *          - Thread on SHARE_CPU accesses data with use mode SHARED (read only)
     *          Note: works on MESIF and non-MESIF protocols (copy on SHARE_CPU will be in Forward state for MESIF, thus SHRAE_CPU should be as far away from first CPU as posible)
     * Owned:   - Thread on selected CPU accesses data with use mode MODIFIED
     *          - Thread on SHARE_CPU accesses data with use mode SHARED (read only)
     *          Note: only works if coherency protocol is MOESI (otherwise both lines will be in shared state)
     */
    
   /* set pointer to parameter structure of currently selected CPU */
   if (PARAMS->thread_id) THREAD_PARAMS=(param_t*) (data->threaddata[PARAMS->thread_id].aligned_addr + data->buffersize/2 + data->offset);
   else THREAD_PARAMS=NULL;
   /* set pointer to parameter structure of SHARE_CPU if required*/
   if ((data->USE_MODE_R&(MODE_SHARED|MODE_OWNED|MODE_FORWARD))||(data->USE_MODE_W&(MODE_SHARED|MODE_OWNED|MODE_FORWARD)))
     SHARE_CPU_PARAMS=(param_t*) (data->threaddata[data->SHARE_CPU].aligned_addr + data->buffersize/2 + data->offset);
   else SHARE_CPU_PARAMS=NULL;

   /* setup all information needed during measurement into the param_t structure */
   PARAMS->runs=data->runs;
   switch(data->burst_length) /* all but burst_length=3 perform 32 accesses per loop within the measurement routine, (burst_length_3 requires a multiple of 6) */
   {
     case 3: PARAMS->accesses_per_loop=36;break;
     default: PARAMS->accesses_per_loop=32;break;
   }
   switch (data->function)  /* setup data type size - determines the shift of the write buffer in each run */
   {
     case USE_MOV:
     case USE_MOVNTI:
       PARAMS->alignment=8;  /* 64 Bit operations*/
       break;
     default: 
       PARAMS->alignment=16; /* 128 Bit operations*/
       break;
   }
   PARAMS->read_local=data->read_local;
   PARAMS->write_local=data->write_local;
   PARAMS->passes=memsize/(PARAMS->alignment*PARAMS->accesses_per_loop);
   PARAMS->memsize=PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment; 
   if (data->layout == LAYOUT_CONT) PARAMS->addr_1=aligned_addr-((PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment)/2); 
   else if (data->layout == LAYOUT_FIXED) PARAMS->addr_1=aligned_addr-data->buffersize/2; 
   PARAMS->addr_2=aligned_addr+((sizeof(param_t)+data->alignment)/data->alignment)*data->alignment;
   PARAMS->layout=data->layout;
   PARAMS->use_direction=data->USE_DIRECTION;
   PARAMS->num_uses=data->NUM_USES;
   PARAMS->default_use_mode_1=data->USE_MODE_R;
   PARAMS->default_use_mode_2=data->USE_MODE_W;
   #ifdef USE_PAPI
   PARAMS->Eventset=data->Eventset;
   PARAMS->num_events=data->num_events;
   PARAMS->values=data->values;
   #endif
   if (data->method==METHOD_SCALE) PARAMS->factor=data->factor;
   PARAMS->value=data->init_value;
   /* configure cache flushes */
   PARAMS->flush_mode=data->FLUSH_MODE;
   PARAMS->num_flushes=data->NUM_FLUSHES;
   PARAMS->flushaddr=(void*)(data->cache_flush_area);
   PARAMS->flushsize=0;
   if (!strcmp(data->cpuinfo->vendor,"AuthenticAMD")) /* exclusive caches */
   {
     for (PARAMS->i=data->cpuinfo->Cachelevels;PARAMS->i>0;PARAMS->i--)
     {   
       if (data->settings&FLUSH(PARAMS->i))
       {
         PARAMS->flushsize=0;
         /* flushsize = sum of all cache levels */
         for (PARAMS->j=PARAMS->i;PARAMS->j>0;PARAMS->j--) PARAMS->flushsize+=data->cpuinfo->U_Cache_Size[PARAMS->j-1]+data->cpuinfo->D_Cache_Size[PARAMS->j-1];
         if(memsize<=PARAMS->flushsize) PARAMS->flushsize = 0;
       }
     }
   }
   if (!strcmp(data->cpuinfo->vendor,"GenuineIntel")) /* inclusive caches */
   {
     for (PARAMS->i=data->cpuinfo->Cachelevels;PARAMS->i>0;PARAMS->i--)
     {   
       if (data->settings&FLUSH(PARAMS->i))
       {
         /* flushsize = size of selected level */
         PARAMS->flushsize==data->cpuinfo->U_Cache_Size[PARAMS->i-1]+data->cpuinfo->D_Cache_Size[PARAMS->i-1];;
         if(memsize<=PARAMS->flushsize) PARAMS->flushsize = 0;
       }
     }
   } 
   /* increase size to increase cache preasure */
   PARAMS->flushsize*=12;
   PARAMS->flushsize/=10;   

   /* setup parameter structure of selected CPU */
   if (PARAMS->thread_id) {
     memcpy(THREAD_PARAMS,PARAMS,sizeof(param_t));
     /* setup pointers to thread private streams */
     if (data->layout == LAYOUT_CONT) THREAD_PARAMS->addr_1=(data->threaddata[PARAMS->thread_id].aligned_addr+ data->buffersize/2 + data->offset)-((PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment)/2); 
     else if (data->layout == LAYOUT_FIXED) THREAD_PARAMS->addr_1=data->threaddata[PARAMS->thread_id].aligned_addr+data->offset;
     THREAD_PARAMS->addr_2=(data->threaddata[PARAMS->thread_id].aligned_addr+ data->buffersize/2 + data->offset)+((sizeof(param_t)+data->alignment)/data->alignment)*data->alignment;
     THREAD_PARAMS->flushaddr=(void*)(data->threaddata[PARAMS->thread_id].cache_flush_area);  
   }
   
   /* setup parameter structure of SHARE_CPU */
   if (SHARE_CPU_PARAMS!=NULL) {
     memcpy(SHARE_CPU_PARAMS,PARAMS,sizeof(param_t));
     SHARE_CPU_PARAMS->thread_id=data->SHARE_CPU;
     SHARE_CPU_PARAMS->flushaddr=(void*)(data->threaddata[SHARE_CPU_PARAMS->thread_id].cache_flush_area); 
     /* addr_1 and addr_2 are set to the selected streams within the loop */
   }   

   /* clflush everything that is not needed any more
    * - can help to reduce L1 anomalies on AMD K8/K10 if memsize is greater than half of the 2-way set associative L1
    * - however, clflushes create TLB entries for the unused area, thus might result in performance reduction by evicting usefull TLB entries
    * TODO add option to deactivate this
    */
   __asm__ __volatile__("mfence;"::);
   __asm__ __volatile__("clflush (%%rax);":: "a" ((unsigned long long)&memsize));
   for(PARAMS->i = sizeof(cpu_info_t)/64;PARAMS->i>=0;PARAMS->i--)
   {
      __asm__ __volatile__("clflush (%%rax);":: "a" (((unsigned long long)(data->cpuinfo))+64*PARAMS->i));
   }
   for(PARAMS->i = (sizeof(threaddata_t)*data->num_threads)/64;PARAMS->i>=0;PARAMS->i--)
   {
      __asm__ __volatile__("clflush (%%rax);":: "a" (((unsigned long long)(data->threaddata))+64*PARAMS->i));
   }
   for(PARAMS->i = sizeof(mydata_t)/64;PARAMS->i>=0;PARAMS->i--)
   {
      __asm__ __volatile__("clflush (%%rax);":: "a" (((unsigned long long)data)+64*PARAMS->i));
   }
   __asm__ __volatile__("clflush (%%rax);":: "a" ((unsigned long long)&data));
   __asm__ __volatile__("mfence;"::);
  
  
   /* perform measurement only if memsize is large enough for at least one iteration of the measurement loop */
   if (PARAMS->passes) 
   {
    PARAMS->tmax=0x7fffffffffffffff;
    /* perform the selected number of runs (BENCHIT_KERNEL_RUNS) */
    for (PARAMS->iter=0;PARAMS->iter<PARAMS->runs;PARAMS->iter++)
    {
      /* rotate memory layout each run */
      if (PARAMS->layout == LAYOUT_ALT1) 
      {
       if (PARAMS->iter%2 == 1){
         PARAMS->addr_1=aligned_addr-((PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment)/2);
         if (PARAMS->thread_id) THREAD_PARAMS->addr_1=(data->threaddata[PARAMS->thread_id].aligned_addr+ data->buffersize/2 + data->offset)-((PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment)/2);
       }
       else { 
         PARAMS->addr_1=aligned_addr-data->buffersize/2;
         if (PARAMS->thread_id) THREAD_PARAMS->addr_1=data->threaddata[PARAMS->thread_id].aligned_addr+data->offset;
       }
      }
      /* switch layout after half of the runns */
      if (PARAMS->layout == LAYOUT_ALT2) 
      {
       if (PARAMS->iter>=PARAMS->runs/2) {
         PARAMS->addr_1=aligned_addr-((PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment)/2);
         if (PARAMS->thread_id) THREAD_PARAMS->addr_1=(data->threaddata[PARAMS->thread_id].aligned_addr+ data->buffersize/2 + data->offset)-((PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment)/2);
       }
       else {
         PARAMS->addr_1=aligned_addr-data->buffersize/2;
         if (PARAMS->thread_id) THREAD_PARAMS->addr_1=data->threaddata[PARAMS->thread_id].aligned_addr+data->offset;
       }
      }
     
     /* slightly shift write buffer each run to reduce bank conflicts 
      * - deactivated for contiguous memory as this is supposed to show bank conflicts */   
     if (PARAMS->layout!=LAYOUT_CONT){
       PARAMS->addr_2+=PARAMS->alignment;
       if (PARAMS->thread_id) THREAD_PARAMS->addr_2+=PARAMS->alignment;
     }
    
     /* access remote memory to warm up TLB before other threads use the data 
      * cached data will be invalidated by other threads' accesses */
     if (PARAMS->thread_id>0)
     {
       if (!(PARAMS->read_local)) PARAMS->addr_1=THREAD_PARAMS->addr_1;
       if (!(PARAMS->write_local)) PARAMS->addr_2=THREAD_PARAMS->addr_2;
       /* use modified as this is the fastest option */
       PARAMS->use_mode_1=MODE_MODIFIED;
       PARAMS->use_mode_2=MODE_MODIFIED;
       /* local streams will be touched later on so they can be skiped here*/
       if ((PARAMS->read_local)) PARAMS->use_mode_1=MODE_DISABLED;
       if ((PARAMS->write_local)) PARAMS->use_mode_2=MODE_DISABLED;
       use_memory(PARAMS);
     }

     /* copy pointers to selected streams to SHARE_CPU parameters */
     if (SHARE_CPU_PARAMS!=NULL) {
       SHARE_CPU_PARAMS->addr_1=PARAMS->addr_1;
       SHARE_CPU_PARAMS->addr_2=PARAMS->addr_2;
     }        

      /* thread on SHARE_CPU acceses selected streams with target use mode forward */
      if (SHARE_CPU_PARAMS!=NULL)
       {
        /* see USE MODE ADAPTION */
        if (SHARE_CPU_PARAMS->default_use_mode_1==MODE_FORWARD) SHARE_CPU_PARAMS->use_mode_1=MODE_EXCLUSIVE;
        else SHARE_CPU_PARAMS->use_mode_1=MODE_DISABLED;
        if (SHARE_CPU_PARAMS->default_use_mode_2==MODE_FORWARD) SHARE_CPU_PARAMS->use_mode_2=MODE_EXCLUSIVE;
        else SHARE_CPU_PARAMS->use_mode_2=MODE_DISABLED;
        //TODO remove accesses to mydata_t (data) to reduce cache footprint
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[SHARE_CPU_PARAMS->thread_id]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[SHARE_CPU_PARAMS->thread_id]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 1\n");
        data->ack=0;
        while (!data->done); //printf("wait for done 1\n");
        data->done=0;
      }
      
      /* access local streams if running locally (thread_id is 0) or if streams are selected to be
       * allways accessed locally (BENCHIT_KERNEL_READ_LOCAL/BENCHIT_KERNEL_WRITE_LOCAL) */
      if (PARAMS->thread_id==0) 
      {
        /* see USE MODE ADAPTION */
        if (PARAMS->default_use_mode_1==MODE_SHARED) PARAMS->use_mode_1=MODE_EXCLUSIVE;
        else if (PARAMS->default_use_mode_1==MODE_OWNED) PARAMS->use_mode_1=MODE_MODIFIED;
        else PARAMS->use_mode_1=PARAMS->default_use_mode_1;
        if (PARAMS->default_use_mode_2==MODE_SHARED) PARAMS->use_mode_2=MODE_EXCLUSIVE;
        else if (PARAMS->default_use_mode_2==MODE_OWNED) PARAMS->use_mode_2=MODE_MODIFIED;
        else PARAMS->use_mode_2=PARAMS->default_use_mode_2;                
        use_memory(PARAMS);
      }
      else if ((PARAMS->read_local)||(PARAMS->write_local))
      {        
        if (PARAMS->read_local)
        {
          /* see USE MODE ADAPTION */
          if (PARAMS->default_use_mode_1==MODE_SHARED) PARAMS->use_mode_1=MODE_EXCLUSIVE;
          else if (PARAMS->default_use_mode_1==MODE_OWNED) PARAMS->use_mode_1=MODE_MODIFIED;
          else PARAMS->use_mode_1=PARAMS->default_use_mode_1;
        }
        else PARAMS->use_mode_1=MODE_DISABLED;
        if (PARAMS->read_local)
        {
          /* see USE MODE ADAPTION */
          if (PARAMS->default_use_mode_2==MODE_SHARED) PARAMS->use_mode_2=MODE_EXCLUSIVE;
          else if (PARAMS->default_use_mode_2==MODE_OWNED) PARAMS->use_mode_2=MODE_MODIFIED;
          else PARAMS->use_mode_2=PARAMS->default_use_mode_2; 
        }
        else PARAMS->use_mode_2=MODE_DISABLED;
        use_memory(PARAMS);
      }

      /* thread on selected CPU accesses remote streams */
      if (PARAMS->thread_id>0)
      {
        /* see USE MODE ADAPTION */
        if (THREAD_PARAMS->default_use_mode_1==MODE_SHARED) THREAD_PARAMS->use_mode_1=MODE_EXCLUSIVE;
        else if (THREAD_PARAMS->default_use_mode_1==MODE_OWNED) THREAD_PARAMS->use_mode_1=MODE_MODIFIED;
        else THREAD_PARAMS->use_mode_1=THREAD_PARAMS->default_use_mode_1;
        if (THREAD_PARAMS->default_use_mode_2==MODE_SHARED) THREAD_PARAMS->use_mode_2=MODE_EXCLUSIVE;
        else if (THREAD_PARAMS->default_use_mode_2==MODE_OWNED) THREAD_PARAMS->use_mode_2=MODE_MODIFIED;
        else THREAD_PARAMS->use_mode_2=THREAD_PARAMS->default_use_mode_2;
        /* disable unused remote streams to reduce cache prasure */
        if (PARAMS->read_local) THREAD_PARAMS->use_mode_1=MODE_DISABLED;
        if (PARAMS->write_local) THREAD_PARAMS->use_mode_2=MODE_DISABLED;
        //TODO remove accesses to mydata_t (data) to reduce cache footprint
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[PARAMS->thread_id]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[PARAMS->thread_id]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 2\n");
        data->ack=0;
        while (!data->done); //printf("wait for done 2\n");
        data->done=0;    
      }
     
      /* flush cachelevels as specified in PARAMETERS */
      if (PARAMS->flushsize) flush_caches(PARAMS);

      /* thread on SHARE_CPU acceses selected streams with target use mode owned or shared */
      if (SHARE_CPU_PARAMS!=NULL)
       {
        /* see USE MODE ADAPTION */
        if (SHARE_CPU_PARAMS->default_use_mode_1&(MODE_SHARED|MODE_OWNED)) SHARE_CPU_PARAMS->use_mode_1=SHARE_CPU_PARAMS->default_use_mode_1;
        else SHARE_CPU_PARAMS->use_mode_1=MODE_DISABLED;
        if (SHARE_CPU_PARAMS->default_use_mode_2&(MODE_SHARED|MODE_OWNED)) SHARE_CPU_PARAMS->use_mode_2=SHARE_CPU_PARAMS->default_use_mode_2;
        else SHARE_CPU_PARAMS->use_mode_2=MODE_DISABLED;
        //TODO remove accesses to mydata_t (data) to reduce cache footprint
        __asm__ __volatile__("mfence;"::);
        data->thread_comm[SHARE_CPU_PARAMS->thread_id]=THREAD_USE_MEMORY;
        while (!data->ack);
        data->ack=0;
        data->thread_comm[SHARE_CPU_PARAMS->thread_id]=THREAD_WAIT;    
        //wait for other thread using the memory
        while (!data->ack); //printf("wait for ack 3\n");
        data->ack=0;
        while (!data->done); //printf("wait for done 3\n");
        data->done=0;
      }
           
      /* call ASM implementation */
      asm_work(PARAMS);
      
      if (PARAMS->rax<PARAMS->tmax) PARAMS->tmax=PARAMS->rax;
    }
    /* disabled, not needed as RUN_LINEAR is enforced in PARAMETERS file
     * // clear caches for next run with different (probably smaller) memsize
     * params->use_mode_1=MODE_INVALID;
     * params->use_mode_2=MODE_INVALID;
     * use_memory(params);
     */

   }
   else PARAMS->tmax=0;
  
   if (PARAMS->tmax>0) (*results)[PARAMS->thread_id+1]=(((double)(PARAMS->passes*PARAMS->accesses_per_loop*PARAMS->alignment))/(((double)(PARAMS->tmax-data->cpuinfo->rdtsc_latency)/data->cpuinfo->clockrate)))/1000000000;
   else (*results)[PARAMS->thread_id+1]=INVALID_MEASUREMENT;
   
   #ifdef USE_PAPI
    for (PARAMS->i=0;PARAMS->i<data->num_events;PARAMS->i++)
    {
       data->papi_results[PARAMS->i]=(double)data->values[PARAMS->i]/(double)(PARAMS->passes*PARAMS->accesses_per_loop);
    }
   #endif
  }
}
