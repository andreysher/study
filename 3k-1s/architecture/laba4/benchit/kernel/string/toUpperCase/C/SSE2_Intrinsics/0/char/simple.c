/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: simple.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/string/toUpperCase/C/SSE2_Intrinsics/0/char/simple.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: SSE String Operations
 *******************************************************************/

#include "simple.h"
#include <emmintrin.h>
 
myinttype toUpperCaseSSE(char * field, myinttype size)
{
	myinttype i=0;
	__m128i load_field;
	__m128i a_field;
	__m128i z_field;
	__m128i a_to_A_difference;
	__m128i lt_mask;
	__m128i gt_mask;
	__m128i change_mask;
	__m128i sub;
	char a_to_A='a'-'A';
	a_field=_mm_set1_epi8('a'-1);
	z_field=_mm_set1_epi8('z'+1);
	a_to_A_difference=_mm_set1_epi8(a_to_A);
	long long result[2]={0,0};
	// goto alignment :P
	unsigned long long adress=(unsigned long long)field;
	int before=(16-adress%16)%16;
	for (i=0;i<before;i++)
		if ((field[i]>='a')&&(field[i]<='z'))
			field[i]-=a_to_A;
	for (i=before;i<size;i=i+16)
	{
		load_field=_mm_load_si128(&field[i]);
		// get mask for chars to change
		// must be greater equal then 'a' , greater then 'a'-1
		gt_mask = _mm_cmpgt_epi8(load_field,a_field);
		// must be smaller equal then 'z' , smaller then 'z'-1
		lt_mask = _mm_cmplt_epi8(load_field,z_field);
		// both has to fit
		change_mask=_mm_and_si128(gt_mask,lt_mask);
		// build values to add
		sub=_mm_and_si128(change_mask,a_to_A_difference);
		load_field=_mm_subs_epi8(load_field,sub);
		_mm_store_si128(&field[i],load_field);
	}
	i=i-16;
	for (i;i<size;i++)
		if ((field[i]>='a')&&(field[i]<='z'))
			field[i]-=a_to_A;
  return 1;
}
myinttype toUpperCaseSSEunaligned(char * field, myinttype size)
{
	myinttype i=0;
	__m128i load_field;
	__m128i a_field;
	__m128i z_field;
	__m128i a_to_A_difference;
	__m128i lt_mask;
	__m128i gt_mask;
	__m128i change_mask;
	__m128i sub;
	char a_to_A='a'-'A';
	a_field=_mm_set1_epi8('a'-1);
	z_field=_mm_set1_epi8('z'+1);
	a_to_A_difference=_mm_set1_epi8(a_to_A);
	long long result[2]={0,0};
	for (i=0;i<size;i=i+16)
	{
		load_field=_mm_loadu_si128(&field[i]);
		// get mask for chars to change
		// must be greater equal then 'a' , greater then 'a'-1
		gt_mask = _mm_cmpgt_epi8(load_field,a_field);
		// must be smaller equal then 'z' , smaller then 'z'-1
		lt_mask = _mm_cmplt_epi8(load_field,z_field);
		// both has to fit
		change_mask=_mm_and_si128(gt_mask,lt_mask);
		// build values to add
		sub=_mm_and_si128(change_mask,a_to_A_difference);
		load_field=_mm_subs_epi8(load_field,sub);
		_mm_storeu_si128(&field[i],load_field);
	}
	i=i-16;
	for (i;i<size;i++)
		if ((field[i]>='a')&&(field[i]<='z'))
			field[i]-=a_to_A;
  return 1;
}

myinttype toUpperCase(char * field, myinttype size)
{
	myinttype i=0;
	char a_to_A='a'-'A';
	for (i=0;i<size;i++)
		if ((field[i]>='a')&&(field[i]<='z'))
			field[i]-=a_to_A;
  return 1;
}

