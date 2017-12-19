/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: soccer.h 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/applications/reflection/C/0/0/soccergoal/soccer.h $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: compare the possibility to hit the goal when the pole have a squared or round form
 *******************************************************************/

#ifndef __work_h
#define __work_h

#if (defined (_CRAYMPP) || \
     defined (_USE_SHORT_AS_INT))
typedef short myinttype;
#elif (defined (_USE_LONG_AS_INT))
typedef long myinttype;
#else
typedef int myinttype;
#endif

typedef struct point {
   double x;
   double y;
} point_t;

/* x = s + a * t */
typedef struct line {
   double sx;
   double sy;
   double tx;
   double ty;
} line_t;

extern double soccergoal(myinttype *, myinttype *);

#endif


