/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: simple.c 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/sort/heapsort/C/0/0/double/simple.c $
 * For license details see COPYING in the package base directory
 *******************************************************************/

#include "simple.h"
#include "interface.h"

double simple(myinttype * pi_prob_size)
{
  double dresult = 1.0;
  myinttype ii = 0, pre = 0, prepre = 0;
  

  switch (* pi_prob_size)
  {
    case 0:
            break;
    case 1:
            break;
    default:
            pre = *pi_prob_size - 1;
            prepre = pre - 1;
            dresult = (double) (simple(&pre) + simple(&prepre));
  }

              
/*  for (ii=*pi_prob_size; ii>0; ii--)
  {
    dresult = dresult * ii;
    dresult = sqrt(dresult);
  }
*/
  return dresult;
}


void heapsortd(double *pfsort, long lnumber)
   {
   int  ii, ij, ik;
   double fh;

   /*initialize variables*/
   ii = 0;
   ik = 0;
   fh = 0;
   ij = 0;

   /*creating heap*/
   for (ii = lnumber >> 1; ii > 0; ii--)
      {
      /*all nodes beginning in the 2nd level from below 
        are pushed down to*/
      /*a lower level if necessary*/
      fh = pfsort[ii];
      ik = ii;
      /*as long as there is a lower level*/
      while (ik <= lnumber >> 1)
         {
         /*find out which the bigger one of the two children*/
         ij = ik + ik;
         if (ij < lnumber)
            {
            if (pfsort[ij] < pfsort[ij + 1])
               {
               ij = ij + 1;
               }
            }
         /*if the childs are smaller -> ok break up*/
         if (fh >= pfsort[ij])
            {
            break;
            }
         /*if not -> write the bigger child to
           the place of the parent*/
         pfsort[ik] = pfsort[ij];
         /*and continue with the level below*/
         ik = ij;
         }
      /*write the examined number to the correct place*/
      pfsort[ik] = fh;
      }
   /*dismantle heap*/
   while (lnumber > 0)
      {
      /*the biggest element is always the root element
        -> remove root element -> change it with the last element*/
      fh = pfsort[1];
      pfsort[1] = pfsort[lnumber];
      pfsort[lnumber] = fh;
      /*heap is smaller now (root element was removed)*/
      lnumber--;
      /*now recreate the heap by moving the element that was changed
        with the root element (->element on place 1)
        to the correct place */
      ik = 1;
      fh = pfsort[ik];
      /*as long as there is a lower level */
      while (ik <= lnumber >> 1)
         {
         /*find out which the bigger one of the two children */
         ij = ik + ik;
         if (ij < lnumber)
            if (pfsort[ij] < pfsort[ij + 1])
               ij = ij + 1;
         /*if the childs are smaller -> ok break up */
         if (fh >= pfsort[ij])
            break;
         /*if not -> write the bigger child
           to the place of the parent*/
         pfsort[ik] = pfsort[ij];
         /*and continue with the level below */
         ik = ij;
         }
      /*write the examined number to the correct place */
      pfsort[ik] = fh;
      }
   }


int verifyd(double *pfprobe, long lelements)
   {
   int ii;

   /*initialize variables*/
   ii = 0;

   /*any element on position n+1 has to be larger or equal to element on
     position n...*/
   for (ii = 2; ii < lelements + 1; ii++)
      {
      if (pfprobe[ii - 1] > pfprobe[ii])
         {
         return 0;
         }
      }

   /*"1" means success */
   return 1;
   }

