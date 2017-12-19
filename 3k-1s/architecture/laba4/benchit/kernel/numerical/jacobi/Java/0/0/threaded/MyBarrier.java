/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: MyBarrier.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/Java/0/0/threaded/MyBarrier.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         number of threads or for change of dimension
 *******************************************************************/

/* This class is the barrier needed to synchronize the calculating threads. */
public class MyBarrier {
   private int maxn, c1, c2, c3;

   public MyBarrier(int maxn) {
      this.maxn = maxn;
      this.c1 = 0;
      this.c2 = 0;
      this.c3 = 0;
   }

   public synchronized void runIntoBarrier1() {
      c1 += 1;
      if (c1 == maxn) {
         c1 = 0;
         notifyAll();
      }
      while (c1 > 0) {
         try {
            wait();
         }
         catch (InterruptedException ie) {
         }
      }
   }

   public synchronized void runIntoBarrier2() {
      c2 += 1;
      if (c2 == maxn) {
         c2 = 0;
         notifyAll();
      }
      while (c2 > 0) {
         try {
            wait();
         }
         catch (InterruptedException ie) {
         }
      }
   }

   public synchronized void runIntoBarrier3() {
      c3 += 1;
      if (c3 == maxn) {
         c3 = 0;
         notifyAll();
      }
      while (c3 > 0) {
         try {
            wait();
         }
         catch (InterruptedException ie) {
         }
      }
   }
}

