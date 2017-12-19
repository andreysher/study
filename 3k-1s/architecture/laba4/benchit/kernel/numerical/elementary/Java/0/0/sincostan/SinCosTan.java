/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: SinCosTan.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/elementary/Java/0/0/sincostan/SinCosTan.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Execution time of mathematical operations
 *         SINUS / COSINUS / TANGENT addict to input value
 *******************************************************************/

public class SinCosTan {
   double temp;

   protected double sin(double d, long p) {
      temp = d;
      for (int i=0; i<p; i++) {
         temp = Math.sin(temp);
      }
      return temp;
   }

   protected double cos(double d, long p) {
      temp = d;
      for (int i=0; i<p; i++) {
         temp = Math.cos(temp);
      }
      return temp;
   }

   protected double tan(double d, long p) {
      temp = d;
      for (int i=0; i<p; i++) {
         temp = Math.tan(temp);
      }
      return temp;
   }

   protected double getOverhead(double d, long p) {
      temp = 0;
      for (int i=0; i<p; i++) {
         temp += d;
      }
      return temp;
   }
}

