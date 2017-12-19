/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: MatMul.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/Java/0/0/double/MatMul.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix multiply (Java)
 *******************************************************************/

/** This class definies the measured algorithm.
  * @author Robert Wloch, wloch@zhr.tu-dresden.de
  */
public class MatMul {
   private DataObject data;
   public MatMul( DataObject data ) {
      this.data = data;
   }
   public void testCall() {
      int i, j, k;
      int s = data.maxn;
   }
   void init() {
      int i, j;
      int s = data.maxn;
      for( i = 0; i < s; i++) {
         for( j = 0; j < s; j++) {
            int index = i * s + j;
            data.c[index] = 0.0;
            data.a[index] = (double)index;
            data.b[index] = (double)j;
         }
      }
   }
   void matmul_ijk() {
      int i, j, k;
      int s = data.maxn;
      for( i = 0; i < s; i++) {
         for( j = 0; j < s; j++) {
            for( k = 0; k < s; k++) {
               int index = j * s;
               data.c[index + i] += data.a[k * s + i] * data.b[index + k];
            }
         }
      }
   }
   void matmul_ikj() {
      int i, j, k;
      int s = data.maxn;
      for( i = 0; i < s; i++) {
         for( k = 0; k < s; k++) {
            for( j = 0; j < s; j++) {
               int index = j * s;
               data.c[index + i] += data.a[k * s + i] * data.b[index + k];
            }
         }
      }
   }
   void matmul_kij() {
      int i, j, k;
      int s = data.maxn;
      for( k = 0; k < s; k++) {
         for( i = 0; i < s; i++) {
            for( j = 0; j < s; j++) {
               int index = j * s;
               data.c[index + i] += data.a[k * s + i] * data.b[index + k];
            }
         }
      }
   }
   void matmul_kji() {
      int i, j, k;
      int s = data.maxn;
      for( k = 0; k < s; k++) {
         for( j = 0; j < s; j++) {
            for( i = 0; i < s; i++) {
               int index = j * s;
               data.c[index + i] += data.a[k * s + i] * data.b[index + k];
            }
         }
      }
   }
   void matmul_jik() {
      int i, j, k;
      int s = data.maxn;
      for( j = 0; j < s; j++) {
         for( i = 0; i < s; i++) {
            for( k = 0; k < s; k++) {
               int index = j * s;
               data.c[index + i] += data.a[k * s + i] * data.b[index + k];
            }
         }
      }
   }
   void matmul_jki() {
      int i, j, k;
      int s = data.maxn;
      for( j = 0; j < s; j++) {
         for( k = 0; k < s; k++) {
            for( i = 0; i < s; i++) {
               int index = j * s;
               data.c[index + i] += data.a[k * s + i] * data.b[index + k];
            }
         }
      }
   }
}


