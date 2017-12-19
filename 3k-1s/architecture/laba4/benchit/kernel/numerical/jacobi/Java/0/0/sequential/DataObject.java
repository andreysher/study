/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: DataObject.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/Java/0/0/sequential/DataObject.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension
 *******************************************************************/

/* This class handles all the data that will be processed by the
 * measured algorithm. */
public class DataObject {
   public int maxn = -1;
   public int maxIterations = -1;
   public int mitsdone = -1;
   public double h = -1.0;
   public double diffnorm = 0.0;
   // data to be processed
   public double[] a = null;
   public double[] b = null;
   public double[] f = null;
   
   /* constructor */
   public DataObject() {
   }
}

