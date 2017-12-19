/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: KernelMain.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/Java/0/0/sequential/KernelMain.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension
 *******************************************************************/

import java.util.Date;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/* This class is the Interface between the measured algorithm and BenchIT. */
public class KernelMain implements BIJavaKernel {
   /* if it is static it has to be calculated just one time */
   private static double timeroverhead = 0.0;
   /* Here we define a minimum time that our kernel needs
    * we do this to avoid a divide by zero. */
   private static double MINTIME = 5.0e-6;

   private int RUNS = 2;               /* variations (ij and ji = 2) */
   private int NUM_FUNCTIONS = 2;      /* flop/s and time */

   /* attributes */
   private int MAXIMUM = 5;
   private BIEnvHash rwe = null;
   protected int bi_problemsize_min = 0, bi_problemsize_max = 0,
                 bi_problemsize_increment = 0, maxIterations = 0;

   /* The constructor. */
   public KernelMain() {
      rwe = BIEnvHash.getInstance();
   }

   /* Reads the environment variables used by this kernel. */
   private void evaluate_environment() {
      String s_bi_psize_min = null,  s_bi_psize_max = null,
             s_bi_psize_incr = null, s_jmits = null;
      int i_bi_psize_min = 0,  i_bi_psize_max = 0, i_bi_psize_incr = 0,
          i_jmits = 0;

      s_bi_psize_min = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_MIN");
      s_bi_psize_max = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_MAX");
      s_bi_psize_incr = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT");
      if ((s_bi_psize_min == null) || (s_bi_psize_max == null) || (s_bi_psize_incr == null)) {
         System.out.println("At least one of your kernel specific " + 
                            "environment variables is not set!" );
         System.exit(127);
      }
      try {
         i_bi_psize_min = (new Integer(s_bi_psize_min)).intValue();
         i_bi_psize_max = (new Integer(s_bi_psize_max)).intValue();
         i_bi_psize_incr = (new Integer(s_bi_psize_incr)).intValue();
      }
      catch ( NumberFormatException nfe ) {
         System.out.println( "Use numbers as environment variables!" );
         System.exit(127);
      }
      bi_problemsize_min = i_bi_psize_min;
      bi_problemsize_max = i_bi_psize_max;
      bi_problemsize_increment = i_bi_psize_incr;

      s_jmits = rwe.bi_getEnv("BENCHIT_KERNEL_JACOBI_JMITS");
      if ((s_jmits == null)) {
         System.out.println("At least one of your kernel specific " + 
                            "environment variables is not set!" );
         System.exit(127);
      }
      try {
         i_jmits = (new Integer(s_jmits)).intValue();
      }
      catch ( NumberFormatException nfe ) {
         System.out.println( "Use numbers as environment variables!" );
         System.exit(127);
      }
      maxIterations = i_jmits;
   }

   /* The implementation of the bi_getinfo from the BenchIT interface.
    * Here the info is filled with informations about the kernel.
    * @param info An Object to hold the info about the kernel. */
   public int bi_getinfo( InfoObject info ) {
      evaluate_environment();
      info.codesequence = "for (j = js; j < je; j++)#" +
                          " for (i = is; i < ie; i++) {#" +
                          "  index = j * MAXN + i;#" +
                          "  a[index] = 0.25 * (b[index - MAXN] + b[index - 1] + " + 
                          "b[index + MAXN] + b[index + 1]) - h * h * f[index];#" + 
                          "  ...#" +
                          "  b[index] = 0.25 * (a[index - MAXN] + a[index - 1] + " + 
                          "a[index + MAXN] + a[index + 1]) - h * h * f[index];#" + 
                          "  ...#" +
                          "  diff = a[index] - b[index];#" +
                          "  sum += diff * diff;#" +
                          " }";
      info.xaxistext = "X-Y-Dimension";
      info.log_xaxis = 0;
      info.base_xaxis = 0;
      info.maxproblemsize = (bi_problemsize_max - bi_problemsize_min + 1) /
                            bi_problemsize_increment;
      if ((bi_problemsize_max - bi_problemsize_min + 1) % bi_problemsize_increment != 0)
         info.maxproblemsize++;
      info.kernel_execs_mpi1 = 0;
      info.kernel_execs_mpi2 = 0;
      info.kernel_execs_pvm = 0;
      info.kernel_execs_omp = 0;
      info.kernel_execs_pthreads = 0;
      info.kernel_execs_javathreads = 0;
      info.numfunctions = RUNS * NUM_FUNCTIONS;

      /* allocating memory for y axis texts and properties */
      info.yaxistexts = new String[info.numfunctions];
      info.selected_result = new int[info.numfunctions];
      info.legendtexts = new String[info.numfunctions];
      info.log_yaxis = new int[info.numfunctions];
      info.base_yaxis = new int[info.numfunctions];

      info.yaxistexts[0] = "flop/s";
      info.selected_result[0] = 0;
      info.log_yaxis[0] = 0;
      info.base_yaxis[0] = 0;
      info.legendtexts[0] = "flop/s (ij)";

      info.yaxistexts[1] = "s";
      info.selected_result[1] = 1;
      info.log_yaxis[1] = 0;
      info.base_yaxis[1] = 0;
      info.legendtexts[1] = "time (ij)";

      info.yaxistexts[2] = "flop/s";
      info.selected_result[2] = 0;
      info.log_yaxis[2] = 0;
      info.base_yaxis[2] = 0;
      info.legendtexts[2] = "flop/s (ji)";

      info.yaxistexts[3] = "s";
      info.selected_result[3] = 1;
      info.log_yaxis[3] = 0;
      info.base_yaxis[3] = 0;
      info.legendtexts[3] = "time (ji)";

      return 0;
   }

   /* Implementation of the bi_init of the BenchIT interface. */
   public Object bi_init( int problemsizemax ) {
      if ( problemsizemax > bi_problemsize_max ) {
         System.out.println( "Kernel: Illegal problem size!" );
         System.exit(127);
      }

      DataObject dataObject = new DataObject();
      dataObject.maxIterations = maxIterations;

      return (Object)dataObject;
   }

   /* The central function within each kernel. This function
    * is called for each measurment step seperately.
    * @param  dObject      an Object to the attributes initialized in bi_init,
    *                      it is the Object bi_init returns
    * @param  problemsize  the actual problemsize
    * @param  results      an array of doubles, the
    *                      size of the array depends on the number
    *                      of functions, there are #functions+1
    *                      doubles
    * @return 0 if the measurment was sucessfull, something
    *         else in the case of an error */
   public int bi_entry( Object dObject, int problemsize, double[] results ) {
      /* ts, te: the start and end time of the measurement */
      /* timeinsecs: the time for a single measurement in seconds */
      double ts = 0.0, te = 0.0;
      double[] timeinsecs = new double[2];
      double[] flop = new double[2];
      int imyproblemsize = 0;
      double INVALID_MEASUREMENT = -7.77E7;

      int sweep2dflop = 0, diff2dflop = 0;
      DataObject dataObject = (DataObject)dObject;
      Jacobi worker = null;

      /* check wether the pointer to store the results in is valid or not */
      if ( results == null )
         return 1;

      /* calculate overhead if it is not done yet */
      if (timeroverhead == 0.0) {
         timeroverhead = gettimeroverhead();
         /* maybe we have a VERY fast timer ;-) 
          * the /10000 is because we measure the timeroverhead with 10000 calls 
          * to the timer. if this can't be measured with the timers resolution
          * 1 call has to be 10000 times faster than the resolution of the timer
          */
         if (timeroverhead == 0.0)
            timeroverhead = MINTIME/10000;
      }

      imyproblemsize = bi_problemsize_min + 
                       (problemsize-1) * bi_problemsize_increment;
      dataObject.maxn = imyproblemsize;

      /* initialize for first run */
      dataObject.a = new double[dataObject.maxn * dataObject.maxn];
      dataObject.b = new double[dataObject.maxn * dataObject.maxn];
      dataObject.f = new double[dataObject.maxn * dataObject.maxn];
      dataObject.h = 1.0 / ((double)(dataObject.maxn - 2 + 1));

      sweep2dflop = 7 * (dataObject.maxn - 2) * (dataObject.maxn - 2);
      diff2dflop  = 3 * (dataObject.maxn - 2) * (dataObject.maxn - 2);

      /* run the 2 different algorithms */
      for (int r = 0; r < RUNS; r++) {
         worker = new Jacobi( dataObject );
         worker.twodinit();

         /* start ij or ji */
         switch (r) {
            case 0:
               ts = JBI.bi_gettime();
               worker.run_ij();
               te = JBI.bi_gettime();
               break;
            case 1:
            default:
               ts = JBI.bi_gettime();
               worker.run_ji();
               te = JBI.bi_gettime();
         }

         /* calculate the used time and FLOPS */
         timeinsecs[r] = te - ts;
         timeinsecs[r] -= timeroverhead;
         flop[r] = dataObject.mitsdone * (2.0 * sweep2dflop + diff2dflop);
      }

      /* the index 0 always keeps the value for the x axis
       * the xaxis value needs to be stored only once! */
      results[0] = (double)dataObject.maxn;

      /* store the results in results[1], results[2], ...
       * [1] for the first function, [2] for the second function
       * and so on ... */
      for (int r = 0; r < RUNS; r++) {
         if (timeinsecs[r] < MINTIME) {
            results[r*RUNS + 1] = INVALID_MEASUREMENT;
            results[r*RUNS + 2] = INVALID_MEASUREMENT;
         } else {
            results[r*RUNS + 1] = flop[r] / timeinsecs[r];
            results[r*RUNS + 2] = timeinsecs[r];
         }
      }
      return(0);
   }

   /* Tries to measure the timer overhead for a single call to bi_gettime().
    * @return the calculated overhead in seconds */
   private double gettimeroverhead() {
      int s;
      double ts, te, t;
      ts = JBI.bi_gettime();
      for ( s = 0; s < 10000; s++ ) {
         t = JBI.bi_gettime();
      }
      te = JBI.bi_gettime();
      return ( te - ts ) / 10000.0;
   }
}

