/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: KernelMain.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/elementary/Java/0/0/sincostan/KernelMain.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Execution time of mathematical operations
 *         SINUS / COSINUS / TANGENT addict to input value
 *******************************************************************/

import java.util.Date;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/** This class is the Interface between the measured algorithm and BenchIT.
 * @author Robert Wloch, wloch@zhr.tu-dresden.de
 */
public class KernelMain implements BIJavaKernel {
   /* if it is static it has to be calculated just one time */
   private static double timeroverhead = 0.0;
   /**
    * Here we define a minimum time that our kernel needs
    * we do this to avoid a divide by zero.
    */
   private static double MINTIME = 5.0e-6;
   
   /* attributes */
   /**
    * These variables will help us to keep the overview over the arrays
    * we access for our functions/data.
    */
   /**
    * Number of different ways an algorithm will be measured.
    * sin, cos, tan -> n_of_works=3;
    */
   private int n_of_works;
   /**
    * Number of fixed functions we have per measurement.
    * Example: execution time and MFLOPS are measured for each loop order
    * -> n_of_sure_funcs_per_work=2
    */
   private int n_of_sure_funcs_per_work;
   private int num_funcs = 0;
   private int bi_angle_start, bi_angle_stop, bi_angle_increment;
   private BIEnvHash rwe = null;
   
   
   /**
    * The constructor.
    */
   public KernelMain() {
      rwe = BIEnvHash.getInstance();
   }
   
   
   /**
    * The implementation of the bi_getinfo from the BenchIT interface.
    * Here the info is filled with informations about the kernel.
    * @param info An Object to hold the info about the kernel.
    */
   public int bi_getinfo( InfoObject info ) {
      evaluate_environment();
      info.codesequence = "for( a=0; a<LOOPS; a++) do the mathematical operation; ";
      info.xaxistext = "Angle (degree)";
      info.log_xaxis = 0;
      info.base_xaxis = 0;
      info.maxproblemsize = (bi_angle_stop-bi_angle_start+1)/bi_angle_increment;
      if ((bi_angle_stop-bi_angle_start+1) % bi_angle_increment != 0) info.maxproblemsize++;
      info.kernel_execs_mpi1 = 0;
      info.kernel_execs_mpi2 = 0;
      info.kernel_execs_pvm = 0;
      info.kernel_execs_omp = 0;
      info.kernel_execs_pthreads = 0;
      info.kernel_execs_javathreads = 0;
      /* B ########################################################*/
      n_of_works = 3; /* number versions of this algorithm sin,cos,tan = 3 */
      n_of_sure_funcs_per_work = 1; /* FLOPS */
      /*########################################################*/
      num_funcs = n_of_works * n_of_sure_funcs_per_work;
      info.numfunctions = num_funcs;
      
      /* allocating memory for y axis texts and properties */
      info.yaxistexts = new String[info.numfunctions];
      info.selected_result = new int[info.numfunctions];
      info.legendtexts = new String[info.numfunctions];
      info.log_yaxis = new int[info.numfunctions];
      info.base_yaxis = new int[info.numfunctions];
      /* setting up y axis texts and properties */
      for ( int j = 0; j < n_of_works; j++ ) {
         /* B ########################################################*/
         int index1 = 0 * n_of_works + j;
         info.yaxistexts[index1] = "Op/s";
         info.selected_result[index1] = 0;
         info.log_yaxis[index1] = 0;
         info.base_yaxis[index1] = 0;
         /*########################################################*/
         switch ( j ) {
            /* B ########################################################*/
            case 0: // 1th version legend text
               info.legendtexts[index1] = "sinus";
               break;
            case 1: // 2th version legend text
               info.legendtexts[index1] = "cosinus";
               break;
            case 2: // 3th version legend text
               info.legendtexts[index1] = "tangent";
               break;
            default:
               System.out.println( "Error: corrupte n_of_works" );
               System.exit( 127 );
            /*########################################################*/
         }
      }
      return 0;
   }
   
   
   /**
    * Implementation of the bi_init of the BenchIT interface.
    */
   public Object bi_init( int problemsizemax ) {
      if ( problemsizemax > bi_angle_stop ) {
         System.out.println( "Kernel: Illegal problem size!" );
         System.exit( 127 );
      }
      DataObject dataObject = new DataObject();
      return (Object)dataObject;
   }
   
   
   /**
    * The central function within each kernel. This function
    * is called for each measurment step seperately.
    * @param  dObject      an Object to the attributes initialized in bi_init,
    *                      it is the Object bi_init returns
    * @param  problemsize  the actual problemsize
    * @param  results      an array of doubles, the
    *                      size of the array depends on the number
    *                      of functions, there are #functions+1
    *                      doubles
    * @return 0 if the measurment was sucessfull, something
    *         else in the case of an error
    */
   public int bi_entry( Object dObject, int problemsize, double[] results ) {
      /* ts, te: the start and end time of the measurement */
      /* timeinsecs: the time for a single measurement in seconds */
      double ts = 0.0, te = 0.0, calloverhead = 0.0;
      double[] timeinsecs = new double[n_of_works];
      double maxn = 0.0;
      long howManyRepeats = 1;
      double dummy;
      
      SinCosTan worker = new SinCosTan();
      
      maxn = bi_angle_start + (problemsize-1) * bi_angle_increment;
//      System.out.println("problemsize: " + (int)maxn);

      /* convert degree to radian */
      maxn = Math.toRadians((double)maxn);
      
      /* calculate overhead if it is not done yet */
      if ( timeroverhead == 0.0 ) {
         timeroverhead = gettimeroverhead();
         /* maybe we have a VERY fast timer ;-) 
          * the /10000 is because we measure the timeroverhead with 10000 calls 
          * to the timer. if this can't be measured with the timers resolution
          * 1 call has to be 10000 times faster than the resolution of the timer
          */
         if ( timeroverhead == 0.0 ) timeroverhead = MINTIME/10000;
      }
      
      /* check wether the pointer to store the results in is valid or not */
      if ( results == null ) return 1;
      
      do {
         for (int j=0; j<n_of_works; j++) {
            /* reset of reused values */
            ts = 0.0;
            te = 0.0;
            timeinsecs[j] = 0.0;

            ts = JBI.bi_gettime();
            dummy = worker.getOverhead(maxn, howManyRepeats);
            te = JBI.bi_gettime();
            calloverhead = te - ts;

            /* choose version of algorithm */
            switch ( j ) {
               case 0: // 1th version legend text
                  /* take start time, do measurment, and take end time */
                  ts = JBI.bi_gettime();
                  dummy = worker.sin(maxn, howManyRepeats);
                  te = JBI.bi_gettime();
                  break;
               case 1: // 2th version legend text
                  /* take start time, do measurment, and take end time */
                  ts = JBI.bi_gettime();
                  dummy = worker.cos(maxn, howManyRepeats);
                  te = JBI.bi_gettime();
                  break;
               case 2: // 3th version legend text
                  /* take start time, do measurment, and take end time */
                  ts = JBI.bi_gettime();
                  dummy = worker.tan(maxn, howManyRepeats);
                  te = JBI.bi_gettime();
                  break;
               default:
                  System.out.println( "Error: corrupte n_of_works" );
                  System.exit( 127 );
            }

            /* calculate the used time and FLOPS */
            timeinsecs[j] = te - ts;
            timeinsecs[j] -= calloverhead;
            timeinsecs[j] -= timeroverhead;
         }
         howManyRepeats = howManyRepeats * 8;

         if( (timeinsecs[0]>MINTIME) && (timeinsecs[1]>MINTIME) && (timeinsecs[2]>MINTIME) ) {
            break;
         }
      } while(true);

      /* reset last multiplication on howManyRepeats */
      howManyRepeats = (long)howManyRepeats / 8;

      /**
       * the index 0 always keeps the value for the x axis
       * the xaxis value needs to be stored only once!
       */
      results[0] = (double)(bi_angle_start + (problemsize-1) * bi_angle_increment);

      /**
       * store the results in results[1], results[2], ...
       * [1] for the first function, [2] for the second function
       * and so on ...
       */
      for (int j=0; j<n_of_works; j++) {
         results[j + 1] = howManyRepeats / timeinsecs[j];
      }
      return(0);
   }
   
   
   /**
    * Tries to measure the timer overhead for a single call to bi_gettime().
    * @return the calculated overhead in seconds
    */
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
   
   
   /**
    * Reads the environment variables used by this kernel.
    */
   private void evaluate_environment() {
      String s_bi_angle_start = null,  s_bi_angle_stop = null, s_bi_angle_increment = null;
      int i_bi_angle_start = -1, i_bi_angle_stop = -1, i_bi_angle_increment = 500;
      s_bi_angle_start = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_MIN");
      s_bi_angle_stop = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_MAX");
      s_bi_angle_increment = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT");
      if ((s_bi_angle_start == null) || (s_bi_angle_stop == null) || (s_bi_angle_increment == null)) {
         i_bi_angle_start = 1;
         i_bi_angle_stop = 360;
         i_bi_angle_increment = 1;
      }
      try {
         i_bi_angle_start = (new Integer(s_bi_angle_start)).intValue();
         i_bi_angle_stop = (new Integer(s_bi_angle_stop)).intValue();
         i_bi_angle_increment = (new Integer(s_bi_angle_increment)).intValue();
      }
      catch ( NumberFormatException nfe ) {
         i_bi_angle_start = 1;
         i_bi_angle_stop = 360;
         i_bi_angle_increment = 1;
      }
      bi_angle_start = i_bi_angle_start;
      bi_angle_stop = i_bi_angle_stop;
      bi_angle_increment = i_bi_angle_increment;
   }
}

