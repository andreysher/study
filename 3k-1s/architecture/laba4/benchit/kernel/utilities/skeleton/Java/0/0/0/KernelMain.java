/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: SkeletonKernelMain.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/utilities/skeleton/Java/0/0/0/SkeletonKernelMain.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: java kernel skeleton
 * this file: the interface between Work and JBI
 *******************************************************************/

public class KernelMain implements BIJavaKernel {
   /* if it is static it has to be calculated just one time */
   private static double timeroverhead = 0.0;
   /**
    * Here we define a minimum time that our kernel needs
    * we do this to avoid a divide by zero.
    **/
   private static double MINTIME = 5.0e-6;
   /* attributes */
   /**
    * These variables will help us to keep the overview over the arrays
    * we access for our functions/data.
    **/
   /**
    * Number of different ways an algorithm will be measured.
    * Example: loop orders: ijk, ikj, jki, jik, kij, kji -> n_of_works=6 with
    * each different loop order in an own function.
    **/
   private int n_of_works;
   /**
    * Number of fixed functions we have per measurement.
    * Example: execution time and MFLOPS are measured for each loop order
    * -> n_of_sure_funcs_per_work=2
    **/
   private int n_of_sure_funcs_per_work;
   private int num_funcs = 0;
   private int MAXIMUM = 500;
   private int STEPSIZE = 1;
   private BIEnvHash rwe = null;
   /**
    * The constructor.
    **/
   public KernelMain() {
      rwe = BIEnvHash.getInstance();
   }
   /**
    * The implementation of the bi_getinfo from the BenchIT interface.
    * Here the info is filled with informations about the kernel.
    * @param info An Object to hold the info about the kernel.
    **/
   public int bi_getinfo( InfoObject info ) {
      evaluate_environment();
      info.codesequence = "work_[1|2]()";
      info.xaxistext = "Problem Size";
      info.log_xaxis = 0;
      info.base_xaxis = 0;
      info.maxproblemsize = MAXIMUM;
      info.kernel_execs_mpi1 = 0;
      info.kernel_execs_mpi2 = 0;
      info.kernel_execs_pvm = 0;
      info.kernel_execs_omp = 0;
      info.kernel_execs_pthreads = 0;
      info.kernel_execs_javathreads = 0;
      /* B ########################################################*/
      n_of_works = 2; /* number versions of this algorithm (ijk, ikj, kij, ... = 6 */
      n_of_sure_funcs_per_work = 2; /* time measurement and FLOPS (calculated) */
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
         int index2 = 1 * n_of_works + j;
         //int index3 = 2 * n_of_works + j;
         // 1st function
         info.yaxistexts[index1] = "s";
         info.selected_result[index1] = 1;
         info.log_yaxis[index1] = 0;
         info.base_yaxis[index1] = 0;
         // 2nd function
         info.yaxistexts[index2] = "FLOPS";
         info.selected_result[index2] = 0;
         info.log_yaxis[index2] = 0;
         info.base_yaxis[index2] = 0;
         /*########################################################*/
         // 3rd function
         //info.yaxistexts[index3] = "";
         //info.selected_result[index3] = 0;
         //info.log_yaxis[index3] = 0;
         //info.base_yaxis[index3] = 0.0;
         switch ( j ) {
            /* B ########################################################*/
            case 1: // 2nd version legend text; maybe (ikj)
               info.legendtexts[index1] = "Calculation Time in s (2)";
               info.legendtexts[index2] = "FLOPS (2)";
               break;
            case 0: // 1st version legend text; maybe (ijk)
            default:
               info.legendtexts[index1] = "Calculation Time in s (1)";
               info.legendtexts[index2] = "FLOPS (1)";
            /*########################################################*/
         }
      }
      return 0;
   }
   /**
    * Implementation of the bi_init of the BenchIT interface.
    * Making usage always of the same memory is faster.
    * HAVE A LOOK INTO THE HOWTO !
    **/
   public Object bi_init( int problemsizemax ) {
      SkeletonDataObject dataObject = new SkeletonDataObject();
      /* initialize your own arrays in here */
/*
      dataObject.maxn = problemsizemax;
      dataObject.a = new double[problemsizemax * problemsizemax];
*/
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
    **/
   public int bi_entry( Object dObject, int problemsize, double[] results ) {
      /**
       * In java results is the total result array, not just a subarray
       * as in c. The offset points to the start part of the current entry.
       **/
      //int offset = ( num_funcs + 1 ) * ( problemsize - 1 );
      /* ts, te: the start and end time of the measurement */
      /* timeinsecs: the time for a single measurement in seconds */
      double ts = 0, te = 0;
      double timeinsecs = 0.0;
      /* flops stores the calculated FLOPS */
      double flops = 0.0;
      /* Cast Object reference */
      SkeletonDataObject dataObject = (SkeletonDataObject)dObject;

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
      /* B ########################################################*/
      /* maybe some more init stuff in here */
      SkeletonWork work = new SkeletonWork( dataObject );
      /*########################################################*/

      for ( int j = 0; j < n_of_works; j++ )
      {
         /* B ########################################################*/
         int index1 = 0 * n_of_works + j;
         int index2 = 1 * n_of_works + j;
         /* reset of reused values */
         ts = 0;
         te = 0;
         timeinsecs = 0.0;
         /* choose version of algorithm */
         switch ( j ) {
            case 1: // 2nd version legend text; maybe (ikj)
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               work.work_2();
               te = JBI.bi_gettime();
               break;
            case 0: // 1st version legend text; maybe (ijk)
            default:
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               work.work_1();
               te = JBI.bi_gettime();
         }
         /* calculate the used time and FLOPS */
         timeinsecs = te - ts;
         timeinsecs -= timeroverhead;
         // this flops value is a made up! this calulations should be replaced
         // by something right for the choosen algorithm
         flops = (double)problemsize;
         /* check for divide by zero, if timeinsecs is zero
         * timeinsecs=0 means that our operation does not need any time! */
         if ( timeinsecs < MINTIME ) timeinsecs = MINTIME;
         /* store the results in results[1], results[2], ...
         * [1] for the first function, [2] for the second function
         * and so on ...
         * the index 0 always keeps the value for the x axis
         */
         /* B ########################################################*/
         // the xaxis value needs to be stored only once!
         if ( j == 0 ) results[0] = (double)problemsize;
         results[index1 + 1] = timeinsecs;
         results[index2 + 1] = flops;
         /*########################################################*/
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
    **/
   private void evaluate_environment() {
      String smax = null, sstep = null;
      int max = -1, maxDef = 500, step = -1, stepDef = 1;
      smax = rwe.bi_getEnv( "BENCHIT_SKELETONS_SKELETON_JAVA_0_0_0_STEPS" );
      sstep = rwe.bi_getEnv( "BENCHIT_SKELETONS_SKELETON_JAVA_0_0_0_INCREMENT" );
      if ( ( smax == null ) || ( sstep == null ) ) {
         max = maxDef;
         step = stepDef;
      }
      try {
         max = (new Integer( smax )).intValue();
         step = (new Integer( sstep )).intValue();
      }
      catch ( NumberFormatException nfe ) {
         max = maxDef;
         step = stepDef;
      }
      MAXIMUM = max;
      STEPSIZE = step;
   }
}
