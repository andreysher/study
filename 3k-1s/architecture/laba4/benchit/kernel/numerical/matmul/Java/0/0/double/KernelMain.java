/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: KernelMain.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/Java/0/0/double/KernelMain.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix multiply (Java)
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
   private int bi_matmul_start, bi_matmul_stop, bi_matmul_increment;
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
      info.codesequence = "for( i=0; i<s; i++)#" +
                          "  for( j=0; j<s; j++)#" +
                          "    for( k=0; k<s; k++)#" +
                          "    {#" +
                          "      c[j*s+i]+=a[k*s+i]*b[j*s+k];#" +
                          "    }";
      info.xaxistext = "Matrix Size";
      info.log_xaxis = 0;
      info.base_xaxis = 0;
      info.maxproblemsize = (bi_matmul_stop-bi_matmul_start+1)/bi_matmul_increment;
      if ((bi_matmul_stop-bi_matmul_start+1) % bi_matmul_increment != 0) info.maxproblemsize++;
      info.kernel_execs_mpi1 = 0;
      info.kernel_execs_mpi2 = 0;
      info.kernel_execs_pvm = 0;
      info.kernel_execs_omp = 0;
      info.kernel_execs_pthreads = 0;
      info.kernel_execs_javathreads = 0;
      /* B ########################################################*/
      n_of_works = 6; /* number versions of this algorithm (ijk, ikj, kij, ... = 6 */
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
         info.yaxistexts[index1] = "FLOPS";
         info.selected_result[index1] = 0;
         info.log_yaxis[index1] = 0;
         info.base_yaxis[index1] = 0;
         /*########################################################*/
         switch ( j ) {
            /* B ########################################################*/
            case 5: // 6th version legend text
               info.legendtexts[index1] = "FLOPS (kji)";
               break;
            case 4: // 5th version legend text
               info.legendtexts[index1] = "FLOPS (kij)";
               break;
            case 3: // 4th version legend text
               info.legendtexts[index1] = "FLOPS (jki)";
               break;
            case 2: // 3rd version legend text
               info.legendtexts[index1] = "FLOPS (jik)";
               break;
            case 1: // 2nd version legend text
               info.legendtexts[index1] = "FLOPS (ikj)";
               break;
            case 0: // 1st version legend text
            default:
               info.legendtexts[index1] = "FLOPS (ijk)";
            /*########################################################*/
         }
      }
      return 0;
   }
   /**
    * Implementation of the bi_init of the BenchIT interface.
    **/
   public Object bi_init( int problemsizemax ) {
      if ( problemsizemax > bi_matmul_stop ) {
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
      DataObject dataObject = (DataObject)dObject;
      dataObject.maxn = bi_matmul_start + (problemsize-1) * bi_matmul_increment;
      dataObject.a = new double[dataObject.maxn * dataObject.maxn];
      dataObject.b = new double[dataObject.maxn * dataObject.maxn];
      dataObject.c = new double[dataObject.maxn * dataObject.maxn];

      //System.out.println("problemsize: " + dataObject.maxn);

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
      MatMul worker = new MatMul( dataObject );
      double flos = 2.0 * dataObject.maxn * dataObject.maxn * dataObject.maxn;
      worker.init();
      /*########################################################*/

      for ( int j = 0; j < n_of_works; j++ )
      {
         /* B ########################################################*/
         int index1 = 0 * n_of_works + j;
         /* reset of reused values */
         ts = 0;
         te = 0;
         timeinsecs = 0.0;
         /* choose version of algorithm */
         switch ( j ) {
            case 5: // 6th version legend text
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               worker.matmul_kji();
               te = JBI.bi_gettime();
               break;
            case 4: // 5th version legend text
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               worker.matmul_kij();
               te = JBI.bi_gettime();
               break;
            case 3: // 4th version legend text
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               worker.matmul_jki();
               te = JBI.bi_gettime();
               break;
            case 2: // 3rd version legend text
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               worker.matmul_jik();
               te = JBI.bi_gettime();
               break;
            case 1: // 2nd version legend text
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               worker.matmul_ikj();
               te = JBI.bi_gettime();
               break;
            case 0: // 1st version legend text
            default:
               /* take start time, do measurment, and take end time */
               ts = JBI.bi_gettime();
               worker.matmul_ijk();
               te = JBI.bi_gettime();
         }
         /* calculate the used time and FLOPS */
         timeinsecs = te - ts; 
         timeinsecs -= timeroverhead;
         /* check for divide by zero, if timeinsecs is zero
         * timeinsecs=0 means that our operation does not need any time! */
         if ( timeinsecs < MINTIME ) timeinsecs = MINTIME;
         // this flops value is a made up! this calulations should be replaced
         // by something right for the choosen algorithm
         flops = (double)(flos / timeinsecs);
         /* store the results in results[1], results[2], ...
         * [1] for the first function, [2] for the second function
         * and so on ...
         * the index 0 always keeps the value for the x axis
         */
         /* B ########################################################*/
         // the xaxis value needs to be stored only once!
         if ( j == 0 ) results[0] = (double)dataObject.maxn;
         results[index1 + 1] = flops;
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
      String s_bi_matmul_start = null,  s_bi_matmul_stop = null, s_bi_matmul_increment = null;
      int i_bi_matmul_start = -1, i_bi_matmul_stop = -1, i_bi_matmul_increment = 500;
      s_bi_matmul_start = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_MIN");
      s_bi_matmul_stop = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_MAX");
      s_bi_matmul_increment = rwe.bi_getEnv("BENCHIT_KERNEL_PROBLEMSIZE_INCREMENT");
      if ((s_bi_matmul_start == null) || (s_bi_matmul_stop == null) || (s_bi_matmul_increment == null)) {
         i_bi_matmul_start = 1;
         i_bi_matmul_stop = 500;
         i_bi_matmul_increment = 1;
      }
      try {
         i_bi_matmul_start = (new Integer(s_bi_matmul_start)).intValue();
         i_bi_matmul_stop = (new Integer(s_bi_matmul_stop)).intValue();
         i_bi_matmul_increment = (new Integer(s_bi_matmul_increment)).intValue();
      }
      catch ( NumberFormatException nfe ) {
         i_bi_matmul_start = 1;
         i_bi_matmul_stop = 500;
         i_bi_matmul_increment = 1;
      }
      bi_matmul_start = i_bi_matmul_start;
      bi_matmul_stop = i_bi_matmul_stop;
      bi_matmul_increment = i_bi_matmul_increment;
   }
}


