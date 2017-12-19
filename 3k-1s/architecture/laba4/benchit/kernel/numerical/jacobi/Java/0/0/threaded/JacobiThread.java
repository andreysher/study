/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: JacobiThread.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/Java/0/0/threaded/JacobiThread.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         number of threads or for change of dimension
 *******************************************************************/

/* This class definies the measured algorithm. */
public class JacobiThread extends Thread {
   protected int myID;
   public boolean run_ij;
   protected DataObject dataObject;

   /* Attention, to avoid ArrayOutOfBoundsExceptions and to ensure
    * correct working of the program choose a nxy to satisfy this
    * conditions: (maxn MOD nxy = 0) AND (maxn >= nxy) .*/
   public JacobiThread(int ID, DataObject dataObject, boolean ij) {
      this.myID = ID;
      this.dataObject = dataObject;
      this.run_ij = ij;
   }

   /* This method inititializes static attributes that need
    * to be initialized only once. */
   public void twodinit() {
      dataObject.diffnormArray = new double[dataObject.numThreads];
      dataObject.cubes = (dataObject.maxn - 2) / dataObject.nxy;
      for (int j = 0; j < dataObject.numThreads; j++) {
         dataObject.diffnormArray[j] = 0.0;
      }
      for (int i = 0; i < dataObject.maxn; i++) {
         for (int j = 0; j < dataObject.maxn; j++) {
            dataObject.a[i * dataObject.maxn + j] = 0.0;
            dataObject.b[i * dataObject.maxn + j] = 0.0;
            dataObject.f[i * dataObject.maxn + j] = 0.0;
         }
      }
      for (int j = 1; j < dataObject.maxn - 1; j++) {
         int index = j * dataObject.maxn;
         dataObject.a[index] = 1.0;
         dataObject.b[index] = 1.0;
      }
      for (int i = 1; i < dataObject.maxn - 1; i++) {
         int index = i;
         dataObject.a[index] = 1.0;
         dataObject.b[index] = 1.0;
      }
   }

   /* Computation of the stencil writing to b[].
    * This method expects is and js to be > 0
    * and ie and je to be < maxn!
    * @param is left x value
    * @param ie right x value
    * @param js top y value
    * @param je bottom y value */
   public void sweep2d_afnb_ji(int is, int ie, int js, int je) {
      int index, i, j;
      for (j = js; j < je; j++) {
         for (i = is; i < ie; i++) {
            index = j * dataObject.maxn + i;
            dataObject.b[index] = 0.25 *
               (dataObject.a[index - dataObject.maxn] + dataObject.a[index - 1] +
               dataObject.a[index + 1] + dataObject.a[index + dataObject.maxn]) -
               dataObject.h * dataObject.h * dataObject.f[index];
         }
      }
   }

   /* Computation of the stencil writing to b[].
    * This method expects is and js to be > 0
    * and ie and je to be < maxn!
    * @param is left x value
    * @param ie right x value
    * @param js top y value
    * @param je bottom y value */
   public void sweep2d_afnb_ij(int is, int ie, int js, int je) {
      int index, i, j;
      for (i = is; i < ie; i++) {
         for (j = js; j < je; j++) {
            index = j * dataObject.maxn + i;
            dataObject.b[index] = 0.25 *
               (dataObject.a[index - dataObject.maxn] + dataObject.a[index - 1] +
               dataObject.a[index + 1] + dataObject.a[index + dataObject.maxn]) -
               dataObject.h * dataObject.h * dataObject.f[index];
         }
      }
   }

   /* Computation of the stencil writing to a[].
    * This method expects is and js to be > 0
    * and ie and je to be < maxn!
    * @param is left x value
    * @param ie right x value
    * @param js top y value
    * @param je bottom y value */
   public void sweep2d_bfna_ji(int is, int ie, int js, int je) {
      int index, i, j;
      for (j = js; j < je; j++) {
         for (i = is; i < ie; i++) {
            index = j * dataObject.maxn + i;
            dataObject.a[index] = 0.25 *
               (dataObject.b[index - dataObject.maxn] + dataObject.b[index - 1] +
               dataObject.b[index + 1] + dataObject.b[index + dataObject.maxn]) -
               dataObject.h * dataObject.h * dataObject.f[index];
         }
      }
   }

   /* Computation of the stencil writing to a[].
    * This method expects is and js to be > 0
    * and ie and je to be < maxn!
    * @param is left x value
    * @param ie right x value
    * @param js top y value
    * @param je bottom y value */
   public void sweep2d_bfna_ij(int is, int ie, int js, int je) {
      int index, i, j;
      for (i = is; i < ie; i++) {
         for (j = js; j < je; j++) {
            index = j * dataObject.maxn + i;
            dataObject.a[index] = 0.25 *
               (dataObject.b[index - dataObject.maxn] + dataObject.b[index - 1] +
               dataObject.b[index + 1] + dataObject.b[index + dataObject.maxn]) -
               dataObject.h * dataObject.h * dataObject.f[index];
         }
      }
   }

   /* Calculation of residual error between sub-arrays a[] and b[].
    * This method expects is and js to be > 0
    * and ie and je to be < maxn!
    * @param is left x value
    * @param ie right x value
    * @param js top y value
    * @param je bottom y value */
   public double diff2d_ji(int is, int ie, int js, int je) {
      int index, i, j;
      double diff = 0.0;
      double sum = 0.0;
      for (j = js; j < je; j++) {
         for (i = is; i < ie; i++) {
            index = j * dataObject.maxn + i;
            diff = dataObject.a[index] - dataObject.b[index];
            sum += diff * diff;
         }
      }
      return sum;
   }

   /* Calculation of residual error between sub-arrays a[] and b[].
    * This method expects is and js to be > 0
    * and ie and je to be < maxn!
    * @param is left x value
    * @param ie right x value
    * @param js top y value
    * @param je bottom y value */
   public double diff2d_ij(int is, int ie, int js, int je) {
      int index, i, j;
      double diff = 0.0;
      double sum = 0.0;
      for (i = is; i < ie; i++) {
         for (j = js; j < je; j++) {
            index = j * dataObject.maxn + i;
            diff = dataObject.a[index] - dataObject.b[index];
            sum += diff * diff;
         }
      }
      return sum;
   }

   /* Returns the ID of this Thread. */
   public int getID() {
      return myID;
   }

   /* Overriding method run() and defining the work to be done by the threads. */
   public void run() {
      if (run_ij)
         run_ij();
      else
         run_ji();
   }

   /* here is defined the job to do by the threads;
    * the very right thread is responsible for diffnorm */
   public void run_ji() { 
      dataObject.diffnorm = 1.0;
      /* DECOMPOSITION: defining the area this threads has to work on
       * is == left x value, ie == right  x value
       * js == top  y value, je == bottom y value */
      int is, js, ie, je;
      is = (myID % dataObject.cubes) * dataObject.nxy + 1;
      js = ((myID -(myID % dataObject.cubes)) / dataObject.cubes) * dataObject.nxy + 1;
      ie = is + dataObject.nxy;
      je = js + dataObject.nxy;
      dataObject.mitsdone = 0;

      /* main loop = iteration */
      for (int it = 0; it < dataObject.maxIterations; it++) { 
         if (dataObject.converged)
            break;
         /* ############
          * ## STEP 1 ##
          * ############ sweep2d(b, f, n, a); */
         sweep2d_bfna_ji(is, ie, js, je);
         dataObject.mb.runIntoBarrier1();
         /* ############
          * ## STEP 2 ##
          * ############ sweep2d(a, f, n, b); */
         sweep2d_afnb_ji(is, ie, js, je);
         dataObject.mb.runIntoBarrier2();
         /* ############
          * ## STEP 3 ##
          * ############ diffnorm = diff2d_abn(); */
         dataObject.incDiffNorm(diff2d_ji(is, ie, js, je)); /* incDiffNorm includes a barrier !! */
         if (myID == dataObject.numThreads - 1) {
            /* this thread has to build the diffnorm value! */
            dataObject.mitsdone++;
            if (dataObject.diffnorm < 1.3e-3) {
               dataObject.converged = true;
            }
            dataObject.diffnorm = 0.0;
            dataObject.mb.runIntoBarrier3();
         } else {
            dataObject.mb.runIntoBarrier3();
         }
      }
      /* end of main-loop */

      if (myID == dataObject.numThreads - 1) {
         /* after dataObject.maxIterations Iterations */
         if (dataObject.diffnorm >= 1.3e-3) {
//            System.out.println("failed to converge");
         }
      }
      dataObject.mb.runIntoBarrier1();
   }

   /* here is defined the job to do by the threads;
    * the very right thread is responsible for diffnorm */
   public void run_ij() {
      dataObject.diffnorm = 1.0;
      /* DECOMPOSITION: defining the area this threads has to work on
       * is == left x value, ie == right  x value
       * js == top  y value, je == bottom y value */
      int is, js, ie, je;
      is = (myID % dataObject.cubes) * dataObject.nxy + 1;
      js = ((myID -(myID % dataObject.cubes)) / dataObject.cubes) * dataObject.nxy + 1;
      ie = is + dataObject.nxy;
      je = js + dataObject.nxy;
      dataObject.mitsdone = 0;

      /* main loop = iteration */
      for (int it = 0; it < dataObject.maxIterations; it++) {
         if (dataObject.converged)
            break;
         /* ############
          * ## STEP 1 ##
          * ############ sweep2d(b, f, n, a); */
         sweep2d_bfna_ij(is, ie, js, je);
         dataObject.mb.runIntoBarrier1();
         /* ############
          * ## STEP 2 ##
          * ############ sweep2d(a, f, n, b); */
         sweep2d_afnb_ij(is, ie, js, je);
         dataObject.mb.runIntoBarrier2();
         /* ############
          * ## STEP 3 ##
          * ############ diffnorm = diff2d_abn(); */
         dataObject.incDiffNorm(diff2d_ij(is, ie, js, je)); /* incDiffNorm includes a barrier !! */
         if (myID == dataObject.numThreads - 1) {
            /* this thread has to build the diffnorm value! */
            dataObject.mitsdone++;
            if (dataObject.diffnorm < 1.3e-3) {
               dataObject.converged = true;
            }
            dataObject.diffnorm = 0.0;
            dataObject.mb.runIntoBarrier3();
         } else {
            dataObject.mb.runIntoBarrier3();
         }
      }
      /* end of main-loop */

      if (myID == dataObject.numThreads - 1) {
         /* after dataObject.maxIterations Iterations */
         if (dataObject.diffnorm >= 1.3e-3) {
//            System.out.println("failed to converge");
         }
      }
      dataObject.mb.runIntoBarrier1();
   }
}

