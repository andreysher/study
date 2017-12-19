/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: InfoObject.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/jbi/InfoObject.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Part of the BenchIT-project (java version)
 *******************************************************************/

public class InfoObject{
   public String codesequence;
   public String xaxistext;
   public String[] yaxistexts = null;
   public String[] legendtexts = null;
   public int maxproblemsize;
   public int numfunctions;
   public int numlibraries;
   public int[] selected_result=null;
   public int kernel_execs_mpi1;
   public int kernel_execs_mpi2;
   public int kernel_execs_pvm;
   public int kernel_execs_omp;
   public int kernel_execs_pthreads;
   public int kernel_execs_javathreads;
   public int log_xaxis;
   public int[] log_yaxis=null;
   public int base_xaxis;
   public int[] base_yaxis=null;
   public InfoObject(){
      codesequence = "";
      xaxistext = "";
      maxproblemsize = 0;
      numfunctions = 0;
      kernel_execs_mpi1 = 0;
      kernel_execs_mpi2 = 0;
      kernel_execs_pvm = 0;
      kernel_execs_omp = 0;
      kernel_execs_pthreads = 0;
      kernel_execs_javathreads = 0;
      log_xaxis = 0;
      base_xaxis = 0;
   }
}

