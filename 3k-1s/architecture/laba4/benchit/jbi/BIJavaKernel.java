/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: BIJavaKernel.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/jbi/BIJavaKernel.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Interface to the BenchIT-project (java version)
 *******************************************************************/
public interface BIJavaKernel {
    /* provides an empty info-struct. has to be filled by the kernel */
    public int bi_getinfo( InfoObject info );

    /* initializes the kernel with the maximum problem-size. returns a pointer to the
       allocated memory. this pointer is given with each other kernel call */
    public Object bi_init( int problemsizemax );

    /* calls the kernel with one problem-size. Expects the results in the *result. */
    public int bi_entry( Object dObject, int problemsize, double[] results );
}

