/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: DataObject.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/matmul/Java/0/0/double/DataObject.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Matrix multiply (Java)
 *******************************************************************/

/** This class handles all the data that will be processed by the
  * measured algorithm.
  * @author Robert Wloch, wloch@zhr.tu-dresden.de
  */
public class DataObject {
	public int maxn = -1;
	// data to be processed
	public double[] a = null;
	public double[] b = null;
	public double[] c = null;
	public DataObject() {
	}
}


