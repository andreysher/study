/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: Jacobi.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/kernel/numerical/jacobi/Java/0/0/sequential/Jacobi.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
/* Kernel: Jacobi algorithm measuring FLOPS (ij, ji) for change of
 *         dimension
 *******************************************************************/

/* This class definies the measured algorithm. */
public class Jacobi {
	/* constants */
	protected DataObject dataObject;

	public Jacobi( DataObject dataObject ) {
		this.dataObject = dataObject;
	}

	/* This methode inititializes static attributes that need
	 * to be initialized only once. */
	public void twodinit() {
		dataObject.mitsdone = 0;
		dataObject.diffnorm = 0.0;
		/* boundary condition: a and b have value 1 on lower bound (0 < x < 1, y = 0) */
		for ( int i = 0; i < dataObject.maxn; i++ ) {
			for ( int j = 0; j < dataObject.maxn; j++ ) {
				dataObject.a[i * dataObject.maxn + j] = 0.0;
				dataObject.b[i * dataObject.maxn + j] = 0.0;
				dataObject.f[i * dataObject.maxn + j] = 0.0;
			}
		}
		for ( int j = 1; j < dataObject.maxn - 1; j++ ) {
			int index = j * dataObject.maxn;
			dataObject.a[index] = 1.0;
			dataObject.b[index] = 1.0;
		}
		for ( int i = 1; i < dataObject.maxn - 1; i++ ) {
			int index = i;
			dataObject.a[index] = 1.0;
			dataObject.b[index] = 1.0;
		}
	}

	/* Computation of the stencil writing to b[]. */
	public void sweep2d_afnb_ji() {
		int index, i, j;
		for ( j = 1; j < dataObject.maxn - 1; j++ ) {
			for ( i = 1; i < dataObject.maxn - 1; i++ ) {
				index = j * dataObject.maxn + i;
				dataObject.b[index] = 0.25 *
					( dataObject.a[index - dataObject.maxn]
               + dataObject.a[index - 1] + dataObject.a[index + 1]
               + dataObject.a[index + dataObject.maxn] ) -
					dataObject.h * dataObject.h * dataObject.f[index];
			}
		}
	}

	public void sweep2d_afnb_ij() {
		int index, i, j;
		for ( i = 1; i < dataObject.maxn - 1; i++ ) {
			for ( j = 1; j < dataObject.maxn - 1; j++ ) {
				index = j * dataObject.maxn + i;
				dataObject.b[index] = 0.25 *
					( dataObject.a[index - dataObject.maxn]
               + dataObject.a[index - 1] + dataObject.a[index + 1]
               + dataObject.a[index + dataObject.maxn] ) -
					dataObject.h * dataObject.h * dataObject.f[index];
			}
		}
	}

	/** Computation of the stencil writing to a[]. */
	public void sweep2d_bfna_ji() {
		int index, i, j;
		for ( j = 1; j < dataObject.maxn - 1; j++ ) {
			for ( i = 1; i < dataObject.maxn - 1; i++ ) {
				index = j * dataObject.maxn + i;
				dataObject.a[index] = 0.25 *
					( dataObject.b[index - dataObject.maxn]
               + dataObject.b[index - 1] + dataObject.b[index + 1]
               + dataObject.b[index + dataObject.maxn] ) -
					dataObject.h * dataObject.h * dataObject.f[index];
			}
		}
	}
	public void sweep2d_bfna_ij() {
		int index, i, j;
		for ( i = 1; i < dataObject.maxn - 1; i++ ) {
			for ( j = 1; j < dataObject.maxn - 1; j++ ) {
				index = j * dataObject.maxn + i;
				dataObject.a[index] = 0.25 *
					( dataObject.b[index - dataObject.maxn]
               + dataObject.b[index - 1] + dataObject.b[index + 1]
               + dataObject.b[index + dataObject.maxn] ) -
					dataObject.h * dataObject.h * dataObject.f[index];
			}
		}
	}

	/* Calculation of residual error between sub-arrays a[] and b[]. */
	public double diff2d_ji() {
		int index, i, j;
		double diff = 0.0;
		double sum = 0.0;
		for ( j = 1; j < dataObject.maxn - 1; j++ ) {
			for ( i = 1; i < dataObject.maxn - 1; i++ ) {
				index = j * dataObject.maxn + i;
				diff = dataObject.a[index] - dataObject.b[index];
				sum += diff * diff;
			}
		}
		return sum;
	}

	public double diff2d_ij() {
		int index, i, j;
		double diff = 0.0;
		double sum = 0.0;
		for ( i = 1; i < dataObject.maxn - 1; i++ ) {
			for ( j = 1; j < dataObject.maxn - 1; j++ ) {
				index = j * dataObject.maxn + i;
				diff = dataObject.a[index] - dataObject.b[index];
				sum += diff * diff;
			}
		}
		return sum;
	}

	/* Helps the programmer to write less code ;-). */
	private void println( String msg ) {
		System.out.println( msg );
	}

	/* If you call this, it will print the b[]-array at any place
	 * during execution. */
	public void printStatB() {
	   StringBuffer sbb = new StringBuffer();
	   int j, i, index = 0;
	   sbb.append( "double[] b:\n[" );
	   for ( j = 0; j < dataObject.maxn; j++ ) {
		   for ( i = 0; i < dataObject.maxn; i++ ) {
			   index = j * dataObject.maxn + i;
			   String st = new String(
               ( new Double( dataObject.b[index] ) ).toString() );
			   sbb.append( "[" );
			   for ( int space = 1; space <= ( 10 - st.length() ); space++ ) {
			      sbb.append( " " );
            }
			   sbb.append( st + "]" );
			   if ( i < ( dataObject.maxn - 1 ) ) {
		   	   sbb.append( "," );
            }
			}
			if ( j < ( dataObject.maxn - 1 ) ) {
			   sbb.append("\n ");
         }
	    }
	    sbb.append( "]" );
	    println( sbb.toString() );
	}

	public void printStatA() {
	   StringBuffer sbb = new StringBuffer();
	   int j, i, index = 0;
	   sbb.append( "double[] a:\n[" );
	   for ( j = 0; j < dataObject.maxn; j++ ) {
			for ( i = 0; i < dataObject.maxn; i++ ) {
			   index = j * dataObject.maxn + i;
			   String st = new String(
               ( new Double( dataObject.a[index] ) ).toString() );
			   sbb.append( "[" );
			   for ( int space = 1; space <= ( 10 - st.length() ); space++ ) {
				   sbb.append( " " );
            }
			   sbb.append( st + "]" );
			   if ( i < ( dataObject.maxn - 1 ) ) {
		      	sbb.append(",");
            }
			}
			if ( j < ( dataObject.maxn - 1 ) ) {
            sbb.append( "\n " );
         }
	    }
	    sbb.append( "]" );
	    println( sbb.toString() );
	}

	public void run_ji() {
      /* here is defined the job to do */
		for ( int it = 0; it < dataObject.maxIterations; it++ ) {
         /* main loop = iteration */
			sweep2d_bfna_ji();
			sweep2d_afnb_ji();
			dataObject.diffnorm = diff2d_ji();
			dataObject.mitsdone++;
			if ( dataObject.diffnorm < 1.3e-3 ) break;
			dataObject.diffnorm = 0.0;
		}
		if ( dataObject.diffnorm >= 1.3e-3 ) {
//			println( "failed to converge" );
		}
	}

	public void run_ij() {
      /* here is defined the job to do */
		for ( int it = 0; it < dataObject.maxIterations; it++ ) {
         /* main loop = iteration */
			sweep2d_bfna_ij();
			sweep2d_afnb_ij();
			dataObject.diffnorm = diff2d_ij();
			dataObject.mitsdone++;
			if ( dataObject.diffnorm < 1.3e-3 ) break;
			dataObject.diffnorm = 0.0;
		}
		if ( dataObject.diffnorm >= 1.3e-3 ) {
//			println( "failed to converge" );
		}
	}
}

