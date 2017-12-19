package org.benchit.bitconnect;

public class RequestCancelledException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2531123814259157251L;

	public RequestCancelledException() {}

	public RequestCancelledException(String arg0) {
		super(arg0);
	}

	public RequestCancelledException(Throwable arg0) {
		super(arg0);
	}

	public RequestCancelledException(String arg0, Throwable arg1) {
		super(arg0, arg1);
	}

}
