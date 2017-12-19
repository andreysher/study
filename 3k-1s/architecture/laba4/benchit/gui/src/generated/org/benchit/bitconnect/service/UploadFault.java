/**
 * UploadFault.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:35 LKT)
 */

package org.benchit.bitconnect.service;

public class UploadFault extends java.lang.Exception {

	private org.benchit.bitconnect.service.types.UploadFaultElem faultMessage;

	public UploadFault() {
		super("UploadFault");
	}

	public UploadFault(java.lang.String s) {
		super(s);
	}

	public UploadFault(java.lang.String s, java.lang.Throwable ex) {
		super(s, ex);
	}

	public void setFaultMessage(org.benchit.bitconnect.service.types.UploadFaultElem msg) {
		faultMessage = msg;
	}

	public org.benchit.bitconnect.service.types.UploadFaultElem getFaultMessage() {
		return faultMessage;
	}
}
