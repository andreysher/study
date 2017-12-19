/**
 * SessionFault.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:35 LKT)
 */

package org.benchit.bitconnect.service;

public class SessionFault extends java.lang.Exception {

	private org.benchit.bitconnect.service.types.SessionFaultElem faultMessage;

	public SessionFault() {
		super("SessionFault");
	}

	public SessionFault(java.lang.String s) {
		super(s);
	}

	public SessionFault(java.lang.String s, java.lang.Throwable ex) {
		super(s, ex);
	}

	public void setFaultMessage(org.benchit.bitconnect.service.types.SessionFaultElem msg) {
		faultMessage = msg;
	}

	public org.benchit.bitconnect.service.types.SessionFaultElem getFaultMessage() {
		return faultMessage;
	}
}
