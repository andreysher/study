/**
 * RegisterFault.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:35 LKT)
 */

package org.benchit.bitconnect.service;

public class RegisterFault extends java.lang.Exception {

	private org.benchit.bitconnect.service.types.RegisterFaultElem faultMessage;

	public RegisterFault() {
		super("RegisterFault");
	}

	public RegisterFault(java.lang.String s) {
		super(s);
	}

	public RegisterFault(java.lang.String s, java.lang.Throwable ex) {
		super(s, ex);
	}

	public void setFaultMessage(org.benchit.bitconnect.service.types.RegisterFaultElem msg) {
		faultMessage = msg;
	}

	public org.benchit.bitconnect.service.types.RegisterFaultElem getFaultMessage() {
		return faultMessage;
	}
}
