/**
 * LoginFault.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:35 LKT)
 */

package org.benchit.bitconnect.service;

public class LoginFault extends java.lang.Exception {

	private org.benchit.bitconnect.service.types.LoginFaultElem faultMessage;

	public LoginFault() {
		super("LoginFault");
	}

	public LoginFault(java.lang.String s) {
		super(s);
	}

	public LoginFault(java.lang.String s, java.lang.Throwable ex) {
		super(s, ex);
	}

	public void setFaultMessage(org.benchit.bitconnect.service.types.LoginFaultElem msg) {
		faultMessage = msg;
	}

	public org.benchit.bitconnect.service.types.LoginFaultElem getFaultMessage() {
		return faultMessage;
	}
}
