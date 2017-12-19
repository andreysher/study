
/**
 * BITConnect.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:35 LKT)
 */

package org.benchit.bitconnect.service;

/*
 *  BITConnect java interface
 */

public interface BITConnect {

	/**
	 * Auto generated method signature
	 * 
	 * @param listBITIdentifiers9
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITIdentifierListType listBITIdentifiers(

	java.lang.String param10) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param listBITIdentifiers9
	 */
	public void startlistBITIdentifiers(

	java.lang.String param10,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param getBITFileUploadStatus13
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public org.benchit.bitconnect.service.types.BITFileUploadStatusType getBITFileUploadStatus(

	java.lang.String filehash14) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param getBITFileUploadStatus13
	 */
	public void startgetBITFileUploadStatus(

	java.lang.String filehash14,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param deleteBITFile17
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public boolean deleteBITFile(

	java.lang.String filename18) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param deleteBITFile17
	 */
	public void startdeleteBITFile(

	java.lang.String filename18,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param getBITFile21
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public org.benchit.bitconnect.service.types.BITFileType getBITFile(

	java.lang.String filename22) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param getBITFile21
	 */
	public void startgetBITFile(

	java.lang.String filename22,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param listBITItemsByConstrain25
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITItemListType listBITItemsByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints26)
			throws java.rmi.RemoteException

			, org.benchit.bitconnect.service.SessionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param listBITItemsByConstrain25
	 */
	public void startlistBITItemsByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints26,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param loginBITAccount29
	 * @throws org.benchit.bitconnect.service.LoginFault :
	 */

	public org.benchit.bitconnect.service.types.BITUserType loginBITAccount(

	java.lang.String login30, java.lang.String password31) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.LoginFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param loginBITAccount29
	 */
	public void startloginBITAccount(

	java.lang.String login30, java.lang.String password31,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param listBITItems34
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITItemListType listBITItems(

	java.lang.String param35) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param listBITItems34
	 */
	public void startlistBITItems(

	java.lang.String param35,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param registerBITAccount38
	 * @throws org.benchit.bitconnect.service.RegisterFault :
	 */

	public boolean registerBITAccount(

	org.benchit.bitconnect.service.types.BITUserType user39) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.RegisterFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param registerBITAccount38
	 */
	public void startregisterBITAccount(

	org.benchit.bitconnect.service.types.BITUserType user39,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param listBITFilesByConstrain42
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITFileListType listBITFilesByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints43)
			throws java.rmi.RemoteException

			, org.benchit.bitconnect.service.SessionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param listBITFilesByConstrain42
	 */
	public void startlistBITFilesByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints43,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param uploadBITFile46
	 * @throws org.benchit.bitconnect.service.UploadFault :
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public boolean uploadBITFile(

	org.benchit.bitconnect.service.types.BITFileType file47) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.UploadFault, org.benchit.bitconnect.service.SessionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param uploadBITFile46
	 */
	public void startuploadBITFile(

	org.benchit.bitconnect.service.types.BITFileType file47,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	/**
	 * Auto generated method signature
	 * 
	 * @param getBITFileMetadata50
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public org.benchit.bitconnect.service.types.BITItemListType getBITFileMetadata(

	java.lang.String filename51) throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault;

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @param getBITFileMetadata50
	 */
	public void startgetBITFileMetadata(

	java.lang.String filename51,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException;

	//
}
