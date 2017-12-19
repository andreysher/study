/**
 * BITConnectCallbackHandler.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:35 LKT)
 */

package org.benchit.bitconnect.service;

/**
 * BITConnectCallbackHandler Callback class, Users can extend this class and implement their own receiveResult and
 * receiveError methods.
 */
public abstract class BITConnectCallbackHandler {

	protected Object clientData;

	/**
	 * User can pass in any object that needs to be accessed once the NonBlocking Web service call is finished and
	 * appropriate method of this CallBack is called.
	 * 
	 * @param clientData Object mechanism by which the user can pass in user data that will be avilable at the time this
	 *          callback is called.
	 */
	public BITConnectCallbackHandler(Object clientData) {
		this.clientData = clientData;
	}

	/**
	 * Please use this constructor if you don't want to set any clientData
	 */
	public BITConnectCallbackHandler() {
		this.clientData = null;
	}

	/**
	 * Get the client data
	 */

	public Object getClientData() {
		return clientData;
	}

	/**
	 * auto generated Axis2 call back method for listBITIdentifiers method override this method for handling normal
	 * response from listBITIdentifiers operation
	 */
	public void receiveResultlistBITIdentifiers(
			org.benchit.bitconnect.service.types.BITIdentifierListType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from listBITIdentifiers
	 * operation
	 */
	public void receiveErrorlistBITIdentifiers(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for getBITFileUploadStatus method override this method for handling normal
	 * response from getBITFileUploadStatus operation
	 */
	public void receiveResultgetBITFileUploadStatus(
			org.benchit.bitconnect.service.types.BITFileUploadStatusType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from getBITFileUploadStatus
	 * operation
	 */
	public void receiveErrorgetBITFileUploadStatus(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for deleteBITFile method override this method for handling normal response
	 * from deleteBITFile operation
	 */
	public void receiveResultdeleteBITFile(boolean result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from deleteBITFile operation
	 */
	public void receiveErrordeleteBITFile(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for getBITFile method override this method for handling normal response from
	 * getBITFile operation
	 */
	public void receiveResultgetBITFile(org.benchit.bitconnect.service.types.BITFileType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from getBITFile operation
	 */
	public void receiveErrorgetBITFile(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for listBITItemsByConstrain method override this method for handling normal
	 * response from listBITItemsByConstrain operation
	 */
	public void receiveResultlistBITItemsByConstrain(
			org.benchit.bitconnect.service.types.BITItemListType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from listBITItemsByConstrain
	 * operation
	 */
	public void receiveErrorlistBITItemsByConstrain(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for loginBITAccount method override this method for handling normal response
	 * from loginBITAccount operation
	 */
	public void receiveResultloginBITAccount(org.benchit.bitconnect.service.types.BITUserType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from loginBITAccount operation
	 */
	public void receiveErrorloginBITAccount(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for listBITItems method override this method for handling normal response
	 * from listBITItems operation
	 */
	public void receiveResultlistBITItems(org.benchit.bitconnect.service.types.BITItemListType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from listBITItems operation
	 */
	public void receiveErrorlistBITItems(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for registerBITAccount method override this method for handling normal
	 * response from registerBITAccount operation
	 */
	public void receiveResultregisterBITAccount(boolean result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from registerBITAccount
	 * operation
	 */
	public void receiveErrorregisterBITAccount(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for listBITFilesByConstrain method override this method for handling normal
	 * response from listBITFilesByConstrain operation
	 */
	public void receiveResultlistBITFilesByConstrain(
			org.benchit.bitconnect.service.types.BITFileListType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from listBITFilesByConstrain
	 * operation
	 */
	public void receiveErrorlistBITFilesByConstrain(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for uploadBITFile method override this method for handling normal response
	 * from uploadBITFile operation
	 */
	public void receiveResultuploadBITFile(boolean result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from uploadBITFile operation
	 */
	public void receiveErroruploadBITFile(java.lang.Exception e) {}

	/**
	 * auto generated Axis2 call back method for getBITFileMetadata method override this method for handling normal
	 * response from getBITFileMetadata operation
	 */
	public void receiveResultgetBITFileMetadata(
			org.benchit.bitconnect.service.types.BITItemListType result) {}

	/**
	 * auto generated Axis2 Error handler override this method for handling error response from getBITFileMetadata
	 * operation
	 */
	public void receiveErrorgetBITFileMetadata(java.lang.Exception e) {}

}
