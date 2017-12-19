/**
 * BITConnectStub.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:35 LKT)
 */
package org.benchit.bitconnect.service;

/*
 *  BITConnectStub java implementation
 */

public class BITConnectStub extends org.apache.axis2.client.Stub implements BITConnect {
	protected org.apache.axis2.description.AxisOperation[] _operations;

	// hashmaps to keep the fault mapping
	private java.util.HashMap faultExceptionNameMap = new java.util.HashMap();
	private java.util.HashMap faultExceptionClassNameMap = new java.util.HashMap();
	private java.util.HashMap faultMessageMap = new java.util.HashMap();

	private static int counter = 0;

	private static synchronized String getUniqueSuffix() {
		// reset the counter if it is greater than 99999
		if (counter > 99999) {
			counter = 0;
		}
		counter = counter + 1;
		return Long.toString(System.currentTimeMillis()) + "_" + counter;
	}

	private void populateAxisService() throws org.apache.axis2.AxisFault {

		// creating the Service with a unique name
		_service = new org.apache.axis2.description.AxisService("BITConnect" + getUniqueSuffix());
		addAnonymousOperations();

		// creating the operations
		org.apache.axis2.description.AxisOperation __operation;

		_operations = new org.apache.axis2.description.AxisOperation[11];

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"listBITIdentifiers"));
		_service.addOperation(__operation);

		_operations[0] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"getBITFileUploadStatus"));
		_service.addOperation(__operation);

		_operations[1] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"deleteBITFile"));
		_service.addOperation(__operation);

		_operations[2] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"getBITFile"));
		_service.addOperation(__operation);

		_operations[3] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"listBITItemsByConstrain"));
		_service.addOperation(__operation);

		_operations[4] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"loginBITAccount"));
		_service.addOperation(__operation);

		_operations[5] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"listBITItems"));
		_service.addOperation(__operation);

		_operations[6] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"registerBITAccount"));
		_service.addOperation(__operation);

		_operations[7] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"listBITFilesByConstrain"));
		_service.addOperation(__operation);

		_operations[8] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"uploadBITFile"));
		_service.addOperation(__operation);

		_operations[9] = __operation;

		__operation = new org.apache.axis2.description.OutInAxisOperation();

		__operation.setName(new javax.xml.namespace.QName("http://benchit.org/bitconnect/service/",
				"getBITFileMetadata"));
		_service.addOperation(__operation);

		_operations[10] = __operation;

	}

	// populates the faults
	private void populateFaults() {

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.types.PermissionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.types.PermissionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.types.PermissionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "loginFaultElem"),
				"org.benchit.bitconnect.service.LoginFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "loginFaultElem"),
				"org.benchit.bitconnect.service.LoginFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "loginFaultElem"),
				"org.benchit.bitconnect.service.types.LoginFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "registerFaultElem"),
				"org.benchit.bitconnect.service.RegisterFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "registerFaultElem"),
				"org.benchit.bitconnect.service.RegisterFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "registerFaultElem"),
				"org.benchit.bitconnect.service.types.RegisterFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "uploadFaultElem"),
				"org.benchit.bitconnect.service.UploadFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "uploadFaultElem"),
				"org.benchit.bitconnect.service.UploadFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "uploadFaultElem"),
				"org.benchit.bitconnect.service.types.UploadFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.SessionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "sessionFaultElem"),
				"org.benchit.bitconnect.service.types.SessionFaultElem");

		faultExceptionNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultExceptionClassNameMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.PermissionFault");
		faultMessageMap.put(new javax.xml.namespace.QName(
				"http://benchit.org/bitconnect/service/types/", "permissionFaultElem"),
				"org.benchit.bitconnect.service.types.PermissionFaultElem");

	}

	/**
	 * Constructor that takes in a configContext
	 */

	public BITConnectStub(org.apache.axis2.context.ConfigurationContext configurationContext,
			java.lang.String targetEndpoint) throws org.apache.axis2.AxisFault {
		this(configurationContext, targetEndpoint, false);
	}

	/**
	 * Constructor that takes in a configContext and useseperate listner
	 */
	public BITConnectStub(org.apache.axis2.context.ConfigurationContext configurationContext,
			java.lang.String targetEndpoint, boolean useSeparateListener)
			throws org.apache.axis2.AxisFault {
		// To populate AxisService
		populateAxisService();
		populateFaults();

		_serviceClient = new org.apache.axis2.client.ServiceClient(configurationContext, _service);

		configurationContext = _serviceClient.getServiceContext().getConfigurationContext();

		_serviceClient.getOptions().setTo(
				new org.apache.axis2.addressing.EndpointReference(targetEndpoint));
		_serviceClient.getOptions().setUseSeparateListener(useSeparateListener);

	}

	/**
	 * Default Constructor
	 */
	public BITConnectStub(org.apache.axis2.context.ConfigurationContext configurationContext)
			throws org.apache.axis2.AxisFault {

		this(configurationContext, "https://www.benchit.org/lib/service/service.php");

	}

	/**
	 * Default Constructor
	 */
	public BITConnectStub() throws org.apache.axis2.AxisFault {

		this("https://www.benchit.org/lib/service/service.php");

	}

	/**
	 * Constructor taking the target endpoint
	 */
	public BITConnectStub(java.lang.String targetEndpoint) throws org.apache.axis2.AxisFault {
		this(null, targetEndpoint);
	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#listBITIdentifiers
	 * @param listBITIdentifiers54
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITIdentifierListType listBITIdentifiers(

	java.lang.String param55)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[0].getName());
			_operationClient.getOptions().setAction(
					"http://benchit.org/bitconnect/service/listBITIdentifiers");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.ListBITIdentifiers dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), param55,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "listBITIdentifiers")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.ListBITIdentifiersResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getListBITIdentifiersResponseIdentifierList((org.benchit.bitconnect.service.ListBITIdentifiersResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startlistBITIdentifiers
	 * @param listBITIdentifiers54
	 */
	public void startlistBITIdentifiers(

	java.lang.String param55,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[0].getName());
		_operationClient.getOptions().setAction(
				"http://benchit.org/bitconnect/service/listBITIdentifiers");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.ListBITIdentifiers dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), param55,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "listBITIdentifiers")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.ListBITIdentifiersResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultlistBITIdentifiers(getListBITIdentifiersResponseIdentifierList((org.benchit.bitconnect.service.ListBITIdentifiersResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorlistBITIdentifiers(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErrorlistBITIdentifiers((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								callback.receiveErrorlistBITIdentifiers(new java.rmi.RemoteException(ex
										.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITIdentifiers(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITIdentifiers(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITIdentifiers(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITIdentifiers(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITIdentifiers(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITIdentifiers(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITIdentifiers(f);
							}
						} else {
							callback.receiveErrorlistBITIdentifiers(f);
						}
					} else {
						callback.receiveErrorlistBITIdentifiers(f);
					}
				} else {
					callback.receiveErrorlistBITIdentifiers(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorlistBITIdentifiers(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[0].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[0].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#getBITFileUploadStatus
	 * @param getBITFileUploadStatus58
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public org.benchit.bitconnect.service.types.BITFileUploadStatusType getBITFileUploadStatus(

	java.lang.String filehash59)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[1].getName());
			_operationClient.getOptions().setAction(
					"http://benchit.org/bitconnect/service/getBITFileUploadStatus");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.GetBITFileUploadStatus dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filehash59,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "getBITFileUploadStatus")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.GetBITFileUploadStatusResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getGetBITFileUploadStatusResponseStatus((org.benchit.bitconnect.service.GetBITFileUploadStatusResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
							throw (org.benchit.bitconnect.service.PermissionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startgetBITFileUploadStatus
	 * @param getBITFileUploadStatus58
	 */
	public void startgetBITFileUploadStatus(

	java.lang.String filehash59,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[1].getName());
		_operationClient.getOptions().setAction(
				"http://benchit.org/bitconnect/service/getBITFileUploadStatus");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.GetBITFileUploadStatus dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filehash59,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "getBITFileUploadStatus")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.GetBITFileUploadStatusResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultgetBITFileUploadStatus(getGetBITFileUploadStatusResponseStatus((org.benchit.bitconnect.service.GetBITFileUploadStatusResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorgetBITFileUploadStatus(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErrorgetBITFileUploadStatus((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
									callback
											.receiveErrorgetBITFileUploadStatus((org.benchit.bitconnect.service.PermissionFault) ex);
									return;
								}

								callback.receiveErrorgetBITFileUploadStatus(new java.rmi.RemoteException(ex
										.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileUploadStatus(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileUploadStatus(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileUploadStatus(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileUploadStatus(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileUploadStatus(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileUploadStatus(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileUploadStatus(f);
							}
						} else {
							callback.receiveErrorgetBITFileUploadStatus(f);
						}
					} else {
						callback.receiveErrorgetBITFileUploadStatus(f);
					}
				} else {
					callback.receiveErrorgetBITFileUploadStatus(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorgetBITFileUploadStatus(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[1].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[1].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#deleteBITFile
	 * @param deleteBITFile62
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public boolean deleteBITFile(

	java.lang.String filename63)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[2].getName());
			_operationClient.getOptions()
					.setAction("http://benchit.org/bitconnect/service/deleteBITFile");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.DeleteBITFile dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filename63,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "deleteBITFile")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.DeleteBITFileResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getDeleteBITFileResponseResponse((org.benchit.bitconnect.service.DeleteBITFileResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
							throw (org.benchit.bitconnect.service.PermissionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startdeleteBITFile
	 * @param deleteBITFile62
	 */
	public void startdeleteBITFile(

	java.lang.String filename63,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[2].getName());
		_operationClient.getOptions().setAction("http://benchit.org/bitconnect/service/deleteBITFile");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.DeleteBITFile dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filename63,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "deleteBITFile")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.DeleteBITFileResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultdeleteBITFile(getDeleteBITFileResponseResponse((org.benchit.bitconnect.service.DeleteBITFileResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrordeleteBITFile(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErrordeleteBITFile((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
									callback
											.receiveErrordeleteBITFile((org.benchit.bitconnect.service.PermissionFault) ex);
									return;
								}

								callback.receiveErrordeleteBITFile(new java.rmi.RemoteException(ex.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrordeleteBITFile(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrordeleteBITFile(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrordeleteBITFile(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrordeleteBITFile(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrordeleteBITFile(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrordeleteBITFile(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrordeleteBITFile(f);
							}
						} else {
							callback.receiveErrordeleteBITFile(f);
						}
					} else {
						callback.receiveErrordeleteBITFile(f);
					}
				} else {
					callback.receiveErrordeleteBITFile(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrordeleteBITFile(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[2].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[2].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#getBITFile
	 * @param getBITFile66
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public org.benchit.bitconnect.service.types.BITFileType getBITFile(

	java.lang.String filename67)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[3].getName());
			_operationClient.getOptions().setAction("http://benchit.org/bitconnect/service/getBITFile");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.GetBITFile dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filename67,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "getBITFile")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.GetBITFileResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getGetBITFileResponseFile((org.benchit.bitconnect.service.GetBITFileResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
							throw (org.benchit.bitconnect.service.PermissionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startgetBITFile
	 * @param getBITFile66
	 */
	public void startgetBITFile(

	java.lang.String filename67,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[3].getName());
		_operationClient.getOptions().setAction("http://benchit.org/bitconnect/service/getBITFile");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.GetBITFile dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filename67,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "getBITFile")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.GetBITFileResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultgetBITFile(getGetBITFileResponseFile((org.benchit.bitconnect.service.GetBITFileResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorgetBITFile(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback.receiveErrorgetBITFile((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
									callback
											.receiveErrorgetBITFile((org.benchit.bitconnect.service.PermissionFault) ex);
									return;
								}

								callback.receiveErrorgetBITFile(new java.rmi.RemoteException(ex.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFile(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFile(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFile(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFile(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFile(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFile(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFile(f);
							}
						} else {
							callback.receiveErrorgetBITFile(f);
						}
					} else {
						callback.receiveErrorgetBITFile(f);
					}
				} else {
					callback.receiveErrorgetBITFile(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorgetBITFile(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[3].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[3].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#listBITItemsByConstrain
	 * @param listBITItemsByConstrain70
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITItemListType listBITItemsByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints71)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[4].getName());
			_operationClient.getOptions().setAction(
					"http://benchit.org/bitconnect/service/listBITItemsByConstrain");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.ListBITItemsByConstrain dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()),
					constraints71, dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "listBITItemsByConstrain")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.ListBITItemsByConstrainResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getListBITItemsByConstrainResponseItemList((org.benchit.bitconnect.service.ListBITItemsByConstrainResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startlistBITItemsByConstrain
	 * @param listBITItemsByConstrain70
	 */
	public void startlistBITItemsByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints71,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[4].getName());
		_operationClient.getOptions().setAction(
				"http://benchit.org/bitconnect/service/listBITItemsByConstrain");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.ListBITItemsByConstrain dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), constraints71,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "listBITItemsByConstrain")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.ListBITItemsByConstrainResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultlistBITItemsByConstrain(getListBITItemsByConstrainResponseItemList((org.benchit.bitconnect.service.ListBITItemsByConstrainResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorlistBITItemsByConstrain(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErrorlistBITItemsByConstrain((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								callback.receiveErrorlistBITItemsByConstrain(new java.rmi.RemoteException(ex
										.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItemsByConstrain(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItemsByConstrain(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItemsByConstrain(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItemsByConstrain(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItemsByConstrain(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItemsByConstrain(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItemsByConstrain(f);
							}
						} else {
							callback.receiveErrorlistBITItemsByConstrain(f);
						}
					} else {
						callback.receiveErrorlistBITItemsByConstrain(f);
					}
				} else {
					callback.receiveErrorlistBITItemsByConstrain(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorlistBITItemsByConstrain(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[4].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[4].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#loginBITAccount
	 * @param loginBITAccount74
	 * @throws org.benchit.bitconnect.service.LoginFault :
	 */

	public org.benchit.bitconnect.service.types.BITUserType loginBITAccount(

	java.lang.String login75, java.lang.String password76)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.LoginFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[5].getName());
			_operationClient.getOptions().setAction(
					"http://benchit.org/bitconnect/service/loginBITAccount");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.LoginBITAccount dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), login75,
					password76, dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "loginBITAccount")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.LoginBITAccountResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getLoginBITAccountResponseResponse((org.benchit.bitconnect.service.LoginBITAccountResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.LoginFault) {
							throw (org.benchit.bitconnect.service.LoginFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startloginBITAccount
	 * @param loginBITAccount74
	 */
	public void startloginBITAccount(

	java.lang.String login75, java.lang.String password76,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[5].getName());
		_operationClient.getOptions()
				.setAction("http://benchit.org/bitconnect/service/loginBITAccount");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.LoginBITAccount dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), login75,
				password76, dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "loginBITAccount")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.LoginBITAccountResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultloginBITAccount(getLoginBITAccountResponseResponse((org.benchit.bitconnect.service.LoginBITAccountResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorloginBITAccount(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.LoginFault) {
									callback
											.receiveErrorloginBITAccount((org.benchit.bitconnect.service.LoginFault) ex);
									return;
								}

								callback.receiveErrorloginBITAccount(new java.rmi.RemoteException(ex.getMessage(),
										ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorloginBITAccount(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorloginBITAccount(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorloginBITAccount(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorloginBITAccount(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorloginBITAccount(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorloginBITAccount(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorloginBITAccount(f);
							}
						} else {
							callback.receiveErrorloginBITAccount(f);
						}
					} else {
						callback.receiveErrorloginBITAccount(f);
					}
				} else {
					callback.receiveErrorloginBITAccount(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorloginBITAccount(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[5].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[5].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#listBITItems
	 * @param listBITItems79
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITItemListType listBITItems(

	java.lang.String param80)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[6].getName());
			_operationClient.getOptions().setAction("http://benchit.org/bitconnect/service/listBITItems");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.ListBITItems dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), param80,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "listBITItems")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.ListBITItemsResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getListBITItemsResponseItemList((org.benchit.bitconnect.service.ListBITItemsResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startlistBITItems
	 * @param listBITItems79
	 */
	public void startlistBITItems(

	java.lang.String param80,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[6].getName());
		_operationClient.getOptions().setAction("http://benchit.org/bitconnect/service/listBITItems");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.ListBITItems dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), param80,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "listBITItems")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.ListBITItemsResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultlistBITItems(getListBITItemsResponseItemList((org.benchit.bitconnect.service.ListBITItemsResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorlistBITItems(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErrorlistBITItems((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								callback.receiveErrorlistBITItems(new java.rmi.RemoteException(ex.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItems(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItems(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItems(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItems(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItems(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItems(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITItems(f);
							}
						} else {
							callback.receiveErrorlistBITItems(f);
						}
					} else {
						callback.receiveErrorlistBITItems(f);
					}
				} else {
					callback.receiveErrorlistBITItems(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorlistBITItems(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[6].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[6].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#registerBITAccount
	 * @param registerBITAccount83
	 * @throws org.benchit.bitconnect.service.RegisterFault :
	 */

	public boolean registerBITAccount(

	org.benchit.bitconnect.service.types.BITUserType user84)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.RegisterFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[7].getName());
			_operationClient.getOptions().setAction(
					"http://benchit.org/bitconnect/service/registerBITAccount");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.RegisterBITAccount dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), user84,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "registerBITAccount")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.RegisterBITAccountResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getRegisterBITAccountResponseResponse((org.benchit.bitconnect.service.RegisterBITAccountResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.RegisterFault) {
							throw (org.benchit.bitconnect.service.RegisterFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startregisterBITAccount
	 * @param registerBITAccount83
	 */
	public void startregisterBITAccount(

	org.benchit.bitconnect.service.types.BITUserType user84,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[7].getName());
		_operationClient.getOptions().setAction(
				"http://benchit.org/bitconnect/service/registerBITAccount");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.RegisterBITAccount dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), user84,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "registerBITAccount")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.RegisterBITAccountResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultregisterBITAccount(getRegisterBITAccountResponseResponse((org.benchit.bitconnect.service.RegisterBITAccountResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorregisterBITAccount(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.RegisterFault) {
									callback
											.receiveErrorregisterBITAccount((org.benchit.bitconnect.service.RegisterFault) ex);
									return;
								}

								callback.receiveErrorregisterBITAccount(new java.rmi.RemoteException(ex
										.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorregisterBITAccount(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorregisterBITAccount(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorregisterBITAccount(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorregisterBITAccount(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorregisterBITAccount(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorregisterBITAccount(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorregisterBITAccount(f);
							}
						} else {
							callback.receiveErrorregisterBITAccount(f);
						}
					} else {
						callback.receiveErrorregisterBITAccount(f);
					}
				} else {
					callback.receiveErrorregisterBITAccount(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorregisterBITAccount(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[7].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[7].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#listBITFilesByConstrain
	 * @param listBITFilesByConstrain87
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public org.benchit.bitconnect.service.types.BITFileListType listBITFilesByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints88)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[8].getName());
			_operationClient.getOptions().setAction(
					"http://benchit.org/bitconnect/service/listBITFilesByConstrain");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.ListBITFilesByConstrain dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()),
					constraints88, dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "listBITFilesByConstrain")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.ListBITFilesByConstrainResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getListBITFilesByConstrainResponseFileList((org.benchit.bitconnect.service.ListBITFilesByConstrainResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startlistBITFilesByConstrain
	 * @param listBITFilesByConstrain87
	 */
	public void startlistBITFilesByConstrain(

	org.benchit.bitconnect.service.types.BITItemListType constraints88,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[8].getName());
		_operationClient.getOptions().setAction(
				"http://benchit.org/bitconnect/service/listBITFilesByConstrain");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.ListBITFilesByConstrain dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), constraints88,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "listBITFilesByConstrain")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.ListBITFilesByConstrainResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultlistBITFilesByConstrain(getListBITFilesByConstrainResponseFileList((org.benchit.bitconnect.service.ListBITFilesByConstrainResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorlistBITFilesByConstrain(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErrorlistBITFilesByConstrain((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								callback.receiveErrorlistBITFilesByConstrain(new java.rmi.RemoteException(ex
										.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITFilesByConstrain(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITFilesByConstrain(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITFilesByConstrain(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITFilesByConstrain(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITFilesByConstrain(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITFilesByConstrain(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorlistBITFilesByConstrain(f);
							}
						} else {
							callback.receiveErrorlistBITFilesByConstrain(f);
						}
					} else {
						callback.receiveErrorlistBITFilesByConstrain(f);
					}
				} else {
					callback.receiveErrorlistBITFilesByConstrain(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorlistBITFilesByConstrain(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[8].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[8].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#uploadBITFile
	 * @param uploadBITFile91
	 * @throws org.benchit.bitconnect.service.UploadFault :
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 */

	public boolean uploadBITFile(

	org.benchit.bitconnect.service.types.BITFileType file92)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.UploadFault, org.benchit.bitconnect.service.SessionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[9].getName());
			_operationClient.getOptions()
					.setAction("http://benchit.org/bitconnect/service/uploadBITFile");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.UploadBITFile dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), file92,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "uploadBITFile")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.UploadBITFileResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getUploadBITFileResponseResponse((org.benchit.bitconnect.service.UploadBITFileResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.UploadFault) {
							throw (org.benchit.bitconnect.service.UploadFault) ex;
						}

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startuploadBITFile
	 * @param uploadBITFile91
	 */
	public void startuploadBITFile(

	org.benchit.bitconnect.service.types.BITFileType file92,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[9].getName());
		_operationClient.getOptions().setAction("http://benchit.org/bitconnect/service/uploadBITFile");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.UploadBITFile dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), file92,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "uploadBITFile")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.UploadBITFileResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultuploadBITFile(getUploadBITFileResponseResponse((org.benchit.bitconnect.service.UploadBITFileResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErroruploadBITFile(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.UploadFault) {
									callback
											.receiveErroruploadBITFile((org.benchit.bitconnect.service.UploadFault) ex);
									return;
								}

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErroruploadBITFile((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								callback.receiveErroruploadBITFile(new java.rmi.RemoteException(ex.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErroruploadBITFile(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErroruploadBITFile(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErroruploadBITFile(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErroruploadBITFile(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErroruploadBITFile(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErroruploadBITFile(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErroruploadBITFile(f);
							}
						} else {
							callback.receiveErroruploadBITFile(f);
						}
					} else {
						callback.receiveErroruploadBITFile(f);
					}
				} else {
					callback.receiveErroruploadBITFile(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErroruploadBITFile(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[9].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[9].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * Auto generated method signature
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#getBITFileMetadata
	 * @param getBITFileMetadata95
	 * @throws org.benchit.bitconnect.service.SessionFault :
	 * @throws org.benchit.bitconnect.service.PermissionFault :
	 */

	public org.benchit.bitconnect.service.types.BITItemListType getBITFileMetadata(

	java.lang.String filename96)

	throws java.rmi.RemoteException

	, org.benchit.bitconnect.service.SessionFault, org.benchit.bitconnect.service.PermissionFault {
		org.apache.axis2.context.MessageContext _messageContext = null;
		try {
			org.apache.axis2.client.OperationClient _operationClient = _serviceClient
					.createClient(_operations[10].getName());
			_operationClient.getOptions().setAction(
					"http://benchit.org/bitconnect/service/getBITFileMetadata");
			_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

			addPropertyToOperationClient(_operationClient,
					org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

			// create a message context
			_messageContext = new org.apache.axis2.context.MessageContext();

			// create SOAP envelope with that payload
			org.apache.axiom.soap.SOAPEnvelope env = null;
			org.benchit.bitconnect.service.GetBITFileMetadata dummyWrappedType = null;
			env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filename96,
					dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
							"http://benchit.org/bitconnect/service/", "getBITFileMetadata")));

			// adding SOAP soap_headers
			_serviceClient.addHeadersToEnvelope(env);
			// set the message context with that soap envelope
			_messageContext.setEnvelope(env);

			// add the message contxt to the operation client
			_operationClient.addMessageContext(_messageContext);

			// execute the operation client
			_operationClient.execute(true);

			org.apache.axis2.context.MessageContext _returnMessageContext = _operationClient
					.getMessageContext(org.apache.axis2.wsdl.WSDLConstants.MESSAGE_LABEL_IN_VALUE);
			org.apache.axiom.soap.SOAPEnvelope _returnEnv = _returnMessageContext.getEnvelope();

			java.lang.Object object = fromOM(_returnEnv.getBody().getFirstElement(),
					org.benchit.bitconnect.service.GetBITFileMetadataResponse.class,
					getEnvelopeNamespaces(_returnEnv));

			return getGetBITFileMetadataResponseFiledata((org.benchit.bitconnect.service.GetBITFileMetadataResponse) object);

		} catch (org.apache.axis2.AxisFault f) {

			org.apache.axiom.om.OMElement faultElt = f.getDetail();
			if (faultElt != null) {
				if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
					// make the fault by reflection
					try {
						java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
								.get(faultElt.getQName());
						java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
						java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
						// message class
						java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
								.getQName());
						java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
						java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
						java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
								new java.lang.Class[]{messageClass});
						m.invoke(ex, new java.lang.Object[]{messageObject});

						if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
							throw (org.benchit.bitconnect.service.SessionFault) ex;
						}

						if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
							throw (org.benchit.bitconnect.service.PermissionFault) ex;
						}

						throw new java.rmi.RemoteException(ex.getMessage(), ex);
					} catch (java.lang.ClassCastException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.ClassNotFoundException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.NoSuchMethodException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.reflect.InvocationTargetException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.IllegalAccessException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					} catch (java.lang.InstantiationException e) {
						// we cannot intantiate the class - throw the original Axis fault
						throw f;
					}
				} else {
					throw f;
				}
			} else {
				throw f;
			}
		} finally {
			_messageContext.getTransportOut().getSender().cleanup(_messageContext);
		}
	}

	/**
	 * Auto generated method signature for Asynchronous Invocations
	 * 
	 * @see org.benchit.bitconnect.service.BITConnect#startgetBITFileMetadata
	 * @param getBITFileMetadata95
	 */
	public void startgetBITFileMetadata(

	java.lang.String filename96,

	final org.benchit.bitconnect.service.BITConnectCallbackHandler callback)

	throws java.rmi.RemoteException {

		org.apache.axis2.client.OperationClient _operationClient = _serviceClient
				.createClient(_operations[10].getName());
		_operationClient.getOptions().setAction(
				"http://benchit.org/bitconnect/service/getBITFileMetadata");
		_operationClient.getOptions().setExceptionToBeThrownOnSOAPFault(true);

		addPropertyToOperationClient(_operationClient,
				org.apache.axis2.description.WSDL2Constants.ATTR_WHTTP_QUERY_PARAMETER_SEPARATOR, "&");

		// create SOAP envelope with that payload
		org.apache.axiom.soap.SOAPEnvelope env = null;
		final org.apache.axis2.context.MessageContext _messageContext = new org.apache.axis2.context.MessageContext();

		// Style is Doc.
		org.benchit.bitconnect.service.GetBITFileMetadata dummyWrappedType = null;
		env = toEnvelope(getFactory(_operationClient.getOptions().getSoapVersionURI()), filename96,
				dummyWrappedType, optimizeContent(new javax.xml.namespace.QName(
						"http://benchit.org/bitconnect/service/", "getBITFileMetadata")));

		// adding SOAP soap_headers
		_serviceClient.addHeadersToEnvelope(env);
		// create message context with that soap envelope
		_messageContext.setEnvelope(env);

		// add the message context to the operation client
		_operationClient.addMessageContext(_messageContext);

		_operationClient.setCallback(new org.apache.axis2.client.async.AxisCallback() {
			public void onMessage(org.apache.axis2.context.MessageContext resultContext) {
				try {
					org.apache.axiom.soap.SOAPEnvelope resultEnv = resultContext.getEnvelope();

					java.lang.Object object = fromOM(resultEnv.getBody().getFirstElement(),
							org.benchit.bitconnect.service.GetBITFileMetadataResponse.class,
							getEnvelopeNamespaces(resultEnv));
					callback
							.receiveResultgetBITFileMetadata(getGetBITFileMetadataResponseFiledata((org.benchit.bitconnect.service.GetBITFileMetadataResponse) object));

				} catch (org.apache.axis2.AxisFault e) {
					callback.receiveErrorgetBITFileMetadata(e);
				}
			}

			public void onError(java.lang.Exception error) {
				if (error instanceof org.apache.axis2.AxisFault) {
					org.apache.axis2.AxisFault f = (org.apache.axis2.AxisFault) error;
					org.apache.axiom.om.OMElement faultElt = f.getDetail();
					if (faultElt != null) {
						if (faultExceptionNameMap.containsKey(faultElt.getQName())) {
							// make the fault by reflection
							try {
								java.lang.String exceptionClassName = (java.lang.String) faultExceptionClassNameMap
										.get(faultElt.getQName());
								java.lang.Class exceptionClass = java.lang.Class.forName(exceptionClassName);
								java.lang.Exception ex = (java.lang.Exception) exceptionClass.newInstance();
								// message class
								java.lang.String messageClassName = (java.lang.String) faultMessageMap.get(faultElt
										.getQName());
								java.lang.Class messageClass = java.lang.Class.forName(messageClassName);
								java.lang.Object messageObject = fromOM(faultElt, messageClass, null);
								java.lang.reflect.Method m = exceptionClass.getMethod("setFaultMessage",
										new java.lang.Class[]{messageClass});
								m.invoke(ex, new java.lang.Object[]{messageObject});

								if (ex instanceof org.benchit.bitconnect.service.SessionFault) {
									callback
											.receiveErrorgetBITFileMetadata((org.benchit.bitconnect.service.SessionFault) ex);
									return;
								}

								if (ex instanceof org.benchit.bitconnect.service.PermissionFault) {
									callback
											.receiveErrorgetBITFileMetadata((org.benchit.bitconnect.service.PermissionFault) ex);
									return;
								}

								callback.receiveErrorgetBITFileMetadata(new java.rmi.RemoteException(ex
										.getMessage(), ex));
							} catch (java.lang.ClassCastException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileMetadata(f);
							} catch (java.lang.ClassNotFoundException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileMetadata(f);
							} catch (java.lang.NoSuchMethodException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileMetadata(f);
							} catch (java.lang.reflect.InvocationTargetException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileMetadata(f);
							} catch (java.lang.IllegalAccessException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileMetadata(f);
							} catch (java.lang.InstantiationException e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileMetadata(f);
							} catch (org.apache.axis2.AxisFault e) {
								// we cannot intantiate the class - throw the original Axis fault
								callback.receiveErrorgetBITFileMetadata(f);
							}
						} else {
							callback.receiveErrorgetBITFileMetadata(f);
						}
					} else {
						callback.receiveErrorgetBITFileMetadata(f);
					}
				} else {
					callback.receiveErrorgetBITFileMetadata(error);
				}
			}

			public void onFault(org.apache.axis2.context.MessageContext faultContext) {
				org.apache.axis2.AxisFault fault = org.apache.axis2.util.Utils
						.getInboundFaultFromMessageContext(faultContext);
				onError(fault);
			}

			public void onComplete() {
				try {
					_messageContext.getTransportOut().getSender().cleanup(_messageContext);
				} catch (org.apache.axis2.AxisFault axisFault) {
					callback.receiveErrorgetBITFileMetadata(axisFault);
				}
			}
		});

		org.apache.axis2.util.CallbackReceiver _callbackReceiver = null;
		if (_operations[10].getMessageReceiver() == null
				&& _operationClient.getOptions().isUseSeparateListener()) {
			_callbackReceiver = new org.apache.axis2.util.CallbackReceiver();
			_operations[10].setMessageReceiver(_callbackReceiver);
		}

		// execute the operation client
		_operationClient.execute(false);

	}

	/**
	 * A utility method that copies the namepaces from the SOAPEnvelope
	 */
	private java.util.Map getEnvelopeNamespaces(org.apache.axiom.soap.SOAPEnvelope env) {
		java.util.Map returnMap = new java.util.HashMap();
		java.util.Iterator namespaceIterator = env.getAllDeclaredNamespaces();
		while (namespaceIterator.hasNext()) {
			org.apache.axiom.om.OMNamespace ns = (org.apache.axiom.om.OMNamespace) namespaceIterator
					.next();
			returnMap.put(ns.getPrefix(), ns.getNamespaceURI());
		}
		return returnMap;
	}

	private javax.xml.namespace.QName[] opNameArray = null;
	private boolean optimizeContent(javax.xml.namespace.QName opName) {

		if (opNameArray == null) {
			return false;
		}
		for (int i = 0; i < opNameArray.length; i++) {
			if (opName.equals(opNameArray[i])) {
				return true;
			}
		}
		return false;
	}
	// https://www.benchit.org/lib/service/service.php
	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.ListBITIdentifiers param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.ListBITIdentifiers.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.ListBITIdentifiersResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.ListBITIdentifiersResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.types.SessionFaultElem param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.types.SessionFaultElem.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.GetBITFileUploadStatus param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.GetBITFileUploadStatus.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.GetBITFileUploadStatusResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(
					org.benchit.bitconnect.service.GetBITFileUploadStatusResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.types.PermissionFaultElem param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.types.PermissionFaultElem.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(org.benchit.bitconnect.service.DeleteBITFile param,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.DeleteBITFile.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.DeleteBITFileResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.DeleteBITFileResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(org.benchit.bitconnect.service.GetBITFile param,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.GetBITFile.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.GetBITFileResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.GetBITFileResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.ListBITItemsByConstrain param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.ListBITItemsByConstrain.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.ListBITItemsByConstrainResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(
					org.benchit.bitconnect.service.ListBITItemsByConstrainResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(org.benchit.bitconnect.service.LoginBITAccount param,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.LoginBITAccount.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.LoginBITAccountResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.LoginBITAccountResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.types.LoginFaultElem param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.types.LoginFaultElem.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(org.benchit.bitconnect.service.ListBITItems param,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.ListBITItems.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.ListBITItemsResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.ListBITItemsResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.RegisterBITAccount param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.RegisterBITAccount.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.RegisterBITAccountResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.RegisterBITAccountResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.types.RegisterFaultElem param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.types.RegisterFaultElem.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.ListBITFilesByConstrain param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.ListBITFilesByConstrain.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.ListBITFilesByConstrainResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(
					org.benchit.bitconnect.service.ListBITFilesByConstrainResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(org.benchit.bitconnect.service.UploadBITFile param,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.UploadBITFile.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.UploadBITFileResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.UploadBITFileResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.types.UploadFaultElem param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.types.UploadFaultElem.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.GetBITFileMetadata param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.GetBITFileMetadata.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.om.OMElement toOM(
			org.benchit.bitconnect.service.GetBITFileMetadataResponse param, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			return param.getOMElement(org.benchit.bitconnect.service.GetBITFileMetadataResponse.MY_QNAME,
					org.apache.axiom.om.OMAbstractFactory.getOMFactory());
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			java.lang.String param1, org.benchit.bitconnect.service.ListBITIdentifiers dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.ListBITIdentifiers wrappedType = new org.benchit.bitconnect.service.ListBITIdentifiers();

			wrappedType.setParam(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.ListBITIdentifiers.MY_QNAME,
							factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITIdentifierListType getListBITIdentifiersResponseIdentifierList(
			org.benchit.bitconnect.service.ListBITIdentifiersResponse wrappedType) {

		return wrappedType.getIdentifierList();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			java.lang.String param1,
			org.benchit.bitconnect.service.GetBITFileUploadStatus dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.GetBITFileUploadStatus wrappedType = new org.benchit.bitconnect.service.GetBITFileUploadStatus();

			wrappedType.setFilehash(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.GetBITFileUploadStatus.MY_QNAME,
							factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITFileUploadStatusType getGetBITFileUploadStatusResponseStatus(
			org.benchit.bitconnect.service.GetBITFileUploadStatusResponse wrappedType) {

		return wrappedType.getStatus();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			java.lang.String param1, org.benchit.bitconnect.service.DeleteBITFile dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.DeleteBITFile wrappedType = new org.benchit.bitconnect.service.DeleteBITFile();

			wrappedType.setFilename(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.DeleteBITFile.MY_QNAME, factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private boolean getDeleteBITFileResponseResponse(
			org.benchit.bitconnect.service.DeleteBITFileResponse wrappedType) {

		return wrappedType.getResponse();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			java.lang.String param1, org.benchit.bitconnect.service.GetBITFile dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.GetBITFile wrappedType = new org.benchit.bitconnect.service.GetBITFile();

			wrappedType.setFilename(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.GetBITFile.MY_QNAME, factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITFileType getGetBITFileResponseFile(
			org.benchit.bitconnect.service.GetBITFileResponse wrappedType) {

		return wrappedType.getFile();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			org.benchit.bitconnect.service.types.BITItemListType param1,
			org.benchit.bitconnect.service.ListBITItemsByConstrain dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.ListBITItemsByConstrain wrappedType = new org.benchit.bitconnect.service.ListBITItemsByConstrain();

			wrappedType.setConstraints(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.ListBITItemsByConstrain.MY_QNAME,
							factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITItemListType getListBITItemsByConstrainResponseItemList(
			org.benchit.bitconnect.service.ListBITItemsByConstrainResponse wrappedType) {

		return wrappedType.getItemList();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			java.lang.String param1, java.lang.String param2,
			org.benchit.bitconnect.service.LoginBITAccount dummyWrappedType, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.LoginBITAccount wrappedType = new org.benchit.bitconnect.service.LoginBITAccount();

			wrappedType.setLogin(param1);

			wrappedType.setPassword(param2);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType
							.getOMElement(org.benchit.bitconnect.service.LoginBITAccount.MY_QNAME, factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITUserType getLoginBITAccountResponseResponse(
			org.benchit.bitconnect.service.LoginBITAccountResponse wrappedType) {

		return wrappedType.getResponse();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			java.lang.String param1, org.benchit.bitconnect.service.ListBITItems dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.ListBITItems wrappedType = new org.benchit.bitconnect.service.ListBITItems();

			wrappedType.setParam(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.ListBITItems.MY_QNAME, factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITItemListType getListBITItemsResponseItemList(
			org.benchit.bitconnect.service.ListBITItemsResponse wrappedType) {

		return wrappedType.getItemList();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			org.benchit.bitconnect.service.types.BITUserType param1,
			org.benchit.bitconnect.service.RegisterBITAccount dummyWrappedType, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.RegisterBITAccount wrappedType = new org.benchit.bitconnect.service.RegisterBITAccount();

			wrappedType.setUser(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.RegisterBITAccount.MY_QNAME,
							factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private boolean getRegisterBITAccountResponseResponse(
			org.benchit.bitconnect.service.RegisterBITAccountResponse wrappedType) {

		return wrappedType.getResponse();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			org.benchit.bitconnect.service.types.BITItemListType param1,
			org.benchit.bitconnect.service.ListBITFilesByConstrain dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.ListBITFilesByConstrain wrappedType = new org.benchit.bitconnect.service.ListBITFilesByConstrain();

			wrappedType.setConstraints(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.ListBITFilesByConstrain.MY_QNAME,
							factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITFileListType getListBITFilesByConstrainResponseFileList(
			org.benchit.bitconnect.service.ListBITFilesByConstrainResponse wrappedType) {

		return wrappedType.getFileList();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			org.benchit.bitconnect.service.types.BITFileType param1,
			org.benchit.bitconnect.service.UploadBITFile dummyWrappedType, boolean optimizeContent)
			throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.UploadBITFile wrappedType = new org.benchit.bitconnect.service.UploadBITFile();

			wrappedType.setFile(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.UploadBITFile.MY_QNAME, factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private boolean getUploadBITFileResponseResponse(
			org.benchit.bitconnect.service.UploadBITFileResponse wrappedType) {

		return wrappedType.getResponse();

	}

	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory,
			java.lang.String param1, org.benchit.bitconnect.service.GetBITFileMetadata dummyWrappedType,
			boolean optimizeContent) throws org.apache.axis2.AxisFault {

		try {
			org.benchit.bitconnect.service.GetBITFileMetadata wrappedType = new org.benchit.bitconnect.service.GetBITFileMetadata();

			wrappedType.setFilename(param1);

			org.apache.axiom.soap.SOAPEnvelope emptyEnvelope = factory.getDefaultEnvelope();

			emptyEnvelope.getBody().addChild(
					wrappedType.getOMElement(org.benchit.bitconnect.service.GetBITFileMetadata.MY_QNAME,
							factory));

			return emptyEnvelope;
		} catch (org.apache.axis2.databinding.ADBException e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
	}

	/* methods to provide back word compatibility */

	private org.benchit.bitconnect.service.types.BITItemListType getGetBITFileMetadataResponseFiledata(
			org.benchit.bitconnect.service.GetBITFileMetadataResponse wrappedType) {

		return wrappedType.getFiledata();

	}

	/**
	 * get the default envelope
	 */
	private org.apache.axiom.soap.SOAPEnvelope toEnvelope(org.apache.axiom.soap.SOAPFactory factory) {
		return factory.getDefaultEnvelope();
	}

	private java.lang.Object fromOM(org.apache.axiom.om.OMElement param, java.lang.Class type,
			java.util.Map extraNamespaces) throws org.apache.axis2.AxisFault {

		try {

			if (org.benchit.bitconnect.service.ListBITIdentifiers.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITIdentifiers.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.ListBITIdentifiersResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITIdentifiersResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.GetBITFileUploadStatus.class.equals(type)) {

				return org.benchit.bitconnect.service.GetBITFileUploadStatus.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.GetBITFileUploadStatusResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.GetBITFileUploadStatusResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.PermissionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.PermissionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.DeleteBITFile.class.equals(type)) {

				return org.benchit.bitconnect.service.DeleteBITFile.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.DeleteBITFileResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.DeleteBITFileResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.PermissionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.PermissionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.GetBITFile.class.equals(type)) {

				return org.benchit.bitconnect.service.GetBITFile.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.GetBITFileResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.GetBITFileResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.PermissionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.PermissionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.ListBITItemsByConstrain.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITItemsByConstrain.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.ListBITItemsByConstrainResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITItemsByConstrainResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.LoginBITAccount.class.equals(type)) {

				return org.benchit.bitconnect.service.LoginBITAccount.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.LoginBITAccountResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.LoginBITAccountResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.LoginFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.LoginFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.ListBITItems.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITItems.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.ListBITItemsResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITItemsResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.RegisterBITAccount.class.equals(type)) {

				return org.benchit.bitconnect.service.RegisterBITAccount.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.RegisterBITAccountResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.RegisterBITAccountResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.RegisterFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.RegisterFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.ListBITFilesByConstrain.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITFilesByConstrain.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.ListBITFilesByConstrainResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.ListBITFilesByConstrainResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.UploadBITFile.class.equals(type)) {

				return org.benchit.bitconnect.service.UploadBITFile.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.UploadBITFileResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.UploadBITFileResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.UploadFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.UploadFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.GetBITFileMetadata.class.equals(type)) {

				return org.benchit.bitconnect.service.GetBITFileMetadata.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.GetBITFileMetadataResponse.class.equals(type)) {

				return org.benchit.bitconnect.service.GetBITFileMetadataResponse.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.SessionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.SessionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

			if (org.benchit.bitconnect.service.types.PermissionFaultElem.class.equals(type)) {

				return org.benchit.bitconnect.service.types.PermissionFaultElem.Factory.parse(param
						.getXMLStreamReaderWithoutCaching());

			}

		} catch (java.lang.Exception e) {
			throw org.apache.axis2.AxisFault.makeFault(e);
		}
		return null;
	}

}
