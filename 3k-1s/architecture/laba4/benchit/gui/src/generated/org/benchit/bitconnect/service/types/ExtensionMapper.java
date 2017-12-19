/**
 * ExtensionMapper.java
 *
 * This file was auto-generated from WSDL
 * by the Apache Axis2 version: 1.4.1  Built on : Aug 13, 2008 (05:03:41 LKT)
 */

package org.benchit.bitconnect.service.types;
/**
 * ExtensionMapper class
 */

public class ExtensionMapper {

	public static java.lang.Object getTypeObject(java.lang.String namespaceURI,
			java.lang.String typeName, javax.xml.stream.XMLStreamReader reader)
			throws java.lang.Exception {

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultLoginFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultLoginFaultType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITItemType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITItemType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFileUploadStatusNoSuchFileType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFileUploadStatusNoSuchFileType.Factory
					.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultRegisterLoginFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultRegisterLoginFaultType.Factory
					.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITIdentifierListType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITIdentifierListType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFileUploadStatusSuccessType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFileUploadStatusSuccessType.Factory
					.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultUploadFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultUploadFaultType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFileUploadStatusType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFileUploadStatusType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITItemListType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITItemListType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITUserType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITUserType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultPermissionFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultPermissionFaultType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultSessionFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultSessionFaultType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultUploadDataFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultUploadDataFaultType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultUploadQuotaFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultUploadQuotaFaultType.Factory
					.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFileUploadStatusProcessingType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFileUploadStatusProcessingType.Factory
					.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultRegisterEmailFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultRegisterEmailFaultType.Factory
					.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFileType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFileType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFileUploadStatusFailedType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFileUploadStatusFailedType.Factory
					.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFileListType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFileListType.Factory.parse(reader);

		}

		if ("http://benchit.org/bitconnect/service/types/".equals(namespaceURI)
				&& "BITFaultRegisterFaultType".equals(typeName)) {

			return org.benchit.bitconnect.service.types.BITFaultRegisterFaultType.Factory.parse(reader);

		}

		throw new org.apache.axis2.databinding.ADBException("Unsupported type " + namespaceURI + " "
				+ typeName);
	}

}
