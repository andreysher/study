/**
 * 
 */
package org.benchit.bitconnect;

import java.io.IOException;
import java.net.*;
import java.security.*;
import java.security.cert.CertificateException;

import javax.net.ssl.*;

import org.apache.commons.httpclient.ConnectTimeoutException;
import org.apache.commons.httpclient.params.HttpConnectionParams;
import org.apache.commons.httpclient.protocol.ProtocolSocketFactory;

/**
 * @author dreiche
 */
public class BITCertSocketFactory implements ProtocolSocketFactory {
	/*
	 * (non-Javadoc)
	 * 
	 * @see org.apache.commons.httpclient.protocol.SecureProtocolSocketFactory#createSocket(java.net.Socket ,
	 * java.lang.String, int, boolean)
	 */
	public Socket createSocket(Socket arg0, String arg1, int arg2, boolean arg3) throws IOException,
			UnknownHostException {
		return createSocket(arg1, arg2);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.apache.commons.httpclient.protocol.ProtocolSocketFactory#createSocket(java.lang.String, int)
	 */
	public Socket createSocket(String arg0, int arg1) throws IOException, UnknownHostException {

		SSLSocket theSocket = null;

		KeyStore keys;
		TrustManagerFactory tmf;
		SSLContext sc;

		try {
			keys = KeyStore.getInstance("jks");
			keys.load(ClassLoader.getSystemResourceAsStream("benchitkeys.jks"),
					"www.benchit.org".toCharArray());

			tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
			tmf.init(keys);

			sc = SSLContext.getInstance("SSL");
			sc.init(null, tmf.getTrustManagers(), null);

			SSLSocketFactory ssf = sc.getSocketFactory();
			theSocket = (SSLSocket) ssf.createSocket(arg0, arg1);
		} catch (KeyStoreException e) {
			e.printStackTrace();
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
		} catch (KeyManagementException e) {
			e.printStackTrace();
		} catch (CertificateException e) {
			e.printStackTrace();
		}

		return theSocket;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.apache.commons.httpclient.protocol.ProtocolSocketFactory#createSocket(java.lang.String, int,
	 * java.net.InetAddress, int)
	 */
	public Socket createSocket(String arg0, int arg1, InetAddress arg2, int arg3) throws IOException,
			UnknownHostException {
		return createSocket(arg0, arg1);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.apache.commons.httpclient.protocol.ProtocolSocketFactory#createSocket(java.lang.String, int,
	 * java.net.InetAddress, int, org.apache.commons.httpclient.params.HttpConnectionParams)
	 */
	public Socket createSocket(String arg0, int arg1, InetAddress arg2, int arg3,
			HttpConnectionParams arg4) throws IOException, UnknownHostException, ConnectTimeoutException {
		return createSocket(arg0, arg1);
	}

}
