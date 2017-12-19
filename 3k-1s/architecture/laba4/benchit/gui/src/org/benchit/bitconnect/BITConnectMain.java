/**
 * @author Daniel Reiche d.reiche@gmx.ch
 */
package org.benchit.bitconnect;

import gui.*;

import java.io.*;
import java.rmi.RemoteException;
import java.util.*;

import javax.activation.*;
import javax.swing.JOptionPane;

import org.apache.axis2.AxisFault;
import org.apache.commons.httpclient.protocol.Protocol;
import org.benchit.bitconnect.gui.GUIControl;
import org.benchit.bitconnect.service.*;
import org.benchit.bitconnect.service.types.*;

import com.twmacinta.util.MD5;

/**
 * @author Daniel Reiche d.reiche@gmx.ch
 * @since October 2008
 */
final public class BITConnectMain {
	private static BITConnectMain instance;

	private final BIGGUI parent;
	private BIGGUIObserverProgress progressbar;
	private BITUserType currUser;
	private String login;
	private String pass;

	public static String debug;

	private BITConnectStub stub;

	/**
	 * Constructor. private due to singleton design pattern
	 * 
	 * @param parent
	 */
	private BITConnectMain(BIGGUI parent) {
		this.parent = parent;
		init();
	}

	/**
	 * Initialize the web service client and GUI elements
	 */
	private void init() {

		// enable logging if the system property is set. will cause _all_ communication to be written to
		// log
		if (System.getProperty("org.benchit.bitconnect.Debug", "false").equals("true")) {
			BITConnectMain.debug = "true";
		} else {
			BITConnectMain.debug = "false";
		}

		// switch on transport layer debug output
		if (debug.equals("true")) {
			System.setProperty("org.apache.commons.logging.Log",
					"org.apache.commons.logging.impl.SimpleLog");
			System.setProperty("org.apache.commons.logging.simplelog.showdatetime", "true");
			System.setProperty("org.apache.commons.logging.simplelog.log.httpclient.wire", "debug");
			System.setProperty("org.apache.commons.logging.simplelog.log.org.apache.commons.httpclient",
					"debug");
		}

		// try to initialize the service stub class used for communication with the web service
		try {
			stub = new BITConnectStub();
		} catch (AxisFault e) {

			e.printStackTrace();
		}

		// turn off transport layer compression, if debug was set.
		if (!debug.equals("true")) {
			stub._getServiceClient().getOptions()
					.setProperty(org.apache.axis2.transport.http.HTTPConstants.MC_GZIP_REQUEST, Boolean.TRUE);
			stub._getServiceClient().getOptions()
					.setProperty(org.apache.axis2.transport.http.HTTPConstants.MC_ACCEPT_GZIP, Boolean.TRUE);
		}

		// initialize the service client stub
		// force HTTP layer session handling for cookies with 'PHPSESSID'
		// reset user agent used by the client, this allows us to track web service access on the server
		// side
		stub._getServiceClient().getOptions().setManageSession(true);
		stub._getServiceClient().getOptions().setProperty("customCookieID", "PHPSESSID");
		stub._getServiceClient().getOptions()
				.setProperty(org.apache.axis2.transport.http.HTTPConstants.USER_AGENT, "BenchIT-GUI");

		// initialize and use a custom protocol, which will make use of our own keys
		// needed because of our root certificate not beeing included with java
		// TODO change to ProtocolSocketFactory-constructor
		Protocol sslProt = new Protocol("https", new BITCertSocketFactory(), 443);
		stub._getServiceClient()
				.getOptions()
				.setProperty(org.apache.axis2.transport.http.HTTPConstants.CUSTOM_PROTOCOL_HANDLER, sslProt);

		// increase socket and connection timeout
		String timeout = System.getProperty("org.benchit.bitconnect.Timeout", "120000");
		stub._getServiceClient().getOptions()
				.setTimeOutInMilliSeconds(new Integer(timeout).longValue());

		// initialize our GUI elements control class
		GUIControl.init(parent);
	}

	/**
	 * Get the singleton instance of this class.
	 * 
	 * @param parent the parent object
	 * @return
	 */
	public static BITConnectMain getInstance(BIGGUI parent) {
		if (BITConnectMain.instance == null) {
			instance = new BITConnectMain(parent);
		}
		return instance;
	}

	/**
	 * Use an ArrayList with absolute file paths to upload the corresponding files to www.benchit.org
	 * 
	 * @param itemPathnames
	 * @return an ArrayList with all successfully uploaded file paths.
	 * @throws IOException
	 */
	public ArrayList<String> uploadFilesFromPathStrings(ArrayList<?> itemPathnames)
			throws IOException {
		ArrayList<String> uploadedItems = new ArrayList<String>();

		initProgressBar(0, itemPathnames.size());

		Iterator<?> it = itemPathnames.iterator();
		int i = 0;
		while (it.hasNext()) {
			String item = uploadFile((String) it.next());
			if (item != null) {
				uploadedItems.add(item);
				setProgressBar(++i);
			}
		}
		return uploadedItems;
	}

	/**
	 * Use a HashSet with file descriptors to upload the corresponding files to www.benchit.org
	 * 
	 * @param items
	 * @return a HashSet with all successfully uploaded file descriptors.
	 * @throws IOException
	 */
	public HashSet<File> uploadFilesFromFileObj(HashSet<?> items) throws IOException {
		HashSet<File> uploadedItems = new HashSet<File>();

		initProgressBar(0, items.size());

		Iterator<?> it = items.iterator();
		int i = 0;
		while (it.hasNext()) {
			File item = uploadFile((File) it.next());
			if (item != null) {
				uploadedItems.add(item);
				setProgressBar(++i);
			}
		}

		return uploadedItems;
	}

	/**
	 * Initialize the BIGGUI progress bar with min and max values
	 * 
	 * @param min
	 * @param max
	 */
	private void initProgressBar(int min, int max) {
		progressbar = parent.getStatusProgress();
		progressbar.setMaximum(max);
		progressbar.setMinimum(min);
		progressbar.setValue(0);
	}

	/**
	 * increase the BIGGUI progress bar.
	 * 
	 * @param num
	 */
	private void setProgressBar(int num) {

		progressbar.setValue(num);
	}

	/**
	 * Login a user to the web service using username and password. A dialog will be used, if no username and password are
	 * present.
	 * 
	 * @return the user as seen from the web service
	 * @throws RequestCancelledException
	 */
	public BITUserType loginToWebservice() throws RequestCancelledException {
		currUser = null;
		if ((login == null || pass == null) || (login.length() == 0 || pass.length() == 0)) {
			GUIControl.showLoginDialog();
			getLoginData();
		}

		while (true) {

			try {
				currUser = stub.loginBITAccount(login, pass);
				break;
			} catch (RemoteException e) {
				GUIControl.showErrorMessage("Connection Error",
						"Connecting to BenchIT-Web failed, please check your internet connection.");
				throw new RequestCancelledException(e);
			} catch (LoginFault e) {
				GUIControl.showLoginDialog(e.getFaultMessage().getLoginFaultElem().getFaultMsg());
				getLoginData();
			}
		}
		return currUser;
	}

	/**
	 * Fetch login data from the gui elements.
	 * 
	 * @see GUIControl.getLoginData()
	 * @throws RequestCancelledException
	 */
	private void getLoginData() throws RequestCancelledException {
		HashMap<?, ?> map = GUIControl.getLoginData();
		login = (String) map.get("login");
		pass = (String) map.get("pass");
	}

	/**
	 * Upload a single file by its complete absolute path to www.benchit.org using the published web service and the web
	 * service client from package org.benchit.bitconnect.service.
	 * 
	 * @param item absolute path of the file to be uploaded
	 * @return the filepath on success, otherwise null
	 * @throws IOException
	 */
	public String uploadFile(String item) throws IOException {

		if (uploadFile(new File(item)) == null)
			return null;
		return item;
	}

	/**
	 * Upload a single file by its file descriptor to www.benchit.org using the published web service and the web service
	 * client from package org.benchit.bitconnect.service.
	 * 
	 * @param file file descriptor of the file to be uploaded
	 * @return the file descriptor on success, otherwise null
	 * @throws IOException
	 */
	public File uploadFile(File file) throws IOException {

		BITFileType bitFile = new BITFileType();
		bitFile.setData(new DataHandler(new FileDataSource(file)));
		bitFile.setFilename(file.getName());
		bitFile.setFilehash(MD5.asHex(MD5.getHash(file)));

		boolean upResponse = false;

		// try to login to the web service
		if (currUser == null) {
			try {
				loginToWebservice();
			} catch (RequestCancelledException e) {
				GUIControl
						.showErrorMessage("Login failed",
								"In order to upload result files to ww.benchit.org, you need to provide some user credentials.");
			}
		}

		// upload file and handle potentially occuring exceptions
		try {
			upResponse = stub.uploadBITFile(bitFile);
		} catch (RemoteException e) {
			GUIControl.showErrorMessage("Connection Error",
					"File upload failed, please check your internet connection.");
		} catch (UploadFault e) {
			GUIControl.showErrorMessage("Upload Error", "Error while uploading file:\n" + file.getName()
					+ "\n\nReason was:\n" + e.getFaultMessage().getUploadFaultElem().getFaultMsg());
		} catch (SessionFault e) {
			// try to (re-)login, SessionFault is thrown only on not logged in operations
			try {
				loginToWebservice();
			} catch (RequestCancelledException e1) {
				GUIControl
						.showErrorMessage("Login failed",
								"In order to upload result files to ww.benchit.org, you need to provide some user credentials.");
			}

			// retry uploading file
			try {
				upResponse = stub.uploadBITFile(bitFile);
			} catch (UploadFault e1) {
				GUIControl.showErrorMessage("Upload Error", "Error while uploading\n\n" + file.getName()
						+ "\n\nReason was:\n" + e1.getFaultMessage().getUploadFaultElem().getFaultMsg());
			} catch (SessionFault e1) {
				GUIControl
						.showErrorMessage(
								"Session Error",
								"The file\n\n"
										+ file.getName()
										+ "\n\ncould not be uploaded to www.benchit.org due to a session error.\nPlease try again later, this may be a temporarily problem.");
				e1.printStackTrace();
			}
		}

		// output status and return
		if (upResponse) {
			System.out.println("[SUCCESS] " + file.getName() + " has been uploaded successfully.");
			return file;
		} else {
			System.out.println("[ERROR] " + file.getName() + " was not uploaded.");
			return null;
		}

	}

	/**
	 * Check the upload status of a file.
	 * 
	 * @param file file descriptor
	 * @return true if file is uploaded and parsed successfully, otherwise false
	 * @throws IOException
	 */
	public boolean isFileUploadedSuccessfully(File file) throws IOException {
		BITFileUploadStatusType filestat = null;

		try {
			filestat = stub.getBITFileUploadStatus(MD5.asHex(MD5.getHash(file)));
		} catch (RemoteException e) {
			JOptionPane.showMessageDialog(null, "Error uploading file: " + e.getMessage());
			e.printStackTrace();
		} catch (SessionFault e) {
			// if we got a session fault, try to login and recheck the status.
			try {
				loginToWebservice();
				isFileUploadedSuccessfully(file);
			} catch (RequestCancelledException e1) {
				// if (re)-login fails or is canceled, print stack-trace and fail
				e1.printStackTrace();
			}
		} catch (PermissionFault e) {
			JOptionPane.showMessageDialog(null, "No permissions to upload file!");
			return false;
		}

		if (filestat == null)
			return false;
		else if (filestat.getClass().getName().equals("BITFileUploadSuccessType"))
			return true;
		else if (filestat.getClass().getName().equals("BITFileUploadFailedType"))
			return false;

		return false;
	}

	/**
	 * Clean-up and destroy.
	 */
	@Override
	public void finalize() {
		GUIControl.close();

	}
}
