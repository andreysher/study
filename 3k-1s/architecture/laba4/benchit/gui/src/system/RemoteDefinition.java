package system;

import java.io.*;
import java.util.*;

import ch.ethz.ssh2.*;
import conn.ConnectionHelper;

/**
 * <p>
 * Überschrift: BenchIT
 * </p>
 * <p>
 * Beschreibung:
 * </p>
 * <p>
 * Copyright: Copyright (c) 2004
 * </p>
 * <p>
 * Organisation: ZHR TU Dresden
 * </p>
 * 
 * @author Robert Schoene
 * @version 1.0
 */

public class RemoteDefinition {
	// ip or dns-name of remote machine
	private final String ip;
	// port for ssh;
	private final int port;
	// username on remotemachine
	private final String userName;
	// name of folder on remote machine
	private final String folderName;
	// selected in dialog
	private boolean isSelected;
	// use scp (please dont)
	private final boolean useSCP;
	// password is asked once per session and stored
	private String password = null;
	// hostname of the system
	private String hostName;
	// use specific localdef?
	private boolean useLocaldef = false;
	// name of the localdef
	private String localDef = null;
	// bash --login or sh or whatever
	private String shellCmd = null;

	public RemoteDefinition(String ip, String username, String folderName, boolean isSelected,
			boolean useSCP, String hostname, boolean useLocaldef, String localdef, String shellCmd,
			int port) {
		this.ip = ip;
		this.useSCP = useSCP;
		userName = username;
		this.folderName = folderName;
		this.isSelected = isSelected;
		hostName = hostname;
		this.useLocaldef = useLocaldef;
		localDef = localdef;
		this.shellCmd = shellCmd;
		this.port = port;
	}

	public RemoteDefinition(String ip, String username, String folderName, boolean isSelected,
			boolean useSCP, String hostname, boolean useLocaldef, String localdef, String shellCmd) {
		this(ip, username, folderName, isSelected, useSCP, hostname, useLocaldef, localdef, shellCmd,
				22);
	}

	@Override
	public String toString() {
		return userName + "@" + ip + ":~/" + folderName
				+ ((getLOCALDEF() != null) ? "(" + getLOCALDEF() + ")" : "");
	}

	public String getUsername() {
		return userName;
	}

	public String getIP() {
		return ip;
	}

	public int getPort() {
		return port;
	}

	public String getFoldername() {
		return folderName;
	}

	public void setSelected(boolean b) {
		isSelected = b;
	}

	public boolean isSelected() {
		return isSelected;
	}

	public String retrieveHostname(Session session) {
		String hostName = null;

		StreamGobbler[] in = new StreamGobbler[2];
		String ret = "";
		try {
			in[0] = new StreamGobbler(session.getStderr());
			in[1] = new StreamGobbler(session.getStdout());
			session.execCommand("hostname && echo \"FINISHED\"\n");
			System.out.println("Opening ssh-session to " + this.ip + " to get the hostname.");
		} catch (IOException ex) {
			ex.printStackTrace();
			return null;
		}
		try {
			byte[] buffer = new byte[8192];
			int nums = in[1].read(buffer);
			while (nums > -1) {
				ret = ret + new String(buffer, 0, nums);
				nums = in[1].read(buffer);
				if (ret.endsWith("FINISHED")) {
					break;
				}
			}
		} catch (IOException ex1) {
			ex1.printStackTrace();
		}
		try {
			hostName = ret.substring(0, ret.indexOf("\nFINISHED"));
		} catch (Exception e) {
			System.out.println("Getting hostname failed.");
		}
		if (hostName != null && hostName.lastIndexOf("\n") > 0) {
			hostName = hostName.substring(hostName.lastIndexOf("\n") + 1);
		}
		this.hostName = hostName;

		return hostName;
	}

	/**
	 * get the hostname of a remotedefinition
	 * 
	 * @return String the result of the commandline call hostname
	 */
	public String getHostname() {
		// had been searched for. so no need to look twice
		if (hostName != null)
			return hostName;

		Session session = ConnectionHelper.openSession(this, null);

		if (session == null) {
			System.err.println("Couldn't open Session");
			return null;
		}

		retrieveHostname(session);
		ConnectionHelper.closeSession(session);
		return this.hostName;
	}

	public boolean isValid() {
		return ip != null && userName != null && folderName != null && !userName.isEmpty()
				&& !ip.isEmpty() && !folderName.isEmpty() && folderName != "/";
	}

	public String getXML() {
		StringBuffer sb = new StringBuffer();
		sb.append("<RemoteDefinition>\n");
		sb.append("   <ip>");
		sb.append(ip);
		sb.append("</ip>\n");
		sb.append("   <username>");
		sb.append(userName);
		sb.append("</username>\n");
		sb.append("   <foldername>");
		sb.append(folderName);
		sb.append("</foldername>\n");
		sb.append("   <isSelected>");
		sb.append(isSelected);
		sb.append("</isSelected>\n");
		sb.append("   <hostname>");
		sb.append(hostName);
		sb.append("</hostname>\n");
		sb.append("   <useSCP>");
		sb.append(useSCP);
		sb.append("</useSCP>\n");
		sb.append("   <useLOCALDEF>");
		sb.append(useLocaldef);
		sb.append("</useLOCALDEF>\n");
		sb.append("   <LOCALDEF>");
		sb.append(localDef);
		sb.append("</LOCALDEF>\n");
		if (shellCmd != null) {
			sb.append("   <SHELLCMD>");
			sb.append(shellCmd);
			sb.append("</SHELLCMD>\n");
		}
		sb.append("   <port>");
		sb.append(port);
		sb.append("</port>\n");
		sb.append("</RemoteDefinition>\n");

		return sb.toString();
	}

	public static String getXMLStarting() {
		return "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n" + "<!DOCTYPE RemoteDefinition [\n"
				+ "   <!ELEMENT RemoteDefinition ( ip,username,foldername,isSelected,hostname ) >\n"
				+ "   <!ELEMENT ip         (#PCDATA)>\n" + "   <!ELEMENT username   (#PCDATA)>\n"
				+ "   <!ELEMENT foldername (#PCDATA)>\n" + "   <!ELEMENT isSelected (#PCDATA)>\n"
				+ "   <!ELEMENT useSCP (#PCDATA)>\n" + "   <!ELEMENT useLOCALDEF (#PCDATA)>\n"
				+ "   <!ELEMENT LOCALDEF (#PCDATA)>\n" + "   <!ELEMENT SHELLCMD (#PCDATA)>\n"
				+ "   <!ELEMENT port (#PCDATA)>\n" + "]>\n\r";
	}

	public static String getFromXMLTag(String filePart, String tagName) {
		String start = "<" + tagName + ">";
		String end = "</" + tagName + ">";
		int iStart = filePart.indexOf(start);
		if (iStart < 0)
			return null;
		// TODO: trim() call is made as old styled xml files added whitespaces at: <name> value </name>; remove that call
		// somewhen
		return filePart.substring(iStart + start.length(), filePart.indexOf(end)).trim();
	}

	public static List<RemoteDefinition> getRemoteDefsFromXML(File f) {
		// read File to String
		ArrayList<RemoteDefinition> remoteDefs = new ArrayList<RemoteDefinition>();
		if (!f.exists() || !f.isFile())
			return remoteDefs;
		String filecontent = BIGFileHelper.getFileContent(f);
		String curDef = getFromXMLTag(filecontent, "RemoteDefinition");
		while (curDef != null) {
			String ip = getFromXMLTag(curDef, "ip");
			String username = getFromXMLTag(curDef, "username");
			String foldername = getFromXMLTag(curDef, "foldername");
			boolean isSelected = getFromXMLTag(curDef, "isSelected").equals("true");
			String hostname = getFromXMLTag(curDef, "hostname");
			boolean useSCP = getFromXMLTag(curDef, "useSCP").equals("true");
			boolean useLocaldef = getFromXMLTag(curDef, "useLOCALDEF").equals("true");

			String localdef = getFromXMLTag(curDef, "LOCALDEF");

			String shellCmd = getFromXMLTag(curDef, "SHELLCMD");

			int port;
			try {
				port = Integer.parseInt(getFromXMLTag(curDef, "port"));
			} catch (NumberFormatException e) {
				port = 22;
			}
			remoteDefs.add(new RemoteDefinition(ip, username, foldername, isSelected, useSCP, hostname,
					useLocaldef, localdef, shellCmd, port));
			filecontent = filecontent.substring(filecontent.indexOf("</RemoteDefinition>\n") + 19);
			curDef = getFromXMLTag(filecontent, "RemoteDefinition");
		}
		return remoteDefs;
	}

	public static void saveRemoteDefsToXML(List<RemoteDefinition> remoteDefs, File f) {
		StringBuffer sb = new StringBuffer();
		sb.append(RemoteDefinition.getXMLStarting());
		for (int i = 0; i < remoteDefs.size(); i++) {
			sb.append(remoteDefs.get(i).getXML());
		}
		BIGFileHelper.saveToFile(sb.toString(), f);
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String p) {
		password = p;
	}

	public boolean getUseSCP() {
		return useSCP;
	}

	public boolean getUseLocalDef() {
		return useLocaldef;
	}

	public String getLOCALDEF() {
		if (!useLocaldef)
			return getHostname();
		if (localDef == null || localDef.isEmpty() || localDef.equals("null"))
			return getHostname();
		return localDef;
	}
	public String getShellCommand() {
		String command = "bash --login";
		try {
			command = BIGInterface.getInstance().getBIGConfigFileParser()
					.stringCheckOut("remoteShellCmd");
		} catch (Exception ex) {
			System.out.println("remoteShellCmd not set in BGUI.cfg. Using default \"bash --login\"");
			System.out.println("To run BenchIT on a system without bash, close the GUI, open the file");
			System.out.println("BGUI.cfg and set the flag \"remoteShellCmd=sh\"");
		}

		if (shellCmd == null)
			return command;
		if (shellCmd.equals(""))
			return command;
		return shellCmd;

	}

}
