package org.benchit.bitconnect.gui;

import java.util.HashMap;

import javax.swing.*;

import org.benchit.bitconnect.RequestCancelledException;

public final class GUIControl {

	private static HashMap<String, LoginDataDialog> gui;
	private static JFrame parentFrame;

	private GUIControl() {

	}

	public static void init(JFrame frame) {
		parentFrame = frame;
		gui = new HashMap<String, LoginDataDialog>();
	}

	public static void showLoginDialog() {
		if (!gui.containsKey("LoginDialog")) {
			gui.put("LoginDialog", new LoginDataDialog(parentFrame, true));
		}

		LoginDataDialog loginDlg = gui.get("LoginDialog");
		loginDlg.setVisible(true);
	}

	public static HashMap<String, String> getLoginData() throws RequestCancelledException {
		HashMap<String, String> data = new HashMap<String, String>();
		LoginDataDialog loginDlg = gui.get("LoginDialog");

		if (loginDlg.getOptionValue().equals(new Integer(JOptionPane.CANCEL_OPTION))
				|| loginDlg.getOptionValue().equals(new Integer(JOptionPane.CLOSED_OPTION))
				|| loginDlg.getOptionValue().equals(JOptionPane.UNINITIALIZED_VALUE))
			throw new RequestCancelledException("No credentials provided");
		else {
			data.put("login", loginDlg.getLogin());
			data.put("pass", loginDlg.getPass());
			loginDlg.resetOptionValue();
		}
		return data;
	}

	public static void showLoginDialog(String msg) {
		if (!gui.containsKey("LoginDialog")) {
			gui.put("LoginDialog", new LoginDataDialog(parentFrame, true));
		}

		LoginDataDialog loginDlg = gui.get("LoginDialog");
		loginDlg.setMessage(msg);
		loginDlg.setVisible(true);
	}

	public static void close() {
		if (!gui.isEmpty()) {
			gui.clear();
		}
	}

	public static void showErrorMessage(String title, String msg) {
		JOptionPane.showMessageDialog(parentFrame, msg, title, JOptionPane.ERROR_MESSAGE);

	}

}
