/*
 * LoginDataDialog.java Created on 30. September 2008, 10:09
 */

package org.benchit.bitconnect.gui;

import java.awt.Color;
import java.awt.event.*;
import java.beans.*;

import javax.swing.*;

/**
 * This code was edited or generated using CloudGarden's Jigloo SWT/Swing GUI Builder, which is free
 * for non-commercial use. If Jigloo is being used commercially (ie, by a corporation, company or
 * business for any purpose whatever) then you should purchase a license for each developer using
 * Jigloo. Please visit www.cloudgarden.com for details. Use of Jigloo implies acceptance of these
 * licensing terms. A COMMERCIAL LICENSE HAS NOT BEEN PURCHASED FOR THIS MACHINE, SO JIGLOO OR THIS
 * CODE CANNOT BE USED LEGALLY FOR ANY CORPORATE OR COMMERCIAL PURPOSE.
 */
/**
 * @author dreiche
 */
public class LoginDataDialog extends javax.swing.JDialog
		implements
			ActionListener,
			PropertyChangeListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2364333715488035802L;
	private JOptionPane optionPane;
	private String msg;
	private JPasswordField passField;
	private JTextField loginField;
	private JLabel passLabel;
	private JLabel loginLabel;

	/** Creates new form LoginDataDialog */
	public LoginDataDialog(java.awt.Frame parent, boolean modal) {
		super(parent, modal);
		initComponents();
		setLocationRelativeTo(parent);
	}

	/**
	 * This method is called from within the constructor to initialize the form.
	 */
	private void initComponents() {
		setTitle("BenchIT-Web Login");

		msg = "Please provide some valid user credentials:\n";

		loginLabel = new JLabel("Login:");
		passLabel = new JLabel("Password:");

		loginField = new JTextField();
		passField = new JPasswordField();

		Object[] array = {msg, loginLabel, loginField, passLabel, passField};

		optionPane = new JOptionPane(array, JOptionPane.QUESTION_MESSAGE, JOptionPane.OK_CANCEL_OPTION);

		setContentPane(optionPane);

		setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);

		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				optionPane.setValue(new Integer(JOptionPane.CLOSED_OPTION));
			}
		});

		addComponentListener(new ComponentAdapter() {
			@Override
			public void componentShown(ComponentEvent ce) {
				loginField.requestFocusInWindow();
			}
		});

		loginField.addActionListener(this);
		passField.addActionListener(this);

		optionPane.addPropertyChangeListener(this);

		pack();
	}

	public Object getOptionValue() {
		return optionPane.getValue();
	}

	public String getLogin() {
		return loginField.getText();
	}

	public String getPass() {
		return new String(passField.getPassword());
	}

	public void actionPerformed(ActionEvent e) {
		optionPane.setValue(new Integer(JOptionPane.OK_OPTION));
	}

	public void setMessage(String msg) {
		Object[] arr = {msg, loginLabel, loginField, passLabel, passField};
		optionPane.setMessage(arr);
		optionPane.setBackground(new Color(255, 0, 0));
		pack();
	}

	public void propertyChange(PropertyChangeEvent e) {
		String prop = e.getPropertyName();

		if (isVisible()
				&& (e.getSource() == optionPane)
				&& (JOptionPane.VALUE_PROPERTY.equals(prop) || JOptionPane.INPUT_VALUE_PROPERTY
						.equals(prop))) {

			Object value = optionPane.getValue();

			if (value == JOptionPane.UNINITIALIZED_VALUE)
				// ignore reset
				return;

			if (value.equals(new Integer(JOptionPane.OK_OPTION))) {
				if (loginField.getText().length() != 0 && getPass().length() != 0) {
					// we're done; clear and dismiss the dialog
					clearAndHide();
				} else {
					// text was invalid
					JOptionPane.showMessageDialog(this,
							"Sorry, you have to provide some user credentials.\n", "Try again!\n",
							JOptionPane.ERROR_MESSAGE);
					optionPane.setValue(JOptionPane.UNINITIALIZED_VALUE);
					loginField.requestFocusInWindow();
				}
			} else { // user closed dialog or clicked cancel
				clearAndHide();
			}
		}
	}

	private void clearAndHide() {
		setVisible(false);
	}

	public void resetOptionValue() {
		optionPane.setValue(JOptionPane.UNINITIALIZED_VALUE);
	}
}
