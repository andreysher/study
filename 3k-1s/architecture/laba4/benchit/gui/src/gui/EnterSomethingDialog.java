package gui;

import java.awt.BorderLayout;
import java.awt.event.*;

import javax.swing.*;

/**
 * This dialog displays a number of text lines and a text field. The text field can either be plain text or a password
 * field.
 */
public class EnterSomethingDialog extends JDialog {
	private static final long serialVersionUID = 1L;

	JTextField answerField;
	JPasswordField passwordField;

	final boolean isPassword;

	public String answer;

	public EnterSomethingDialog(JFrame parent, String title, String content, boolean isPassword) {
		this(parent, title, new String[]{content}, isPassword);
	}

	public EnterSomethingDialog(JFrame parent, String title, String[] content, boolean isPassword) {
		super(parent, title, true);

		this.isPassword = isPassword;

		JPanel pan = new JPanel();
		pan.setLayout(new BoxLayout(pan, BoxLayout.Y_AXIS));

		for (int i = 0; i < content.length; i++) {
			if (content[i] == null || content[i].isEmpty()) {
				continue;
			}
			JLabel contentLabel = new JLabel(content[i]);
			pan.add(contentLabel);
		}

		answerField = new JTextField(20);
		passwordField = new JPasswordField(20);

		if (isPassword) {
			pan.add(passwordField);
		} else {
			pan.add(answerField);
		}

		KeyAdapter kl = new KeyAdapter() {
			@Override
			public void keyTyped(KeyEvent e) {
				if (e.getKeyChar() == '\n') {
					finish();
				}
			}
		};

		answerField.addKeyListener(kl);
		passwordField.addKeyListener(kl);

		getContentPane().add(BorderLayout.CENTER, pan);

		setResizable(false);
		pack();
		setLocationRelativeTo(null);
	}

	public void setDefault(String text) {
		if (isPassword) {
			passwordField.setText(text);
		} else {
			answerField.setText(text);
		}
	}

	private void finish() {
		if (isPassword) {
			answer = new String(passwordField.getPassword());
		} else {
			answer = answerField.getText();
		}

		dispose();
	}
}