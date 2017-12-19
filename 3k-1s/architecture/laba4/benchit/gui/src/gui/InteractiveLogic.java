package gui;

import java.io.IOException;

import javax.swing.JFrame;

import system.RemoteDefinition;
import ch.ethz.ssh2.InteractiveCallback;

/**
 * The logic that one has to implement if "keyboard-interactive" autentication shall be supported.
 */
public class InteractiveLogic implements InteractiveCallback {
	int promptCount = 0;
	String lastError;
	RemoteDefinition def = null;
	JFrame loginFrame;

	public InteractiveLogic(String lastError, RemoteDefinition def, JFrame loginFrame) {
		this.lastError = lastError;
		this.def = def;
		this.loginFrame = loginFrame;
	}

	/*
	 * the callback may be invoked several times, depending on how many questions-sets the server sends
	 */

	public String[] replyToChallenge(String name, String instruction, int numPrompts,
			String[] prompt, boolean[] echo) throws IOException {
		String[] result = new String[numPrompts];

		for (int i = 0; i < numPrompts; i++) {
			if (def.getPassword() == null) {
				/* Often, servers just send empty strings for "name" and "instruction" */
				String title = "Keyboard Interactive Authentication for " + def.getUsername() + "@"
						+ def.getIP();
				String[] content = new String[]{lastError, title, " ", name, instruction, prompt[i]};
				lastError = null;
				EnterSomethingDialog esd = new EnterSomethingDialog(loginFrame, title, content, !echo[i]);

				esd.setVisible(true);
				if (esd.answer == null)
					throw new IOException("Login aborted by user");

				result[i] = esd.answer;
				def.setPassword(esd.answer);
			} else {
				result[i] = def.getPassword();
			}
			promptCount++;
		}

		return result;
	}

	/*
	 * We maintain a prompt counter - this enables the detection of situations where the ssh server is signaling
	 * "authentication failed" even though it did not send a single prompt.
	 */

	public int getPromptCount() {
		return promptCount;
	}
}