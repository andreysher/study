package conn;

import gui.*;

import java.io.*;
import java.util.HashMap;

import javax.swing.JFrame;

import system.RemoteDefinition;
import ch.ethz.ssh2.*;

public class ConnectionHelper {
	static final String sshDir = System.getProperty("user.home") + File.separator + ".ssh"
			+ File.separator;
	static final String knownHostPath = sshDir + "known_hosts";
	static final String idDSAPath = sshDir + "id_dsa";
	static final String idRSAPath = sshDir + "id_rsa";

	private static KnownHosts database = new KnownHosts();
	private static final HashMap<Object, Connection> conns = new HashMap<Object, Connection>();

	private static boolean isInit = false;

	private static String lastError = "";

	private static void Init() {
		try {
			database.addHostkeys(new File(knownHostPath));
		} catch (IOException ex) {
			System.err.println("Couldn't find " + knownHostPath);
		}
		isInit = true;
	}

	private static boolean TryAuthXSA(RemoteDefinition def, Connection conn, String keyFile,
			String password, JFrame parentFrame, String authMethod) throws IOException {
		File key = new File(keyFile);
		if (!key.exists())
			return false;

		if (password == null) {
			if (conn.authenticateWithPublicKey(def.getUsername(), key, "")) {
				return true;
			}
			String title = authMethod + " Authentication for " + def.getUsername() + "@" + def.getIP();
			EnterSomethingDialog esd = new EnterSomethingDialog(parentFrame, title, new String[]{
					lastError, title, " ", "Enter " + authMethod + " private key password:"}, true);
			esd.setVisible(true);

			if (esd.answer == null)
				return false; // Login aborted by user

			if (conn.authenticateWithPublicKey(def.getUsername(), key, esd.answer)) {
				def.setPassword(esd.answer);
				return true;
			}
		} else {
			if (conn.authenticateWithPublicKey(def.getUsername(), key, password)) {
				return true;
			}
		}
		lastError = authMethod + " authentication failed.";
		return false;
	}

	private static Connection openConnection(RemoteDefinition def, JFrame parentFrame) {
		if (!isInit) {
			Init();
		}
		lastError = "";
		Connection conn = new Connection(def.getIP());
		String[] hostkeyAlgos = database.getPreferredServerHostkeyAlgorithmOrder(def.getIP());
		String password = def.getPassword();

		try {
			// CONNECT AND VERIFY SERVER HOST KEY
			boolean enableKeyboardInteractive = true;
			boolean enableDSA = true;
			boolean enableRSA = true;

			while (true) {
				conn.close();
				conn = new Connection(def.getIP());

				if (hostkeyAlgos != null)
					conn.setServerHostKeyAlgorithms(hostkeyAlgos);
				conn.connect(new SimpleVerifier(database));

				if (conn.isAuthMethodAvailable(def.getUsername(), "publickey")) {
					if (enableDSA) {
						if (TryAuthXSA(def, conn, idDSAPath, password, parentFrame, "DSA"))
							return conn;
						enableDSA = false; // do not try again
						continue;
					}
					if (enableRSA) {
						if (TryAuthXSA(def, conn, idRSAPath, password, parentFrame, "RSA"))
							return conn;
						enableRSA = false; // do not try again
						continue;
					}
				}

				if (enableKeyboardInteractive
						&& conn.isAuthMethodAvailable(def.getUsername(), "keyboard-interactive")) {
					InteractiveLogic il = new InteractiveLogic(lastError, def, parentFrame);
					if (conn.authenticateWithKeyboardInteractive(def.getUsername(), il))
						return conn;

					if (il.getPromptCount() == 0) {
						// aha. the server announced that it supports "keyboard-interactive", but when
						// we asked for it, it just denied the request without sending us any prompt.
						// That happens with some server versions/configurations.
						// We just disable the "keyboard-interactive" method and notify the user.

						lastError = "Keyboard-interactive does not work.";
						enableKeyboardInteractive = false; // do not try again
					} else {
						lastError = "Keyboard-interactive auth failed."; // try again, if possible
					}

					continue;
				}

				if (conn.isAuthMethodAvailable(def.getUsername(), "password")) {
					if (password == null) {
						final EnterSomethingDialog esd = new EnterSomethingDialog(parentFrame,
								"Password Authentication for " + def.getUsername() + "@" + def.getIP(),
								new String[]{lastError,
										"Enter password for " + def.getUsername() + "@" + def.getIP()}, true);
						esd.setVisible(true);

						if (esd.answer == null)
							throw new IOException("Login aborted by user");
						try {
							if (conn.authenticateWithPassword(def.getUsername(), esd.answer)) {
								def.setPassword(esd.answer);
								return conn;
							}
						} catch (IOException ex2) {
							lastError = "Password authentication failed.";
						}
					} else {
						try {
							if (conn.authenticateWithPassword(def.getUsername(), password))
								return conn;
						} catch (IOException ex3) {
							lastError = "Password authentication failed."; // try again, if possible
						}
						password = null;
					}

					lastError = "Password authentication failed."; // try again, if possible
					continue;
				}

				throw new IOException("No supported authentication methods available.");
			}
		} catch (Exception exc) {
			System.err.println("Couldn't connect to " + def.getIP());
		}
		return null;
	}

	/***
	 * @param def
	 * @param parentFrame
	 * @return Session
	 */
	public static Session openSession(RemoteDefinition def, JFrame parentFrame) {
		Connection conn = openConnection(def, parentFrame);
		if (conn == null)
			return null;
		Session sess;
		try {
			sess = conn.openSession();
		} catch (IOException e) {
			conn.close();
			return null;
		}
		conns.put(sess, conn);
		return sess;
	}

	public static SCPClient openSCPClient(RemoteDefinition def, JFrame parentFrame) {
		Connection conn = openConnection(def, parentFrame);
		if (conn == null)
			return null;
		SCPClient sess;
		try {
			sess = conn.createSCPClient();
		} catch (IOException e) {
			conn.close();
			return null;
		}
		conns.put(sess, conn);
		return sess;
	}

	public static void closeSession(Session sess) {
		// TODO: Not really working. Got to remove timeout and check other solutions
		sess.waitForCondition(ChannelCondition.EXIT_SIGNAL, 0);
		sess.close();
		if (conns.get(sess) != null) {
			conns.get(sess).close();
			conns.remove(sess);
		}
	}

	public static void put(SCPClient scp, File f, String directory) {
		if (f.isFile()) {
			try {
				scp.put(f.getAbsolutePath(), f.length(), directory, "0600").close();
			} catch (IOException ex1) {}
			return;
		}
		File[] files = f.listFiles();
		for (File fileOrDir : files) {
			put(scp, fileOrDir, directory + "/" + f.getName());
		}
	}
}
