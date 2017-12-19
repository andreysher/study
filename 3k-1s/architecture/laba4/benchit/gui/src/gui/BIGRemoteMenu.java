package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import java.util.List;

import javax.swing.*;

import org.apache.tools.tar.*;

import plot.data.BIGOutputParser;
import system.*;
import ch.ethz.ssh2.Session;
import conn.ConnectionHelper;

/**
 * <p>
 * Ueberschrift: BenchIT BIGRemoteMenu
 * </p>
 * <p>
 * Beschreibung: Menu for Remote Folder support (execute on other system via ssh)
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

public class BIGRemoteMenu extends JMenu {
	/** Default Serial Version UID to get rid of warnings */
	private static final long serialVersionUID = 1L;

	/** name of file that contains items that should be copied to a remote folder */
	static final String WHITELIST_FILENAME = "copywhitelist.txt";

	/** name of file that contains items that should NOT be copied to a remote folder */
	static final String BLACKLIST_FILENAME = "copyblacklist.txt";

	/** path to the whitelist file and the blacklist file */
	static final String PATH_TO_LISTS = BIGInterface.getInstance().getConfigPath() + File.separator;

	JFrame loginFrame = null;

	static boolean debug = false;
	public List<RemoteDefinition> remoteDefs;
	private BIGGUI bgui;

	// MenuItems
	JMenuItem execRemote;
	JMenuItem copyRemote;
	JMenuItem removeOutput;
	JMenuItem removeFolder;
	JMenuItem getOutput;
	JMenuItem checkForOutput;
	JMenuItem remoteFolder;
	JMenuItem copyExec;
	JMenuItem justExec;

	/**
	 * Creates a menu for a menu bar containing all JMenuItems for operations on remote machines.
	 * 
	 * @param gui The BIGGUI.
	 **/
	public BIGRemoteMenu(BIGGUI gui) {
		createMenus(gui);
	}

	/**
	 * Creates the JMenuItems for this BIGRemoteMenu.
	 * 
	 * @param gui The BIGGUI.
	 **/
	private void createMenus(BIGGUI gui) {
		bgui = gui;
		if (BIGInterface.getInstance().getDebug("BIGRemoteMenu") > 0) {
			debug = true;
		}
		// if(BIGInterface.getSystem()==BIGInterface.WINDOWS_SYSTEM) return;
		try {
			String name = BIGInterface.getInstance().getConfigPath() + File.separator
					+ BIGInterface.getInstance().getBIGConfigFileParser().stringCheckOut("remoteXMLFile");
			File xmlFile = new File(name);

			remoteDefs = RemoteDefinition.getRemoteDefsFromXML(xmlFile);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Couldn't load \"remoteXMLFile\" from BGUI.cfg");
			remoteDefs = new ArrayList<RemoteDefinition>();
		}
		/*
		 * try { this.sshCommand = BIGInterface.getInstance(). getBIGConfigFileParser().stringCheckOut("sshCommand"); }
		 * catch (Exception e) { System.out.println( "Couldn't load \"sshCommand\" from BGUI.cfg, using \"ssh\""); } try {
		 * this.scpCommand = BIGInterface.getInstance(). getBIGConfigFileParser().stringCheckOut("scpCommand"); } catch
		 * (Exception e) { System.out.println( "Couldn't load \"scpCommand\" from BGUI.cfg, using \"scp\" instead"); }
		 * String additional = ""; try { additional = BIGInterface.getInstance(). getBIGConfigFileParser().stringCheckOut(
		 * "additionalSCPFlags"); if (!additional.equals("")) { additional = " " + additional; } } catch (Exception e) {
		 * System.out.println( "Couldn't load \"additionalSCPFlags\" from BGUI.cfg, using none"); } this.scpCommand =
		 * this.scpCommand + additional; try { this.tarCommand = BIGInterface.getInstance().
		 * getBIGConfigFileParser().stringCheckOut("tarCommand"); } catch (Exception e) { System.out.println(
		 * "Couldn't load \"tarCommand\" from BGUI.cfg, using \"tar\" instead"); }
		 */
		setText("Remote Mode");

		remoteFolder = new JMenuItem("Create a remote folder");
		remoteFolder.setToolTipText("Create a remote folder at a computer");
		remoteFolder.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {
						createRemoteFolderDialog();
					}
				}).start();
			}
		});

		checkForOutput = new JMenuItem("Check for output");
		checkForOutput.setToolTipText("Check a remote folder for output data");
		checkForOutput.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {

						if (remoteDefs.size() == 0) {
							showEmptyDialog();
						} else {
							checkForOutput();
						}
					}
				}).start();

			}
		});

		getOutput = new JMenuItem("Get remote results");
		getOutput.setToolTipText("Get the result files from a remote folder");
		getOutput.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				if (remoteDefs.size() == 0) {
					showEmptyDialog();
				} else {
					getOutput();
				}
			}
		});

		removeFolder = new JMenuItem("Remove remote folder");
		removeFolder.setToolTipText("Removes a remote folder");
		removeFolder.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {
						if (remoteDefs.size() == 0) {
							showEmptyDialog();
						} else {
							removeRemoteFolder();
						}
					}
				}).start();
			}
		});

		removeOutput = new JMenuItem("Remove remote output");
		removeOutput.setToolTipText("Removes output data and results in remote folder");
		removeOutput.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {
						if (remoteDefs.size() == 0) {
							showEmptyDialog();
						} else {
							removeRemoteOutput();
						}
					}
				}).start();
			}
		});
		execRemote = new JMenuItem("Execute in remote folder");
		execRemote.setToolTipText("Executes selected kernels in remote folder");
		execRemote.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {
						if (remoteDefs.size() == 0) {
							showEmptyDialog();
						} else {
							bgui.executeSelectedOnOtherSystem();
						}
					}
				}).start();

			}
		});
		copyRemote = new JMenuItem("Copy to remote folder");
		copyRemote.setToolTipText("Copies selected kernel files (sources etc.) to remote folder");
		copyRemote.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {
						if (remoteDefs.size() == 0) {
							showEmptyDialog();
						} else {
							bgui.copySelectedToOtherSystem();
						}
					}
				}).start();

			}
		});
		copyExec = new JMenuItem("Copy executables to remote folder");
		copyExec.setToolTipText("Copy selected available executables to remote folder");
		copyExec.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {
						if (remoteDefs.size() == 0) {
							showEmptyDialog();
						} else {
							List<RemoteDefinition> remDefs = getDefsDialog("to copy the executables to");
							for (int i = 0; i < remDefs.size(); i++) {
								copyExecutables(remDefs.get(i));
							}
						}
					}
				}).start();
			}
		});
		justExec = new JMenuItem("Run remote executables");
		justExec.setToolTipText("Run selected available executables in remote folder");
		justExec.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				(new Thread() {
					@Override
					public void run() {
						if (remoteDefs.size() == 0) {
							showEmptyDialog();
						} else {
							List<RemoteDefinition> remDefs = getDefsDialog("to start the executables at");
							for (int i = 0; i < remDefs.size(); i++) {
								if (onlyRunCompilations(remDefs.get(i), "") == 0) {
									startRemoteProc(remDefs.get(i));
								}
							}
						}
					}
				}).start();

			}
		});
	}

	/**
	 * Returns this BIGRemoteMenu's JMenuItems in ordered Lists according to their category.
	 * 
	 * @return an array of 3 Lists of JMenuItems.
	 **/
	public Map<String, List<JMenuItem>> getRemoteMenus() {
		Map<String, List<JMenuItem>> retval = new HashMap<String, List<JMenuItem>>();
		retval.put("setup", new ArrayList<JMenuItem>());
		retval.put("measure", new ArrayList<JMenuItem>());
		retval.put("evaluate", new ArrayList<JMenuItem>());
		// add menu items for category: setup
		retval.get("setup").add(remoteFolder);
		retval.get("setup").add(removeFolder);
		// add menu items for category: measure
		retval.get("measure").add(execRemote);
		retval.get("measure").add(copyRemote);
		retval.get("measure").add(copyExec);
		retval.get("measure").add(justExec);
		// add menu items for category: evaluate
		retval.get("evaluate").add(checkForOutput);
		retval.get("evaluate").add(getOutput);
		retval.get("evaluate").add(removeOutput);
		return retval;
	}

	/**
	 * This method lists all files in temporary directory and saves the files in a Vector.
	 * 
	 * @param temp - file or directory which should be listed
	 * @param vec - Vector for saving entries
	 */
	private void filesToVector(File temp, Vector<File> vec) {
		// store all files in file or directory to Vector

		// list all files of temp
		File[] actualFiles = temp.listFiles();
		// a second Vector that contains only entries from actualFiles
		Vector<File> all_Files = new Vector<File>();

		// add entries to both Vectors
		for (int i = 0; i < actualFiles.length; i++) {
			File f = actualFiles[i];
			all_Files.add(f);
			vec.add(f);
		}

		// look for directories in all_Files
		// directories have to be handled again
		for (int i = 0; i < all_Files.size(); i++) {
			File actual = (all_Files.get(i));
			if (actual.isDirectory()) {
				filesToVector(actual, vec);
			}
		}
	}

	class CheckFolderKey extends KeyAdapter {
		BIGWizard wiz;

		public CheckFolderKey(BIGWizard wiz) {
			this.wiz = wiz;
		}
		@Override
		public void keyTyped(KeyEvent ke) {
			boolean acceptable = true;
			if (!Character.isLetterOrDigit(ke.getKeyChar())) {
				acceptable = false;
			}

			if (ke.getKeyChar() == '\n') {
				acceptable = true;
				wiz.next(null);
			}
			if (ke.getKeyChar() == '\b') {
				acceptable = true;
			}
			if (ke.getKeyChar() == '/') {
				acceptable = true;
			}
			if (ke.getKeyChar() > 122) {
				acceptable = false;
			}
			if (ke.getKeyChar() == '-') {
				acceptable = true;
			}
			if (ke.getKeyChar() == '_') {
				acceptable = true;
			}
			if (!acceptable) {
				ke.consume();
				final JWindow w = new JWindow();
				JLabel l = new JLabel("Unacceptable input. Not a letter, neither a number.");
				l.setBorder(new javax.swing.border.EtchedBorder());
				l.setForeground(Color.RED);
				l.setBackground(Color.WHITE);
				w.getContentPane().add(l);
				Point p = ((JTextField) ke.getSource()).getLocationOnScreen();
				p.move((int) p.getX(), (int) p.getY() + ((JTextField) ke.getSource()).getHeight());
				w.setLocation(p);
				w.pack();
				w.setVisible(true);
				(new Thread() {
					@Override
					public void run() {
						try {
							Thread.sleep(2000);
						} catch (InterruptedException ex) {}
						w.setVisible(false);
					}
				}).start();
			}
		}
	}

	class CheckFolderKeyFocus extends FocusAdapter {
		private final JTextField folderField;

		public CheckFolderKeyFocus(JTextField folderField) {
			this.folderField = folderField;
		}

		@Override
		public void focusLost(FocusEvent evt) {
			String text = folderField.getText();
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < text.length(); i++) {
				char c = text.charAt(i);
				boolean acceptable = false;
				if (Character.isLetterOrDigit(c)) {
					acceptable = true;
				}
				if (c > 122) {
					acceptable = false;
				}

				if (c == '/') {
					acceptable = true;
				}
				if (c == '-') {
					acceptable = true;
				}
				if (c == '_') {
					acceptable = true;
				}
				if (acceptable) {
					sb.append(text.charAt(i));
				}
			}
			if (!sb.toString().equals(text)) {

				final JWindow w = new JWindow();
				JLabel l = new JLabel("Removed unacceptable characters.");
				l.setBorder(new javax.swing.border.EtchedBorder());
				l.setForeground(Color.RED);
				l.setBackground(Color.WHITE);
				w.getContentPane().add(l);
				Point p = folderField.getLocationOnScreen();
				p.move((int) p.getX(), (int) p.getY() + folderField.getHeight());
				w.setLocation(p);
				w.pack();
				w.setVisible(true);
				(new Thread() {
					@Override
					public void run() {
						try {
							Thread.sleep(2000);
						} catch (InterruptedException ex) {}
						w.setVisible(false);
					}
				}).start();
				folderField.setText(sb.toString());
				folderField.validate();
			}
		}
	}

	class RemoteGuiDialogEntries {
		JComboBox<String> ipField;
		JComboBox<String> userField;
		JTextField folderField;
		JCheckBox useSCP;
		JCheckBox useLOCALDEF;
		JTextField LOCALDEFtf;
		JTextField cmd;
		JSpinner portSpinner;
		JLabel LOCALDEFlabel;
		BIGWizard wiz;
		RemoteDefinition def;
	}

	class CopyFilesToRemoteFolderAction extends AbstractAction {
		private static final long serialVersionUID = 1L;

		private final RemoteGuiDialogEntries gui;
		private final BIGConsole bcc;

		public CopyFilesToRemoteFolderAction(RemoteGuiDialogEntries gui, BIGConsole bcc) {
			this.gui = gui;
			this.bcc = bcc;
		}

		private boolean DeleteDir(File dir) {
			File[] files = dir.listFiles();
			if (files != null) {
				for (int i = 0; i < files.length; i++) {
					if (files[i].isDirectory()) {
						DeleteDir(files[i]);
					} else {
						files[i].delete();
					}
				}
				return dir.delete();
			}
			return false;
		}

		public void actionPerformed(ActionEvent evt) {
			try {
				// def contains all information of the first wizard screen
				// e.g. ip, user name, folder name, ssh/scp switch, localdef related information and
				// commands
				RemoteDefinition def = new RemoteDefinition((String) gui.ipField.getSelectedItem(),
						(String) gui.userField.getSelectedItem(), gui.folderField.getText(), false,
						gui.useSCP.isSelected(), null, gui.useLOCALDEF.isSelected(), gui.LOCALDEFtf.getText(),
						gui.cmd.getText(), (Integer) gui.portSpinner.getValue());

				if (!def.isValid()) {
					JOptionPane
							.showMessageDialog(
									null,
									"Your input is incomplete. Please fill out all fields.\r\nPlease note that the root folder of the remote host is not valid!",
									"Error", JOptionPane.ERROR_MESSAGE);
					gui.wiz.back(null);
					return;
				}

				gui.def = def;

				bcc.postMessage("Preparing temp folder...");
				// temporary folder where the files where stored first and than copied to remote folder
				final File temp = new File(BIGInterface.getInstance().getTempPath() + File.separator
						+ def.getFoldername());

				if (temp.exists() && !DeleteDir(temp)) {
					JOptionPane.showMessageDialog(null, "Could not empty temp directory!", "Error",
							JOptionPane.ERROR_MESSAGE);
					gui.wiz.setVisible(false);
					return;
				}

				// first: copy all files named in the whitelist to temporary folder
				// check if necessary whitelist and blacklist exist
				File whitelist = new File(BIGRemoteMenu.PATH_TO_LISTS + BIGRemoteMenu.WHITELIST_FILENAME);
				File blacklist = new File(BIGRemoteMenu.PATH_TO_LISTS + BIGRemoteMenu.BLACKLIST_FILENAME);
				if (!(whitelist.exists() && blacklist.exists())) {
					// at least one of both list DOES NOT exist
					// -> show an error dialog
					String errorMessage = new String("You need the files ");
					errorMessage = errorMessage + BIGRemoteMenu.WHITELIST_FILENAME + " and "
							+ BIGRemoteMenu.BLACKLIST_FILENAME + " in the folder " + BIGRemoteMenu.PATH_TO_LISTS
							+ ". See www.benchit.org to get these files.";
					JOptionPane.showMessageDialog(null, errorMessage, "Error", JOptionPane.ERROR_MESSAGE);
					gui.wiz.setVisible(false);
					return;
				}

				bcc.postMessage("Copying files to temp directory...");

				// parse the whitelist file line by line
				// copy all specified files/folders to temporary folder
				String benchitPath = BIGInterface.getInstance().getBenchItPath();
				String s, line = null;
				File parentFile;
				boolean ok = false;
				try {
					FileReader wlFileReader = new FileReader(whitelist);
					BufferedReader wlBufferedReader = new BufferedReader(wlFileReader);
					try {
						while ((line = wlBufferedReader.readLine()) != null) {
							// handle each line of whitelist file
							if (line.startsWith("#")) {
								// line is a comment so skip it
								continue;
							}
							// replace "/" with system specific file separator
							line = line.replace("/", File.separator);
							String pathOfCopyFile = benchitPath + File.separator + line;

							// get parent file of actual item
							parentFile = new File(pathOfCopyFile).getParentFile();
							// now we want to get the folder structure relative to the BenchIT directory
							s = parentFile.getAbsolutePath().substring(benchitPath.length());
							system.BIGFileHelper.copyToFolder(new File(pathOfCopyFile),
									new File(temp.getAbsolutePath() + s), true);
						}
						ok = true;
					} finally {
						wlBufferedReader.close();
					}
				} catch (FileNotFoundException fnfe) {
					bcc.postMessage("Couldn't find file " + whitelist.getAbsolutePath() + ".",
							BIGConsole.ERROR);
				} catch (IOException ioe) {
					bcc.postMessage("Exception while reading file " + whitelist.getAbsolutePath() + ".",
							BIGConsole.ERROR);
				}
				if (!ok) {
					gui.wiz.setFinished(true);
					return;
				}

				// in addition make a bin folder
				boolean binFolderCreated = new File(temp.getAbsolutePath() + File.separator + "bin")
						.mkdir();
				if (!binFolderCreated) {
					bcc.postMessage("Error while creating bin folder in " + temp.getAbsolutePath(),
							BIGConsole.ERROR);
					gui.wiz.setFinished(true);
					return;
				}

				bcc.postMessage("Removing unneccessary files...");

				// store all files in temporary directory to Vector
				Vector<File> vectorOfFiles = new Vector<File>();
				filesToVector(temp, vectorOfFiles);

				// now we are working with the blacklist
				// this list will contain entries that should NOT be copied to remote folder
				ArrayList<String> unwanted = new ArrayList<String>();
				ok = false;
				try {
					FileReader blFileReader = new FileReader(blacklist);
					BufferedReader blBufferedReader = new BufferedReader(blFileReader);
					try {
						line = null;
						while ((line = blBufferedReader.readLine()) != null) {
							// handle each line of whitelist file
							if (line.startsWith("#") || line.isEmpty()) {
								// line is a comment or empty -> so skip it
								continue;
							}

							s = temp.getAbsolutePath() + File.separator + line.replace("/", File.separator);
							// add line to unwanted items
							unwanted.add(s);
						}
						ok = true;
					} finally {
						blBufferedReader.close();
					}
				} catch (FileNotFoundException fnfe) {
					bcc.postMessage("Couldn't find file " + blacklist.getAbsolutePath() + ".",
							BIGConsole.ERROR);
				} catch (IOException ioe) {
					bcc.postMessage("Exception while reading file " + blacklist.getAbsolutePath() + ".",
							BIGConsole.ERROR);
				}
				if (!ok) {
					gui.wiz.setFinished(true);
					return;
				}

				for (int k = 0; k < unwanted.size(); k++) {
					// actualFilterFileName can contain * and ?
					String actualFilterFileName = unwanted.get(k);
					for (int i = 0; i < vectorOfFiles.size(); i++) {
						if (!vectorOfFiles.get(i).exists()) {
							vectorOfFiles.remove(i);
							i--;
						} else {
							if (isRegEx(
									actualFilterFileName.substring(temp.getAbsolutePath().length() + 1),
									vectorOfFiles.get(i).getAbsolutePath()
											.substring(temp.getAbsolutePath().length() + 1))) {
								BIGFileHelper.remove(vectorOfFiles.get(i));
								vectorOfFiles.remove(i);
								i--;
							}
						}
					}
				}

				if (def.getUseSCP()) {
					String command = "scp -r " + temp.getAbsolutePath() + " " + def.getUsername() + "@"
							+ def.getIP() + ":~/";
					Process p = null;
					bcc.postMessage(
							"Please enter your password at the console, you started the BenchIT-GUI with.",
							BIGConsole.DEBUG);

					try {
						p = Runtime.getRuntime().exec(command);
						bcc.addStream(p.getInputStream(), BIGConsole.DEBUG);
						bcc.addStream(p.getErrorStream(), BIGConsole.ERROR);
						p.waitFor();
					} catch (IOException ex) {
						bcc.postMessage("Couldn't copy folder to remote machine.", BIGConsole.ERROR);
						throw ex;
					} catch (InterruptedException ex1) {
						bcc.postMessage("Didn't wait for scp to end!", BIGConsole.ERROR);
						throw new Exception();
					}
				} else {
					Session sess = ConnectionHelper.openSession(def, loginFrame);
					if (sess == null) {
						bcc.postMessage(
								"Error while copying basic folder to remote system. \r\nConnection not available!",
								BIGConsole.ERROR);
						gui.wiz.setFinished(true);
						return;
					}
					try {
						try {
							bcc.addStream(sess.getStdout(), BIGConsole.DEBUG);
							bcc.addStream(sess.getStderr(), BIGConsole.ERROR);
						} catch (Exception e) {
							bcc.postMessage("Error(1) while copying basic folder to remote system",
									BIGConsole.ERROR);
							throw e;
						}
						TarOutputStream tarOut = null;
						try {
							tarOut = new TarOutputStream(sess.getStdin());
						} catch (Exception e) {
							bcc.postMessage("Error(2) while copying basic folder to remote system",
									BIGConsole.ERROR);
							throw e;
						}
						tarOut.setLongFileMode(TarOutputStream.LONGFILE_GNU);
						File[] allFiles = null;
						allFiles = system.BIGFileHelper.getAllSubFiles(temp, allFiles, 0);
						String path = "";
						if (def.getFoldername().startsWith("/")) {
							path = "-C /";
						}
						try {
							sess.execCommand("tar xBf - " + path);
							if (debug) {
								System.err.println("tar xBf - " + path);
							}
						} catch (IOException ex4) {
							bcc.postMessage("Error(3) while copying basic folder to remote system",
									BIGConsole.ERROR);
							try {
								tarOut.close();
							} catch (Exception ignored) {}
							throw ex4;
						}
						try {
							byte buffer[] = new byte[8192];
							// Open archive file
							boolean isWin = false;
							if (BIGInterface.getSystem() == BIGInterface.WINDOWS_SYSTEM) {
								isWin = true;
							}
							if (allFiles != null) {
								// skip temp dir!
								for (int i = 1; i < allFiles.length; i++) {
									File file = allFiles[i];
									if (file == null || !file.exists()) {
										continue;
									}

									// Add archive entry
									TarEntry tarAdd = new TarEntry(file);
									tarAdd.setModTime(file.lastModified());
									tarAdd.setName(file.getAbsolutePath().substring(
											temp.getAbsolutePath().lastIndexOf("temp") + 5));
									if (isWin) {
										tarAdd.setName(tarAdd.getName().replaceAll("\\\\", "/"));
									}
									tarOut.putNextEntry(tarAdd);
									// Write file to archive
									if (file.isFile()) {
										FileInputStream in = new FileInputStream(file);
										while (true) {
											int nRead = in.read(buffer, 0, buffer.length);
											if (nRead <= 0) {
												break;
											}
											tarOut.write(buffer, 0, nRead);
										}
										in.close();
									}
									tarOut.flush();
									tarOut.closeEntry();
								}
							}
						} catch (IOException ex3) {
							ex3.printStackTrace();
							bcc.postMessage("Error(5) while copying basic folder to remote system",
									BIGConsole.ERROR);
							try {
								tarOut.close();
							} catch (Exception ignored) {}
							throw ex3;
						} finally {
							try {
								tarOut.close();
							} catch (Exception ignored) {}
						}
						ConnectionHelper.closeSession(sess);
					} catch (Exception e) {
						ConnectionHelper.closeSession(sess);
						e.printStackTrace();
						return;
					}
				}
				// Retrieve hostname to be stored in config
				def.getHostname();
				bcc.postMessage("Succesfully copied main data to new remote-folder", BIGConsole.DEBUG);
				addRemoteDef(def);
			} catch (Exception e) {
				bcc.postMessage("Error occured: " + e.getLocalizedMessage());
				e.printStackTrace();
				gui.wiz.setFinished(true);
			}
		}
	}

	class CreateRemoteLocalDefAction extends AbstractAction {
		private static final long serialVersionUID = 1L;

		private final RemoteGuiDialogEntries gui;
		private final BIGConsole bcc;

		public CreateRemoteLocalDefAction(RemoteGuiDialogEntries gui, BIGConsole bcc) {
			this.gui = gui;
			this.bcc = bcc;
		}

		public void actionPerformed(ActionEvent evt) {
			try {
				RemoteDefinition def = gui.def;

				File localDefFile = new File(BIGInterface.getInstance().getBenchItPath() + File.separator
						+ "LOCALDEFS" + File.separator + def.getLOCALDEF());
				if (localDefFile.exists()) {
					bcc.postMessage("LOCALDEFS for " + def.getLOCALDEF() + " already exist, skip creating.");
					return;
				}

				Session s = ConnectionHelper.openSession(def, loginFrame);
				if (s == null) {
					bcc.postMessage("Connection failed!");
					gui.wiz.setFinished(true);
					return;
				}
				try {
					InputStream[] is = new InputStream[2];
					is[0] = s.getStdout();
					is[1] = s.getStderr();
					bcc.addStream(is[0], BIGConsole.DEBUG);
					bcc.addStream(is[1], BIGConsole.WARNING);
					String otherLocaldef = def.getLOCALDEF();
					String name = def.getHostname();
					String cmd =
					// go in remote dir
					"cd " + def.getFoldername() + " && echo \"start creating LOCALDEFs\" " +
					// add executable to tools
							" && chmod -R +x tools &&" +
							// dont ask for stuff
							" export BENCHIT_INTERACTIVE=0 &&" + " echo \"y\" | sh tools/FIRSTTIME &&";
					// add localdefname if necessary
					if (otherLocaldef != null && !name.equals(otherLocaldef)) {
						cmd = cmd + " mv LOCALDEFS/" + name + " LOCALDEFS/" + otherLocaldef + " && ";
						cmd = cmd + " mv LOCALDEFS/" + name + "_input_architecture LOCALDEFS/" + otherLocaldef
								+ "_input_architecture && ";
						cmd = cmd + " mv LOCALDEFS/" + name + "_input_display LOCALDEFS/" + otherLocaldef
								+ "_input_display && ";
					}
					cmd = cmd + " echo \"finished creating LOCALDEFs\" ";

					s.execCommand(cmd);

					// s.waitForCondition(ChannelCondition.STDOUT_DATA | ChannelCondition.STDERR_DATA
					// | ChannelCondition.EOF, 1000);
					ConnectionHelper.closeSession(s);
				} catch (IOException ex) {
					ConnectionHelper.closeSession(s);
					System.err.println("Couldnt start FIRSTTIME");
					throw ex;
				}
			} catch (Exception e) {
				bcc.postMessage("Error occured: " + e.getLocalizedMessage());
				e.printStackTrace();
				gui.wiz.setFinished(true);
			}
		}
	}

	class GetRemoteLocalDefAction extends AbstractAction {
		private static final long serialVersionUID = 1L;

		private final RemoteGuiDialogEntries gui;
		private final BIGConsole bcc;

		public GetRemoteLocalDefAction(RemoteGuiDialogEntries gui, BIGConsole bcc) {
			this.gui = gui;
			this.bcc = bcc;
		}
		public void actionPerformed(ActionEvent evt) {
			try {
				RemoteDefinition def = gui.def;

				// if we have the LOCALDEFS dont create em
				File localDefFile = new File(BIGInterface.getInstance().getBenchItPath() + File.separator
						+ "LOCALDEFS" + File.separator + def.getLOCALDEF());
				if (!localDefFile.exists()) {
					if (getLOCALDEFSFromRemote(def) != 0) {
						bcc.postMessage("Error getting remote LOCALDEFs! You may try this again!");
						return;
					}
				}
				bcc.postMessage("Press Finish to finish creating a remotefolder.\n"
						+ "You may load the LOCALDEFS of the remote-system\n" + def.getHostname()
						+ " by Loading LOCALDEFs.");
			} catch (Exception e) {
				bcc.postMessage("Error occured: " + e.getLocalizedMessage());
				e.printStackTrace();
				gui.wiz.setFinished(true);
			}
		}
	}

	/**
	 * shows a Dialog, which can create a Remote Folder
	 */
	private void createRemoteFolderDialog() {
		final RemoteGuiDialogEntries gui = new RemoteGuiDialogEntries();

		gui.wiz = new BIGWizard("Create new Session", false);
		JPanel jp = new JPanel(new GridLayout(14, 1));
		Vector<String> insets = new Vector<String>();
		for (int i = 0; i < remoteDefs.size(); i++) {
			if (!insets.contains(remoteDefs.get(i).getIP())) {
				insets.add(remoteDefs.get(i).getIP());
			}
		}
		gui.ipField = new JComboBox<String>(insets);
		gui.ipField.setMaximumRowCount(20);
		gui.ipField.setEditable(true);
		gui.folderField = new JTextField(20);
		gui.folderField.setEditable(true);
		gui.folderField
				.setToolTipText("insert the foldername here, special characters are prohibited.");
		gui.folderField.addKeyListener(new CheckFolderKey(gui.wiz));
		gui.folderField.addFocusListener(new CheckFolderKeyFocus(gui.folderField));
		gui.useSCP = new JCheckBox("use SCP instead of SSH/tar");
		gui.useSCP.setSelected(false);
		gui.useLOCALDEF = new JCheckBox("Define specific LOCALDEF for folder");
		gui.useLOCALDEF.setSelected(false);
		gui.LOCALDEFtf = new JTextField();
		gui.LOCALDEFtf.setEnabled(false);
		gui.LOCALDEFlabel = new JLabel("Specific LOCALDEF for this remote folder");
		gui.LOCALDEFlabel.setEnabled(false);
		gui.useLOCALDEF.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				gui.LOCALDEFtf.setEnabled(gui.useLOCALDEF.isSelected());
				gui.LOCALDEFlabel.setEnabled(gui.useLOCALDEF.isSelected());
			}
		});
		if (BIGInterface.getSystem() == BIGInterface.WINDOWS_SYSTEM) {
			gui.useSCP.setEnabled(false);
			gui.useSCP.setText(gui.useSCP.getText() + "(UNIX-Option only)");
		}
		insets = new Vector<String>();
		for (int i = 0; i < remoteDefs.size(); i++) {
			if (!insets.contains(remoteDefs.get(i).getUsername())) {
				insets.add(remoteDefs.get(i).getUsername());
			}
		}

		gui.userField = new JComboBox<String>(insets);
		gui.userField.setMaximumRowCount(20);
		gui.userField.setEditable(true);
		gui.portSpinner = new JSpinner();
		gui.portSpinner.setValue(new Integer(22));
		jp.add(new JLabel("IP or DNS-name of the remote computer"));
		jp.add(gui.ipField);
		jp.add(new JLabel("SSH Port"));
		jp.add(gui.portSpinner);
		jp.add(new JLabel("Username on the remote-computer"));
		jp.add(gui.userField);
		jp.add(new JLabel("Foldername within the home-directory"));
		jp.add(gui.folderField);
		jp.add(new JLabel("Shell command for remote computer"));
		gui.cmd = new JTextField("bash --login");
		jp.add(gui.cmd);
		jp.add(gui.useLOCALDEF);
		jp.add(gui.LOCALDEFlabel);
		jp.add(gui.LOCALDEFtf);
		jp.add(gui.useSCP);

		gui.wiz.addPanel(jp, "Set the information for the new Session");

		try {
			BIGConsole bcc = new BIGConsole(200, 200, null);
			JPanel pan = bcc.getDisplayPanel();
			pan.setBorder(BorderFactory.createTitledBorder("Output and Messages for copying"));
			Action act = new CopyFilesToRemoteFolderAction(gui, bcc);
			gui.wiz.addPanel(pan, act);

			bcc = new BIGConsole(200, 200, null);
			pan = bcc.getDisplayPanel();
			act = new CreateRemoteLocalDefAction(gui, bcc);
			gui.wiz.addPanel(pan, act);

			bcc = new BIGConsole(200, 200, null);
			pan = bcc.getDisplayPanel();
			act = new GetRemoteLocalDefAction(gui, bcc);
			gui.wiz.addPanel(pan, act);
		} catch (IOException ex) {
			JOptionPane.showMessageDialog(null, "Error creating console");
		}
		gui.wiz.start();
	}

	// stores the position of the vertical scrollbar
	int defsDialogPosition = 0;

	private List<RemoteDefinition> getDefsDialog(final String reason) {
		JCheckBox[] checkBoxes = new JCheckBox[remoteDefs.size()];
		for (int i = 0; i < checkBoxes.length; i++) {
			checkBoxes[i] = new JCheckBox(remoteDefs.get(i).toString(), remoteDefs.get(i).isSelected());
		}
		JPanel pan = new JPanel(new GridLayout(checkBoxes.length + 1, 1));
		pan.add(new JLabel("Select the Remote Folder " + reason));
		for (int i = 0; i < checkBoxes.length; i++) {
			pan.add(checkBoxes[i]);
		}
		final JScrollPane sp = new JScrollPane(pan);
		sp.getVerticalScrollBar().setValue(defsDialogPosition);
		sp.setPreferredSize(new Dimension(500, 500));
		sp.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		sp.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
		int value = JOptionPane.showConfirmDialog(bgui, sp, "select remote folders " + reason,
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		defsDialogPosition = sp.getVerticalScrollBar().getValue();
		List<RemoteDefinition> retRemDef = new ArrayList<RemoteDefinition>();
		if (value != JOptionPane.CANCEL_OPTION && value != JOptionPane.CLOSED_OPTION) {
			for (int i = 0; i < checkBoxes.length; i++) {
				remoteDefs.get(i).setSelected(checkBoxes[i].isSelected());
				if (checkBoxes[i].isSelected()) {
					retRemDef.add(remoteDefs.get(i));
				}
			}
		}
		return retRemDef;
	}

	/**
	 * opens a Dialog where to check for output-data
	 */
	private void checkForOutput() {
		if (remoteDefs.size() == 0) {
			showEmptyDialog();
			return;
		}
		List<RemoteDefinition> remDefs = getDefsDialog("to check for output");
		for (int j = 0; j < remDefs.size(); j++) {
			RemoteDefinition defs = remDefs.get(j);
			checkForOutputs(defs);
		}
	}

	/**
	 * shows a dialog where to get output data from
	 */
	private void getOutput() {
		(new Thread() {
			@Override
			public void run() {
				if (remoteDefs.size() == 0) {
					showEmptyDialog();
					return;
				}
				List<RemoteDefinition> remDefs = getDefsDialog("to get the output from");
				if (remDefs.size() == 0)
					return;
				for (int j = 0; j < remDefs.size(); j++) {
					getOutputFromRemote(remDefs.get(j));
					Thread t = new Thread() {
						@Override
						public void run() {
							bgui.getResultTree().updateResultTree(null);
							System.out.println("Remote results loaded!");
						}
					};
					SwingUtilities.invokeLater(t);
				}
			}
		}).start();
	}

	/**
	 * remove a remote folder
	 */
	private void removeRemoteFolder() {
		List<RemoteDefinition> defs = getDefsDialog("you want to remove");
		if (defs.size() == 0)
			return;
		int delCount = 0;
		for (int i = 0; i < defs.size(); i++) {
			if (deleteRemoteFolder(defs.get(i)) == 0) {
				System.out.println("Removed " + defs.get(i));
				delCount++;
			} else {
				System.out.println("Removing of " + defs.get(i) + " failed!");
			}
		}
		if (delCount > 0)
			JOptionPane.showMessageDialog(bgui, "Deletion of " + delCount
					+ " remote definitions finished!");
		else
			JOptionPane.showMessageDialog(bgui, "Nothing deleted!");
	}

	/**
	 * remove output from a remote folder
	 */
	private void removeRemoteOutput() {
		List<RemoteDefinition> defs = getDefsDialog("to remove output from");
		for (int i = 0; i < defs.size(); i++) {
			if (deleteRemoteOutput(defs.get(i)) == 0) {
				System.out.println("Removed output from " + defs.get(i));
			} else {
				System.out.println("Removing of output from" + defs.get(i) + " failed!");
			}
		}
	}

	/**
	 * starts the scp command to copy KernelDirectories and LOCALDEFS
	 * 
	 * @param def RemoteDefinition destination
	 * @param tempDir File source tempDir
	 * @return int returnstatus of scp
	 */
	private int scpCopyKernelsAndLOCDEFs(RemoteDefinition def, File tempDir) {
		if (def.getUseSCP()) {
			showTypeDialog(def);
			String command = "scp -r " + tempDir.getAbsolutePath() + " " + def.getUsername() + "@"
					+ def.getIP() + ":~/";
			Process p = null;
			BIGConsole bcc2 = null;
			try {
				bcc2 = new BIGConsole(0, 0, null);
				p = Runtime.getRuntime().exec(command);
				bcc2.addStream(p.getInputStream(), BIGConsole.DEBUG);
				bcc2.addStream(p.getErrorStream(), BIGConsole.ERROR);

			} catch (IOException ex) {
				System.err.println("Couldn't copy folder to remote machine.");
				return -1;
			}
			try {
				return p.waitFor();
			} catch (InterruptedException ex1) {
				System.err.println("Didn't wait for scp to end!");
				return -1;
			}

		} else {
			String path = "";
			if (def.getFoldername().startsWith("/")) {
				path = "-C /";
			}
			Session sess = ConnectionHelper.openSession(def, loginFrame);
			if (sess == null) {
				System.err.println("Error(0) while copying basic folder to remote system");
				return -1;

			}
			try {
				/*
				 * try { InputStream is[] = new InputStream[2]; is[0] = sess.getStdout(); is[1] = sess.getStderr(); BIGConsole c
				 * = new BIGConsole(300, 600, is); c.show() ; } catch (IOException ex2) { ex2.printStackTrace(); }
				 */
				TarOutputStream tarOut = null;

				try {
					tarOut = new TarOutputStream(sess.getStdin());
				} catch (Exception e) {
					System.err.println("Error(1) while copying basic folder to remote system");
					return -1;
				}
				tarOut.setLongFileMode(TarOutputStream.LONGFILE_GNU);
				File[] allFiles = null;
				allFiles = system.BIGFileHelper.getAllSubFiles(tempDir, allFiles, 0);
				try {
					sess.execCommand("tar xBf - " + path);
					if (debug) {
						System.err.println("tar xBf - " + path);
					}
				} catch (IOException ex4) {
					System.err.println("Error(2) while copying basic folder to remote system");
					try {
						tarOut.close();
					} catch (Exception e) {}
					return -1;
				}
				try {
					byte buffer[] = new byte[8192];
					// Open archive file

					if (allFiles != null) {
						boolean isWin = false;
						if (BIGInterface.getSystem() == BIGInterface.WINDOWS_SYSTEM) {
							isWin = true;
						}
						for (int i = 0; i < allFiles.length; i++) {
							File file = allFiles[i];
							if (file == null || !file.exists()) {
								continue;
							}

							// Add archive entry
							TarEntry tarAdd = new TarEntry(file);
							tarAdd.setModTime(file.lastModified());
							tarAdd.setName(file.getAbsolutePath().substring(
									tempDir.getAbsolutePath().lastIndexOf(
											File.separator + "temp" + File.separator + def.getFoldername()) + 6));
							if (isWin) {
								tarAdd.setName(tarAdd.getName().replaceAll("\\\\", "/"));
							}
							tarOut.putNextEntry(tarAdd);
							// Write file to archive
							if (file.isFile()) {
								FileInputStream in = new FileInputStream(file);
								while (true) {
									int nRead = in.read(buffer, 0, buffer.length);
									if (nRead <= 0) {
										break;
									}
									tarOut.write(buffer, 0, nRead);
								}
								in.close();
							}
							tarOut.flush();
							tarOut.closeEntry();
						}
						tarOut.flush();
						tarOut.close();
					}
				} catch (IOException ex3) {
					ex3.printStackTrace();
					System.err.println("Error(3) while copying basic folder to remote system");
					return -1;
				}
			} finally {
				ConnectionHelper.closeSession(sess);
			}
			return 0;

			/*
			 * TarArchive arch = null ; try { sess.execCommand( "tar xf - " + path ) ; arch = new TarArchive( sess.getStdin()
			 * , tempDir ) ; } catch ( Exception ex4 ) { System.err.println(
			 * "Error(1) while copying something to remote directory" ) ; } try { arch.writeEntry( new TarEntry( tempDir ) ,
			 * true ) ; } catch ( Exception ex3 ) { System.err.println( "Error(2) while copying something to remote directory"
			 * ) ; } try { BIGInterface.getInstance().getConsole().addStream( sess. getStdout() , BIGConsole.DEBUG ) ; } catch
			 * ( Exception ex2 ) { System.err.println( "Error(3) while copying something to remote directory" ) ; return -1 ;
			 * } try { arch.closeArchive() ; } catch ( IOException ex ) { }
			 */
		}
	}

	/**
	 * starts ssh command, that checks for output
	 * 
	 * @param def RemoteDefinition destination
	 * @return int returnstatus of ssh
	 */
	private int checkForOutputs(RemoteDefinition def) {
		Session sess = null;
		sess = ConnectionHelper.openSession(def, loginFrame);
		if (sess == null)
			return 0;
		BIGConsole bcc = null;
		try {
			InputStream[] in = new InputStream[2];
			in[0] = sess.getStdout();
			in[1] = sess.getStderr();
			bcc = new BIGConsole(400, 400, in);
			bcc.setTitle("Results on " + def.getIP() + "(" + def.getFoldername() + ")");
			bcc.setVisible(true);
			sess.execCommand("echo \"Found results\" && find " + def.getFoldername()
					+ " -name *.bit && echo \"FINISHED\"");
			if (debug) {
				System.err.println("find " + def.getFoldername() + " -name *.bit");
			}

		} catch (IOException ex2) {
			System.err.println("Error(2) while checking for output");
			return -1;

		}

		while (bcc.getText().indexOf("\nFINISHED") == -1) {
			;
		}
		if (bcc.getText().equals("Found results\nFINISHED")) {
			System.out.println("No results were found.");
		}
		ConnectionHelper.closeSession(sess);
		return 0;
	}

	/**
	 * starts scp-command to get outputdata from a RemoteDefinition
	 * 
	 * @param def RemoteDefinition to get from this
	 * @return int returnstatus from scp
	 */
	private int getOutputFromRemote(RemoteDefinition def) {
		if (def.getUseSCP()) {

			showTypeDialog(def);
			String command = "scp -r " + def.getUsername() + "@" + def.getIP() + ":~/"
					+ def.getFoldername() + "/" + "output " + " "
					+ BIGInterface.getInstance().getBenchItPath();
			Process p = null;
			BIGConsole bcc2 = null;
			try {
				bcc2 = new BIGConsole(0, 0, null);
				p = Runtime.getRuntime().exec(command);
				bcc2.addStream(p.getInputStream(), BIGConsole.DEBUG);
				bcc2.addStream(p.getErrorStream(), BIGConsole.ERROR);

			} catch (IOException ex) {
				System.err.println("Couldn't copy folder to remote machine.");
				return -1;
			}
			try {
				return p.waitFor();
			} catch (InterruptedException ex1) {
				System.err.println("Didn't wait for scp to end!");
				return -1;
			}

		} else {
			try {
				JLabel status = BIGInterface.getInstance().getStatusLabel();
				String path = "";
				if (def.getFoldername().startsWith("/")) {
					path = "-C / ";
				}
				File tempFile = new File("temp.tar");
				Session sess = ConnectionHelper.openSession(def, loginFrame);
				if (sess == null)
					return -1;
				try {
					status.setText("open Session");
					sess.execCommand("tar cBf - " + path + def.getFoldername() + "/output");
					status.setText("get output");
					OutputStream out = new FileOutputStream(tempFile);

					InputStream is = sess.getStdout();

					byte[] buff = new byte[8192];

					while (true) {
						if (is.available() < 0) {
							break;
						}
						int len = is.read(buff);
						if (len < 0) {
							break;
						}
						out.write(buff, 0, len);
					}

					out.flush();
					out.close();
					String tempContent = BIGFileHelper.getFileContent(tempFile);
					if (tempContent.indexOf(def.getFoldername()) < 0) {
						JOptionPane.showMessageDialog(null, "The folder seems to contain no output-data");
						status.setText("done");
						return -1;
					}
					tempContent = tempContent.substring(tempContent.indexOf(def.getFoldername()));
					BIGFileHelper.saveToFile(tempContent, tempFile);

					File tempDir = (new File(BIGInterface.getInstance().getTempPath()));
					TarInputStream tarIn = new TarInputStream(new FileInputStream(tempFile));
					try {
						TarEntry actualEntry = tarIn.getNextEntry();
						while (actualEntry != null) {
							if (actualEntry.isDirectory()) {
								File newDirectory = new File(tempDir.getAbsolutePath() + File.separator
										+ actualEntry.getName());
								newDirectory.mkdirs();
							} else {
								// byte[] buffer = new byte[8192];
								File newFile = new File(tempDir.getAbsolutePath() + File.separator
										+ actualEntry.getName());
								FileOutputStream outNew = new FileOutputStream(newFile);
								tarIn.copyEntryContents(outNew);
								/*
								 * FileInputStream in = new FileInputStream( actualEntry.getFile() ) ; while ( true ) { int nRead =
								 * in.read( buffer , 0 , buffer.length ) ; if ( nRead <= 0 ) break ; outNew.write( buffer , 0 , nRead )
								 * ; } in.close() ;
								 */
								outNew.close();
							}
							actualEntry = tarIn.getNextEntry();
						}
					} finally {
						tarIn.close();
					}
					/*
					 * TarArchive arch = new TarArchive( new FileInputStream( tempFile ) ) ; File tempDir = ( new File(
					 * BIGInterface.getInstance().getTempPath() ) ) ; try { arch.extractContents( tempDir ) ; } catch (
					 * IOException ex6 ) { ex6.printStackTrace() ; System.err.println( "Couldn't extract temporary tar-file" ) ;
					 * return -1 ; }
					 */

					BIGFileHelper.copyToFolder(
							new File(tempDir.getAbsolutePath() + File.separator + def.getFoldername()
									+ File.separator + "output"), new File(BIGInterface.getInstance()
									.getBenchItPath()), false);
					status.setText("removing temporary files");
					BIGFileHelper.remove(tempDir);
					BIGFileHelper.remove(tempFile);
				} finally {
					ConnectionHelper.closeSession(sess);
				}
				status.setText("done");

				return 0;
			} catch (IOException ex2) {
				ex2.printStackTrace();
				return -1;
			}

		}

	}
	/**
	 * starts scp-command to get outputdata from a RemoteDefinition
	 * 
	 * @param def RemoteDefinition to get from this
	 * @return int returnstatus from scp
	 */
	private int getLOCALDEFSFromRemote(RemoteDefinition def) {
		if (def.getUseSCP()) {
			showTypeDialog(def);
			String command = "scp -r " + def.getUsername() + "@" + def.getIP() + ":~/"
					+ def.getFoldername() + "/" + "LOCALDEFS " + " "
					+ BIGInterface.getInstance().getBenchItPath();
			Process p = null;
			BIGConsole bcc2 = null;
			try {
				bcc2 = new BIGConsole(0, 0, null);
				p = Runtime.getRuntime().exec(command);
				bcc2.addStream(p.getInputStream(), BIGConsole.DEBUG);
				bcc2.addStream(p.getErrorStream(), BIGConsole.ERROR);

			} catch (IOException ex) {
				System.err.println("Couldn't copy LOCALDEFS from remote machine.");
				return -1;
			}
			try {
				return p.waitFor();
			} catch (InterruptedException ex1) {
				System.err.println("Didn't wait for scp to end!");
				return -1;
			}
		} else {
			try {
				JLabel status = BIGInterface.getInstance().getStatusLabel();
				String path = "";
				if (def.getFoldername().startsWith("/")) {
					path = "-C / ";
				}
				File tempFile = new File("temp.tar");
				Session sess = ConnectionHelper.openSession(def, loginFrame);
				if (sess == null)
					return -1;
				try {
					status.setText("open Session");
					sess.execCommand("tar cBf - " + path + def.getFoldername() + "/LOCALDEFS");
					status.setText("get LOCALDEFS");
					OutputStream out = new FileOutputStream(tempFile);

					InputStream is = sess.getStdout();

					byte[] buff = new byte[8192];

					while (is.available() >= 0) {
						int len = is.read(buff);
						if (len < 0) {
							break;
						}
						out.write(buff, 0, len);
					}

					out.flush();
					out.close();
					String tempContent = BIGFileHelper.getFileContent(tempFile);
					if (tempContent.indexOf(def.getFoldername()) < 0) {
						JOptionPane.showMessageDialog(null, "The folder seems to contain no LOCALDEFS");
						status.setText("done");
						return -1;
					}
					tempContent = tempContent.substring(tempContent.indexOf(def.getFoldername()));
					BIGFileHelper.saveToFile(tempContent, tempFile);

					File tempDir = new File(BIGInterface.getInstance().getTempPath());
					TarInputStream tarIn = new TarInputStream(new FileInputStream(tempFile));
					try {
						TarEntry actualEntry = tarIn.getNextEntry();
						while (actualEntry != null) {
							if (actualEntry.isDirectory()) {
								File newDirectory = new File(tempDir.getAbsolutePath() + File.separator
										+ actualEntry.getName());
								newDirectory.mkdirs();
							} else {
								// byte[] buffer = new byte[8192];
								File newFile = new File(tempDir.getAbsolutePath() + File.separator
										+ actualEntry.getName());
								FileOutputStream outNew = new FileOutputStream(newFile);
								tarIn.copyEntryContents(outNew);
								outNew.close();
							}
							actualEntry = tarIn.getNextEntry();
						}
					} finally {
						tarIn.close();
					}
					BIGFileHelper.copyToFolder(
							new File(tempDir.getAbsolutePath() + File.separator + def.getFoldername()
									+ File.separator + "LOCALDEFS"), new File(BIGInterface.getInstance()
									.getBenchItPath()), false);
					status.setText("removing temporary files");
					BIGFileHelper.remove(tempDir);
					BIGFileHelper.remove(tempFile);
				} finally {
					ConnectionHelper.closeSession(sess);
				}
				status.setText("done");
				return 0;
			} catch (IOException ex2) {
				ex2.printStackTrace();
				return -1;
			}
		}
	}

	/**
	 * starts ssh - command to remove a remote-folders output data from a remote computer
	 * 
	 * @param remDefPos int position in this.remoteDefs
	 * @return int returnstatus of the ssh command
	 */
	private int deleteRemoteOutput(RemoteDefinition def) {
		Session sess = ConnectionHelper.openSession(def, loginFrame);
		if (sess == null) {
			System.err.println("Couldn't remove remote output.");
			return -1;
		}

		String home = "~";
		if (def.getFoldername().startsWith("/")) {
			home = "";
		}
		try {
			sess.execCommand("rm -r " + home + "/" + def.getFoldername() + "/output/*");
		} catch (IOException ex2) {
			ConnectionHelper.closeSession(sess);
			System.err.println("Couldn't remove remote folder.");
			return -1;
		}
		ConnectionHelper.closeSession(sess);
		return 0;
	}

	/**
	 * starts ssh - command to remove a remote-folder from a remote computer and from this.remoteDefs
	 * 
	 * @param remDefPos int position in this.remoteDefs
	 * @return int returnstatus of the ssh command
	 */
	private int deleteRemoteFolder(RemoteDefinition def) {
		Session sess = ConnectionHelper.openSession(def, loginFrame);
		if (sess == null) {
			if (JOptionPane.showConfirmDialog(bgui, "Could not connect to " + def.getIP()
					+ ".\nDelete RemoteDefinition?", "Connection failed", JOptionPane.YES_NO_OPTION,
					JOptionPane.QUESTION_MESSAGE) == JOptionPane.YES_OPTION) {
				removeRemoteDef(def);
				return 0;
			}
			System.err.println("Couldn't remove remote folder.");
			return -2;
		}

		String home = "~";
		if (def.getFoldername().startsWith("/")) {
			home = "";
		}

		try {
			sess.execCommand("rm -r " + home + "/" + def.getFoldername());
		} catch (IOException ex2) {
			ConnectionHelper.closeSession(sess);
			if (JOptionPane.showConfirmDialog(bgui,
					"Could not delete remote folder. Error: " + ex2.getLocalizedMessage()
							+ "\n\nDelete RemoteDefinition?", "Deleting failed", JOptionPane.YES_NO_OPTION,
					JOptionPane.QUESTION_MESSAGE) == JOptionPane.YES_OPTION) {
				removeRemoteDef(def);
				return 0;
			}
			System.err.println("Couldn't remove remote folder.");
			return -1;
		}
		ConnectionHelper.closeSession(sess);
		removeRemoteDef(def);
		return 0;
	}
	private void addRemoteDef(RemoteDefinition def) {
		remoteDefs.add(def);
		try {
			saveRemoteDefs();
		} catch (Exception ex) {
			System.err.println("Couldn't save \"RemoteXMLFile\". Can't save new entry");
		}
	}

	private void removeRemoteDef(RemoteDefinition def) {
		remoteDefs.remove(def);
		try {
			saveRemoteDefs();
		} catch (Exception ex) {
			ex.printStackTrace();
			System.out.println("Couldn't save \"RemoteXMLFile\".Can't remove entry");
		}
	}

	public void saveRemoteDefs() throws Exception {
		synchronized (this) {
			RemoteDefinition.saveRemoteDefsToXML(remoteDefs, new File(BIGInterface.getInstance()
					.getConfigPath()
					+ File.separator
					+ BIGInterface.getInstance().getBIGConfigFileParser().stringCheckOut("remoteXMLFile")));
		}
	}

	/**
	 * @return List<RemoteDefinition> this.remoteDefs
	 */
	public List<RemoteDefinition> getRemoteDefs() {
		return remoteDefs;
	}

	private void showTypeDialog(RemoteDefinition def) {
		if (BIGInterface.getSystem() == BIGInterface.UNIX_SYSTEM) {
			boolean b = BIGInterface.getInstance().getBIGConfigFileParser()
					.boolCheckOut("showRemoteTypeDialog", true);
			if (b) {
				JOptionPane.showMessageDialog(bgui, "Please type the password for " + def
						+ " at the console you started BenchIT at.");
			}
		}
	}

	private void showEmptyDialog() {
		JOptionPane.showMessageDialog(bgui, "There are no remotefolders defined, add some.");
	}

	private BIGStrings createPermissionsScript(RemoteDefinition def) {
		BIGStrings script = new BIGStrings();

		String directory = "${PWD}/" + def.getFoldername();
		if (def.getFoldername().startsWith("/")) {
			directory = def.getFoldername();
		}
		// header of file
		script.add("#!/bin/sh");
		// header of file
		script.add("PWD=`pwd`");
		script.add("echo \"##### permissions in bin- and tools-folder will be set to +rwx! ##### \" ");
		script.add("chmod -R +rwx " + directory + "/bin");
		script.add("chmod -R +rwx " + directory + "/tools");
		script.add("chmod -R +rwx " + directory + "/COMPILE.SH");
		script.add("chmod -R +rwx " + directory + "/RUN.SH");
		script.add("rm $0");
		return script;
	}

	/**
	 * Copies files for selected kernel to remote folder ready for manual compilation etc.
	 * 
	 * @param def
	 * @param runEntries
	 * @param generatePostProcScript
	 */
	public void copyFilesToRemoteMachine(RemoteDefinition def, List<BIGRunEntry> runEntries,
			boolean generatePostProcScript) {
		BIGConsole console = BIGInterface.getInstance().getConsole();
		File tempDir = new File(BIGInterface.getInstance().getTempPath() + File.separator
				+ def.getFoldername());
		tempDir.mkdirs();
		File kernelDir = new File(tempDir.getAbsolutePath() + File.separator + "kernel");
		kernelDir.mkdirs();
		File locDefDir = new File(tempDir.getAbsolutePath() + File.separator + "LOCALDEFS");
		locDefDir.mkdirs();
		// copy kernels to tempdir
		while (!kernelDir.exists() || !locDefDir.exists()) {}
		// System.err.println("Kernels:"+kernels);
		for (int i = 0; i < runEntries.size(); i++) {
			String kernelName = runEntries.get(i).kernelName;

			BIGKernel k = bgui.getKernelTree().findKernel(kernelName);
			if (k == null) {
				System.err.println("An error occured while searching kernel " + kernelName);
				return;
			}
			File f3 = new File(kernelDir.getAbsolutePath() + File.separator + k.getRelativePath())
					.getParentFile();
			f3.mkdirs();
			BIGFileHelper.copyToFolder(new File(k.getAbsolutePath()), f3, true);
		}
		// copy tools-folder
		BIGFileHelper.copyToFolder(new File(BIGInterface.getInstance().getBenchItPath()
				+ File.separator + "tools"), tempDir, true);
		// and benchit.c
		BIGFileHelper.copyToFolder(new File(BIGInterface.getInstance().getBenchItPath()
				+ File.separator + "benchit.c"), tempDir, true);
		// and interface.h
		BIGFileHelper.copyToFolder(new File(BIGInterface.getInstance().getBenchItPath()
				+ File.separator + "interface.h"), tempDir, true);
		BIGFileHelper.copyToFolder(new File(BIGInterface.getInstance().getBenchItPath()
				+ File.separator + "COMPILE.SH"), tempDir, true);
		BIGFileHelper.copyToFolder(new File(BIGInterface.getInstance().getBenchItPath()
				+ File.separator + "RUN.SH"), tempDir, true);

		if (generatePostProcScript) {
			// generate shellscript
			if (generatePostProcScript(runEntries, def) == -1)
				return;
		} else {
			createPermissionsScript(def).saveToFile(
					BIGInterface.getInstance().getTempPath() + File.separator + def.getFoldername()
							+ File.separator + "setPermissions.sh");
		}
		// if not LOCALDEF-check copy all
		BIGFileHelper.copyToFolder(new File(BIGInterface.getInstance().getBenchItPath()
				+ File.separator + "LOCALDEFS"), tempDir, true);
		// remove unwanted files and folders
		String unwanted = "";
		try {
			unwanted = BIGInterface.getInstance().getBIGConfigFileParser().stringCheckOut("dontCopyList");
		} catch (Exception ex) {
			BIGInterface.getInstance().getBIGConfigFileParser()
					.set("dontCopyList", "tools/kernel-gen*,jbi/jni/libjbi.so");
		}
		StringTokenizer tokenizer = new StringTokenizer(unwanted, ",");
		// store all files in directory to Vector
		File[] actualFiles = tempDir.listFiles();
		Vector<File> allFiles = new Vector<File>();
		for (int i = 0; i < actualFiles.length; i++) {
			allFiles.add(actualFiles[i]);
		}
		for (int i = 0; i < allFiles.size(); i++) {
			File actual = (allFiles.get(i));
			if (actual.isDirectory()) {
				actualFiles = actual.listFiles();
				for (int k = 0; k < actualFiles.length; k++) {
					allFiles.add(actualFiles[k]);
				}
			}
		}
		while (tokenizer.hasMoreTokens()) {
			// actualFile can contain * and ?
			String actualFilterFileName = tokenizer.nextToken().trim();
			for (int i = 0; i < allFiles.size(); i++) {
				if (!allFiles.get(i).exists()) {
					allFiles.remove(i);
					i--;
				} else {
					if (isRegEx(actualFilterFileName,
							allFiles.get(i).getAbsolutePath().substring(tempDir.getAbsolutePath().length() + 1))) {
						BIGFileHelper.remove(allFiles.get(i));
						allFiles.remove(i);
						i--;
					}
				}
			}
		}
		console.postMessage("Copying files to remote machine...");
		scpCopyKernelsAndLOCDEFs(def, tempDir);
		BIGFileHelper.remove(tempDir);
		console.postMessage("Files copied!");
		if (!generatePostProcScript) {
			console.postMessage("Setting permissions");
			if (execRemoteCommand(def, def.getFoldername() + "/setPermissions.sh"))
				console.postMessage("Permissions set!");
			else
				console
						.postMessage("Could not set permissions. Run setPermissions.sh to set them manually.");
		}
	}

	/**
	 * opens a dialog, that asks on which remote folder the kernels shall be executed
	 * 
	 * @param kernels BIGStrings the kernels (matmul_double)
	 * @param scripts BIGStrings the scripts (COMPILE.SH,RUN.SH,)
	 */
	public void startWorkOnAnotherMachine(List<BIGRunEntry> runEntries) {
		if (remoteDefs.size() == 0) {
			showEmptyDialog();
			return;
		}
		List<RemoteDefinition> remDefs = getDefsDialog("to start selected kernels at");
		BIGConsole console = BIGInterface.getInstance().getConsole();
		for (int j = 0; j < remDefs.size(); j++) {
			RemoteDefinition def = remDefs.get(j);
			console.postMessage("Preparing execution for " + def);
			copyFilesToRemoteMachine(def, runEntries, true);
			console.postMessage("Starting execution...");
			startRemoteProc(def);
		}
	}

	/**
	 * opens a dialog, that asks on which remote folder the kernels shall be executed
	 * 
	 * @param kernels BIGStrings the kernels (matmul_double)
	 * @param scripts BIGStrings the scripts (COMPILE.SH,RUN.SH,)
	 */
	public void copyFilesToAnotherMachine(List<BIGRunEntry> runEntries) {
		if (remoteDefs.size() == 0) {
			showEmptyDialog();
			return;
		}
		List<RemoteDefinition> remDefs = getDefsDialog("to copy files to");
		BIGConsole console = BIGInterface.getInstance().getConsole();
		for (int j = 0; j < remDefs.size(); j++) {
			RemoteDefinition def = remDefs.get(j);
			console.postMessage("Preparing file copy for " + def);
			copyFilesToRemoteMachine(def, runEntries, false);
		}
		SwingUtilities.invokeLater(new Thread() {
			@Override
			public void run() {
				JOptionPane.showMessageDialog(null, "All files copied to remote folder(s)!",
						"File copy complete", JOptionPane.INFORMATION_MESSAGE);
			}
		});
	}

	private boolean execRemoteCommand(RemoteDefinition def, String cmd) {
		String shellCmd = def.getShellCommand();
		final Session sess = ConnectionHelper.openSession(def, loginFrame);
		if (sess == null) {
			System.err.println("Couldn't start remote-proc.");
			return false;
		}
		try {
			sess.execCommand("nohup " + shellCmd + " " + cmd);
		} catch (IOException e) {
			ConnectionHelper.closeSession(sess);
			e.printStackTrace();
			System.err.println("Couldn't start remote procedure.");
			return false;
		}
		ConnectionHelper.closeSession(sess);
		return true;
	}

	private int startRemoteProc(RemoteDefinition def) {
		String shellCmd = def.getShellCommand();
		final Session sess = ConnectionHelper.openSession(def, loginFrame);
		if (sess == null) {
			System.err.println("Couldn't start remote-proc.");
			return -1;
		}

		InputStream[] ins = new InputStream[2];
		try {
			ins[0] = sess.getStderr();
			ins[1] = sess.getStdout();
			BIGConsole bcc = new BIGConsole(600, 400, ins) {
				/** Default Serial Version UID to get rid of warnings */
				private static final long serialVersionUID = 1L;

				@Override
				public void dispose() {
					ConnectionHelper.closeSession(sess);
				}
			};
			bcc.setTitle("Kernel(s) running on " + def.getIP() + "(" + def.getFoldername() + ")");
			bcc.setVisible(true);
			sess.execCommand("nohup " + shellCmd + " " + def.getFoldername() + "/remoteproc.sh | tee "
					+ def.getFoldername() + "/output.txt");

		} catch (IOException ex1) {
			ConnectionHelper.closeSession(sess);
			ex1.printStackTrace();
			System.err.println("Couldn't start remote procedure.");
			return -1;
		}
		ConnectionHelper.closeSession(sess);
		return 0;
	}

	private int generatePostProcScript(List<BIGRunEntry> runEntries, RemoteDefinition def) {
		String comment;
		String hostname = def.getHostname();
		if (hostname == null) {
			System.err.println("Couldn't get hostname");
			return -1;
		}

		if (def.getLOCALDEF().equals(BIGInterface.getInstance().getHost())) {
			comment = BIGInterface.getInstance().getEntry("<hostname>", "BENCHIT_FILENAME_COMMENT")
					.getValue().toString();
		} else {
			BIGOutputParser parser;
			try {
				parser = new BIGOutputParser(BIGInterface.getInstance().getBenchItPath() + File.separator
						+ "LOCALDEFS" + File.separator + def.getLOCALDEF());
				comment = parser.getValue("BENCHIT_FILENAME_COMMENT");
			} catch (BIGParserException e) {
				comment = "";
			}
		}
		if (comment == null || comment.isEmpty()) {
			comment = "0";
		}
		String noParam = "";
		String targetRun = " --target=" + def.getLOCALDEF();
		String targetCompile = " --target=" + def.getLOCALDEF() + " ";

		try {
			if (BIGInterface.getInstance().getBIGConfigFileParser()
					.boolCheckOut("settedTargetActiveCompileRemote")) {
				targetCompile = " --target="
						+ BIGInterface.getInstance().getBIGConfigFileParser()
								.stringCheckOut("settedTargetCompileRemote");
			}
			if (BIGInterface.getInstance().getBIGConfigFileParser()
					.boolCheckOut("settedTargetActiveRunRemote")) {
				targetRun = " --target="
						+ BIGInterface.getInstance().getBIGConfigFileParser()
								.stringCheckOut("settedTargetRunRemote");
			}
			if (BIGInterface.getInstance().getBIGConfigFileParser()
					.boolCheckOut("runRemoteWithoutParameter")) {
				noParam = " --no-parameter-file";
			}

		} catch (Exception ex) {
			System.err.println("At least one variable for compiling /"
					+ " running on remote systems was not set" + " within BGUI.cfg. Using defaults.");
		}

		BIGStrings script = new BIGStrings();

		String directory = "${PWD}/" + def.getFoldername();
		if (def.getFoldername().startsWith("/")) {
			directory = def.getFoldername();
		}
		// header of file
		script.add("#!/bin/sh");
		script.add("echo \"##### starting postproceeding / selected kernels at " + def.getHostname()
				+ " with comment " + comment + "\"");
		// header of file
		script.add("PWD=`pwd`");
		script.add("echo \"##### permissions in bin- and tools-folder will be set to +rwx! ##### \" ");
		script.add("chmod -R +rwx " + directory + "/bin");
		script.add("chmod -R +rwx " + directory + "/tools");
		script.add("chmod -R +rwx " + directory + "/COMPILE.SH");
		script.add("chmod -R +rwx " + directory + "/RUN.SH");
		// kernel starts
		for (int pos = 0; pos < runEntries.size(); pos++) {
			BIGRunEntry entry = runEntries.get(pos);
			BIGKernel k = bgui.getKernelTree().findKernel(entry.kernelName);
			if (k == null) {
				System.err.println("couldn't find kernel " + entry.kernelName);
				continue;
			}
			// maybe BENCHIT_FILENAME_COMMENT is defined in PARAMETERS
			File files[] = k.getFiles();
			for (int l = 0; l < files.length; l++) {
				if (files[l].getName().equals("PARAMETERS")) {
					BIGOutputParser parser;
					try {
						parser = new BIGOutputParser(files[l].getName());
					} catch (BIGParserException e) {
						continue;
					}
					String paramComment = parser.getValue("BENCHIT_FILENAME_COMMENT");
					if (paramComment != null && !paramComment.isEmpty()) {
						comment = paramComment;
					}
				}
			}

			script.add("echo \"##### starting \\\"" + k.getName() + "\\\"\"");

			if (entry.type == BIGRunType.Both) {
				script.add(directory + "/COMPILE.SH" + targetCompile + " " + k.getNameAfterSorting(0));
				String run = directory + "/RUN.SH";
				run = run + targetRun + noParam + " " + k.getNameAfterSorting(0);
				run = run + "." + comment;
				run = run + " --output-dir=" + directory + "/output";
				script.add("if [ $? -eq 0 ]; then");
				script.add(run);
				script.add("fi");
			} else if (entry.type == BIGRunType.Compile) {
				script.add(directory + "/COMPILE.SH" + targetCompile + " " + k.getNameAfterSorting(0));
			} else
				return onlyRunCompilations(def, k.getNameAfterSorting(0));
			// TODO: Fix above! This method should create the shell script!
			script.add("echo \"##### finished \\\"" + entry.kernelName + "\\\"\"");
		}
		// write file
		try {
			script.saveToFile(BIGInterface.getInstance().getTempPath() + File.separator
					+ def.getFoldername() + File.separator + "remoteproc.sh");
		} catch (Exception e) {
			System.err.println("BIGExecute: Error during writing of remoteproc.sh\n" + e);
			return -1;
		}
		return 0;
	}

	public int onlyRunCompilations(RemoteDefinition def, String match) {
		if (debug) {
			System.err.println("--------------onlyRun..()");
			System.err.println("Def:" + def);
		}
		// Session sess=null;
		String output = new String();
		BIGConsole bcc = null;
		Session sess = null;
		try {
			sess = ConnectionHelper.openSession(def, loginFrame);
			if (sess == null) {
				System.err.println("Couldn't run remote executables.");
				return -1;
			}
			try {
				InputStream[] is = new InputStream[2];
				is[1] = sess.getStderr();
				is[0] = sess.getStdout();

				bcc = new BIGConsole(1, 1, is);
				// "find " + def.getFoldername() +
				// "/bin/ -name *.*.*"
				// sess.getStdin().write(("ls " + def.getFoldername() + "/bin/*\n").getBytes());
				sess.execCommand("ls " + def.getFoldername() + "/bin/* && echo \"FINISHED\"");
			} finally {
				ConnectionHelper.closeSession(sess);
			}
		} catch (IOException ex3) {
			ex3.printStackTrace();
			System.err.println("Couldn't check for executables");
			return -1;
		}
		while (bcc.getText().indexOf("\nFINISHED") == -1) {
			;
		}
		ConnectionHelper.closeSession(sess);
		output = bcc.getText();

		StringTokenizer st = null;
		st = new StringTokenizer(output, "\n");
		Vector<String> kernels = new Vector<String>();
		// int nextToWrite = 0 ;
		String directory = def.getFoldername() + "/bin/";
		// first the executables (c,f)
		String temp;
		// System.err.println(st.countTokens());
		while (st.hasMoreElements()) {
			temp = (String) st.nextElement();
			if (debug) {
				System.err.println("Element:" + temp);
			}
			if (temp.startsWith(directory)) {
				if (debug) {
					System.err.println("starts with dir");
				}
				kernels.add(temp);
				if (kernels.lastElement().endsWith(":") && (temp.indexOf(match) != -1)) {
					if (debug) {
						System.err.println("is dir");
					}
					kernels.add(temp.substring(0, temp.length() - 1) + "/RUN.SH");
					// nextToWrite++ ;
					// now the directories(java)
					while (st.hasMoreElements()) {
						temp = (String) st.nextElement();
						if (temp.startsWith(directory) && temp.endsWith(":") && (temp.indexOf(match) != -1)) {
							kernels.add(temp.substring(0, temp.length() - 1) + "/RUN.SH");
						}
					}
				}
				// nextToWrite++;
			} else {
				if ((temp.indexOf("RUN.SH") != -1) && (temp.indexOf(match) != -1)) {
					kernels.add(temp.substring(0, temp.lastIndexOf("/") + 1) + "RUN.SH "
							+ temp.substring(temp.lastIndexOf("/") + 1, temp.length()));
					// nextToWrite++ ;
				}
			}
		}
		if (kernels.size() < 1) {
			System.out.println("There are no compiled kernels within the binary folder");
			return -1;
		}
		// while ((numberOfKernels<kernels.length)&&(kernels[numberOfKernels]!=null))
		// numberOfKernels++;
		String[] allKernels = new String[kernels.size()];
		for (int i = 0; i < allKernels.length; i++) {
			allKernels[i] = kernels.get(i);
		}

		JList<String> selectionList = new JList<String>(allKernels);
		// selectionList.setVisibleRowCount(10);
		selectionList.setBorder(BorderFactory.createTitledBorder("available compilations"));
		JTextField jt = new JTextField();
		jt.setBorder(BorderFactory.createTitledBorder("regular expression (includes *'s and ?'s)"));
		Object[] o = new Object[3];
		o[0] = "Found kernels:";
		o[1] = selectionList;
		o[2] = jt;
		int value = JOptionPane.showConfirmDialog(null, o, "select remote compilations",
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		if (value == JOptionPane.CANCEL_OPTION)
			return -1;
		else {
			String add = "";
			try {
				add = " --target=" + def.getLOCALDEF();
				if (BIGInterface.getInstance().getBIGConfigFileParser()
						.boolCheckOut("settedTargetActiveRunRemote")) {
					add = " --target="
							+ BIGInterface.getInstance().getBIGConfigFileParser()
									.stringCheckOut("settedTargetRunRemote");
				}
				if (BIGInterface.getInstance().getBIGConfigFileParser()
						.boolCheckOut("runRemoteWithoutParameter")) {
					add = add + " --no-parameter-file";
				}

			} catch (Exception e) {
				System.err.println("At least one variable in BGUI.cfg not set. Using defaults.");
			}
			String[] returnStrings;
			// regular expression
			if ((jt.getText() != null) && (!jt.getText().equals(""))) {
				Vector<String> vOfEexecs = new Vector<String>();
				// getting matches in JList for regex
				for (int i = 0; i < allKernels.length; i++) {
					if (isRegEx(jt.getText(), allKernels[i])) {
						vOfEexecs.add(allKernels[i]);
					}
				}
				// turn Objects into Strings
				o = vOfEexecs.toArray();
				returnStrings = new String[o.length];
				if (returnStrings.length == 0)
					return -1;
				for (int i = 0; i < o.length; i++) {
					returnStrings[i] = (String) o[i];
					String kernelDir = returnStrings[i].substring(0);
					while (kernelDir.indexOf(".") > -1) {
						kernelDir = kernelDir.substring(0, kernelDir.indexOf(".")) + "/"
								+ kernelDir.substring(kernelDir.indexOf(".") + 1, kernelDir.length());
					}
					kernelDir = kernelDir.substring(0, kernelDir.lastIndexOf("/"));
					while (kernelDir.charAt(0) == '/') {
						kernelDir = kernelDir.substring(1);
					}
					returnStrings[i] = returnStrings[i].substring(def.getFoldername().length() + 5);
					returnStrings[i] = def.getFoldername() + "/RUN.SH" + add + " " + returnStrings[i]
							+ " --output-dir=${PWD}/" + def.getFoldername() + "/output";
				}
			} else
			// selected in list
			{
				// turn Objects into Strings
				returnStrings = new String[selectionList.getSelectedValuesList().size()];
				for (int i = 0; i < returnStrings.length; i++) {
					returnStrings[i] = (selectionList.getSelectedValuesList().get(i));
					returnStrings[i] = returnStrings[i].substring(def.getFoldername().length() + 5);

					returnStrings[i] = def.getFoldername() + "/RUN.SH" + add + " " + returnStrings[i]
							+ " --output-dir=${PWD}/" + def.getFoldername() + "/output";
				}
				if (returnStrings.length == 0)
					return -1;
			}
			File tempDir = new File(BIGInterface.getInstance().getTempPath() + File.separator
					+ def.getFoldername());
			tempDir.mkdirs();
			StringBuffer fileContent = new StringBuffer();
			fileContent.append("#!/bin/sh\n");
			fileContent.append("PWD=`pwd`\n");
			fileContent
					.append("echo \"##### permissions in bin- and tools-folder will be set to +rwx! ##### \" \n");
			fileContent.append("chmod -R +rwx ${PWD}/" + def.getFoldername() + "/bin\n");
			fileContent.append("chmod -R +rwx ${PWD}/" + def.getFoldername() + "/tools\n");
			fileContent.append("chmod -R +rwx ${PWD}/" + def.getFoldername() + "/COMPILE.SH\n");
			fileContent.append("chmod -R +rwx ${PWD}/" + def.getFoldername() + "/RUN.SH\n");
			fileContent.append("echo \"##### starting selected kernels\"\n");

			for (int i = 0; i < returnStrings.length; i++) {
				fileContent.append(returnStrings[i] + "\n");
			}
			fileContent.append("echo \"##### finished selected kernels\"\n");

			// generate shellscript
			system.BIGFileHelper.saveToFile(fileContent.toString(),
					new File((BIGInterface.getInstance().getTempPath() + File.separator + def.getFoldername()
							+ File.separator + "remoteproc.sh")));
			BIGFileHelper.copyToFolder(new File(BIGInterface.getInstance().getBenchItPath()
					+ File.separator + "LOCALDEFS"), new File(BIGInterface.getInstance().getTempPath()),
					false

			);
			int retInt = scpCopyKernelsAndLOCDEFs(def, tempDir);
			if (retInt != 0)
				return retInt;
			// BIGFileHelper.remove( tempDir ) ;
			return 0; // startRemoteProc( def ) ;
		}
	}

	/**
	 * opens a dialog to select some executables which will be copied to a remote folder
	 * 
	 * @param def RemoteDefinition remotefolder to copy to
	 * @return int errorReturn by ssh/scp commands
	 */
	public int copyExecutables(RemoteDefinition def) {
		File f = new File(BIGInterface.getInstance().getBenchItPath() + File.separator + "bin"
				+ File.separator);
		File[] files = f.listFiles(new FileFilter() {
			public boolean accept(File fileInBinPath) {
				String temp = fileInBinPath.getName();
				int numberOfDots = 0;
				while (temp.indexOf(".") > -1) {
					numberOfDots++;
					temp = temp.substring(temp.indexOf(".") + 1, temp.length());
				}
				if (numberOfDots > 5)
					return true;

				// if ( fileInBinPath.isDirectory() )return true ;
				return false;
			}
		});
		if (files.length == 0) {
			System.out.println("No executables were found");
			return -1;
		}
		JList<File> selectionList = new JList<File>(files);
		// just show the filename not the path
		selectionList.setCellRenderer(new DefaultListCellRenderer() {
			/** Default Serial Version UID to get rid of warnings */
			private static final long serialVersionUID = 1L;

			@Override
			public Component getListCellRendererComponent(JList<?> list, Object value, int index,
					boolean isSelected, boolean cellHasFocus) {
				Component c = super.getListCellRendererComponent(list, value, index, isSelected,
						cellHasFocus);
				setText(((File) value).getName());
				return c;
			}
		});
		selectionList.setVisibleRowCount(10);
		selectionList.setBorder(BorderFactory.createTitledBorder("available compilations"));
		JTextField jt = new JTextField();
		jt.setBorder(BorderFactory.createTitledBorder("regular expression (includes *'s and ?'s)"));
		Object[] o = new Object[3];
		o[0] = "Found executables:";
		o[1] = selectionList;
		o[2] = jt;
		int value = JOptionPane.showConfirmDialog(null, o, "select executables to copy",
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		if (value == JOptionPane.CANCEL_OPTION)
			return -1;
		else {

			File[] returnStrings;
			// regular expression
			if ((jt.getText() != null) && (!jt.getText().equals(""))) {
				Vector<File> vOfEexecs = new Vector<File>();
				// getting matches in JList for regex
				for (int i = 0; i < files.length; i++) {
					if (isRegEx(jt.getText(), files[i].getName())) {
						vOfEexecs.add(files[i]);
					}
				}
				// turn Objects into Strings
				o = vOfEexecs.toArray();
				returnStrings = new File[o.length];
				for (int i = 0; i < o.length; i++) {
					returnStrings[i] = (File) o[i];
				}
			} else
			// selected in list
			{
				// turn Objects into Strings
				returnStrings = new File[selectionList.getSelectedValuesList().size()];
				for (int i = 0; i < returnStrings.length; i++) {
					returnStrings[i] = (selectionList.getSelectedValuesList().get(i));
				}
				if (returnStrings.length == 0)
					return -1;
			}
			File tempDir = new File(BIGInterface.getInstance().getTempPath() + File.separator
					+ def.getFoldername());
			File binDir = new File(tempDir.getPath() + File.separator + "bin");
			binDir.mkdirs();
			for (int i = 0; i < returnStrings.length; i++) {
				BIGFileHelper.copyToFolder(returnStrings[i], binDir, true);
			}
			return scpCopyKernelsAndLOCDEFs(def, tempDir);
		}
	}

	/**
	 * This thread consumes output from the remote server and displays it in the terminal window.
	 */
	class RemoteConsumer extends Thread {
		private boolean run = false;
		StringBuffer sb = new StringBuffer();
		InputStream in = null;

		RemoteConsumer(InputStream in) {
			this.in = in;
		}

		private void addText(byte[] data, int len) {
			for (int i = 0; i < len; i++) {
				char c = (char) (data[i] & 0xff);
				if (data[i] != 0) {
					sb.append(c);
				}
			}
		}

		@Override
		public void run() {
			run = true;
			byte[] buff = new byte[8192];

			try {
				while (run) {
					int len = in.read(buff);
					if (len == -1)
						return;
					addText(buff, len);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		public String getText() {
			return sb.toString();
		}

		public void stopIt() {
			run = false;
		}
	}

	/*
	*/

	/**
	 * checks for sth. like regular expressions but only supports *'s and ?'s
	 * 
	 * @param regEx String the pseudo-regular-expression
	 * @param check String the sentence to check
	 * @return boolean is check expressed by regEx?
	 */
	public static boolean isRegEx(String regEx, String check) {
		// System.err.println("IsRegEx:\tregEx = " + regEx + "\t\tcheck = " + check);

		if (regEx.length() > check.length())
			return false;
		if ((regEx.length() == 0) && (check.length() == 0))
			return true;
		if (regEx.length() == 0)
			return false;
		if (check.length() == 0)
			return false;
		int i = 0;
		{
			// ?s
			if (regEx.charAt(i) == '?')
				return isRegEx(regEx.substring(1, regEx.length()), check.substring(1, check.length()));
			else
			// *s
			if (regEx.charAt(i) == '*') {
				// if the last char is a *
				if (i == regEx.length() - 1)
					return true;
				boolean retBool = false;
				for (int j = 0; j < check.length(); j++) {
					retBool = (retBool || isRegEx(regEx.substring(1, regEx.length()),
							check.substring(j, check.length())));
				}
				return retBool;
			} else
			// other characters
			{
				if (regEx.charAt(i) == check.charAt(i))
					return isRegEx(regEx.substring(1, regEx.length()), check.substring(1, check.length()));
				else
					return false;
			}
		}
	}
}
