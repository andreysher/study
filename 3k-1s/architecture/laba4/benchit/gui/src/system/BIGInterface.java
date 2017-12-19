/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGInterface.java Author:
 * SWTP Nagel 1 Last change by: $Author: tschuet $ $Revision: 1.15 $ $Date: 2008/06/02 10:10:37 $
 ******************************************************************************/
package system;

import gui.*;

import java.io.*;
import java.net.UnknownHostException;
import java.util.*;

import javax.swing.JLabel;

/**
 * The main interface of the programm, what the GUI and the underlayer should use for communicating. So it is the holder
 * of the intern database.
 * <P>
 * There exists only one instance of the interface which could be get through <code>getInstance</code>.
 * 
 * @author <a href="mailto:fx@fx-world.de">Pascal Weyprecht</a>
 * @see BIGEntry
 **/
public class BIGInterface {
	// this is the one and only instance existing
	private static BIGInterface bigInterface;

	// this is the structure, where all entries are placed in
	private Map<String, Map<String, BIGEntry>> db;

	private class HostName extends Observable {
		private String hostName = null;
		public String get() {
			return hostName;
		}
		public void set(String hostName) {
			if (hostName.equals(this.hostName))
				return;
			this.hostName = hostName;
			setChanged();
			notifyObservers();
		}
		@Override
		public String toString() {
			return get();
		}
	}
	private final HostName hostname = new HostName();
	// the actually name of the computer this programm is running
	private String origHostname = null;

	// the path where BenchIT is
	private String benchItPath;

	// the name of the BIG config file
	private String configFilename = null;

	// whether to show the "New"-Button in Admin-Mode or not
	// private boolean showAdminNew=false;

	// tells GUI at start to supress daily tip and display result tree
	private boolean restart = false;

	// the debug modes for each class
	private Map<String, Integer> debug;

	// the console frame
	private BIGConsole console;

	// the look and feel
	private String lookAndFeel = "default";

	// should the admin tool to be loaded or not
	private boolean loadAdminTool = false;

	// A ProgressBar for several Prozesses
	private Observer progressBar;

	// private JProgressBar startProgress;

	// status bar components
	private JLabel statusLabel = null;
	private Observer statusProgress = null;

	// save the args comming over command line
	private String[] arguments;

	private static int DB_FILE_INITCAPACITY = 5;
	private static int DB_ENTRY_INITCAPACITY = 500;

	// args from the Config File BGUI.cfg and functions to get a value
	private BIGConfigFileParser cfgFileParser = null;

	// the actual System runs Windows
	public static int WINDOWS_SYSTEM = 0;
	// the actual System runs Unix
	public static int UNIX_SYSTEM = 1;

	/**
	 * Constructs a BIGInterface. But because of it should exist only one instance use getInstance instead.
	 **/
	private BIGInterface() {
		db = new HashMap<String, Map<String, BIGEntry>>(DB_FILE_INITCAPACITY);
		debug = new HashMap<String, Integer>();
		// getting path:

		File relativeDir = new File(System.getProperties().getProperty("user.dir"));
		while (true) {
			File[] filesInRelDir = relativeDir.listFiles(new FilenameFilter() {
				public boolean accept(File f, String s) {
					if (s.equals("benchit.c"))
						return true;
					return false;
				}
			});
			if ((filesInRelDir != null) && (filesInRelDir.length > 0)) {
				break;
			}
			relativeDir = relativeDir.getParentFile();
		}
		benchItPath = relativeDir.getAbsolutePath();

	}

	/**
	 * This method can be used to help finding bugs in the database.
	 */
	public void debugDB() {
		if (db != null) {
			System.out.println("BIGInterface @142:");
			System.out.println(db.keySet());
			Iterator<String> it = db.keySet().iterator();
			while (it.hasNext()) {
				String dbKeyName = it.next();
				System.out.println(dbKeyName);
				if (db.get(dbKeyName) == null) {
					continue;
				}
				HashMap<?, ?> dbValueMap = (HashMap<?, ?>) db.get(dbKeyName);
				Iterator<?> it2 = dbValueMap.keySet().iterator();
				while (it2.hasNext()) {
					String dbValueKeyName = (String) it2.next();
					if (dbValueMap.get(dbValueKeyName) == null) {
						continue;
					}
					BIGEntry dbEntry = (BIGEntry) dbValueMap.get(dbValueKeyName);
					if (dbEntry.getPos() != 1) {
						System.out.println(dbEntry.getName() + ": pos=" + dbEntry.getPos());
					}
				}
			}
		}
	}

	/**
	 * Returns the one and only instance of the interface. If there doesn't exist one, it creates one.
	 * 
	 * @return the instance pointer
	 **/
	public static BIGInterface getInstance() {
		if (bigInterface == null) {
			bigInterface = new BIGInterface();
		}
		return bigInterface;
	}

	/**
	 * sets the ConfigFileParser (and with it its values) to this.cfgFilePArser
	 * 
	 * @author roberts
	 **/
	public void setBIGConfigFileParser(BIGConfigFileParser pars) {
		cfgFileParser = pars;
	}

	public BIGConfigFileParser getBIGConfigFileParser() {
		return cfgFileParser;
	}

	/** Sets the JLabel usable for status messages. */
	public void setStatusLabel(JLabel label) {
		statusLabel = label;
	}

	public void setStatusProgress(Observer progressBar) {
		statusProgress = progressBar;
	}

	/** Gets the JLabel usable for status messages. */
	public JLabel getStatusLabel() {
		return statusLabel;
	}

	public Observer getStatusProgress() {
		return statusProgress;
	}

	/**
	 * sets some values of the Config File and saves it (writes it to disk)
	 */
	public int setConfigFileContents(BIGGUI jf) {
		// sets size and position fromthe windows
		java.awt.Rectangle where = jf.getBounds();
		cfgFileParser.set("mainWindowXPos", (new Integer((int) where.getX()).toString()));
		cfgFileParser.set("mainWindowYPos", (new Integer((int) where.getY()).toString()));
		cfgFileParser.set("mainWindowXSize", (new Integer((int) where.getWidth()).toString()));
		cfgFileParser.set("mainWindowYSize", (new Integer((int) where.getHeight()).toString()));
		cfgFileParser.set("mainWindowRightSplit", (new Integer(jf.getRightSplit()).toString()));
		cfgFileParser.set("mainWindowMainSplit", (new Integer(jf.getMainSplit()).toString()));
		cfgFileParser.set("detailLevel", jf.getDetailLevel().name());
		this.setConsole();
		int i = cfgFileParser.save();
		return i;
	}

	/**
	 * sets some values of the Config File and saves it (writes it to disk)
	 */
	/*
	 * public int setConfigFileContents(BIGAdmin jf) { java.awt.Rectangle where =jf.getBounds();
	 * this.cfgFileParser.set("adminToolXPos",(new Integer((int)where.getX()).toString()));
	 * this.cfgFileParser.set("adminToolYPos",(new Integer((int)where.getY()).toString()));
	 * this.cfgFileParser.set("adminToolXSize",(new Integer((int)where.getWidth()).toString()));
	 * this.cfgFileParser.set("adminToolYSize",(new Integer((int)where.getHeight()).toString())); this.setConsole(); int
	 * i=this.cfgFileParser.save(); return i; }
	 */
	/**
	 * sets console values of the Config File
	 */
	public void setConsole() {
		// sets openConsole
		if (console.isShowing()) {
			cfgFileParser.set("openConsole", "1");
		} else {
			cfgFileParser.set("openConsole", "0");
		}
		java.awt.Rectangle where = console.getBounds();

		cfgFileParser.set("consoleWindowXPos", (new Integer((int) where.getX()).toString()));
		cfgFileParser.set("consoleWindowYPos", (new Integer((int) where.getY()).toString()));
		cfgFileParser.set("consoleWindowXSize", (new Integer((int) where.getWidth()).toString()));
		cfgFileParser.set("consoleWindowYSize", (new Integer((int) where.getHeight()).toString()));
	}

	/**
	 * Loads in the whole database with the actually hostname.<BR>
	 * In normal mode it starts the XMLParser and the DefFileParser.<BR>
	 * When the adminTool is activated, it only runs the XML Parser.
	 */
	public void load() throws BIGParserException {
		// saving the old db and creating a new
		Map<String, Map<String, BIGEntry>> oldDB = db;
		db = new HashMap<String, Map<String, BIGEntry>>(DB_FILE_INITCAPACITY);

		try {
			// loading of the XMLconfig file
			BIGXMLParser myXMLParser = new BIGXMLParser();
			if (progressBar != null) {
				myXMLParser.getObservable().addObserver(progressBar);
			}
			if (statusProgress != null) {
				myXMLParser.getObservable().addObserver(statusProgress);
			}
			if (statusLabel != null)
				statusLabel.setText("Loading " + getConfigFile());
			myXMLParser.read(new File(getConfigFile()).toURI().toURL(), false);

			try {
				Thread.sleep(0, 1);
			} catch (InterruptedException ie) {}

			if (loadAdminTool == false) {
				// loading of the defFiles
				BIGDefFileParser myDefFileParser = new BIGDefFileParser();
				if (progressBar != null) {
					myDefFileParser.getObservable().addObserver(progressBar);
				}
				if (statusProgress != null) {
					myDefFileParser.getObservable().addObserver(statusProgress);
				}
				Iterator<String> tempIterator = getAllFilenames().iterator();
				while (tempIterator.hasNext()) {
					// System.err.println(System.currentTimeMillis()-time);
					String filename = tempIterator.next();
					if (statusLabel != null) {
						statusLabel.setText("Loading " + filename.replaceAll("<hostname>", hostname.get()));
					}
					Iterator<BIGEntry> entryIterator = iterator(filename);
					// System.err.println("entries for filename:"+(System.currentTimeMillis()-time));
					while (entryIterator.hasNext()) {
						BIGEntry entry = entryIterator.next();
						if (entry.getValue() != null) {
							entry.setPrototypeValue(entry.getValue());
						}
					}
					myDefFileParser.read(benchItPath + File.separator + "LOCALDEFS" + File.separator
							+ filename.replaceAll("<hostname>", hostname.get()), filename,
							getAllEntrynames(filename));
					if (statusLabel != null) {
						statusLabel.setText("done");
					}
					try {
						Thread.sleep(0, 1);
					} catch (InterruptedException ie) {}
					if (checkForFilled(filename)) {
						myDefFileParser.write(benchItPath + File.separator + "LOCALDEFS" + File.separator
								+ filename.replace("<hostname>", hostname.get()), filename,
								getAllEntrynames(filename));
					}
				}
			}
		} catch (BIGParserException e) {
			db = oldDB;
			throw e;
		} catch (Exception e) {
			e.printStackTrace();
			throw new BIGParserException(
					"BIGInterface: unreporteted Exception from one of the Parser\nBIGInterface: " + e);
		}
	}

	/**
	 * is nearly the same as load, but here we read just the first tab, because the other tabs
	 * (input_display/architecture) arent used so often those get now just loaded when someone presses the second tab
	 * 
	 * @param first if true, only the <hostname> file will be loaded, if false the 2 input files will be loaded.
	 */
	/*
	 * public void loadFirst( boolean first ) throws BIGParserException { // saving the old db and creating a new Map
	 * oldDB = db; if ( first ) db = new HashMap( DB_FILE_INITCAPACITY ); try { // loading of the XMLconfig file
	 * BIGXMLParser myXMLParser = new BIGXMLParser(); if ( progressBar != null )
	 * myXMLParser.getObservable().addObserver(progressBar); if ( statusProgress != null )
	 * myXMLParser.getObservable().addObserver( statusProgress ); if ( statusLabel != null ) statusLabel.setText(
	 * "Loading " + getConfigFilename() ); myXMLParser.read( getConfigFilename(), first ); try { Thread.sleep( 0, 1 ); }
	 * catch ( InterruptedException ie ) {} if ( loadAdminTool == false ) { // loading of the defFiles BIGDefFileParser
	 * myDefFileParser = new BIGDefFileParser(); if ( progressBar != null ) myDefFileParser.getObservable().addObserver(
	 * progressBar ); if ( statusProgress != null ) myDefFileParser.getObservable().addObserver( statusProgress );
	 * Iterator tempIterator = getAllFilenames().iterator(); while ( tempIterator.hasNext() ) { String filename =
	 * (String)tempIterator.next(); if ( first ) { if ( filename.endsWith( "_input_architecture" ) || filename.endsWith(
	 * "_input_display" ) ) continue; } else // !first { if ( !( filename.endsWith( "_input_architecture" ) ||
	 * filename.endsWith( "_input_display" ) ) ) continue; } if ( statusLabel != null ) statusLabel.setText( "Loading " +
	 * filename.replaceAll( "<hostname>", hostname ) ); Iterator entryIterator = iterator( filename ); while (
	 * entryIterator.hasNext() ) { BIGEntry entry = (BIGEntry)entryIterator.next(); if ( entry.getValue() != null ) {
	 * entry.setPrototypeValue( entry.getValue() ); } } // loading defFile of host myDefFileParser.read( benchItPath +
	 * File.separator + "LOCALDEFS" + File.separator + filename.replaceAll( "<hostname>", hostname ), filename,
	 * getAllEntrynames( filename ) ); if ( statusLabel != null ) statusLabel.setText( "done" ); try { Thread.sleep( 0, 1
	 * ); } catch ( InterruptedException ie ) {} } } } catch ( BIGParserException e ) { db = oldDB; throw e; } catch (
	 * Exception e ) { throw new BIGParserException( "BIGInterface: unreporteted" +
	 * " Exception from one of the Parser\nBIGInterface: " + e ); } }
	 */

	/**
	 * Saves the whole database with the actually hostname.<BR>
	 * In normal mode it starts the DefFileParser.<BR>
	 * When adminTool activated, it runs only the XMLParser.
	 * 
	 * @param adminCall if true, it means the adminTool panel calls this method.
	 */
	public void save(boolean adminCall) throws BIGParserException {
		// boolean lATChanged = false;
		if (loadAdminTool && adminCall) {
			loadAdminTool = false;
			// lATChanged = true;
		}
		if (!loadAdminTool && !adminCall) {
			// we don't have to save the config.xml, but the localdefs
			BIGDefFileParser myDefFileParser = new BIGDefFileParser();
			if ((progressBar != null) && !adminCall) {
				myDefFileParser.getObservable().addObserver(progressBar);
			}
			if (statusProgress != null) {
				myDefFileParser.getObservable().addObserver(statusProgress);
			}

			Iterator<String> tempIterator = getAllFilenames().iterator();
			while (tempIterator.hasNext()) {
				String filename = tempIterator.next();
				boolean exception = false;
				if (statusLabel != null) {
					statusLabel.setText("Saving " + filename.replace("<hostname>", hostname.get()));
				}
				try {
					myDefFileParser.write(benchItPath + File.separator + "LOCALDEFS" + File.separator
							+ filename.replace("<hostname>", hostname.get()), filename,
							getAllEntrynames(filename));
				} catch (BIGParserException e) {
					exception = true;
					throw e;
				} catch (Exception e) {
					exception = true;
					e.printStackTrace();
					throw new BIGParserException("BIGInterface: unreporteted "
							+ "Exception from one of the Parser\nBIGInterface: " + e);
				}
				if (exception) {
					statusLabel.setText("Some error occured! " + "For details check the console output. ");
				} else {
					statusLabel.setText("done");
				}
			}
		} else {
			// we have to save the config.xml, but not the localdefs
			// verifing db
			boolean db_ok = true;
			Iterator<BIGEntry> it = iterator();
			while (it.hasNext() && db_ok) {
				db_ok = db_ok && it.next().verify();
			}
			if (db_ok) {
				// saving the XML file
				Thread thread = null;
				BIGXMLParser myXMLParser = new BIGXMLParser();
				if ((progressBar != null) && !adminCall) {
					myXMLParser.getObservable().addObserver(progressBar);
				}
				if (statusProgress != null) {
					myXMLParser.getObservable().addObserver(statusProgress);
				}
				if (statusLabel != null) {
					statusLabel.setText("Saving " + getConfigFile());
				}
				try {
					thread = myXMLParser.write(getConfigFile());
				} catch (BIGParserException e) {
					throw e;
				} catch (Exception e) {
					e.printStackTrace();
					throw new BIGParserException("BIGInterface: unreporteted "
							+ "Exception from one of the Parser\nBIGInterface: " + e);
				}
				// If admin call is true the admin tool
				// is integrated in another gui and calling this
				// method with adminCall == false. In that case it
				// happens, that the entry values switch to their
				// defaults. As this is not wanted in an integrated
				// gui we need to reload the LOCALDEF files after
				// saving the config.xml file.
				final Thread writer = thread;
				// final boolean setLAT = lATChanged;
				if (adminCall && (writer != null)) {
					Thread loadThread = new Thread() {
						@Override
						public void run() {
							try {
								writer.join();
							} catch (InterruptedException ie) {}
							try {
								load();
							} catch (BIGParserException bpe) {
								console.postMessage("Cannot load file\n" + bpe, BIGConsole.ERROR);
								System.err.println("Cannot load file\n" + bpe);
							}
						}
					};
					loadThread.start();
				}
			} else {
				System.err
						.println("BIGInterface: Database verifing " + "failed. XML file hasn't be saved.");
			}
		}
	}

	/**
	 * Sets the actually hostname, the name of the computer.
	 * 
	 * @param hostname the name of the actually computer
	 **/
	public void setHost(String hostname) {
		if (hostname == null) {
			this.hostname.set("noname");
		} else if (hostname.indexOf(".") > 0) {
			this.hostname.set(hostname.substring(0, hostname.indexOf(".")));
		} else {
			this.hostname.set(hostname);
		}
	}

	/**
	 * Returns the actually hostname, the name of the computer.
	 * 
	 * @return the name of the actually computer
	 **/
	public String getHost() {
		return hostname.get();
	}

	public Observable getHostObservable() {
		return hostname;
	}

	/**
	 * Returns the original hostname, the name of the computer.
	 * 
	 * @return the name of the actually computer
	 **/
	public String getOriginalHost() {
		if (origHostname == null) {
			try {
				String name = java.net.InetAddress.getLocalHost().getCanonicalHostName();
				// if ( name.indexOf( "." ) > 0 )
				// retval = name.substring( 0, name.indexOf( "." ) );
				// else
				if (name.indexOf(".") > -1) {
					origHostname = name.substring(0, name.indexOf("."));
				} else {
					origHostname = name;
				}
			} catch (UnknownHostException ex) {}
		}
		return origHostname;
	}

	/**
	 * Sets the original hostname, the name of the computer. This method is and should be called only in BIGArgsParser
	 * 
	 * @param newHost the name of the actually computer
	 **/
	public void setOriginalHost(String newHost) {
		if (newHost == null) {
			newHost = "noname";
		}
		origHostname = newHost;
	}

	/**
	 * Adds a DEF-file to the database.<BR>
	 * When the file already exists an <code>BIGDoubleIdentifierException</code> is thrown.
	 * 
	 * @param filename the name of the new file
	 **/
	public void addFile(String filename) throws BIGDoubleIdentifierException {
		if (db.containsKey(filename))
			throw new BIGDoubleIdentifierException("File " + filename + " already exits");
		else {
			db.put(filename, new HashMap<String, BIGEntry>(DB_ENTRY_INITCAPACITY));
		}
	}

	/**
	 * Deletes a DEF-File from the db.
	 * 
	 * @param filename the name of the file to delete
	 */
	public void removeFile(String filename) {
		if (db.containsKey(filename)) {
			db.remove(filename);
		} else {
			System.err.println("File \"" + filename + "\" does not exists.");
		}

	}

	/**
	 * Adds an entry to the database. When an entry with the same name already exists an
	 * <code>BIGDoubleIdentifierException</code> is thrown.
	 * 
	 * @param filename the name of the file where to add the entry
	 * @param entry the entry which to add
	 **/
	public void addEntry(String filename, BIGEntry entry) throws BIGDoubleIdentifierException {
		Map<String, BIGEntry> fileEntry = db.get(filename);
		if (fileEntry == null) {
			addFile(filename);
			fileEntry = db.get(filename);
		}
		if (fileEntry.containsKey(entry.getName()))
			throw new BIGDoubleIdentifierException("Entry " + entry.getName() + " already exits");
		else {
			fileEntry.put(entry.getName(), entry);
		}
	}

	/**
	 * Removes an entry from the database. When the entry does not exist, doesn't matter, otherwise it is erased.
	 * 
	 * @param filename the name of the file where to remove the entry
	 * @param entryName the name of the entry which to remove
	 **/
	public void removeEntry(String filename, String entryName) {
		Map<String, BIGEntry> fileEntry = db.get(filename);
		if (fileEntry != null) {
			fileEntry.remove(entryName);
		}
	}

	/**
	 * Returns an entry of the database.
	 * 
	 * @param filename the name of the file where the entry is
	 * @param entryName the name of the entry
	 * @return the entry, that represents a parameter
	 **/
	public BIGEntry getEntry(String filename, String entryName) {
		Map<String, BIGEntry> tempMap = db.get(filename);
		if (tempMap == null)
			return null;
		else
			return tempMap.get(entryName);
	}

	/**
	 * Returns all files in the database. These are NOT the filenames, these are the <code>Map</code>s representing a
	 * file.<BR>
	 * The keys are the parameternames and the values are the entries.
	 * 
	 * @return all files, an <code>ArrayList</code> with a <code>Map</code> inside.
	 **/
	public ArrayList<Map<String, BIGEntry>> getAllFiles() {
		ArrayList<Map<String, BIGEntry>> result = new ArrayList<Map<String, BIGEntry>>();
		Collection<Map<String, BIGEntry>> tempCollection = db.values();
		Iterator<Map<String, BIGEntry>> tempIterator = tempCollection.iterator();
		while (tempIterator.hasNext()) {
			result.add(tempIterator.next());
		}
		return result;
	}

	/**
	 * Returns all filenames in the database.
	 * 
	 * @return all filenames of the database
	 **/
	public BIGStrings getAllFilenames() {
		BIGStrings result = new BIGStrings();
		Iterator<String> tempIterator;
		ArrayList<String> resultCollection = new ArrayList<String>();
		Collection<String> tempCollection = db.keySet();
		if (tempCollection == null)
			return new BIGStrings();

		tempIterator = tempCollection.iterator();
		while (tempIterator.hasNext()) {
			resultCollection.add(tempIterator.next());
		}

		Collections.sort(resultCollection);

		tempIterator = resultCollection.iterator();
		while (tempIterator.hasNext()) {
			result.add(tempIterator.next());
		}
		return result;
	}

	/**
	 * Returns all entries of one file out of the database. The entries are sorted be its <code>position</code>.
	 * 
	 * @param filename the name of the file where the entries are from
	 * @return all entries of one file
	 **/
	public ArrayList<BIGEntry> getAllEntries(String filename) {
		ArrayList<BIGEntry> result;
		Map<String, BIGEntry> fileEntry = db.get(filename);
		if (fileEntry == null) {
			result = null;
		} else {
			result = new ArrayList<BIGEntry>(fileEntry.values());
			Collections.sort(result);
		}
		return result;
	}

	/**
	 * @return the count of entries in the database
	 */
	public int getAllEntriesCount() {
		int size = 0;
		BIGStrings fileList = getAllFilenames();
		Iterator<String> fileNames = fileList.iterator();
		while (fileNames.hasNext()) {
			size = size + getAllEntries(fileNames.next()).size();
		}
		return size;
	}

	/**
	 * Returns all entrynames of one file out of the database. The entries are sorted be its <code>position</code>.
	 * 
	 * @param filename the name of the file where the entries are from
	 * @return all entrienames of one file
	 **/
	public BIGStrings getAllEntrynames(String filename) {
		BIGStrings result;
		result = new BIGStrings();
		Collection<BIGEntry> tempCollection = getAllEntries(filename);
		if (tempCollection != null) {
			Iterator<BIGEntry> tempIterator = tempCollection.iterator();
			while (tempIterator.hasNext()) {
				result.add(tempIterator.next().getName());
			}
			return result;
		} else
			return null;
	}

	/**
	 * Returns all entries of all files out of the database.
	 * 
	 * @return all entries of all files
	 **/
	public ArrayList<BIGEntry> getAllEntries() {
		ArrayList<BIGEntry> result = new ArrayList<BIGEntry>();
		BIGStrings fileList = getAllFilenames();
		Iterator<String> tempIterator = fileList.iterator();
		while (tempIterator.hasNext()) {
			result.addAll(getAllEntries(tempIterator.next()));
		}
		return result;
	}

	/**
	 * Returns an iterator over the elements in this list in proper sequence, over the given file. The entries are sorted
	 * be its <code>position</code>.
	 * 
	 * @param filename the name of the file, of which the iterator should be.
	 * @return an iterator over the elements in this list in proper sequence.
	 **/
	public Iterator<BIGEntry> iterator(String filename) {
		Collection<BIGEntry> tempCollection = getAllEntries(filename);
		if (tempCollection == null)
			return null;
		else
			return tempCollection.iterator();
	}

	/**
	 * Returns an iterator over the elements in this list in proper sequence, over all files. First all elements, in order
	 * of the <code>position</code>, of the first file, then the second file and so on.
	 * 
	 * @return an iterator over the elements in this list in proper sequence.
	 **/
	public Iterator<BIGEntry> iterator() {
		Collection<BIGEntry> tempCollection = getAllEntries();
		if (tempCollection == null)
			return null;
		else
			return tempCollection.iterator();
	}

	/**
	 * Returns the content of one of the defFiles. It only reads the text from disk not what is changed yet.
	 * <P>
	 * Errors are catched, if the file does not exist one the disk simply an empty text is returned.
	 * 
	 * @return the original text from the disk
	 **/
	public BIGStrings getDefFileText(String dbFilename) {
		BIGStrings result = new BIGStrings();
		try {
			result.readFromFile(benchItPath + File.separator + "LOCALDEFS" + File.separator
					+ dbFilename.replaceAll("<hostname>", hostname.get()));
		} catch (Exception e) {
			return new BIGStrings();
		}
		return result;
	}

	/**
	 * Writes the text to one of the defFiles. It only writes the text to disk , changes nothing in the database.
	 * <P>
	 * Errors are catched, if the file does not exist it will be created.
	 * 
	 * @param text the to write to disk
	 * @param dbFilename the name of the file to write to as it is called in the database
	 **/
	public void setDefFileText(BIGStrings text, String dbFilename) throws IOException {
		text.saveToFile(benchItPath + File.separator + "LOCALDEFS" + File.separator
				+ dbFilename.replaceAll("<hostname>", hostname.get()));
	}

	/**
	 * Returns the path where all the BenchIT files are.
	 * 
	 * @return the path were the benchIt files are
	 **/
	public String getBenchItPath() {
		return benchItPath;
	}

	public String getOutputPath() {
		return getBenchItPath() + File.separator + "output";
	}

	public String getGUIPath() {
		return getBenchItPath() + File.separator + "gui";
	}

	public String getHelpPath() {
		return getGUIPath() + File.separator + "help";
	}

	public String getImgPath() {
		return getGUIPath() + File.separator + "img";
	}

	public String getTempPath() {
		return getGUIPath() + File.separator + "temp";
	}

	public String getConfigPath() {
		return getGUIPath() + File.separator + "cfg";
	}

	/**
	 * Sets the path where all the BenchIT files are.
	 * 
	 * @param path the path were the benchIt files are
	 **/
	public void setBenchItPath(String path) {
		try {
			benchItPath = ((new File(path)).getCanonicalPath());
		} catch (IOException ioe) {
			benchItPath = ((new File(path)).getAbsolutePath());
		}
	}

	/**
	 * Sets the name of the config file which the XML Parser reads.
	 * 
	 * @see BIGXMLParser
	 **/
	public void setConfigFile(String fileName) {
		configFilename = fileName;
	}

	/**
	 * Returns the name of the config file which the XML Parser reads.
	 * 
	 * @see BIGXMLParser
	 * @return the name of the config file
	 **/
	public String getConfigFile() {
		if (configFilename != null && !configFilename.isEmpty())
			return configFilename;
		return getConfigPath() + File.separator + "config.xml";
	}

	/**
	 * Sets the debug level of a given class.
	 * 
	 * @param what the name of the class
	 * @param level the debug level, 0 is off
	 **/
	public void setDebug(String what, int level) {
		debug.put(what, new Integer(level));
	}

	/**
	 * Sets the globally debug level.
	 * 
	 * @param level the debug level, 0 is off
	 **/
	public void setDebug(int level) {
		setDebug("All", level);
	}

	/**
	 * Returns the debug level of a given class.
	 * 
	 * @param what the name of the class
	 * @return the debug level, 0 is off
	 **/
	public int getDebug(String what) {
		Integer temp = debug.get(what);
		return temp == null ? getDebug() : temp.intValue();
	}

	/**
	 * Returns the globally debug level.
	 * 
	 * @return the debug level, 0 is off
	 **/
	public int getDebug() {
		Integer temp = debug.get("All");
		return temp == null ? 0 : temp.intValue();
	}

	/**
	 * Returns the <code>BIGConsole</code> frame.
	 * 
	 * @return the console
	 **/
	public BIGConsole getConsole() {
		return console;
	}

	/**
	 * Sets the <code>BIGConsole</code> frame.
	 * 
	 * @param console the console
	 **/
	public void setConsole(BIGConsole console) {
		this.console = console;
	}

	/**
	 * :) use "win" to set windows laf.
	 */
	public void setLookAndFeel(String laf) {
		lookAndFeel = laf;
	}

	public String getLookAndFeel() {
		return lookAndFeel;
	}

	/**
	 * Set should the admin tool loaded
	 */
	/*
	 * public void setLoadAdminTool(boolean load) { loadAdminTool = load; } /** Should the admin tool be loaded
	 */
	/*
	 * public boolean getLoadAdminTool() { return loadAdminTool; } /** Sets the restart property. The GUI does a restart
	 * when it's starting after having shut down only for a measurement run. This property tells the GUI to supress the
	 * daily tip on a restart and to display the result tree after the measurements.
	 */
	public void setRestart(boolean restart) {
		this.restart = restart;
	}

	/**
	 * Gets the state of the restart property.
	 * 
	 * @return true, if GUI restarts after a measurement, else false.
	 */
	public boolean isRestart() {
		return restart;
	}

	public static <T extends Enum<T>> String[] enumNameToStringArray(T[] values) {
		int i = 0;
		String[] result = new String[values.length];
		for (T value : values) {
			result[i++] = value.name();
		}
		return result;
	}

	/**
	 * Adds a ProgressBar to the actually Prozess
	 **/
	public void addProgressObserver(Observer o) {
		progressBar = o;
	}

	/*
	 * public void setStartProgress(JProgressBar bar) { startProgress = bar; }
	 */

	public void setArguments(String[] args) {
		arguments = new String[args.length];
		for (int i = 0; i < args.length; i++) {
			arguments[i] = args[i];
		}
	}

	public String[] getArguments() {
		return arguments;
	}

	public String getArgumentsString() {
		if (arguments == null)
			return "";
		String str = "";
		for (int i = 0; i < arguments.length; i++) {
			str += arguments[i] + " ";
		}
		return str;
	}

	/*
	 * public void setShowAdminNew(boolean b) { this.showAdminNew=b; } public boolean getShowAdminNew() { return true; }
	 */
	/**
	 * returns the actual System (which it is is guessed by checking the separator char from File)
	 * 
	 * @return this.WINDOWS_SYSTEM or this.UNIX_SYSTEM
	 */
	public static int getSystem() {
		if (File.separatorChar == '\\')
			return WINDOWS_SYSTEM;
		return UNIX_SYSTEM;
	}

	private boolean checkForFilled(String filename) {/*
																										 * boolean changed=false; BIGEntry tempEntry; String check; if
																										 * (filename.equals("<hostname>_input_architecture" )) { if
																										 * (getSystem()==WINDOWS_SYSTEM) { return true; }
																										 * tempEntry=this.getEntry(filename ,"processorname");
																										 * check=(String)tempEntry.getValue(); if
																										 * ((check==null)||(check.equals(""))) {
																										 * check=this.showInput(tempEntry); changed=true; }
																										 * this.getEntry(filename,"processorname" ).setValue(check);
																										 * tempEntry=this.getEntry(filename ,"processorclockrate");
																										 * check=(String)tempEntry.getValue(); if
																										 * ((check==null)||(check.equals(""))) {
																										 * check=this.showInput(tempEntry); changed=true; }
																										 * this.getEntry(filename,"processorclockrate" ).setValue(check); }
																										 * else if (filename.equals("<hostname>")) { if ( getSystem() ==
																										 * WINDOWS_SYSTEM ) { JOptionPane.showMessageDialog( null ,
																										 * "Found WINDOWS System,so EMPTY LOCALDEFS are created." ) ;
																										 * this.getEntry( filename , "CC" ).setValue( "empty" ) ;
																										 * this.getEntry( filename , "F77" ).setValue( "empty" ) ;
																										 * this.getEntry( filename , "HAVE_CC" ).setValue( new Boolean(
																										 * false ) ) ; this.getEntry( filename , "HAVE_F77" ).setValue( new
																										 * Boolean( false ) ) ; this.getEntry( filename ,
																										 * "BENCHIT_ARCH_SHORT" ).setValue( "empty" ) ; this.getEntry(
																										 * filename , "BENCHIT_ARCH_SPEED" ).setValue( "empty" ) ; return
																										 * true ; } tempEntry=this.getEntry(filename,"CC");
																										 * check=(String)tempEntry.getValue(); if
																										 * ((check==null)||((check.equals("")))) {
																										 * check=this.showInput(tempEntry); changed=true; }
																										 * this.getEntry(filename,"CC").setValue(check); this
																										 * .getEntry(filename,"HAVE_CC").setValue(new Boolean(true));
																										 * tempEntry=this.getEntry(filename,"F77"); check = ( String )
																										 * tempEntry.getValue() ; if ((check==null)||((check.equals("")))) {
																										 * check = this.showInput( tempEntry ) ; this.getEntry( filename ,
																										 * "F77" ).setValue( check ) ; if ( ! ( ( check == null ) || (
																										 * check.equals( "" ) ) ) ) this.getEntry( filename , "HAVE_F77"
																										 * ).setValue( new Boolean( true ) ) ; } tempEntry=this.getEntry(
																										 * filename,"BENCHIT_ARCH_SHORT");
																										 * check=(String)tempEntry.getValue(); if ((check
																										 * ==null)||((check.equals(""))||(check.equals ("NONE")))) {
																										 * check=this.showInput(tempEntry); changed=true; }
																										 * this.getEntry(filename,"BENCHIT_ARCH_SHORT" ).setValue(check);
																										 * tempEntry=this.getEntry(filename ,"BENCHIT_ARCH_SPEED");
																										 * check=(String)tempEntry.getValue(); check=(String
																										 * )this.getEntry(filename,"BENCHIT_ARCH_SPEED" ).getValue(); if
																										 * ((check==null)||((check.equals (""))||(check.equals("NONE")))) {
																										 * check=this.showInput(tempEntry); changed=true; }
																										 * this.getEntry(filename,"BENCHIT_ARCH_SPEED" ).setValue(check); }
																										 * return changed;
																										 */
		return false;
	}

	private File lastWorkPath = null;

	public void setLastWorkPath(File f) {
		lastWorkPath = f;
	}

	public File getLastWorkPath() {
		if (lastWorkPath == null) {
			lastWorkPath = new File(benchItPath);
		}
		return lastWorkPath;
	}

	private int width = 1024;
	private int height = 600;

	public int getSaveWidth() {
		return width;
	}

	public int getSaveHeight() {
		return height;
	}

	public void setSaveWidth(int width) {
		this.width = width;
	}

	public void setSaveHeight(int height) {
		this.height = height;
	}

	public static double getVersion() {
		File versionTxt = new File(getInstance().getConfigPath() + File.separator + "version.txt");
		double version = 0.0;
		try {
			version = Double.parseDouble(BIGFileHelper.getFileContent(versionTxt));
		} catch (Exception ex) {
			System.err.println("Can't find own version number.");
		}

		return version;
	}

	public static double getJobVersion() {
		File jobVersionTxt = new File(getInstance().getConfigPath() + File.separator + "jobVersion.txt");
		double version = 0.0;
		try {
			version = Double.parseDouble(BIGFileHelper.getFileContent(jobVersionTxt));
		} catch (Exception ex) {
			System.err.println("Can't find own version number.");
		}

		return version;
	}
}
/*****************************************************************************
 * Log-History
 *****************************************************************************/
