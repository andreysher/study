/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGOutputParser.java Author:
 * SWTP Nagel 1 Last change by: $Author: rschoene $ $Revision: 1.10 $ $Date: 2006/07/04 11:12:40 $
 ******************************************************************************/
package plot.data;

import java.io.File;
import java.util.*;

import system.*;

/**
 * a parser for getting the information from the BenchIT outputfiles (*.bit)
 * 
 * @author <a href="mailto:fx@fx-world.de">Pascal Weyprecht</a>
 * @author <a href="mailto:pisi@pisi.de">Christoph Mueller</a>
 **/
public class BIGOutputParser {

	/**
	 * debugmessages en- or disabled (may be overwritten in the constructor)
	 */
	private boolean debug = false;

	/**
	 * holds the complete content of the file
	 */
	private final BIGStrings text = new BIGStrings();
	private String fileContent = null;

	/**
	 * the name of the file stored in text
	 */
	// private String filename;

	/** The referrence to the Observer class for the progress bar. */
	private final BIGObservable progress = new BIGObservable();

	/** The last loaded fileName */
	private String fileName = null;

	/**
	 * instantiate an OutputParser without initially loaded file
	 */
	public BIGOutputParser() {
		if (BIGInterface.getInstance().getDebug("BIGOutputParser") > 0) {
			debug = true;
		}
	}

	/**
	 * instantiate an OutputParser with initially loaded file
	 * 
	 * @param filename the file to load
	 * @throws BIGParserException there was an error accessing the file
	 */
	public BIGOutputParser(String filename) throws BIGParserException {
		if (BIGInterface.getInstance().getDebug("BIGOutputParser") > 0) {
			debug = true;
		}
		load(filename);
	}

	/**
	 * load the content of a file into the internal BIGStrings text
	 * 
	 * @param filename the name of the file to load
	 * @throws BIGParserException there was an error accessing the file
	 */
	public void load(String filename) throws BIGParserException {
		this.fileName = null;
		text.clear();
		fileContent = "";
		try {
			text.readFromFile(filename);
			fileContent = text.toString();
			if (debug) {
				System.out.println("BIGOutputParser: file [" + filename + "] loaded");
			}
			this.fileName = filename;
		} catch (Exception e) {
			throw new BIGParserException("BIGOutputParser: Error while instantiating an OutputParser:\n"
					+ "BIGOutputParser: could not read from file [" + filename + "]\n"
					+ "BIGOutputParser: cause: [" + e.getMessage() + "]");
		}
	}

	/**
	 * parse the internal BIGStrings text for outputdata
	 * 
	 * @return array d with d[0][] are the x values and d[n][] are the corresponding y values for series n
	 */
	public double[][] getData() {
		return getData(true);
	}

	public double[][] getData(boolean fixFile) {
		if (fileName == null)
			return null;
		// set to data, that doesnt exist or is wrong
		double errorDou = (fixFile) ? 1.0 * system.BIGInterface.getInstance().getBIGConfigFileParser()
				.intCheckOut("errorInt", -1) : -1;

		// whether there is an inf or a NaN
		boolean fileNeedsCleaning = false;
		// looking where is data
		int lineNr = text.size() - 1;

		// get end of datasection
		while (lineNr >= 0 && !text.get(lineNr).trim().equals("endofdata")) {
			lineNr--;
		}
		if (lineNr < 0) {
			if (debug) {
				System.out.println("BIGOutputParser: no endofdata tag");
			}
			if (!fixFile)
				return null;
			System.out.println("Corrupt file, try to clean...");
			lineNr = text.size();
			text.add("endofdata");
			BIGFileHelper.saveToFile(text.toString(), new File(fileName));
			fileContent = text.toString();
		} else if (debug) {
			System.out.println("BIGOutputParser: endofdata tag in line " + lineNr);
		}

		// set max lines number
		int numLines = lineNr;

		// move to begin of datasection
		while (lineNr >= 0 && !text.get(lineNr).trim().equals("beginofdata")) {
			lineNr--;
		}

		if (lineNr < 0) {
			if (debug) {
				System.out.println("BIGOutputParser: no " + "beginofdata tag");
			}
			return new double[0][0];
		}

		if (debug) {
			System.out.println("BIGOutputParser: beginofdata tag in line " + lineNr);
		}

		// subtract begin of data tag
		lineNr++;
		numLines -= lineNr;

		String firstLine = text.get(lineNr);
		int numValsPerLine = 0;
		for (int i = 0; i < firstLine.length(); i++)
			if (firstLine.charAt(i) == '\t')
				numValsPerLine++;

		double result[][] = new double[numValsPerLine][numLines];

		for (int i = 0; i < numLines; i++, lineNr++) {
			progress.setProgress((100 * i) / numLines);
			String line = text.get(lineNr).trim();
			if (debug) {
				System.out.println("BIGOutputParser: line:" + lineNr + " [" + line + "]");
			}

			StringTokenizer tokens = new StringTokenizer(line, "\t");
			boolean errorShown = false;
			for (int j = 0; j < numValsPerLine; j++) {
				if (tokens.hasMoreTokens()) {
					String entry = tokens.nextToken().trim();
					if (entry.equals("-"))
						result[j][i] = errorDou;
					try {
						result[j][i] = (new Double(entry)).doubleValue();
					} catch (NumberFormatException ex) {
						fileNeedsCleaning = true;
					}
					if (debug)
						System.out.println("BIGOutputParser: [" + entry + "] " + j + ":" + i + " \t"
								+ result[j][i]);
				} else {
					result[j][i] = errorDou;
					if (!errorShown) {
						System.err.println("BIGOutputParser: error on parsing line:" + lineNr + " [" + line
								+ "] Missing values!");
						errorShown = true;
						fileNeedsCleaning = true;
					}
				}
			}
		}

		progress.setProgress(100);
		if (fileNeedsCleaning && fileName != null && fixFile) {
			(new Thread() {
				@Override
				public void run() {
					try {
						(new system.BIGUtility()).removeInfinitiesFromFile(fileName).join();
					} catch (InterruptedException ignored) {}
					try {
						load(fileName);
					} catch (BIGParserException e) {
						System.err.println("BIGOutputParser: Unknown error while reopening file");
					}
				}
			}).start();
		}
		return result;
	}

	/**
	 * @param s
	 * @return
	 * @throws NumberFormatException
	 */
	public double getDoubleValue(String s) throws NumberFormatException {
		return Double.parseDouble(getValue(s));
	}

	/**
	 * @param s
	 * @param def
	 * @return
	 */
	public double getDoubleValue(String s, double def) {
		try {
			return getDoubleValue(s);
		} catch (NumberFormatException e) {
			System.err.println("Value for '" + s + "' is not a double.");
			return def;
		}
	}

	/**
	 * @param s
	 * @return
	 * @throws NumberFormatException
	 */
	public Integer getIntValue(String s) throws NumberFormatException {
		return Integer.valueOf(getValue(s));
	}

	/**
	 * @param s
	 * @param def
	 * @return
	 */
	public Integer getIntValue(String s, int def) {
		try {
			return getIntValue(s);
		} catch (NumberFormatException e) {
			System.err.println("Value for '" + s + "' is not an int.");
			return def;
		}
	}

	// TODO: FixME!!!
	public String getValue(String s) {
		if (fileName == null || s == null)
			return null;
		int lastIndex = 0;
		String lineWithData, set;
		do {
			// while we found a not commented version of the string s
			lastIndex = fileContent.indexOf(s, lastIndex);
			if (lastIndex < 0)
				return null;
			while (fileContent.substring(0, lastIndex).lastIndexOf("#") > fileContent.substring(0,
					lastIndex).lastIndexOf("\n")) {
				lastIndex++;
				lastIndex = fileContent.indexOf(s, lastIndex);
				if (lastIndex < 0)
					return null;
			}
			// found line
			lineWithData = fileContent.substring(lastIndex);
			set = lineWithData.substring(0, lineWithData.indexOf("=")).trim();
			lastIndex++;
		} while (!set.equals(s));
		// returning everything after the '=' trimmed
		int endIndex = lineWithData.indexOf("\n");
		while (lineWithData.charAt(endIndex - 1) == '\\') {
			endIndex = lineWithData.indexOf("\n", endIndex + 1);
		}
		lineWithData = lineWithData.substring(lineWithData.indexOf("=") + 1, endIndex);
		lineWithData = lineWithData.replaceAll("\\\n", "");
		lineWithData = lineWithData.trim();
		if (lineWithData.startsWith("\"") && lineWithData.endsWith("\"")) {
			lineWithData = lineWithData.substring(1, lineWithData.length() - 1);
		}
		return lineWithData;

	}

	/**
	 * get the text of the loaded file
	 * 
	 * @return the text of the loaded file
	 */
	public BIGStrings getAllText() {
		return text;
	}

	/**
	 * get the text of the loaded file
	 * 
	 * @return the text of the loaded file
	 */
	public String getText() {
		return fileContent;
	}

	/**
	 * try to find values for given parameters
	 * 
	 * @param names strings representing names of variables
	 * @return a Map keys are the names of the vars values are the values
	 */
	public Map<String, String> getParams(BIGStrings names) {
		Map<String, String> result = new HashMap<String, String>();

		// parse file for names and values
		String row, name, value;
		Iterator<String> namesIt;
		Iterator<String> textIt = text.iterator();

		// variables for using a progress bar
		int s = 0, z = 0;
		int sMax = text.size(), zMax = names.size();
		while (textIt.hasNext()) {
			s++;
			namesIt = names.iterator();
			row = textIt.next();
			row = row.trim();
			if (row.indexOf("=") > 0) {
				while (namesIt.hasNext()) {
					z++;
					// set new value of the progress bar
					// [ ( z ) ( ) ] ( 1 )
					// [ ( - * 100 ) + ( s ) ] * ( - )
					// [ ( Z ) ( ) ] ( S )
					progress.setProgress(((100 * z + s) / zMax) * (1 / sMax));

					name = namesIt.next();
					// we must assure, that names like xyz1 don't
					// interfere with names like xyz112
					if (row.startsWith(name)) {
						String varname = row.substring(0, row.indexOf("="));
						if (varname.compareTo(name) == 0) {
							value = row.substring(row.indexOf("=") + 1);
							// remove eventually existing " at both ends
							if (value.startsWith("\"") && value.endsWith("\"") && value.length() > 1) {
								value = value.substring(value.indexOf("=") + 2, value.length() - 1);
							}
							if (debug) {
								System.out.println("BIGOutputParser: name \"" + name + "\" found with value \""
										+ value + "\"");
							}
							result.put(name, value);
						}
					}
				}
			}
		}
		progress.setProgress(100);
		return result;
	}

	/**
	 * This Method is needed for adding this class Observable to an Observer of a progress bar.
	 * 
	 * @return the progress of this (progress of an internal process)
	 */
	public Observable getObservable() {
		return progress;
	}

	/** The internel class for the progress updates. */
	private class BIGObservable extends Observable {
		private int oldValue = -1;

		public void setProgress(int value) {
			if (oldValue != value) {
				oldValue = value;
				setChanged();
				notifyObservers(new Integer(value));
				try {
					Thread.sleep(0, 1);
				} catch (InterruptedException ie) {}
			}
		}
	}

}
/*****************************************************************************
 * Log-History
 *****************************************************************************/
