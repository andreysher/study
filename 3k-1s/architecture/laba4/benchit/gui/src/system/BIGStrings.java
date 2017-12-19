/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGStrings.java Author: SWTP
 * Nagel 1 Last change by: $Author: rschoene $ $Revision: 1.2 $ $Date: 2006/05/30 11:46:21 $
 ******************************************************************************/
package system;

import java.io.*;
import java.util.*;

/**
 * Its a List of <code>String</code>. <BR>
 * Based on Strings from <code>fXlib</code>, soon available under:
 * <code><a href="http://www.fx-world.de">http://www.fx-world.de</a></code>
 * 
 * @author <a href="mailto:fx@fx-world.de">Pascal Weyprecht</a>
 * @version 1.1
 * @see String
 * @see ArrayList
 **/
public class BIGStrings extends ArrayList<String> {
	private static final long serialVersionUID = 1L;

	public BIGStrings() {
		super();
	}

	public BIGStrings(int size) {
		super(size);
	}

	public BIGStrings(Collection<? extends String> c) {
		super(c);
	}
	/**
	 * Constructs a list filled by the values from the Array by using the method toString.
	 **/
	public BIGStrings(String[] fill) {
		super(fill.length);
		for (int i = 0; i < fill.length; i++) {
			add(fill[i].toString());
		}
	}

	/**
	 * Returns a comma text representation of all strings in the <code>Strings</code>. It's in a form like: <BR>
	 * "hello","this","is a strings","object"
	 * 
	 * @return the commaText representation of <code>Strings</code>
	 **/
	public String getCommaText() {
		String result = "";
		for (int i = 0; i < size(); i++) {
			String tempString = new String(get(i));
			tempString = tempString.replaceAll("\"", "\"\"");
			result = result + "\"" + tempString + "\"";
			if (i + 1 < size()) {
				result = result + ",";
			}
		}
		return result;
	}

	/**
	 * Sets the whole <code>Strings</code> object based upon a commaText. It has to be in a form like: <BR>
	 * "hello","this","is a strings","object" <BR>
	 * otherwise there will be a out of bounce exception.
	 * 
	 * @param str the commaText representation of <code>Strings</code>
	 **/
	public void setCommaText(String str) {
		boolean finished = false;
		int posA = -1;
		int posB = -1;
		clear();
		if (str == null) {
			finished = true;
		}
		while (!finished) {
			posA = str.indexOf("\"", posB + 1);
			// System.out.print(posA+":");
			if (posA >= 0) {
				posB = str.indexOf("\"", posA + 1);
				// System.out.print(posB+"\n");
				while ((posB + 2 <= str.length()) && ("\"\"".equals(str.substring(posB, posB + 2)))) {
					// System.out.print("["+str.substring(posB,posB+2)+"]");
					posB = str.indexOf("\"", posB + 2);
					// System.out.print(posA+":"+posB+"\n");
				}
				add(str.substring(posA + 1, posB));
			} else {
				finished = true;
			}
		}
	}

	/**
	 * Reads a text out of the file into the list. Before reading the list is cleared. Every line in the file gets its own
	 * <code>String</code> in the list.
	 * 
	 * @param filename name of the file where to read from
	 **/
	public void readFromFile(String filename) throws FileNotFoundException, IOException {
		BufferedReader file = new BufferedReader(new FileReader(filename));
		String line = file.readLine();

		clear();
		while (line != null) {
			add(line);
			line = file.readLine();
		}
		file.close();
	}

	/**
	 * Saves the text into the specified file. Every <code>String</code> in the list gets its own line in the file.
	 * 
	 * @param filename name of the file where to save to
	 **/
	public void saveToFile(String filename) {
		BIGFileHelper.saveToFile(toString(), new File(filename));
	}

	/**
	 * Concatenates all strings in the list to one string. After every element there is a new line added.
	 * 
	 * @return Concatenation of all <code>String</code> in the list
	 **/
	@Override
	public String toString() {
		String result = "";
		Iterator<String> it = iterator();
		while (it.hasNext()) {
			result += it.next() + "\n";
		}
		return result;
	}

	/**
	 * Converts the <code>Strings</code> to an array which contains the <code>String</code>s. It is a totally copy, it has
	 * nothing to do with the intern <code>ArrayList</code>, also the <code>String</code>s are copied.
	 * 
	 * @return an array with <code>String</code>
	 **/
	@Override
	public String[] toArray() {
		String[] result = new String[size()];
		for (int l = 0; l < size(); l++) {
			result[l] = new String(get(l));
		}
		return result;
	}

	/**
	 * Returns the line number where the <code>String</code> could be found on the first time. Not the whole
	 * <code>String</code> in the line must match also if the toFind is a part of the line the line number is returned.
	 * 
	 * @param toFind the <code>String</code> what should be found
	 * @return line number where the <code>String</code> was found, or -1 when the <code>String</code> wasn't found
	 **/
	public int find(String toFind) {
		return find(toFind, 0);
	}

	/**
	 * Returns the line number where the <code>String</code> could be found on the first time, starting at line
	 * <code>fromIndex</code>. Not the whole <code>String</code> in the line must match also if the toFind is a part of
	 * the line the line number is returned.
	 * 
	 * @param toFind the <code>String</code> what should be found
	 * @param fromIndex the line where to start the search
	 * @return line number where the <code>String</code> was found, or -1 when the <code>String</code> wasn't found
	 **/
	public int find(String toFind, int fromIndex) {
		int lineNr = fromIndex;
		while (lineNr < size()) {
			if (get(lineNr).indexOf(toFind) >= 0)
				return lineNr;
			else {
				lineNr++;
			}
		}
		return -1;
	}

	/**
	 * Sorts the lines alphabetically.
	 **/
	public void sort() {
		Collections.sort(this);
	}

	/**
	 * concatenate a BIGStrings object to this and return it
	 * 
	 * @param bs a BIRStrings object to concat
	 */
	// added by pisi 15.05.2003
	public BIGStrings concat(BIGStrings bs) {
		BIGStrings out = new BIGStrings();
		Iterator<String> it = iterator();
		while (it.hasNext()) {
			out.add(it.next());
		}
		it = bs.iterator();
		while (it.hasNext()) {
			out.add(it.next());
		}
		return out;
	}
}
