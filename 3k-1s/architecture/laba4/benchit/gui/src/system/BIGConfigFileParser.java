/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGConfigFileParser.java
 * Author: Robert Schoene Last change by: $Author: tschuet $ $Revision: 1.9 $ $Date: 2007/05/08
 * 09:43:30 $
 ******************************************************************************/
package system;

import gui.DetailLevel;

import java.awt.Color;
import java.io.*;
import java.util.*;

public class BIGConfigFileParser {
	// Value:Set
	private final Map<String, String> entries = new HashMap<String, String>();
	// the options dialog
	private gui.BIGOptionsDialog optionsDialog = null;

	private File file = null;

	// new Options stored as String[2]
	private final Vector<String> newOptions = new Vector<String>();

	public BIGConfigFileParser(String datei) {
		// first we read the file (and store its important content to the Vectors)
		ReadIt(datei);
		// then we save a link to the file in this.file
		file = new File(datei);
	}

	public void showConfigDialog(java.awt.Frame f, gui.BIGKernelTree tree) {
		if (optionsDialog == null) {
			optionsDialog = new gui.BIGOptionsDialog(f, entries, tree);
		} else {
			optionsDialog.setVisible(true);
		}
	}

	/**
	 * this method sets a new Value to a given attribute
	 */

	public void set(String attr, String value) {
		if (!entries.containsKey(attr))
			newOptions.add(attr);
		entries.put(attr, value);
	}

	public void set(String a, boolean b) {
		set(a, b ? "1" : "0");
	}

	/*
	 * this method rewrites the file with the given attributes
	 */
	public int save() {
		BufferedReader in = null;
		String line;
		// first we write the new content of BGUI.cfg to tmp.tmp
		String content = "";
		try {
			// we read files with in
			in = new BufferedReader(new FileReader(file));
			// and write files with out
			while ((line = in.readLine()) != null) {
				// this String will contain the content before the first '#'
				String notAComment = "";
				// this String the conten from the first '#' to the end
				String comment = "";
				// if there is a '#'
				if (line.indexOf('#') != -1) {
					// we divide the read line into notAComment
					notAComment = line.substring(0, line.indexOf('#'));
					// and the comment
					comment = line.substring(line.indexOf('#'), line.length());
				} else {
					// elsewise the whole line is notAComment
					notAComment = line;
				}
				// notAComment could be a blank line. Then we don't have to
				// parse the line
				if (notAComment.length() > 0) {
					// if there is no '=',we don't have to parse
					if ((notAComment.indexOf('=')) != -1) {
						// this contains all the ' 's and '\t's before the attributes name
						String preName = "";
						// the ' 's and '\t's after the attributes name
						String afterName = "";
						// this contains the attributes name
						String name = notAComment.substring(0, notAComment.indexOf('='));
						// the whole as with name, but with the value/setting
						String preSet = "";
						String afterSet = "";
						String set = notAComment.substring(notAComment.indexOf('=') + 1, notAComment.length());
						// first we set the pren-/aftern-/N-ame
						while (name.startsWith("\t")) {
							name = name.substring(1, name.length());
							preName = preName + "\t";
						}
						while (name.startsWith(" ")) {
							name = name.substring(1, name.length());
							preName = preName + " ";
						}
						while (name.endsWith("\t")) {
							name = name.substring(0, name.length() - 1);
							afterName = "\t" + afterName;
						}
						while (name.endsWith(" ")) {
							name = name.substring(0, name.length() - 1);
							afterName = " " + afterName;
						}
						// then the whole for the value
						while (set.startsWith("\t")) {
							set = set.substring(1, set.length());
							preSet = preSet + "\t";
						}
						while (set.startsWith(" ")) {
							set = set.substring(1, set.length());
							preSet = preSet + " ";
						}
						while (set.endsWith("\t")) {
							set = set.substring(0, set.length() - 1);
							afterSet = "\t" + afterSet;
						}
						while (set.endsWith(" ")) {
							set = set.substring(0, set.length() - 1);
							afterSet = " " + afterSet;
						}
						// now we set the set to the actual set
						set = stringCheckOut(name, "");
						// and rewrite the notAComment part of the line
						notAComment = preName + name + afterName + "=" + preSet + set + afterSet;
					}
				}
				// we add the \n, that was cut out by readLine()
				line = notAComment + comment + "\n";
				content = content + line;

			}
			// we close the streams
			in.close();

		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		}

		// Now we write the content of tmp.tmp to the config file
		// comments??? Look at it!
		if (newOptions.size() > 0) {
			try {
				content = content + "# new entries from "
						+ (java.util.Calendar.getInstance()).get(Calendar.YEAR) + "/"
						+ (java.util.Calendar.getInstance()).get(Calendar.MONTH) + "/"
						+ (java.util.Calendar.getInstance()).get(Calendar.DATE) + "\n";
				for (String option : newOptions) {
					content = content + "\t " + option + " = " + stringCheckOut(option, "") + "\n";
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		BIGFileHelper.saveToFile(content, file);

		return 1;

	}

	/**
	 * returns true, if the setting to the variable "whichToCheckOut" is >0 returns false, if the setting is 0 and an
	 * Exception if the setting is not an Integer or the variable "whichToCheckOut" does not exist
	 */

	public boolean boolCheckOut(String whichToCheckOut) throws Exception {
		String val = stringCheckOut(whichToCheckOut);
		try {
			return new Integer(val) > 0;
		} catch (Exception e) {
			return val.equals("true");
		}
	}

	public boolean boolCheckOut(String whichToCheckOut, boolean def) {
		try {
			return boolCheckOut(whichToCheckOut);
		} catch (Exception e) {
			return def;
		}
	}

	/**
	 * returns the value, if the setting to the variable "whichToCheckOut" is >0 and an Exception if the setting is not an
	 * Integer or there is no variable "whichToCheckOut"
	 */

	public int intCheckOut(String whichToCheckOut) throws Exception {
		String val = stringCheckOut(whichToCheckOut);
		try {
			return new Integer(val);
		} catch (Exception e) {
			return 0;
		}
	}

	public int intCheckOut(String whichToCheckOut, int def) {
		String val = stringCheckOut(whichToCheckOut, "" + def);
		try {
			return new Integer(val);
		} catch (Exception e) {
			return def;
		}
	}

	/**
	 * returns the value, if the setting to the variable "whichToCheckOut" is >0 and an Exception if the setting is not an
	 * Integer or there is no variable "whichToCheckOut"
	 */

	public double doubleCheckOut(String whichToCheckOut) throws Exception {
		String val = stringCheckOut(whichToCheckOut);
		try {
			return new Double(val);
		} catch (Exception e) {
			return 0;
		}
	}

	/**
	 * returns the setting to the variable "whichToCheckOut" and an Exception if there is no variable "whichToCheckOut"
	 */

	public String stringCheckOut(String whichToCheckOut) throws Exception {
		if (entries.containsKey(whichToCheckOut))
			return entries.get(whichToCheckOut);
		throw new Exception();
	}

	public String stringCheckOut(String whichToCheckOut, String def) {
		if (entries.containsKey(whichToCheckOut))
			return entries.get(whichToCheckOut);
		return def;
	}

	public Color colorCheckOut(String whichToCheckOut) throws Exception {
		return new Color(intCheckOut(whichToCheckOut));
	}

	public DetailLevel DetailevelCheckOut(String whichToCheckOut) throws Exception {
		String val = stringCheckOut(whichToCheckOut);
		return DetailLevel.valueOf(val);
	}

	/*
	 * public Color getColorFromString(String s) { if (s.length()!=6) return null; String red=s.substring(0,2); String
	 * green=s.substring(2,4); String blue=s.substring(4,6); return new
	 * Color(getIntFromHex(red),getIntFromHex(green),getIntFromHex(blue)); } public String getStringFromColor(Color c) {
	 * return getHexFromInt(c.getRed())+getHexFromInt(c.getGreen())+c. if (s.length()!=6) return null; String
	 * red=s.substring(0,2); String green=s.substring(2,4); String blue=s.substring(4,6); return new
	 * Color(getIntFromHex(red),getIntFromHex(green),getIntFromHex(blue)); } private int getIntFromHex(String s) { int
	 * i=0; StringBuffer sb = new StringBuffer(s); while (sb.length()>1) { i=i*16; char c = sb.charAt(0);
	 * sb.deleteCharAt(0); switch (c) { case '0': break; case '1': i = i + 1 ; case '2' : i = i + 2 ; break; case '3' : i
	 * = i + 3 ; break; case '4' : i = i + 4 ; break; case '5' : i = i + 5 ; break; case '6' : i = i + 6 ; break; case '7'
	 * : i = i + 7 ; break; case '8' : i = i + 8 ; break; case '9' : i = i + 9 ; break; case 'A' : i = i + 10 ; break;
	 * case 'B' : i = i + 11 ; break; case 'C' : i = i + 12 ; break; case 'D' : i = i + 13 ; break; case 'E' : i = i + 14
	 * ; break; case 'F' : i = i + 15 ; break; default: break; } } return i; } private String getHexFromInt(int s) { int
	 * i=0; int next=0; StringBuffer sb = new StringBuffer(); boolean stop=false; while (!stop) { next=s%16; s=s/16; if
	 * (s==0) stop=true; switch (next) { case 0: sb.insert(0,'0'); break; case 1: sb.insert(0,'1') ; break; case 2 :
	 * sb.insert(0,'2') ; break; case 3 : sb.insert(0,'3') ; break; case 4 : sb.insert(0,'4') ; break; case 5 :
	 * sb.insert(0,'5') ; break; case 6 : sb.insert(0,'6') ; break; case 7 : sb.insert(0,'7') ; break; case 8 :
	 * sb.insert(0,'8') ; break; case 9 : sb.insert(0,'9') ; break; case 10 : sb.insert(0,'A') ; break; case 11 :
	 * sb.insert(0,'B') ; break; case 12 : sb.insert(0,'C') ; break; case 13 : sb.insert(0,'D') ; break; case 14 :
	 * sb.insert(0,'E') ; break; case 15 : sb.insert(0,'F') ; break; default: break; } } return sb.toString(); }
	 */

	private void ReadIt(String datei) {
		entries.clear();
		// this will contain the content of a line
		String line;
		try {
			// cleaning file (linebreaks :( )
			if (BIGInterface.getSystem() == BIGInterface.WINDOWS_SYSTEM) {
				String s = BIGFileHelper.getFileContent(new File(datei));
				if (s != null) {
					if (s.indexOf("\r\n") == -1) {
						for (int i = 0; i < s.length(); i++) {
							if (s.charAt(i) == '\n') {
								s = s.substring(0, i) + "\r\n" + s.substring(i + 1, s.length());
								i++;
							}
						}
					}
					BIGFileHelper.saveToFile(s, new File(datei));
				}
			}
			// he will read the file
			BufferedReader d = new BufferedReader(new FileReader(datei));
			try {
				// to the end of the file
				while ((line = d.readLine()) != null) {
					// we just parse the line, if there exists a '=' in the line
					if (line.indexOf('=') != -1) {
						// first we cut everything from the '#'
						while (line.indexOf('#') != -1) {
							line = line.substring(0, line.indexOf('#'));
						}
						// then we divide the remaining string
						// (if there is a '=' in it)
						if (line.indexOf('=') != -1) {
							// to the attributes name
							String attribute = line.substring(0, line.indexOf('='));

							while (attribute.startsWith("\t")) {
								attribute = attribute.substring(1, attribute.length());
							}
							while (attribute.startsWith(" ")) {
								attribute = attribute.substring(1, attribute.length());
							}
							while (attribute.endsWith("\t")) {
								attribute = attribute.substring(0, attribute.length() - 1);
							}
							while (attribute.endsWith(" ")) {
								attribute = attribute.substring(0, attribute.length() - 1);
							}

							// and to the attributes value
							String value = line.substring(line.indexOf('=') + 1, line.length());
							while (value.startsWith("\t")) {
								value = value.substring(1, value.length());
							}
							while (value.startsWith(" ")) {
								value = value.substring(1, value.length());
							}
							while (value.endsWith("\t")) {
								value = value.substring(0, value.length() - 1);
							}
							while (value.endsWith(" ")) {
								value = value.substring(0, value.length() - 1);
							}
							entries.put(attribute, value);
						}
					}

				} // Ende while
			} catch (IOException ignored) {}
		} catch (FileNotFoundException notignored) {
			System.out.println("file not found");
		}
	}

	public File getFile() {
		return file;
	}
}
