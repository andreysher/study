package system;

/**
 * <p>
 * Überschrift:
 * </p>
 * <p>
 * Beschreibung:
 * </p>
 * <p>
 * Copyright: Copyright (c) 2004
 * </p>
 * <p>
 * Organisation:
 * </p>
 * 
 * @author unbekannt
 * @version 1.0
 */
import gui.BIGConsole;

import java.awt.*;
import java.io.*;
import java.util.Observable;

import javax.swing.JComponent;

import plot.data.BIGOutputParser;

public class BIGUtility {

	public String[] commands = new String[3];
	public Process p;
	private final BIGObservable progress = new BIGObservable();

	public BIGUtility() {}

	public static void addComponent(JComponent container, JComponent component,
			GridBagLayout gridbag, GridBagConstraints c) {
		gridbag.setConstraints(component, c);
		container.add(component);
	}

	public static void setConstraints(GridBagConstraints c, int x, int y, int gw, int gh) {
		c.gridx = x;
		c.gridy = y;
		c.gridwidth = gw;
		c.gridheight = gh;
	}

	public Thread removeInfinitiesFromFile(final String fileName) {
		final BIGObservable obs = (BIGObservable) getObservable();
		Thread t1 = new Thread() {
			@Override
			public void run() {
				// Removing Infinity from File:" + fileName
				obs.setProgress(0);

				BufferedWriter outputStream = null;
				// first we got to read the file
				BIGOutputParser parser;
				try {
					parser = new BIGOutputParser(fileName);
				} catch (BIGParserException exception) {
					BIGInterface.getInstance().getConsole()
							.postMessage("Couldn't open file! :", BIGConsole.ERROR);
					obs.setProgress(100);
					return;
				}
				// read file
				obs.setProgress(100 / 7);

				double[][] dataArray = parser.getData(false);
				obs.setProgress(400 / 7);
				try {
					outputStream = new BufferedWriter(new FileWriter(fileName));
				} catch (IOException ex2) {
					BIGInterface.getInstance().getConsole()
							.postMessage("Couldn't open write-stream!\nFileName: " + fileName, BIGConsole.ERROR);
					return;
				}
				BIGStrings fileContent = parser.getAllText();
				int lineToFile = 0;
				StringBuffer outSB = new StringBuffer();
				do {
					lineToFile++;
					outSB.append(fileContent.get(lineToFile) + "\n");
				} while (!fileContent.get(lineToFile).equals("beginofdata"));
				obs.setProgress(500 / 7);
				for (int j = 0; j < dataArray.length; j++) {
					for (int k = 0; k < dataArray[0].length; k++) {
						if (dataArray[j][k] >= 0)
							outSB.append(dataArray[j][k]);
						else
							outSB.append("-");
						outSB.append("\t");
					}
					outSB.append("\n");
				}
				obs.setProgress(600 / 7);
				try {
					outSB.append("endofdata\n");
					outputStream.write(outSB.toString());
					outputStream.close();
				} catch (IOException ex4) {
					BIGInterface.getInstance().getConsole()
							.postMessage("Error while writing!", BIGConsole.ERROR);
				}
				obs.setProgress(100);
			}
		};
		t1.start();
		return t1;
	}

	public static void saveToCsv(File file, double[][] values, String[] identifier,
			String[] additionalInformation) {
		try {
			file.createNewFile();
		} catch (IOException ioe) {
			System.err.println("Couldn't write file. No rights?");
		}

		StringBuffer strOutput = new StringBuffer();
		strOutput.append("BenchIT-generated csv\n");
		for (int i = 0; i < additionalInformation.length; i++) {
			strOutput.append(additionalInformation[i] + "\n");
		}
		for (int i = 0; i < identifier.length; i++) {
			strOutput.append(identifier[i] + ";");
		}
		strOutput.deleteCharAt(strOutput.length() - 1); // removing the last ;
		strOutput.append("\n");
		for (int i = 0; i < values[0].length; i++) {
			for (int j = 0; j < values.length; j++) {
				strOutput.append(values[j][i]);
				while (strOutput.lastIndexOf(".") > (strOutput.length() - 4)) {
					strOutput.append(0);
				}
				strOutput.append(";");

			}
			strOutput.deleteCharAt(strOutput.length() - 1); // removing the last ;
			strOutput.append("\n");
		}
		try {
			FileWriter outputStream = new FileWriter(file);

			outputStream.write(strOutput.toString());
			outputStream.flush();
			outputStream.close();
		} catch (IOException exception) {
			System.err.println("Error while writing data to file! No rights?");
			return;
		}

	}

	public static String getContent(InputStream in) {
		int i = 0;
		StringBuffer fileContent = new StringBuffer();
		byte[] buffer = new byte[1000];
		int readData = 0;
		try {

			do {
				readData = in.read(buffer);
				if (readData > 0) {
					fileContent.append(new String(buffer, 0, readData));
				}
				if (readData > -1) {
					System.err.println("i:" + (i++) + "/" + readData);
				}
			} while (true);
		} catch (IOException ioe) {
			String s = fileContent.toString();
			return s;

		}

	}

	class BIGObservable extends Observable {
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

	public Observable getObservable() {
		return progress;
	}

	public static void setFonts(org.jfree.chart.axis.NumberAxis axis, String axisName) {
		BIGInterface interf = system.BIGInterface.getInstance();;
		try {
			String name = interf.getBIGConfigFileParser().stringCheckOut(axisName + "Font");
			int style = interf.getBIGConfigFileParser().intCheckOut(axisName + "FontStyle");
			int size = interf.getBIGConfigFileParser().intCheckOut(axisName + "FontSize");
			axis.setLabelFont(new Font(name, style, size));
		} catch (Exception ex1) {
			System.out.println(axisName + "Font Specification not found in BGUI.cfg");
			System.out.println("Using default");
		}

		try {
			String tname = interf.getBIGConfigFileParser().stringCheckOut(axisName + "TickFont");
			int tstyle = interf.getBIGConfigFileParser().intCheckOut(axisName + "TickFontStyle");
			int tsize = interf.getBIGConfigFileParser().intCheckOut(axisName + "TickFontSize");
			axis.setTickLabelFont(new Font(tname, tstyle, tsize));
		} catch (Exception ex) {
			System.out.println(axisName + "TickFont Specification not found in BGUI.cfg");
			System.out.println("Using default");
		}

	}

	public static Font getFont(String name) {
		BIGInterface interf = system.BIGInterface.getInstance();;
		try {
			String fname = interf.getBIGConfigFileParser().stringCheckOut(name + "Font");
			int style = interf.getBIGConfigFileParser().intCheckOut(name + "FontStyle");
			int size = interf.getBIGConfigFileParser().intCheckOut(name + "FontSize");
			return new Font(fname, style, size);
		} catch (Exception ex1) {
			Font f = new Font("Serif", 0, 10);
			System.out.println(name + "Font Specification not found in BGUI.cfg");
			System.out.println("Setting default (Serif,plain,10)");
			interf.getBIGConfigFileParser().save();
			File bguicfg = interf.getBIGConfigFileParser().getFile();
			String content = BIGFileHelper.getFileContent(bguicfg);
			StringBuffer sb = new StringBuffer(content);
			sb.append("\n# start auto filled axis font settings\n");
			sb.append(name + "Font = " + f.getName() + "\n");
			sb.append(name + "FontStyle = " + f.getStyle() + "\n");
			sb.append(name + "FontSize = " + f.getSize() + "\n");
			sb.append("# end auto filled axis font settings");
			BIGFileHelper.saveToFile(sb.toString(), bguicfg);
			interf.setBIGConfigFileParser(new BIGConfigFileParser(bguicfg.getAbsolutePath()));
			return f;
		}

	}

	public static Font getFont(String name, File file) {
		BIGConfigFileParser p = new BIGConfigFileParser(file.getAbsolutePath());
		try {
			String fname = p.stringCheckOut(name + "Font");
			int style = p.intCheckOut(name + "FontStyle");
			int size = p.intCheckOut(name + "FontSize");
			return new Font(fname, style, size);
		} catch (Exception ex1) {
			return null;
		}

	}

	/**
	 * @param str any string
	 * @param substr the substring to search for
	 * @return true if str contains substr, false otherwise
	 */
	public static boolean stringContains(String str, String substr) {
		if (str.indexOf(substr) < 0)
			// substring is not a part of str
			return false;
		else
			// str contains substr
			return true;
	}

	/**
	 * Rounds the value to contain the given number of digits. This includes 1 digit in front of the decimal point
	 * 
	 * @param val
	 * @param digits
	 * @return
	 */
	public static double round(double val, int digits) {
		if (val == 0)
			return val;
		double tmpVal = Math.abs(val);
		int dig = (int) Math.log10(tmpVal);
		if (dig < 0)
			dig = 0;
		digits -= dig + 1;
		if (digits < 0) {
			double mult = Math.pow(10, -digits);
			return Math.rint(val / mult) * mult;
		} else {
			double mult = Math.pow(10, digits);
			return Math.rint(val * mult) / mult;
		}
	}
}
