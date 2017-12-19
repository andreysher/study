package plot.data;

/**
 * <p>
 * Überschrift: BenchIT
 * </p>
 * <p>
 * Beschreibung: Represents an Object which can be plotted by BIGPlot
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
import java.awt.*;
import java.io.File;
import java.util.*;
import java.util.List;

import javax.swing.*;

import org.jfree.chart.annotations.XYTextAnnotation;

import system.*;
import conn.Graph;

public abstract class BIGPlotable {
	public boolean drawAntiAliased = false;
	/**
	 * fill the shapes for yaxis 1 and 2?
	 */
	public boolean fillShapes[] = {false, false};
	/**
	 * draw lines between shapes for yaxis 1 and 2?
	 */
	public boolean drawLines[] = {false, false};
	/**
	 * Data for all y axis Shall not be reassigned!
	 */
	public final Vector<YAxisData> yAxis = new Vector<YAxisData>();
	public XAxisData xData = new XAxisData(null, this);

	/**
	 * Parser for this plotable
	 */
	protected BIGOutputParser parser = null;
	/**
	 * displayedDataSets means: data[displayedDataSets[0]] used for first yaxis, data[displayedDataSets[1]] used for
	 * second yaxis
	 */
	public int[] displayedDataSets = {};
	private String title = "";
	/**
	 * default title
	 */
	protected String titleDefault = "";
	private String annComment = "";
	// parsed comment
	protected String pComment = null;
	// parsed title
	private String pTitle = null;
	private final Vector<XYTextAnnotation> annots = new Vector<XYTextAnnotation>();
	/**
	 * used for progress
	 */
	protected JProgressBar progressBar = new JProgressBar();
	/**
	 * used for progress
	 */
	protected JLabel progressLabel = new JLabel();
	/**
	 * which fonts are read from config
	 */
	public final static String[] fontNames = {"plotComment", "title", "legend", "XAxis", "XAxisTick",
			"YAxis", "YAxisTick"};
	/**
	 * here the fonts are saved
	 */
	protected Font[] fonts = new Font[fontNames.length];
	/**
	 * where to display the comment
	 */
	protected double commentX = -1;
	protected double commentY = -1;
	/**
	 * how large are the insets for the plot (space to borders to north,west,east and south (other orde))
	 */
	protected Insets insetsDefault = new Insets(4, 8, 4, 32);
	public Insets insets = insetsDefault;
	// -----------------------------------------------------------------
	/**
	 * add a function
	 * 
	 * @param graph Graph of the function
	 */
	abstract public boolean addFunction(Graph graph);

	/**
	 * remove a function
	 * 
	 * @param name String name of the function
	 */
	abstract public void removeFunction(String name);

	/**
	 * initialize plotable
	 */
	public abstract boolean init();

	/**
	 * save plotable
	 */
	abstract public void save();
	/**
	 * return the File this plotable is saved to
	 * 
	 * @return File
	 */
	public abstract File getSaveFile();

	public abstract boolean IsInit();

	/**
	 * Change name of specified dataset
	 * 
	 * @param whichDataSet
	 * @param index
	 * @param newName
	 */
	public abstract void changeName(int whichDataSet, int index, String newName);

	public int getYAxisIndex(String text) {
		for (int i = 0; i < yAxis.size(); i++)
			if (yAxis.get(i).Text.equals(text))
				return i;
		return -1;
	}

	public YAxisData getYAxis(String text) {
		int i = getYAxisIndex(text);
		if (i < 0)
			return null;
		return yAxis.get(i);
	}

	public int getYAxisIndexByName(String name) {
		for (int i = 0; i < yAxis.size(); i++)
			if (yAxis.get(i).getData().getSeriesIndex(name) >= 0)
				return i;
		return -1;
	}

	public YAxisData getYAxisByName(String name) {
		int i = getYAxisIndexByName(name);
		if (i < 0)
			return null;
		return yAxis.get(i);
	}

	private int getDecimals(double value, int max) {
		double maxDiff = Math.pow(10, -max);
		double diff = Math.abs(value - Math.floor(value));
		for (int i = 0; i < max; i++) {
			if (diff < maxDiff)
				return i;
			maxDiff *= 10;
		}
		return max;
	}

	public void calculateScaleX() {
		if (yAxis.size() == 0)
			return;
		calculateScaleX(yAxis.get(0).getData().getValues(), xData);
	}

	public void calculateScaleX(double[][] values, XAxisData axis) {
		calculateScale(values, axis, 0, 1);
	}

	public void calculateScaleY(YAxisData axis) {
		calculateScale(axis.getData().getValues(), axis, 1, axis.getData().getSeriesCount() + 1);
	}

	private void calculateScale(double[][] values, AxisData axis, int start, int end) {
		if (axis.Log > 2)
			return;
		String text = axis.Text.toLowerCase();
		// Allow scaling only in certain cases
		if (text.startsWith("flop") || text.equals("ops") || text.equals("iops") || text.equals("s")
				|| text.equals("seconds") || text.startsWith("byte") || text.startsWith("b /")
				|| text.startsWith("b/") || text.equals("b")) {
			// get absolute minimum
			double minVal = 1e30;
			for (int i = start; i < end; i++)
				for (int j = 0; j < values[i].length; j++) {
					double cur = Math.abs(values[i][j]);
					if (cur > 1e-18 && cur < minVal)
						minVal = cur;
				}
			/* decimal scaling */
			double scaling_base = 1000;
			int scaling_level = 3;
			/* find best scaling level by going through data */
			if (minVal >= scaling_base) {
				while (minVal >= scaling_base) {
					minVal /= scaling_base;
					scaling_level++;
				}
			} else if ((minVal < 1.0) && (axis.Log == 0)) { /* "negative" scaling not possible for base 2 */
				while (minVal < 1.0) {
					minVal *= scaling_base;
					scaling_level--;
				}
			}
			if (scaling_level < 0)
				scaling_level = 0;
			if (scaling_level > 7)
				scaling_level = 7;
			axis.setPre(scaling_level);
			axis.scaleAxisText = true;
		}
		calculateNumberFormat(values, axis, start, end);
	}

	protected void calculateNumberFormatX(double[][] values, AxisData axis) {
		calculateNumberFormat(values, axis, 0, 1);
	}

	protected void calculateNumberFormatY(double[][] values, AxisData axis) {
		calculateNumberFormat(values, axis, 1, values.length);
	}

	private void calculateNumberFormat(double[][] values, AxisData axis, int start, int end) {
		int maxDecimals = 0;
		double mod = BIGDataSet.setModifiers[axis.getPre()];
		for (int i = start; i < end; i++)
			for (int j = 0; j < values[i].length; j++) {
				int cur = getDecimals(values[i][j] * mod, 3);
				if (cur > maxDecimals) {
					maxDecimals = cur;
					if (cur >= 3) {
						i = values.length;
						break;
					}
				}
			}
		if (maxDecimals > 0)
			axis.NumberFormat = "0." + new String(new char[maxDecimals]).replace("\0", "0");
		else
			axis.NumberFormat = "0";
	}

	public void calculateNumberFormatX() {
		calculateNumberFormatX(yAxis.get(0).Data.getValues(), xData);
	}

	public void calculateNumberFormatY(int axis) {
		calculateNumberFormatY(yAxis.get(axis).Data.getValues(), yAxis.get(axis));
	}

	public void calculateNumberFormatY(YAxisData axis) {
		calculateNumberFormatY(axis.Data.getValues(), axis);
	}

	void setXPre(int xPre, boolean setSelf) {
		for (YAxisData axis : yAxis)
			axis.setXPre(xPre);
		if (setSelf)
			xData.setPre(xPre);
	}

	void setXMinAbs(double min, boolean setSelf) {
		for (YAxisData axis : yAxis)
			if (axis.getData() != null)
				axis.getData().setMinimumDomainValue(min, true);
		if (setSelf)
			xData.setMinAbs(min);
	}

	void setXMaxAbs(double max, boolean setSelf) {
		for (YAxisData axis : yAxis)
			if (axis.getData() != null)
				axis.getData().setMaximumDomainValue(max, true);
		if (setSelf)
			xData.setMaxAbs(max);
	}

	public List<String> getLegends() {
		List<String> legends = new ArrayList<String>();
		for (YAxisData axis : yAxis) {
			for (String name : axis.getData().getNames())
				legends.add(name);
		}
		return legends;
	}

	public Insets getDefaultInsets() {
		return insetsDefault;
	}

	/**
	 * initializes standards like fonts or insets
	 * 
	 * @param f File where to read these values from
	 */
	public void initFonts(File f) {
		// default insets
		BIGInterface BigInterface = BIGInterface.getInstance();
		BIGConfigFileParser parser = BigInterface.getBIGConfigFileParser();
		try {

			insets = new Insets(parser.intCheckOut("plotInsetsTop"),
					parser.intCheckOut("plotInsetsLeft"), parser.intCheckOut("plotInsetsBottom"),
					parser.intCheckOut("plotInsetsRight"));
			// use standard if exception
		} catch (Exception ex) {
			insets = insetsDefault;
			parser.set("plotInsetsTop", insets.top + "");
			parser.set("plotInsetsLeft", insets.left + "");
			parser.set("plotInsetsBottom", insets.bottom + "");
			parser.set("plotInsetsRight", insets.right + "");
		}
		// default fillShapes and drawLines
		try {

			fillShapes[0] = parser.boolCheckOut("shapes1Filled");
			drawLines[0] = parser.boolCheckOut("shapes1Lines");
			fillShapes[1] = parser.boolCheckOut("shapes2Filled");
			drawLines[1] = parser.boolCheckOut("shapes2Lines");
			// use standard if exception
		} catch (Exception ex) {
			parser.set("shapes1Filled", "0");
			parser.set("shapes1Lines", "0");
			parser.set("shapes2Filled", "0");
			parser.set("shapes2Lines", "0");
		}
		// read standard fonts
		Font temp = null;
		File std = parser.getFile();
		for (int i = 0; i < fonts.length; i++) {
			fonts[i] = BIGUtility.getFont(fontNames[i], std);
		}

		commentX = parser.intCheckOut("plotCommentPercentX", 80) * .01;
		commentY = parser.intCheckOut("plotCommentPercentY", 90) * .01;
		setAnnComment(parser.stringCheckOut("plotComment", ""));
		// read fonts from f
		if ((f != null) && (f.exists())) {
			// check whether to use std fonts
			boolean standardFont = false;
			try {
				standardFont = parser.boolCheckOut("useStandardFonts");
			} catch (Exception ex2) {
				parser.set("useStandardFonts", "0");
			}
			if (!standardFont) {
				for (int i = 0; i < fonts.length; i++) {
					temp = BIGUtility.getFont(fontNames[i], f);
					if (temp != null) {
						fonts[i] = temp;
					}
				}
			}
			BIGConfigFileParser p = new BIGConfigFileParser(f.getAbsolutePath());
			try {
				setCommentPos(p.intCheckOut("plotCommentPercentX"), p.intCheckOut("plotCommentPercentY"));
				setAnnComment(p.stringCheckOut("plotComment"));
			} catch (Exception ex1) {}

			// and now the insets (space around the plot itself)
			try {
				insets = new Insets(p.intCheckOut("plotInsetsTop"), p.intCheckOut("plotInsetsLeft"),
						p.intCheckOut("plotInsetsBottom"), p.intCheckOut("plotInsetsRight"));
				// use standard if exception
			} catch (Exception ignored) {}

			fillShapes[0] = p.boolCheckOut("shapes1Filled", false);
			drawLines[0] = p.boolCheckOut("shapes1Lines", false);
			fillShapes[1] = p.boolCheckOut("shapes2Filled", false);
			drawLines[1] = p.boolCheckOut("shapes2Lines", false);
		}
		// if fonts could be initialized nowhere: use standard SansSerif,BOLD,14
		for (int i = 0; i < fonts.length; i++) {
			if (fonts[i] == null) {
				if (i == 1) {
					// title-font gets a size of 16pt
					fonts[i] = new Font("SansSerif", Font.BOLD, 16);
				} else {
					// all other fonts get a size of 14pt
					fonts[i] = new Font("SansSerif", Font.BOLD, 14);
				}
			}
		}
	}

	/**
	 * append information about fonts, insets... (everything in initFonts)
	 * 
	 * @param sb StringBuffer where to add the information
	 */
	public void appendFontInformation(StringBuffer sb) {
		// fonts
		sb.append("\n#font informations\n");
		boolean standardFont = false;
		Font localFonts[] = fonts;
		try {
			standardFont = BIGInterface.getInstance().getBIGConfigFileParser()
					.boolCheckOut("useStandardFonts");
		} catch (Exception ex2) {
			BIGInterface.getInstance().getBIGConfigFileParser().set("useStandardFonts", "0");

		}
		if (standardFont && getSaveFile().exists()) {
			Font temp;
			for (int i = 0; i < fonts.length; i++) {
				temp = BIGUtility.getFont(fontNames[i], getSaveFile());
				if (temp != null) {
					localFonts[i] = temp;
				}
			}
		}
		for (int i = 0; i < fonts.length; i++) {
			sb.append(BIGPlotable.fontNames[i] + "Font=" + localFonts[i].getName() + "\n");
			sb.append(BIGPlotable.fontNames[i] + "FontStyle=" + localFonts[i].getStyle() + "\n");
			sb.append(BIGPlotable.fontNames[i] + "FontSize=" + localFonts[i].getSize() + "\n");
		}
		// comment
		sb.append("\n# plotComment\n");
		sb.append("plotComment=" + annComment + "\n");
		sb.append("plotCommentPercentX=" + getCommentXPercent() + "\n");
		sb.append("plotCommentPercentY=" + getCommentYPercent() + "\n");
		// insets
		sb.append("\n# Insets\n");
		sb.append("plotInsetsTop=" + insets.top + "\n");
		sb.append("plotInsetsLeft=" + insets.left + "\n");
		sb.append("plotInsetsBottom=" + insets.bottom + "\n");
		sb.append("plotInsetsRight=" + insets.right + "\n");
		// shapes and lines
		sb.append("\n# Shapes\n");
		sb.append("shapes1Filled=");
		if (fillShapes[0]) {
			sb.append("1\n");
		} else {
			sb.append("0\n");
		}
		sb.append("shapes1Lines=");
		if (drawLines[0]) {
			sb.append("1\n");
		} else {
			sb.append("0\n");
		}
		// ---------------------------------------
		sb.append("AxisTextScaled=");
		if (yAxis.size() > 0 && yAxis.get(0).scaleAxisText) {
			sb.append("1\n");
		} else {
			sb.append("0\n");
		}
		// ---------------------------------------
		sb.append("shapes2Filled=");
		if (fillShapes[1]) {
			sb.append("1\n");
		} else {
			sb.append("0\n");
		}
		sb.append("shapes2Lines=");
		if (drawLines[1]) {
			sb.append("1\n");
		} else {
			sb.append("0\n");
		}
	}

	/**
	 * get the number of funtions from a dataset
	 * 
	 * @param i int which dataset to use
	 * @return int length of data[i] or 0 (if data[i] is not available)
	 */
	public int getNumberOfFunctionsForDataset(int i) {
		// if !exists
		if (yAxis.size() <= i || i < 0)
			// return 0;
			return 0;
		else
			// else return length
			return yAxis.get(i).Data.getSeriesCount();
	}

	/**
	 * gets the number of total functions used in this plotable
	 * 
	 * @return int number of functions
	 */
	public int getTotalFunctions() {
		// add sizes for single datasets
		int size = 0;
		for (int i = 0; i < yAxis.size(); i++) {
			size = size + yAxis.get(i).Data.getSeriesCount();
		}
		return size;
	}

	/**
	 * gets y-Values for a specific function
	 * 
	 * @param functionName String name of the function
	 * @return double[] y Values
	 */
	public double[] getDataForFunction(String functionName) {
		// find function
		for (int i = 0; i < yAxis.size(); i++) {
			for (int j = 0; j < yAxis.get(i).Data.getSeriesCount(); j++) {
				// return data
				if (yAxis.get(i).Data.getSeriesName(j).equals(functionName))
					return yAxis.get(i).Data.getValues()[j + 1];
			}
		}
		// not found return nothing
		return new double[0];
	}

	/**
	 * renames a specific function
	 * 
	 * @param oldName String old name
	 * @param newName String new name
	 */
	public void rename(String oldName, String newName) {
		for (int i = 0; i < yAxis.size(); i++) {
			for (int j = 0; j < yAxis.get(i).Data.getSeriesCount(); j++) {
				// return data
				if (yAxis.get(i).Data.getSeriesName(j).equals(oldName)) {
					// rename it
					yAxis.get(i).Data.setSeriesName(j, newName);
					return;
				}
			}
		}
	}

	/**
	 * sets progressBar and progressLabel (used for init once for guis progress)
	 * 
	 * @param progressBar JProgressBar progressbar from gui
	 * @param progressLabel JLabel progressLabel from gui
	 */
	public void setProgress(JProgressBar progressBar, JLabel progressLabel) {
		this.progressBar = progressBar;
		this.progressLabel = progressLabel;
	}

	/**
	 * gets the annotation for the plot
	 * 
	 * @return XYTextAnnotation
	 */
	public XYTextAnnotation getAnnotationComment() {
		// get the font for annot.
		Font f = getFont("plotComment");
		// not found? use std.
		if (f == null) {
			f = BIGUtility.getFont("plotComment");
		}
		// compute position
		// procentual between min and max
		double xMax = xData.getMax();
		double xMin = xData.getMin();
		double yMax = (yAxis.size() > 0) ? yAxis.get(0).getMax() : 0;
		double yMin = (yAxis.size() > 0) ? yAxis.get(0).getMin() : 0;
		// where to display (value at the axis)
		double xComment = xMin + (xMax - xMin) * commentX;
		double yComment = yMin + (yMax - yMin) * commentY;

		if (xData.Log > 1) {
			double tmp1 = Math.log(xMin);
			double tmp2 = Math.log(xMax);
			xComment = Math.pow(Math.E, ((tmp2 - tmp1) * commentX) + tmp1);
		}
		if (yAxis.size() > 0 && yAxis.get(0).Log > 1) {
			double tmp1 = Math.log(yMin);
			double tmp2 = Math.log(yMax);
			yComment = Math.pow(Math.E, ((tmp2 - tmp1) * commentY) + tmp1);
		}
		// set position, name, font
		XYTextAnnotation an = new XYTextAnnotation(getExtComment(), f, xComment, yComment);
		return an;
	}

	protected String parseParameterString(String s) {
		// real name will be written here
		StringBuffer sb = new StringBuffer();
		// increments over characters
		for (int i = 0; i < s.length(); i++) {
			// starting a flag like <processorname>
			if (s.charAt(i) != '<') {
				sb.append(s.charAt(i));
			} else {
				if (s.indexOf(">", i) == -1) {
					sb.append(s.charAt(i));
				} else {
					// gets the setting for e.g. processorname
					sb.append(getSetting(s.substring(i + 1, s.indexOf(">", i))));
					i = s.indexOf(">", i);
				}
			}
		}
		return sb.toString();
	}

	/**
	 * sets the comment for the plot
	 * 
	 * @param s String comment text
	 */
	public void setAnnComment(String s) {
		// remove beginnig and ending whitespaces
		// save orig
		annComment = s.trim();
		// save parsed
		pComment = parseParameterString(s);
	}

	/**
	 * gets the parsed commet. if not avail, use stdComment
	 * 
	 * @return String
	 */
	public String getExtComment() {
		if (pComment != null)
			return pComment;
		return annComment;
	}

	/**
	 * get std. Comment (not parsed)
	 * 
	 * @return String
	 */
	public String getAnnComment() {
		return annComment;
	}

	/**
	 * get x-pos (percentual) for comment
	 * 
	 * @return int eg 56 for 56%
	 */
	public int getCommentXPercent() {
		return Math.round((float) (commentX * 100.0));
	}

	/**
	 * get y-pos (percentual) for comment
	 * 
	 * @return int eg 56 for 56%
	 */

	public int getCommentYPercent() {
		return Math.round((float) (commentY * 100.0));
	}

	/**
	 * sets position for comment (percentual, e.g. 61 for 61%)
	 * 
	 * @param x int new x-Pos
	 * @param y int new y-Pos
	 */
	public void setCommentPos(double x, double y) {
		commentX = (1.0 * x) * 0.01;
		commentY = (1.0 * y) * 0.01;
	}

	/**
	 * get parsed title. if not avail, get orig title
	 * 
	 * @return String
	 */
	public String getExtTitle() {
		if (pTitle != null)
			return pTitle;
		return title;
	}

	/**
	 * set title to s
	 * 
	 * @param s String new title
	 */
	public void setTitle(String s) {
		// see setComment
		title = s.trim();

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) != '<') {
				sb.append(s.charAt(i));
			} else {
				if (s.indexOf(">", i) == -1) {
					sb.append(s.charAt(i));
				} else {
					sb.append(getSetting(s.substring(i + 1, s.indexOf(">", i))));
					i = s.indexOf(">", i);
				}
			}
		}
		pTitle = sb.toString();
	}

	/**
	 * get setting: used for get setting(processorname) or get(processorclockrate|5|2) 5 means shift it (must be a
	 * number!) 5 positions 2 means set numbers after point to 2
	 * 
	 * @param name String name of the item in the resultfile
	 * @return String setting of the item
	 */
	private String getSetting(String name) {
		// sth wrong
		boolean error = false;
		// how many shifts
		int shift = 0;
		// after point
		int after = 0;
		// if there is an |
		if (name.indexOf('|') > -1) {
			// not at the beginning
			if (name.lastIndexOf('|') > 1) {
				// at least two different |'s
				if (name.lastIndexOf('|') > name.indexOf('|')) {
					// get shifts
					try {
						shift = Integer.parseInt(name.substring(name.indexOf('|') + 1, name.lastIndexOf('|')));
					} catch (NumberFormatException ex) {
						System.err.print("Argument 2 must be an Integer.");
						error = true;
					}
					// get after-point
					try {
						after = Integer.parseInt(name.substring(name.lastIndexOf('|') + 1, name.length()));
					} catch (NumberFormatException ex) {
						System.err.print("Argument 3 must be an Integer.");
						error = true;
					}
					// item before first |
					name = name.substring(0, name.indexOf('|'));
				} else {
					// just one |
					error = true;
				}
			}
		}
		if (error) {
			// display usage information
			System.err.println("Wrong usage");
			System.err.println("Example: \"<processorclockrate|-9|1> \" means ");
			System.err.println("         print in GHz (/10^9) with one digit after point.");
			System.err.println("Or simple: <processorname>");
		}
		// you cant display less then no numbers after the point
		if (after < 0) {
			after = 0;
		}
		// get the setting for the item
		String s = (parser == null) ? null : parser.getValue(name);
		// if it wasn't found: return <itemname>
		if (s == null || s.equals(""))
			return '<' + name + '>';
		// shift?
		if (shift != 0) {
			// if there was an error do nothin
			if (!error) {
				// used for 1!E10!
				String e = null;
				// setting as number
				double d = 0.0;
				try {
					d = Double.parseDouble(s);
				} catch (NumberFormatException ex1) {
					System.err.println("The setting for " + name + " is not a  number.");
					return s;
				}
				// shift it according to to setted shift
				d = d * Math.pow(10.0, 1.0 * shift);
				s = "" + d;
				// remove the E10 part
				if (s.indexOf('E') > -1) {
					e = s.substring(s.indexOf('E'));
					s = s.substring(0, s.indexOf('E'));
				}
				// removing ending numbers (after point) according to setted after
				int commaPos = s.indexOf('.');
				if ((commaPos == -1) && (after != 0)) {
					s = s + ".";
				}
				commaPos = s.indexOf('.');
				if (commaPos > -1) {
					// cut
					if ((s.length() - 1 - commaPos) > after) {
						s = s.substring(0, commaPos + after + 1);
					} else {
						// append zeros
						if ((s.length() - 1 - commaPos) < after) {
							int x = after - (s.length() - commaPos) + 1;
							for (int i = 0; i < x; i++) {
								s = s + "0";
							}

						}
					}
					if (s.charAt(s.length() - 1) == '.') {
						s = s.substring(0, s.length() - 1);
					}
				}
				// add the E10 part
				if (e != null) {
					s = s + e;
				}
			}
		}
		// return the setting
		return s;
	}

	/**
	 * sets the default-configs-standard-title as title
	 */
	public void setStandardTitle() {
		String t = BIGInterface.getInstance().getBIGConfigFileParser()
				.stringCheckOut("standardTitle", "");
		if (t.trim().length() == 0)
			return;
		setTitle(t);
	}

	/**
	 * get the font for the item with name which or null
	 * 
	 * @param which String name
	 * @return Font font for item
	 */
	public Font getFont(String which) {
		for (int i = 0; i < BIGPlotable.fontNames.length; i++)
			if (fontNames[i].equals(which))
				return fonts[i];

		return null;
	}

	/**
	 * get all saved fonts
	 * 
	 * @return Font[] all saved fonts
	 */
	public Font[] getAllFonts() {
		return fonts;
	}

	/**
	 * sets the font for a specific item
	 * 
	 * @param which String item name
	 * @param f Font font for item
	 */
	public void setFont(String which, Font f) {
		for (int i = 0; i < BIGPlotable.fontNames.length; i++)
			if (fontNames[i].equals(which)) {
				fonts[i] = f;
			}
		return;
	}

	public void addAnnotation(String text, double x, double y) {
		annots.add(new XYTextAnnotation(text, x, y));
	}

	public String getTitle() {
		return title;
	}

	public String getDefaultTitle() {
		return titleDefault;
	}

	public String getText() {
		return (parser == null) ? "" : parser.toString();
	}
}
