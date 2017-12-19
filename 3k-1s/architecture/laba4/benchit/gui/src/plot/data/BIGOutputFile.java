package plot.data;

import gui.BIGConsole;

import java.awt.Color;
import java.awt.datatransfer.*;
import java.io.*;
import java.util.*;

import plot.gui.BIGPlotRenderer;
import system.*;
import conn.Graph;

/**
 * <p>
 * Überschrift: BenchIT - BIGOutputFile
 * </p>
 * <p>
 * Beschreibung: Contains a lot of information to an outputfile
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
public class BIGOutputFile extends BIGPlotable implements Transferable {
	private final BIGConsole console = BIGInterface.getInstance().getConsole();

	/**
	 * output file (ends with .bit)
	 */
	private final File outputFile;
	private boolean loaded = false;
	/**
	 * type (e.g.numerical)
	 */
	private String type;
	/**
	 * name (e.g.matmul)
	 */

	private String name;
	/**
	 * name (e.g.C)
	 */
	private String language;
	/**
	 * parallel libs (e.g.MPI)
	 */
	private String parLibs;
	/**
	 * other libs (e.g. glut)
	 */
	private String libs;
	/**
	 * type / datatype (e.g. double)
	 */
	private String dataType;
	/**
	 * name of the file without the .bit
	 */
	private final String fileNameWithoutExtension;

	private static final String UNDEFINED = "undefined";

	/**
	 * when functions for different axis are read, we need to know the order of the original output-file. This is saved
	 * here. file.bit y<order[1]>name means file.bit.gui y2name (1 because the count starts in JAVA with 0, in the
	 * bit-file with 1) Also the order-entry is modified: <numberofaxis-1>*100+order
	 */
	int[] order = new int[0];

	private BIGRawFile rawFile;

	/**
	 * build a new BIGOutputFile by parsing the file f
	 * 
	 * @param f File
	 * @throws FileNotFoundException
	 */
	public BIGOutputFile(File f) throws FileNotFoundException {
		super();

		if (!f.exists())
			throw new FileNotFoundException(f.getAbsolutePath() + " not found!");
		boolean errorOccured = false;
		File temp = null;

		// setting member variables
		outputFile = f;
		fileNameWithoutExtension = f.getName().substring(0, f.getName().lastIndexOf("."));
		temp = f.getParentFile();
		try {
			dataType = temp.getName();
			if (dataType.equals("")) {
				dataType = UNDEFINED;
				errorOccured = true;
			}
		} catch (Exception e) {
			dataType = UNDEFINED;
			errorOccured = true;
		}

		try {
			temp = temp.getParentFile();
			libs = temp.getName();
			if (libs.equals("")) {
				libs = UNDEFINED;
				errorOccured = true;
			}
		} catch (Exception e) {
			libs = UNDEFINED;
			errorOccured = true;
		}

		try {
			temp = temp.getParentFile();
			parLibs = temp.getName();
			if (parLibs.equals("")) {
				parLibs = UNDEFINED;
				errorOccured = true;
			}
		} catch (Exception e) {
			parLibs = UNDEFINED;
			errorOccured = true;
		}

		try {
			temp = temp.getParentFile();
			language = temp.getName();
			if (language.equals("")) {
				language = UNDEFINED;
				errorOccured = true;
			}
		} catch (Exception e) {
			language = UNDEFINED;
			errorOccured = true;
		}

		try {
			temp = temp.getParentFile();
			name = temp.getName();
			if (name.equals("")) {
				name = UNDEFINED;
				errorOccured = true;
			}
		} catch (Exception e) {
			name = UNDEFINED;
			errorOccured = true;
		}

		try {
			temp = temp.getParentFile();
			type = temp.getName();
			if (type.equals("")) {
				type = UNDEFINED;
				errorOccured = true;
			}
		} catch (Exception e) {
			type = UNDEFINED;
			errorOccured = true;
		}
		if (errorOccured) {
			console.postMessage("File " + f.getAbsolutePath()
					+ " violates the BenchIT folder structure. "
					+ "Please read the documentation for information about the necessary folder structure.",
					BIGConsole.ERROR);
		}
		File rawFile = new File(outputFile.getAbsolutePath() + ".raw");
		if (rawFile.exists())
			this.rawFile = new BIGRawFile(rawFile);
	}

	private boolean deleteChecked(String fileName) {
		File file = new File(fileName);
		if (file.exists())
			if (!file.delete()) {
				System.err.println("An Error occured while deleting the result file: " + fileName);
				return false;
			}
		return true;
	}

	public boolean delete() {
		String fileName = getFile().getAbsolutePath();
		int deleted = 0;
		if (deleteChecked(fileName))
			deleted++;
		if (deleteChecked(fileName + ".gp"))
			deleted++;
		if (deleteChecked(fileName + ".gui"))
			deleted++;
		if (deleteChecked(fileName + ".gp.eps"))
			deleted++;
		if (rawFile != null && deleteChecked(rawFile.getFile().getAbsolutePath()))
			deleted++;
		if (deleted > 0) {
			System.out.println("The files for \"" + fileName + "\" were deleted!");
			System.out.flush();
			return true;
		}
		return false;
	}

	/**
	 * gets filename without .bit
	 * 
	 * @return String filename without .bit
	 */
	public String getFilenameWithoutExtension() {
		return fileNameWithoutExtension;
	}

	/**
	 * gets the bit file
	 * 
	 * @return File the bit file for this
	 */
	public File getFile() {
		return outputFile;
	}

	/**
	 * gets the name of the algorithm
	 * 
	 * @return String name of algorithm
	 */
	public String getName() {
		return name;
	}

	/**
	 * gets the type of the algorithm
	 * 
	 * @return String type of algorithm
	 */
	public String getType() {
		return type;
	}

	/**
	 * gets the other libraries of the algorithm
	 * 
	 * @return String other libraries of algorithm
	 */
	public String getLibraries() {
		return libs;
	}

	/**
	 * gets the parallel libraries of the algorithm
	 * 
	 * @return String parallel libraries of algorithm
	 */
	public String getParallelLibraries() {
		return parLibs;
	}

	/**
	 * gets the programming language of the algorithm
	 * 
	 * @return String programming language of algorithm
	 */
	public String getSourceLanguage() {
		return language;
	}

	/**
	 * gets the datatype of the algorithm
	 * 
	 * @return String datatype of algorithm
	 */
	public String getDataType() {
		return dataType;
	}

	public BIGRawFile getRawFile() {
		return rawFile;
	}

	public void removeSavedInformations() {
		if (getSaveFile().exists()) {
			getSaveFile().delete();
		}
	}

	/**
	 * gets the names of the settings with the "sorting"-part leading
	 * 
	 * @param sorting int sort from which
	 * @return String[] names for settings
	 */
	public String[] getSortedNames(int sorting) {
		String[] sortedNames = new String[6];
		// initial setting
		sortedNames[0] = type;
		sortedNames[1] = name;
		sortedNames[2] = language;
		sortedNames[3] = parLibs;
		sortedNames[4] = libs;
		sortedNames[5] = dataType;
		// sort
		String tempString = sortedNames[sorting];
		for (int i = sorting; i > 0; i--) {
			sortedNames[i] = sortedNames[i - 1];
		}
		sortedNames[0] = tempString;
		return sortedNames;
	}

	/**
	 * gets the name of the algorithm with leading setting i
	 * 
	 * @param sorting int which part shall lead the sorting
	 * @return String the sorted name of the algorithm
	 */
	public String getNameAfterSorting(int sorting) {
		// get parts
		String[] names = getSortedNames(sorting);
		String name = new String();
		// combine them with dots
		for (int i = 0; i < names.length; i++) {
			name = name + "." + names[i];
		}
		return name.substring(1);
	}

	/**
	 * used for JTree
	 * 
	 * @return String returns filename without extension
	 */
	@Override
	public String toString() {
		return fileNameWithoutExtension;
	}

	/**
	 * parses output-file for "comment=" and returns setting (used for popup)
	 * 
	 * @return String the setting for comment
	 */
	public String getComment() {
		if (parser == null) {
			try {
				parser = new BIGOutputParser(getFile().getAbsolutePath());
			} catch (BIGParserException e) {
				return "";
			}
		}
		String comment = parser.getValue("comment");
		if (comment != null) {
			String pComment = parseParameterString(comment);
			if (pComment != null)
				comment = pComment;
		}
		return comment;
	}

	/**
	 * saves .bit.gui-file with settings
	 * 
	 * @param f File
	 */
	@Override
	public void save() {
		// if the bit file doesnt exist, dont save gui file
		if (!outputFile.exists())
			return;
		// if this plot is empty dont save
		if (yAxis.size() == 0)
			return;
		// content for gui file
		StringBuffer sb = new StringBuffer("#autoGenerated by BIGGUI\n");
		sb.append("title=" + getTitle() + "\n");
		sb.append("xaxistext=" + xData.Text + "\n");
		sb.append("xoutmin=" + xData.getMinAbs() + "\n");
		sb.append("xoutmax=" + xData.getMaxAbs() + "\n");
		sb.append("xaxislogbase=" + xData.Log + "\n");
		sb.append("xaxisticks=" + xData.Ticks + "\n");
		sb.append("xaxispre=" + xData.getPre() + "\n");
		sb.append("xaxisnumberformat=" + xData.NumberFormat + "\n");
		sb.append("xaxisscaletext=" + xData.scaleAxisText + "\n");
		// now the yaxis/funtion data
		for (int k = 0; k < order.length; k++) {
			// which dataset
			int i = order[k] / 100;
			// which function
			int whichFunction = order[k] % 100;
			int j = whichFunction;
			int howManyBefore = 0;
			for (int l = 0; l < k; l++) {
				if (order[l] / 100 != i) {
					howManyBefore++;
				}
			}
			j = j - howManyBefore - 1;

			YAxisData curAxis = yAxis.get(i);
			BIGDataSet data = curAxis.getData();

			sb.append("y" + whichFunction + "axistext=" + curAxis.Text + "\n");
			sb.append("y" + whichFunction + "outmin=" + curAxis.getMinAbs() + "\n");
			sb.append("y" + whichFunction + "outmax=" + curAxis.getMaxAbs() + "\n");
			sb.append("y" + whichFunction + "axislogbase=" + curAxis.Log + "\n");
			sb.append("y" + whichFunction + "axisnumberformat=" + curAxis.NumberFormat + "\n");
			sb.append("y" + whichFunction + "axisticks=" + curAxis.Ticks + "\n");
			sb.append("tlegendfunction" + whichFunction + "=" + data.getSeriesName(j) + "\n");
			sb.append("y" + whichFunction + "color=" + curAxis.Colors.get(j).getRGB() + "\n");
			// -------------------------------------------------------------------------------
			int tempIndex = -1;
			for (int index = 0; index < BIGPlotRenderer.BIG_SHAPES.length; index++) {
				if (curAxis.Shapes.get(j) == BIGPlotRenderer.BIG_SHAPES[index]) {
					tempIndex = index;
				}
			}
			sb.append("y" + whichFunction + "shape=" + tempIndex + "\n");
			sb.append("y" + whichFunction + "axispre=" + curAxis.getPre() + "\n");
			sb.append("y" + whichFunction + "scaletext=" + curAxis.scaleAxisText + "\n");
			sb.append("y" + whichFunction + "scalelegends=" + data.getPreNamesSelected() + "\n");
		}
		sb.append("numfunctions=" + order.length + "\n");
		super.appendFontInformation(sb);
		sb.append("\n");
		// fonts and stuff

		sb.append("drawAntiAliased=" + drawAntiAliased + "\n");
		BIGFileHelper.saveToFile(sb.toString(), getSaveFile());
	}

	private boolean loadFile(File guiFile, Integer numberOfFunctions) throws Exception {
		if (!guiFile.exists())
			return false;
		BIGOutputParser parser;
		try {
			parser = new BIGOutputParser(guiFile.getAbsolutePath());
		} catch (BIGParserException ex2) {
			System.err.println("Error while parsing " + guiFile.getName() + ":" + ex2.getMessage());
			return false;
		}

		String xPreString = parser.getValue("xaxispre");
		try {
			xData.setPre(Integer.parseInt(xPreString));
		} catch (NumberFormatException ignored) {}
		xData.Text = parser.getValue("xaxistext");
		xData.TextDefault = xData.Text;
		xData.NumberFormat = parser.getValue("xaxisnumberformat");
		xData.setMinAbs(parser.getDoubleValue("xoutmin"));
		xData.setMaxAbs(parser.getDoubleValue("xoutmax"));
		xData.Log = parser.getIntValue("xaxislogbase");
		xData.Ticks = parser.getIntValue("xaxisticks");
		xData.scaleAxisText = Boolean.valueOf(parser.getValue("xaxisscaletext"));
		setTitle(parser.getValue("title"));
		String value = parser.getValue("drawAntiAliased");
		drawAntiAliased = (value != null) && (!value.isEmpty()) && !value.equals("0")
				&& !value.equalsIgnoreCase("false");
		@SuppressWarnings("unchecked") Vector<YAxisData> oldyAxis = (Vector<YAxisData>) yAxis.clone();
		yAxis.clear();
		for (int i = 1; i <= numberOfFunctions; i++) {
			boolean found = false;
			YAxisData curAxis;
			for (int j = 0; j < yAxis.size(); j++) {
				curAxis = yAxis.get(j);
				if (curAxis.Text.equals(parser.getValue("y" + i + "axistext"))) {
					curAxis.Colors.add(new Color(parser.getIntValue("y" + i + "color")));

					int shape;
					try {
						shape = parser.getIntValue("y" + i + "shape");
						curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
					} catch (Exception ex5) {
						shape = new Integer((i - 1) % BIGPlotRenderer.BIG_SHAPES.length);
						curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
					}
					if (curAxis.getMinAbs() > parser.getDoubleValue("y" + i + "outmin"))
						curAxis.setMinAbs(parser.getDoubleValue("y" + i + "outmin"));
					if (curAxis.getMaxAbs() < parser.getDoubleValue("y" + i + "outmax"))
						curAxis.setMaxAbs(parser.getDoubleValue("y" + i + "outmax"));
					if (curAxis.Log > parser.getIntValue("y" + i + "axislogbase"))
						curAxis.Log = parser.getIntValue("y" + i + "axislogbase");
					if (curAxis.Ticks < parser.getIntValue("y" + i + "axisticks"))
						curAxis.Ticks = parser.getIntValue("y" + i + "axisticks");
					curAxis.scaleAxisText = curAxis.scaleAxisText
							&& Boolean.valueOf(parser.getValue("y" + i + "scaletext"));
					curAxis.setScaleLegends(curAxis.getScaleLegends()
							&& Boolean.valueOf(parser.getValue("y" + i + "scalelegends")));
					found = true;
				}
			}
			if (!found) {
				if (parser.getValue("y" + i + "axistext") == null) {
					saveInit(parser);
					throw new Exception();
				}
				curAxis = new YAxisData(parser.getValue("y" + i + "axistext"), null);
				yAxis.add(curAxis);

				curAxis.setPre(parser.getIntValue("y" + i + "axispre"));
				curAxis.Colors.add(new Color(parser.getIntValue("y" + i + "color")));
				int shape;
				try {
					shape = parser.getIntValue("y" + i + "shape");
					curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
				} catch (Exception ex5) {
					shape = new Integer((i - 1) % BIGPlotRenderer.BIG_SHAPES.length);
					curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
				}
				curAxis.setMinAbs(parser.getDoubleValue("y" + i + "outmin"));
				curAxis.setMaxAbs(parser.getDoubleValue("y" + i + "outmax"));
				curAxis.Log = parser.getIntValue("y" + i + "axislogbase");
				curAxis.Ticks = parser.getIntValue("y" + i + "axisticks");
				curAxis.NumberFormat = new String(parser.getValue("y" + i + "axisnumberformat"));
				curAxis.scaleAxisText = Boolean.valueOf(parser.getValue("y" + i + "scaletext"));
				curAxis.setScaleLegends(Boolean.valueOf(parser.getValue("y" + i + "scalelegends")));
			}
		}
		if (yAxis.size() == oldyAxis.size()) {
			for (int i = 0; i < yAxis.size(); i++) {
				YAxisData curAxis = yAxis.get(i);
				YAxisData curAxisOld = oldyAxis.get(i);
				curAxis.TextDefault = curAxisOld.Text;
				curAxis.MinDefault = curAxisOld.getMinAbs();
				curAxis.MaxDefault = curAxisOld.getMaxAbs();
				curAxis.LogDefault = curAxisOld.Log;
				curAxis.TicksDefault = curAxisOld.Ticks;
			}
		} else {
			for (YAxisData curAxis : yAxis) {
				curAxis.TextDefault = curAxis.Text;
				curAxis.MinDefault = curAxis.getMinAbs();
				curAxis.MaxDefault = curAxis.getMaxAbs();
				curAxis.LogDefault = curAxis.Log;
				curAxis.TicksDefault = curAxis.Ticks;
			}
		}
		return true;
	}
	/**
	 * reads the .bit File, set standard settings and init default settings
	 * 
	 * @param f File the .bit File
	 */
	@Override
	public boolean init() {
		if (loaded)
			return true;
		if (!outputFile.exists()) {
			console.postErrorMessage("File " + outputFile.getName() + " does not exist!");
			progressLabel.setText("Error");
			progressBar.setValue(0);
			return false;
		}
		// the settings for the file
		progressBar.setMinimum(0);
		progressBar.setMaximum(11);
		progressLabel.setText("Init: Setting defaults");
		progressBar.setValue(0);

		progressLabel.setText("Init: Parsing outputfile");
		progressBar.setValue(1);
		/*
		 * First: parsing the output File
		 */
		// open the parser
		parser = null;
		try {
			parser = new BIGOutputParser(outputFile.getAbsolutePath());
		} catch (BIGParserException ex) {
			System.err.println("Error while parsing " + outputFile.getName() + ":" + ex.getMessage());
			progressLabel.setText("Error");
			progressBar.setValue(0);
			return false;
		}
		// init a lot of fonts and stuff
		super.initFonts(getSaveFile());

		setStandardTitle();
		if (getTitle() == null || getTitle().isEmpty())
			setTitle(outputFile.getName());
		titleDefault = getTitle();

		// setting xValues (thank god, they are just once)
		xData = new XAxisData(parser.getValue("xaxistext"), this);
		try {
			xData.setMinAbs(parser.getDoubleValue("xoutmin"));
		} catch (NumberFormatException ex7) {}
		xData.MinDefault = xData.getMinAbs();
		try {
			xData.setMaxAbs(parser.getDoubleValue("xoutmax"));
		} catch (NumberFormatException ex6) {}
		xData.MaxDefault = xData.getMaxAbs();
		xData.Log = parser.getIntValue("xaxislogbase");
		xData.LogDefault = xData.Log;
		xData.Ticks = parser.getIntValue("xaxisticks");
		xData.Ticks = xData.Ticks;

		// now the tough stuff the yaxis
		int numberOfFunctions = -1;
		// maybe the file is filled out correctly :)
		try {
			numberOfFunctions = parser.getIntValue("numfunctions");
		} catch (Exception ex1) {}
		// if not check for the data
		if (numberOfFunctions == -1) {
			saveInit(parser);
			return true;
		}

		yAxis.clear();

		Vector<Vector<Integer>> whichFunctionsToWhichAxis = new Vector<Vector<Integer>>();
		order = new int[numberOfFunctions];

		for (int i = 1; i <= numberOfFunctions; i++) {
			boolean found = false;
			YAxisData curAxis;
			String text = parser.getValue("y" + i + "axistext");
			if (text == null) {
				saveInit(parser);
				return true;
			}
			for (int j = 0; j < yAxis.size(); j++) {
				curAxis = yAxis.get(j);
				if (curAxis.Text.equals(text)) {
					whichFunctionsToWhichAxis.get(j).add(i);
					if (curAxis.getMinAbs() > parser.getDoubleValue("y" + i + "outmin"))
						curAxis.setMinAbs(parser.getDoubleValue("y" + i + "outmin"));
					if (curAxis.getMaxAbs() < parser.getDoubleValue("y" + i + "outmax"))
						curAxis.setMaxAbs(parser.getDoubleValue("y" + i + "outmax"));
					if (curAxis.Log > parser.getIntValue("y" + i + "axislogbase", 0))
						curAxis.Log = parser.getIntValue("y" + i + "axislogbase", 0);
					if (curAxis.Ticks < parser.getIntValue("y" + i + "axisticks", -1))
						curAxis.Ticks = parser.getIntValue("y" + i + "axisticks", -1);
					curAxis.Colors.add(null);
					curAxis.Shapes.add(null);

					order[i - 1] = j * 100 + i;

					found = true;
				}
			}
			if (!found) {
				curAxis = new YAxisData(text, null);
				yAxis.add(curAxis);

				Vector<Integer> functionForThisAxis = new Vector<Integer>();
				functionForThisAxis.add(i);
				whichFunctionsToWhichAxis.add(functionForThisAxis);
				try {
					curAxis.setMin(parser.getDoubleValue("y" + i + "outmin"));
				} catch (NumberFormatException ex3) {}
				try {
					curAxis.setMax(parser.getDoubleValue("y" + i + "outmax"));
				} catch (NumberFormatException ex4) {}
				curAxis.Log = parser.getIntValue("y" + i + "axislogbase", 0);
				curAxis.Ticks = parser.getIntValue("y" + i + "axisticks", -1);
				curAxis.Colors.add(null);
				curAxis.Shapes.add(null);
				order[i - 1] = (whichFunctionsToWhichAxis.size() - 1) * 100 + i;
			}
		}
		for (YAxisData curAxis : yAxis) {
			curAxis.TextDefault = curAxis.Text;
			curAxis.MinDefault = curAxis.getMinAbs();
			curAxis.MaxDefault = curAxis.getMaxAbs();
			curAxis.LogDefault = curAxis.Log;
			curAxis.TicksDefault = curAxis.Ticks;
		}
		progressLabel.setText("Init: Parsing outputfiles data");
		progressBar.setValue(3);

		double[][] values = parser.getData();
		progressLabel.setText("Init: Open saved informations");
		progressBar.setValue(5);

		boolean settingsLoaded;
		try {
			settingsLoaded = loadFile(getSaveFile(), numberOfFunctions);
		} catch (Exception e) {
			return true;
		}
		/*
		 * Building the dataSets
		 */
		if (yAxis.size() > 1) {
			displayedDataSets = new int[2];
			displayedDataSets[0] = 0;
			displayedDataSets[1] = 1;
		} else {
			displayedDataSets = new int[1];
			displayedDataSets[0] = 0;
		}
		progressLabel.setText("Init: Initializing Data");
		progressBar.setValue(9);

		if (!settingsLoaded) {
			calculateScaleX(values, xData);
		}

		for (int i = 0; i < whichFunctionsToWhichAxis.size(); i++) {
			Vector<Integer> functionsForThisAxis = whichFunctionsToWhichAxis.get(i);
			double[][] thisAxissData = new double[functionsForThisAxis.size() + 1][values[0].length];
			// xData
			thisAxissData[0] = values[0];
			List<String> namesForThisAxis = new Vector<String>(functionsForThisAxis.size());
			// System.err.println(i+"/"+namesForThisAxis.length+"("+yNames.get(i)+")");
			for (int j = 0; j < functionsForThisAxis.size(); j++) {
				int whichFun = functionsForThisAxis.get(j).intValue();
				thisAxissData[j + 1] = values[whichFun];
				namesForThisAxis.add(parser.getValue("tlegendfunction" + whichFun));
			}
			YAxisData curAxis = yAxis.get(i);
			BIGDataSet data = new BIGDataSet(thisAxissData, namesForThisAxis);
			data.setDefaultMinsAndMaxs(xData.MinDefault, xData.MaxDefault, curAxis.MinDefault,
					curAxis.MaxDefault);
			curAxis.setXPre(xData.getPre());
			data.setMinimumDomainValue(xData.getMin());
			data.setMaximumDomainValue(xData.getMax());
			// System.err.println("DataSet"+i+" hat "+namesForThisAxis.length+" LEGENDEN");
			curAxis.setData(data);
			if (!settingsLoaded)
				calculateScaleY(curAxis);
		}
		progressLabel.setText("Loading raw data");
		progressBar.setValue(10);
		if (rawFile != null && !rawFile.init())
			console.postMessage("Error reading rawFile '" + rawFile.getFile().getName() + "'",
					BIGConsole.ERROR);
		progressLabel.setText("done");
		progressBar.setValue(0);
		loaded = true;
		return true;
	}
	/**
	 * init which only needs beginOfData and endOfData and the data between
	 */
	public void saveInit(BIGOutputParser parser) {
		double[][] data = parser.getData();
		yAxis.clear();
		List<String> names = new Vector<String>(data.length - 1);
		for (int i = 1; i < data.length; i++) {
			names.add("" + i);
		}
		yAxis.add(new YAxisData("y", new BIGDataSet(data, names)));
		displayedDataSets = new int[1];
		displayedDataSets[0] = 0;
		progressLabel.setText("done");
		progressBar.setValue(0);
	}
	/**
	 * does nothing (cant remove from a file)
	 * 
	 * @param s String dont care
	 */
	@Override
	public void removeFunction(String s) {
		return;
	}

	/**
	 * returns false (cant add new to a file)
	 */
	@Override
	public boolean addFunction(Graph graph) {
		return false;
	}

	/**
	 * returns the file, which is used to save information
	 */
	@Override
	public File getSaveFile() {
		return new File(getFile().getAbsolutePath() + ".gui");
	}

	public DataFlavor[] getTransferDataFlavors() {
		DataFlavor[] df = new DataFlavor[1];
		df[0] = getDataFlavorObject();
		return df;
	}

	public boolean isDataFlavorSupported(DataFlavor flavor) {
		if (flavor.equals(getDataFlavorObject()))
			return true;
		return false;
	}

	public Object getTransferData(DataFlavor flavor) throws UnsupportedFlavorException, IOException {
		if (!flavor.equals(getDataFlavorObject()))
			return null;
		return this;
	}

	public DataFlavor getDataFlavorObject() {
		return new DataFlavor(BIGOutputFile.class, "A BenchIT-resultfile " + fileNameWithoutExtension);
	}

	// -----
	@Override
	public void changeName(int whichDataSet, int index, String newName) {
		yAxis.get(displayedDataSets[whichDataSet]).getData().getNames().set(index, newName);
	}
	// -----

	@Override
	public boolean IsInit() {
		return loaded;
	}
}
