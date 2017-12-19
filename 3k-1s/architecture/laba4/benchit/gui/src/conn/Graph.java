/**
 * Title:Graph Description: contains all informations for one Graph Copyright: Copyright (c) 2003
 * Company:ZHR (Center for High Performance Computing)
 * 
 * @author: Robert Schoene (rschoene@zhr.tu-dresden.de)
 * @version 1.0 *
 */
package conn;

import java.awt.datatransfer.*;
import java.util.*;

import plot.data.BIGDataSet;

public class Graph implements Transferable {
	/** id. should be unique ;) */
	int graphID;
	/** name (should also be unique) */
	String graphName = new String();
	/** name for horizontal axis */
	String xAxisName = new String();
	/** name for vertical axis */
	String yAxisName = new String();
	/** x-Values */
	double[] xWerte;
	/** y-Values */
	double[] yWerte;
	/** file-content in lines */
	String[] fileContent;
	// settings[0][x] are the identifiers , // settings[1][x] are the settings to the identifier
	/** settings read from the server */
	String[][] settings;

	// --------------
	/** file names of original bit files */
	List<String> originalBitFiles = null;

	// --------------

	/**
	 * Constructor
	 */
	public Graph(String name, String bitFile) {
		this(name);
		originalBitFiles = new ArrayList<String>();
		originalBitFiles.add(bitFile);
	}

	public Graph(String name, List<String> bitfiles) {
		this(name);
		originalBitFiles = bitfiles;
	}

	/**
	 * Constructor that doesn't handle links to original bitfiles
	 */
	public Graph(String name) {
		this.graphName = name;
	}

	/**
	 * Sets the points of the function
	 * 
	 * @param xs x-values
	 * @param ys y-values
	 */
	public void setPoints(double[] xs, double[] ys) {
		xWerte = xs;
		yWerte = ys;
	}

	/**
	 * sets the number of the identifiers of the function also removes all previously setted identifiers with there
	 * settings
	 * 
	 * @param i the new number of identifiers
	 */
	public void setNumberOfIdentifiers(int i) {
		settings = new String[2][i];
	}

	/**
	 * gets the identifier at position i
	 * 
	 * @param i the position
	 * @return the identifier at position
	 */
	public String getIdentifier(int i) {
		return settings[0][i];
	}

	/**
	 * gets the setting for an identifier
	 * 
	 * @param identifier the identifier
	 * @return the setting for the identifier
	 */
	public String getSetting(String identifier) {
		for (int i = 0; i < settings[0].length; i++) {
			if (settings[0][i].equals(identifier))
				return settings[1][i];
		}
		return "";
	}

	/**
	 * sets the setting for an identifier
	 * 
	 * @param identifier the identifier
	 * @param setting the setting for the identifier
	 */
	public void setIdentifierSettings(String identifier, String setting) {
		if (setting == null) {
			setting = "";
		}
		int free = -1;
		for (int i = 0; i < settings[0].length; i++) {
			if (settings[0][i] == null) {
				free = i;
				break;
			}
			if (settings[0][i].equals(identifier)) {
				settings[1][i] = setting;
				return;
			}
		}
		// if settings is full
		if (free == -1) {
			String[][] newSettings = new String[2][settings.length + 1];
			for (int i = 0; i < settings[0].length; i++) {
				newSettings[0][i] = settings[0][i];
				newSettings[1][i] = settings[1][i];
			}
			newSettings[0][settings.length] = identifier;
			newSettings[1][settings.length] = setting;
			settings = newSettings;
		} else // or not
		{
			settings[0][free] = identifier;
			settings[1][free] = setting;
		}

	}

	/**
	 * gets the identifiers
	 * 
	 * @return the identifiers
	 */
	public String[] getIdentifiers() {
		return settings[0];
	}

	/**
	 * gets the settings
	 * 
	 * @return the settings
	 */
	public String[] getSettings() {
		return settings[1];
	}

	/**
	 * sets the fileContent for this Graph
	 * 
	 * @param f the file (in lines)
	 */
	public void setFileContent(String[] f) {
		fileContent = f;
	}

	/**
	 * gets the file as lines
	 * 
	 * @return the file (separated in lines)
	 */
	public String[] getFileContent() {
		if (fileContent != null)
			return fileContent;
		return new String[0];
	}

	/**
	 * sets the identifiers and settings
	 * 
	 * @param iAnds iAnds[0]...identifiers,iAnds[1]...settings
	 */
	public void setIdentifiersAndSettings(String[][] iAnds) {
		settings = iAnds;
	}

	/**
	 * sets the points for the function
	 * 
	 * @param d d[0]..x-Values,d[1]...y-Values
	 */
	public void setPoints(double[][] d) {
		xWerte = d[0];
		yWerte = d[1];
	}

	/**
	 * sets the xaxis-text
	 * 
	 * @param s the new xaxis-text
	 */
	public void setXAxisText(String s) {
		xAxisName = s;
	}

	/**
	 * sets the yaxis-text
	 * 
	 * @param s the new yaxis-text
	 */
	public void setYAxisText(String s) {
		yAxisName = s;
	}

	/**
	 * @return the xaxis-text of this
	 */
	public String getXAxisText() {
		return xAxisName;
	}

	/**
	 * @return the yaxis-text of this
	 */
	public String getYAxisText() {
		return yAxisName;
	}

	/**
	 * Sets the name of this (is shown in the functionList)
	 * 
	 * @param name the new name of this
	 */
	public void setGraphName(String name) {
		graphName = name;
	}

	/**
	 * @return this graphs name
	 */
	public String getGraphName() {
		return graphName;
	}

	/**
	 * @return the number of points of this graph
	 */
	public int getNumberOfPoints() {
		return yWerte.length;
	}

	/**
	 * @return the x-Values of this
	 */
	public double[] getXValues() {
		return xWerte;
	}

	/**
	 * @return the y-Values of this
	 */
	public double[] getYValues() {
		return yWerte;
	}

	/**
	 * sets the x-Values
	 * 
	 * @param d the new x-Values
	 */
	public void setXValues(double[] d) {
		xWerte = d;
	}

	/**
	 * sets the y-Values
	 * 
	 * @param d the new y-Values
	 */
	public void setYValues(double[] d) {
		yWerte = d;
	}

	/**
	 * @param xValue the value to get the yValue from
	 * @return the yValue to xValue
	 */
	public double getYValue(double xValue) {
		for (int i = 0; i < xWerte.length; i++)
			if (xValue == xWerte[i])
				return yWerte[i];
		return -1E20;
	}

	/**
	 * Removes all values with an x value in the given range
	 * 
	 * @param min
	 * @param max
	 */
	public void keepRange(double min, double max) {
		double[] newX = new double[xWerte.length];
		double[] newY = new double[xWerte.length];
		int ct = 0;
		for (int i = 0; i < xWerte.length; i++) {
			if (xWerte[i] >= min && xWerte[i] <= max) {
				newX[ct] = xWerte[i];
				newY[ct] = yWerte[i];
				ct++;
			}
		}
		xWerte = new double[ct];
		yWerte = new double[ct];
		System.arraycopy(newX, 0, xWerte, 0, ct);
		System.arraycopy(newY, 0, yWerte, 0, ct);
	}

	// --------------
	public List<String> getOriginalBitFiles() {
		return originalBitFiles;
	}

	public String getOriginalBitFilesAsString() {
		String str = "";

		if (originalBitFiles == null)
			// if we have no references to bit-files
			return null;
		for (Iterator<String> iter = originalBitFiles.iterator(); iter.hasNext();) {
			str = str + iter.next() + ",";
		}
		// return the string without the last comma
		return str.substring(0, str.length() - 1);
	}

	// --------------

	@Override
	public String toString() {
		return graphName;
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

	public Object getTransferData(DataFlavor flavor) throws UnsupportedFlavorException,
			java.io.IOException {
		if (!flavor.equals(getDataFlavorObject()))
			return null;
		return this;
	}

	public DataFlavor getDataFlavorObject() {
		return new DataFlavor(Graph.class, "A single result line " + graphName);
	}

	// --------
	public Graph getCopyOfGraph() {
		Graph newGraph = new Graph(graphName, originalBitFiles);
		newGraph.setFileContent(fileContent);
		newGraph.setPoints(xWerte, yWerte);
		newGraph.setIdentifiersAndSettings(settings);
		newGraph.setXAxisText(xAxisName);
		newGraph.setYAxisText(yAxisName);
		newGraph.setPoints(getXValues(), getYValues());
		return newGraph;
	}
	// --------

	public double[][] getValues() {
		return new double[][]{getXValues(), getYValues()};
	}

	public BIGDataSet getDataSet() {
		List<String> names = new Vector<String>();
		names.add(getGraphName());
		return new BIGDataSet(getValues(), names);
	}

}
