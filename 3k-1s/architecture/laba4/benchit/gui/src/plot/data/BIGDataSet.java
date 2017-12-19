/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGDataSet.java Author: SWTP
 * Nagel 1 Last change by: $Author: tschuet $ $Revision: 1.13 $ $Date: 2007/02/27 12:39:03 $
 ******************************************************************************/
package plot.data;

import java.util.List;

import org.jfree.data.*;

import system.BIGInterface;
import conn.Graph;

/**
 * an org.jfree.data.dataset implementation made for the BenchITGUI http://www.jfree.org/jfreechart/index.html
 * 
 * @author <a href="mailto:pisi@pisi.de">Christoph Mueller</a>
 */
public class BIGDataSet extends AbstractSeriesDataset implements XYDataset, DomainInfo, RangeInfo {
	private static final long serialVersionUID = 1L;
	// 3 means no modifier. modifiers are set in sets and modifiers
	private int pre = 3;
	// 3 means no modifier. modifiers are set in sets and modifiers
	private int xpre = 3;
	/**
	 * debugmessages en- or disabled (may be overwritten in the constructor)
	 */
	private boolean debug = false;
	// display the pre also on legends?
	private boolean preSelected = false;
	/**
	 * values [column][row] first column ([0]) are the x values all series have the same x values for each series there is
	 * a column for the y values so we have (series.count + 1) columns
	 */
	private double[][] values;

	/**
	 * the names of the series names[i] belongs to values[i+1][x]
	 */
	private List<String> names;

	/**
	 * min and max values for the achses these valuess will be calculated in the constructor the values ending with an a
	 * are the ones that should not be modified the values ending with a b are for custom viewAreas
	 */
	private double maxXDefault;
	private double minXDefault;
	private double minYDefault;
	private double maxYDefault;
	private double maxX;
	private double minX;
	private double minY;
	private double maxY;
	/**
	 * Strings, how the single y-functions can be modified
	 */
	public static String[] sets = {"n (nano)", "µ (micro)", "m (milli)", " (normal) ", "k (kilo)",
			"M (Mega)", "G (Giga)", "T (Tera)"};
	/**
	 * the corresponing modiers to turn 5,000,000 FLOPS to 5 MFLOPS look in sets where M is (=6) and multiply the values
	 * by setModifiers[6]
	 */
	public static double[] setModifiers = {1.0E+9, 1.0E+6, 1.0E+3, 1.0E+0, 1.0E-3, 1.0E-6, 1.0E-9,
			1.0E-12};

	/**
	 * we get ugly errors when dividing a value with its negative modifier e.g. 4.0 / 1.0E-9 results in 3.9999999996E9
	 * instead of 4.0E9 so we decided to change to multiplication, but to do this we need appropriate modifiers
	 */
	public static double[] setInverseModifiers = {1.0E-9, 1.0E-6, 1.0E-3, 1.0E+0, 1.0E+3, 1.0E+6,
			1.0E+9, 1.0E+12};

	/**
	 * construct a dataset the size of the second dimension of double[] is always the same the size of String[] should be
	 * the size of the first dimension of double[] plus 1
	 * 
	 * @param values a double[][] containing the values (first column ([0]) are the x values - all series have the same x
	 *          values - for each series there is a column for the y values)
	 * @param names a String[] containing the names of the series
	 */
	public BIGDataSet(double[][] values, List<String> names) {
		init(values, names);
	}

	public static String getPrefix(int pre) {
		return sets[pre].substring(0, 1).trim();
	}

	/**
	 * the same as the constructor. re-initialize
	 * 
	 * @param values double[][] containing the values (first column ([0]) are the x values - all series have the same x
	 *          values - for each series there is a column for the y values)
	 * @param names String[] containing the names of the series
	 */
	public void init(double[][] values, List<String> names) {
		// print debug?
		if (BIGInterface.getInstance().getDebug("BIGDataSet") > 0) {
			debug = true;
		}
		if (debug) {
			System.out.println("BIGDataSet()");
		}
		this.names = names;
		this.values = values;
		MinMax mm = getMinMaxX();

		maxXDefault = mm.Max;
		maxX = maxXDefault;
		minXDefault = mm.Min;
		minX = minXDefault;

		mm = getMinMaxY();

		maxYDefault = mm.Max;
		maxY = maxYDefault;
		minYDefault = mm.Min;
		minY = minYDefault;
	}

	private class MinMax {
		public double Min, Max;

		public MinMax() {
			Min = Double.POSITIVE_INFINITY;
			Max = Double.NEGATIVE_INFINITY;
		}

		public MinMax(double min, double max) {
			Min = min;
			Max = max;
		}

		public void add(MinMax other) {
			if (other.Min < Min)
				Min = other.Min;
			if (other.Max > Max)
				Max = other.Max;
		}
	}

	private MinMax calcMinMax(double[] values) {
		// getting minima and maxima for x
		double max = Double.NEGATIVE_INFINITY;
		double min = Double.POSITIVE_INFINITY;
		double errorDou = 1.0 * system.BIGInterface.getInstance().getBIGConfigFileParser()
				.intCheckOut("errorInt", -1);

		// look for min and max in first col (x values)
		for (double val : values) {
			if (val == errorDou)
				continue;
			if (val > max) {
				max = val;
			}
			if (val < min) {
				min = val;
			}
		}

		// if we have only one value, min and max are equal and this will cause problems
		if (max == min) {
			min--;
			max++;
		}
		if (max < 0.0)
			max = 0.0;
		if (min < 0.0)
			min = 0.0;
		if (max <= min)
			max = min + 1.0;
		return new MinMax(min, max);
	}

	public MinMax getMinMaxX() {
		return calcMinMax(values[0]);
	}

	public MinMax getMinMaxY() {
		MinMax mm = new MinMax();

		// look for min and max in all other cols (y values)
		for (int i = 1; i < values.length; i++) {
			mm.add(calcMinMax(values[i]));
		}

		return mm;
	}

	/**
	 * get the number of series
	 * 
	 * @return number of series
	 */
	@Override
	public int getSeriesCount() {
		return values.length - 1;
	}

	/**
	 * get the name of a serie with index i
	 * 
	 * @param i index
	 * @return name of a serie
	 */
	@Override
	public String getSeriesName(int i) {
		String preString = "";
		if (preSelected) {
			preString = sets[pre].substring(0, 1).trim();
		}
		return preString + names.get(i);
	}

	public int getSeriesIndex(String name) {
		return names.indexOf(name);
	}

	/**
	 * sets a new name for a series
	 * 
	 * @param i int number of series
	 * @param s String new name
	 */
	public void setSeriesName(int i, String s) {
		names.set(i, s);
	}

	/**
	 * returns the number of results or items because all series have the same count of items we do not care for the
	 * argument
	 * 
	 * @param arg0 ignored
	 * @return number of items for one row
	 */
	public int getItemCount(int arg0) {
		return values[0].length;
	}

	/**
	 * get the x value of an item
	 * 
	 * @param series ignored
	 * @param item position of the x
	 * @return a Double with the value of the x at position item
	 */
	public Double getXValue(int series, int item) {
		return values[0][item] * BIGDataSet.setModifiers[xpre];
	}

	/**
	 * get the y value of an item
	 * 
	 * @param series serie number
	 * @param item position of the y
	 * @return a Double with the value of the y from serie series at position item
	 */
	public Double getYValue(int series, int item) {
		// we have to add a 1 to series because series 0
		// is our x value and not a series
		if (values[series + 1][item] < 0)
			return null;
		return values[series + 1][item] * BIGDataSet.setModifiers[pre];
	}

	/**
	 * get the minimum x value
	 * 
	 * @return the smallest value in the x series
	 */
	public Double getMinimumDomainValue() {
		return getMinimumDomainValueAbs() * BIGDataSet.setModifiers[xpre];
	}

	public Double getMinimumDomainValueAbs() {
		return minX;
	}

	/**
	 * get the maximum x value
	 * 
	 * @return the highest value in the x series
	 */
	public Double getMaximumDomainValue() {
		return getMaximumDomainValueAbs() * BIGDataSet.setModifiers[xpre];
	}

	public Double getMaximumDomainValueAbs() {
		return maxX;
	}
	/**
	 * get x range as org.jfree.data.Range
	 * 
	 * @return the Range (from this.getMaximumDomainValue() to this.getMinimumDomainValue())
	 */
	public Range getDomainRange() {
		return new Range(minX * BIGDataSet.setModifiers[xpre], maxX * BIGDataSet.setModifiers[xpre]);
	}

	/**
	 * get minimum Y value
	 * 
	 * @return the smallest y value from all series
	 */
	public Double getMinimumRangeValue() {
		return new Double(minY * BIGDataSet.setModifiers[pre]);
	}

	/**
	 * get maximum y value
	 * 
	 * @return the highest y value from all series
	 */
	public Double getMaximumRangeValue() {
		return new Double(maxY * BIGDataSet.setModifiers[pre]);
	}

	/**
	 * get y range as org.jfree.data.Range
	 * 
	 * @return the Range (from this.getMaximumRangeValue() to this.getMinimumRangeValue())
	 */
	public Range getValueRange() {
		return new Range(getMinimumRangeValue(), getMaximumRangeValue());
	}

	/**
	 * set the min value for the Range (vertical achsis)
	 * 
	 * @param d the new minimum for range
	 */
	public void setMinimumRangeValue(double d) {
		setMinimumRangeValue(d * BIGDataSet.setInverseModifiers[xpre], true);
	}

	public void setMinimumRangeValue(double d, boolean absolute) {
		if (d < 0.0)
			return;
		if (absolute) {
			minY = d;
			if (debug) {
				System.out.println("BIGDataSet: min y set to :" + d);
			}
		} else
			setMinimumRangeValue(d);
	}

	/**
	 * set the max value for the Range (vertical achsis)
	 * 
	 * @param d the new maximum for range
	 */
	public void setMaximumRangeValue(double d) {
		setMaximumRangeValue(d * BIGDataSet.setInverseModifiers[xpre], true);
	}

	public void setMaximumRangeValue(double d, boolean absolute) {
		if (d < 0.0)
			return;
		if (absolute) {
			maxY = d;
			if (debug) {
				System.out.println("BIGDataSet: max y set to :" + d);
			}
		} else
			setMaximumRangeValue(d);
	}

	/**
	 * set the min value for the Domain (horizontal achsis)
	 * 
	 * @param d the new minimum for domain
	 */
	public void setMinimumDomainValue(double d) {
		setMinimumDomainValue(d * BIGDataSet.setInverseModifiers[xpre], true);
	}

	public void setMinimumDomainValue(double d, boolean absolute) {
		if (d < 0.0)
			return;
		if (absolute) {
			minX = d;
			if (debug) {
				System.out.println("BIGDataSet: min x set to :" + d);
			}
		} else
			setMinimumDomainValue(d);
	}

	/**
	 * set the max value for the Domain (horizontal achsis)
	 * 
	 * @param d the new maximum for domain
	 */
	public void setMaximumDomainValue(double d) {
		setMaximumDomainValue(d * BIGDataSet.setInverseModifiers[xpre], true);
	}

	public void setMaximumDomainValue(double d, boolean absolute) {
		if (d < 0.0)
			return;
		if (absolute) {
			maxX = d;
			if (debug) {
				System.out.println("BIGDataSet: max x set to :" + d);
			}
		} else
			setMaximumDomainValue(d);
	}
	/**
	 * reset the view range to default (set when parsing the data when creating this object)
	 */
	public void resetAchsisValues() {
		minX = minXDefault;
		maxX = maxXDefault;
		minY = minYDefault;
		maxY = maxYDefault;
		if (debug) {
			System.out.println("BIGDataSet: AchsisValues resetted to:\"" + minXDefault + "\",\""
					+ maxXDefault + "\",\"" + minYDefault + "\",\"" + maxYDefault + "\"");
		}
	}

	/**
	 * sets the defaults min and maxs, when reset is used. the mins and maxs will be set to these values
	 * 
	 * @param minX double minimum for x
	 * @param maxX double maximum for x
	 * @param minY double minimum for y
	 * @param maxY double maximum for y
	 */
	public void setDefaultMinsAndMaxs(double minX, double maxX, double minY, double maxY) {
		if (minXDefault >= 0.0)
			minXDefault = minX;
		if (maxX >= 0.0)
			maxXDefault = maxX;
		if (minY >= 0.0)
			minYDefault = minY;
		if (maxYDefault >= 0.0)
			maxYDefault = maxY;

		resetAchsisValues();
	}

	/**
	 * get all values from this dataset
	 * 
	 * @return double[][] the values, including x
	 */
	public double[][] getValues() {
		return values;
	}

	/**
	 * get the legendnames
	 * 
	 * @return String[] the legendnames
	 */
	public List<String> getNames() {
		return names;
	}

	/**
	 * sets the modifier to change display Do not call from outside AxisData!!!
	 * 
	 * @param i int see this.sets
	 */
	public void setPre(int i) {
		// 0=nano
		// ...
		// 3=normal
		// ...
		// 6= GIGA
		// 7 = Peta
		pre = i;
	}

	public int getPre() {
		return pre;
	}

	/**
	 * sets the modifier to change display Do not call from outside AxisData!!!
	 * 
	 * @param i int see this.sets
	 */
	public void setXPre(int i) {
		// 0=nano
		// ...
		// 3=normal
		// ...
		// 6= GIGA
		// 7 = Peta
		xpre = i;
	}

	/**
	 * turns the containing x and y-values to sth. plotable (Benchit-compatible): xValue1 \t yValue1 \t yValue2 ... \n
	 * xValue2 ...
	 * 
	 * @return String all values as string
	 */
	public String getPlotableFileContent() {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < values[0].length; i++) {
			for (int j = 0; j < values.length; j++) {
				sb.append(values[j][i]);
				sb.append('\t');
			}
			sb.append('\n');
		}
		return sb.toString();
	}

	// -----------------
	/**
	 * checks, if there is already a function name in names[] which equals to newName
	 * 
	 * @param names String[] array of function names
	 * @param newName String function name which should be unique in names[]
	 * @return boolean returns true if and only if newName doesn't already exists in names[]
	 */
	@SuppressWarnings("unused")
	private boolean checkForExistence(String[] names, String newName) {
		int index;
		boolean alreadyExists = false;
		for (index = 0; index < names.length; index++) {
			alreadyExists = newName.equals(names[index]);
			if (alreadyExists)
				return alreadyExists;
		}
		return alreadyExists;
	}

	/**
	 * add a new function
	 * 
	 * @param Graph the graph
	 * @return boolean could be added?
	 */
	public boolean addFromGraph(Graph graph) {
		String newName = add(graph.getValues(), graph.getGraphName());
		graph.setGraphName(newName);
		return true;
	}

	private double[][] mergeXValues(double[] gXValues, int functionCount) {
		double errorDou = 1.0 * system.BIGInterface.getInstance().getBIGConfigFileParser()
				.intCheckOut("errorInt", -1);
		// set to data, that doesnt exist or is wrong
		// the new x-axis. will be shorter later, but now the maximum
		double[] nextXValues = new double[values[0].length + gXValues.length];
		// which element is the one to write
		int whichWriteNext = 0;
		// setting all to errorDou
		for (int i = 0; i < nextXValues.length; i++) {
			nextXValues[i] = errorDou;
		}
		// we get all possible Values
		int oldSize = values[0].length;
		int gSize = gXValues.length;
		int oldIndex = 0, gIndex = 0;
		double[] oldXValues = values[0];
		// go through all values in oldXValues and gXValues
		while ((oldIndex < oldSize) || (gIndex < gSize)) {
			// if we have all old values
			if (oldIndex == oldSize) {
				nextXValues[whichWriteNext] = gXValues[gIndex];
				gIndex++;
			} else {
				// if we have all newValues
				if (gIndex == gSize) {
					nextXValues[whichWriteNext] = oldXValues[oldIndex];
					oldIndex++;
				} else {
					// check which is the next x-Value
					if (oldXValues[oldIndex] < gXValues[gIndex]) {
						nextXValues[whichWriteNext] = oldXValues[oldIndex];
						oldIndex++;
					} else {
						if (oldXValues[oldIndex] > gXValues[gIndex]) {
							nextXValues[whichWriteNext] = gXValues[gIndex];
							gIndex++;
						}
						// are both x-values the same
						else {
							nextXValues[whichWriteNext] = gXValues[gIndex];
							gIndex++;
							oldIndex++;
						}
					}
				}
			}
			whichWriteNext++;
		}
		// first getLength. now the xValues will get shorter (in most cases)
		int newLength = whichWriteNext;
		// writing new data for old graphs
		double xValues[] = new double[newLength];
		for (int i = 0; i < newLength; i++) {
			xValues[i] = nextXValues[i];
		}
		double[][] newDataValues = new double[functionCount + 1][];
		newDataValues[0] = xValues;
		// writing new yValues (or errorDou)
		for (int i = 1; i < values.length; i++) {
			double yValues[] = new double[newLength];
			for (int j = 0; j < newLength; j++) {
				yValues[j] = BIGDataSet.getY(values, i, newDataValues[0][j]);
			}
			newDataValues[i] = yValues;
		}
		return newDataValues;
	}

	/**
	 * add a new function
	 * 
	 * @param newValues double[][] x and y values of new function. newValues.length==2 means add function with
	 *          xs=newValues[0], ys=newValues[1] else the xValues will be updated (only newValues[0] are used)
	 * @param newName String legend/name of function
	 * @return String new Name or null on error
	 */
	public String add(double[][] newValues, String newName) {
		MinMax mmX = new MinMax(minXDefault, maxXDefault);
		mmX.add(calcMinMax(newValues[0]));
		minXDefault = mmX.Min;
		minX = mmX.Min;
		maxXDefault = mmX.Max;
		maxX = mmX.Max;
		// if its a function
		if (newValues.length == 2) {
			// if a function with this name exists, rename it by adding a number
			String newFunctionRenamed = newName;
			// check for existing
			int ct = 0;
			while (getSeriesIndex(newFunctionRenamed) >= 0) {
				// add number
				ct++;
				newFunctionRenamed = newName + ct;
			}
			names.add(newFunctionRenamed);
			double[][] newDataValues = mergeXValues(newValues[0], values.length);
			int newLength = newDataValues[0].length;
			// writing new yValues (or errorDou)
			double yValues[] = new double[newLength];
			for (int j = 0; j < newLength; j++) {
				yValues[j] = BIGDataSet.getY(newValues, 1, newDataValues[0][j]);
			}
			newDataValues[newDataValues.length - 1] = yValues;
			values = newDataValues;
			MinMax mmY = new MinMax(minYDefault, maxYDefault);
			mmY.add(calcMinMax(newValues[1]));
			minYDefault = mmY.Min;
			minY = mmY.Min;
			maxYDefault = mmY.Max;
			maxY = mmY.Max;
			// finaly return
			return newName;
		}
		// else just update xValues
		else {
			values = mergeXValues(newValues[0], values.length - 1);
			// finaly return
			return newName;
		}
	}

	public boolean remove(String oldFunction) {
		// find function
		int index = getSeriesIndex(oldFunction);
		if (index == -1) {
			System.err.println("A function with this name does not exist here");
			return false;
		}
		names.remove(index);
		// ! x axis is one additional
		index = index + 1;
		// writing new values
		double[][] newValues = new double[values.length - 1][values[0].length];
		for (int i = 0; i < index; i++) {
			newValues[i] = values[i];
		}
		for (int i = index + 1; i < values.length; i++) {
			newValues[i - 1] = values[i];
		}
		values = newValues;

		return true;

	}

	/**
	 * gets the y-value to a corresponding x-Value
	 * 
	 * @param values double[][] double[n][] values to search in
	 * @param yPos int 1<=yPos<n position
	 * @param x double x-Value to the searched y-value
	 * @return double the y-value in values, column yPos, belonging to x
	 */
	private static double getY(double[][] values, int yPos, double x) {
		// try to find
		for (int i = 0; i < values[0].length; i++) {
			if (values[0][i] == x)
				return values[yPos][i];
		}
		// return invalid
		double errorDou = 1.0 * system.BIGInterface.getInstance().getBIGConfigFileParser()
				.intCheckOut("errorInt", -1);
		return errorDou;
	}

	/**
	 * @param preSel boolean set it=true, don't = false
	 */
	public void setPreNamesSelected(boolean preSel) {
		preSelected = preSel;
	}

	/**
	 * @param preSel boolean set it=true, don't = false
	 */
	public boolean getPreNamesSelected() {
		return preSelected;
	}
}
/*****************************************************************************
 * Log-History
 *****************************************************************************/
