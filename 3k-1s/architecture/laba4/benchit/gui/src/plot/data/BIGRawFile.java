package plot.data;

import java.io.File;
import java.util.*;

import system.BIGParserException;
import conn.Graph;

public class BIGRawFile {
	private final File file;
	private File OriginFile;
	private String KernelName;
	private String XAxisText;
	private double[] xValues;
	private BIGRawFunction[] Functions;

	public class BIGRawFunction {
		public final String Text, Legend;
		/**
		 * Values[0]: first measurements (e.g. x=1); Values[1]: second measurements (e.g. x=2)...
		 */
		public final double[][] Values;

		public BIGRawFunction(String text, String legend, double[][] values) {
			Text = text;
			Legend = legend;
			Values = values;
		}
	}

	public BIGRawFile(File file) {
		this.file = file;
	}

	public boolean init() {
		// open the parser
		BIGOutputParser parser;
		try {
			parser = new BIGOutputParser(file.getAbsolutePath());
		} catch (BIGParserException e) {
			return false;
		}
		String originFile = parser.getValue("bitfile");
		OriginFile = new File(file.getParent() + File.separator + originFile);
		KernelName = parser.getValue("kernelname");
		XAxisText = parser.getValue("xaxistext");

		int numFunctions = new Integer(parser.getValue("numfunctions"));
		int repeatCt = new Integer(parser.getValue("repeatct"));
		Functions = new BIGRawFunction[numFunctions];
		double[][] values = parser.getData();
		xValues = values[0];
		for (int j = 0; j < numFunctions; j++) {
			double[][] yValues = new double[xValues.length][repeatCt];
			for (int k = 0; k < repeatCt; k++)
				for (int i = 0; i < xValues.length; i++)
					yValues[i][k] = values[j * repeatCt + k + 1][i];
			BIGRawFunction func = new BIGRawFunction(parser.getValue("y" + (j + 1) + "axistext"),
					parser.getValue("tlegendfunction" + (j + 1)), yValues);
			Functions[j] = func;
		}
		return true;
	}

	public File getFile() {
		return file;
	}

	public String getKernelName() {
		return KernelName;
	}

	public String getXAxisText() {
		return XAxisText;
	}

	public int getFunctionIndex(String name) {
		for (int i = 0; i < Functions.length; i++)
			if (Functions[i].Legend.equals(name))
				return i;
		return -1;
	}

	public BIGRawFunction getRawFunction(int funcIndex) {
		if (funcIndex < 0 || funcIndex >= Functions.length)
			return null;
		return Functions[funcIndex];
	}

	private Graph getGraph(BIGRawFunction func, String addText) {
		String name = func.Legend + addText;
		String[] nameParts = KernelName.split("\\.");
		if (nameParts.length == 6) {
			// Valid name structure
			name = nameParts[1] + ": " + name;
		}
		Graph g = new Graph(name, OriginFile.getAbsolutePath());
		g.setXAxisText(XAxisText);
		g.setYAxisText(func.Text);
		g.setXValues(xValues);
		return g;
	}

	private interface ResultSelector {
		/**
		 * Called to select the choosen value
		 * 
		 * @param oldVal Current choosen value
		 * @param newVal Current/new value
		 * @param validCt Number of valid values including current
		 * @return Choosen value
		 */
		public double select(double oldVal, double newVal, int validCt);
		/**
		 * Called before a new value is going to be selected; E.g. new series/new x
		 */
		public void init();
	}

	private Graph getFunctionGraph(int funcIndex, String addText, ResultSelector selector) {
		BIGRawFunction func = getRawFunction(funcIndex);
		if (func == null)
			return null;
		Graph g = getGraph(func, addText);
		double[] yValues = new double[func.Values.length];
		double errorDou = 1.0 * system.BIGInterface.getInstance().getBIGConfigFileParser()
				.intCheckOut("errorInt", -1);
		for (int i = 0; i < yValues.length; i++) {
			int validCt = 0;
			double val = errorDou;
			selector.init();
			for (int j = 0; j < func.Values[0].length; j++) {
				double curY = func.Values[i][j];
				if (curY == errorDou)
					continue;
				validCt++;
				val = selector.select(val, curY, validCt);
			}
			yValues[i] = val;
		}
		g.setYValues(yValues);
		return g;
	}

	public Graph getMin(int funcIndex) {
		return getFunctionGraph(funcIndex, "-min", new ResultSelector() {
			public double select(double oldVal, double newVal, int validCt) {
				if (validCt == 1)
					return newVal;
				return Math.min(newVal, oldVal);
			}

			public void init() {}
		});
	}

	public Graph getMax(int funcIndex) {
		return getFunctionGraph(funcIndex, "-max", new ResultSelector() {
			public double select(double oldVal, double newVal, int validCt) {
				if (validCt == 1)
					return newVal;
				return Math.max(newVal, oldVal);
			}

			public void init() {}
		});
	}

	public Graph getMean(int funcIndex) {
		return getFunctionGraph(funcIndex, "-mean", new ResultSelector() {
			public double select(double oldVal, double newVal, int validCt) {
				if (validCt == 1)
					return newVal;
				return (oldVal * (validCt - 1) + newVal) / validCt;
			}

			public void init() {}
		});
	}

	private double getMedian(List<Double> values, int start, int ct) {
		if ((ct & 1) == 1) {
			// Truncating already gives us element at the middle as list starts with 0
			return values.get(start + ct / 2);
		}
		return (values.get(start + ct / 2 - 1) + values.get(start + ct / 2)) / 2;
	}

	public Graph getMedian(int funcIndex) {
		return getFunctionGraph(funcIndex, "-med", new ResultSelector() {
			private final List<Double> sortedValues = new ArrayList<Double>();
			public double select(double oldVal, double newVal, int validCt) {
				sortedValues.add(newVal);
				Collections.sort(sortedValues);
				return getMedian(sortedValues, 0, validCt);
			}
			public void init() {
				sortedValues.clear();
			}
		});
	}

	public Graph getFirstQuartile(int funcIndex) {
		return getFunctionGraph(funcIndex, "-Q1", new ResultSelector() {
			private final List<Double> sortedValues = new ArrayList<Double>();
			public double select(double oldVal, double newVal, int validCt) {
				sortedValues.add(newVal);
				Collections.sort(sortedValues);
				// Cannot calculate quartiles for less than 4 elements
				// We can give an "approximate", but not for one element (index out of bounds and pointless)
				if (validCt == 1)
					return newVal;
				// Number of elements in quartile. For uneven count it is truncated which is what we want
				// 0,1,2,3 --> 2
				// 0,1,2,3,4 --> 2
				int quartCount = validCt / 2;
				return getMedian(sortedValues, 0, quartCount);
			}
			public void init() {
				sortedValues.clear();
			}
		});
	}

	public Graph getThirdQuartile(int funcIndex) {
		return getFunctionGraph(funcIndex, "-Q3", new ResultSelector() {
			private final List<Double> sortedValues = new ArrayList<Double>();
			public double select(double oldVal, double newVal, int validCt) {
				sortedValues.add(newVal);
				Collections.sort(sortedValues);
				// Cannot calculate quartiles for less than 4 elements
				// We can give an "approximate", but not for one element (index out of bounds and pointless)
				if (validCt == 1)
					return newVal;
				// Number of elements in quartile. For uneven count it is truncated which is what we want
				// 0,1,2,3 --> 2
				// 0,1,2,3,4 --> 2
				int quartCount = validCt / 2;
				// Set startindex to uprounded value
				// 0,1,2,3 --> 2
				// 0,1,2,3,4 --> 3
				int startIndex = (validCt + 1) / 2;
				return getMedian(sortedValues, startIndex, quartCount);
			}
			public void init() {
				sortedValues.clear();
			}
		});
	}

}
