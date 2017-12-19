package plot.data;

/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGResultMixer.java Author:
 * SWTP Nagel 1 Last change by: $Author: tschuet $ $Revision: 1.37 $ $Date: 2009/01/07 12:07:27 $
 */
import gui.*;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.tree.*;

import org.jfree.chart.*;
import org.jfree.chart.event.*;

import plot.gui.*;
import system.*;
import conn.Graph;

public class BIGResultMixer extends BIGPlotable {
	protected JPanel plotPanel = null;
	protected BIGResultTree resultTree = null;
	protected DefaultMutableTreeNode node = null;
	protected Dimension screenSize;
	protected String lastPath = null;
	protected Hashtable<String, Graph> namesAndGraphs = new Hashtable<String, Graph>();

	public class YAxisDataMixed extends YAxisData {
		public YAxisDataMixed(String text, BIGDataSet data) {
			super(text, data);
		}
		Vector<String> originalBitFiles = new Vector<String>();
	}

	/**
	 * constructor
	 * 
	 * @param plotPanel the panel, the graphical result will be shown on
	 * @param textPanel the panel, which will contain nothing, but remove existing information
	 * @param title the title of this new ResultMixer
	 */
	public BIGResultMixer(JPanel plotPanel, String title) {
		super();
		super.initFonts(null);
		screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		setTitle(title);
		titleDefault = title;
		node = new DefaultMutableTreeNode(this);
		this.plotPanel = plotPanel;
	}

	/**
	 * gets the x-Axis-Name from bit-File f
	 * 
	 * @param f File the bit-File
	 * @throws FileNotFoundException if File does not exist
	 * @throws IOException if sth. happened while reading the file
	 * @return String the xaxisname
	 * @see getXAxisName(BIGStrings bs)
	 */
	public String getXAxisName(File f) throws FileNotFoundException, IOException {
		BIGStrings bs = new BIGStrings();
		bs.readFromFile(f.getAbsolutePath());
		return this.getXAxisName(bs);
	}

	/**
	 * returns the x-Axis-Name from a File, stored in BIGStrings
	 * 
	 * @param bs BIGStrings the content of the bit-File
	 * @return String the x-Axis-Name
	 */
	public String getXAxisName(BIGStrings bs) {
		String line = bs.get(bs.find("xaxistext"));
		line = line.substring(line.indexOf("=") + 2, line.length() - 1);
		return line;
	}

	/**
	 * @param f File
	 * @throws FileNotFoundException if File does not exist
	 * @throws IOException if sth. happened while reading the file
	 * @return String[]
	 */
	public String[] getYAxisNames(File f) throws FileNotFoundException, IOException {
		BIGStrings bs = new BIGStrings();
		bs.readFromFile(f.getAbsolutePath());
		return this.getYAxisNames(bs);
	}

	public int getYAxisCount(File f) throws FileNotFoundException, IOException {
		BIGStrings bs = new BIGStrings();
		bs.readFromFile(f.getAbsolutePath());
		return this.getYAxisCount(bs);
	}

	public int getYAxisCount(BIGStrings bs) {
		int foundAt = 0;
		int findWhichNumber = 1;
		do {
			foundAt = bs.find("y" + findWhichNumber + "axistext");
			findWhichNumber++;
		} while (foundAt != -1);
		return findWhichNumber - 2;
	}

	public void cleanUp() {
		plot = null;
		plotPanel = new JPanel();
	}

	public String[] getYAxisNames(BIGStrings bs) {
		String[] textField = bs.toArray();
		String returnField[] = new String[getYAxisCount(bs)];
		int startYaxisTexts = bs.find("y1axistext");
		int endYaxisTexts = bs.find("numfunctions");
		int index = -1;
		for (int i = 0; i < returnField.length; i++) {
			for (int j = startYaxisTexts; j < endYaxisTexts; j++) {
				index = textField[j].indexOf("y" + (i + 1) + "axistext");
				if (index != -1) {
					returnField[i] = textField[j].substring(textField[j].indexOf("=") + 2,
							textField[j].length() - 1);
				}

			}
		}
		return returnField;
	}

	public int getNumberOfFunctions() {
		int size = 0;
		for (int i = 0; i < yAxis.size(); i++) {
			size = size + yAxis.get(i).getData().getSeriesCount();
		}
		return size;
	}

	public void clearData() {
		setTitle("Mixer title");
		node = new DefaultMutableTreeNode(this);
	}

	public void removeFunction(String fileName, String functionLegend) {
		removeSubNode(fileName + "-" + functionLegend);
	}

	public void addFunctionsFromFile(BIGOutputFile file, List<String> functions) {
		ArrayList<String> origin = new ArrayList<String>();
		origin.add(file.getFile().getAbsolutePath());

		for (String function : functions) {
			Graph g = new Graph(file.getName() + "-" + function, origin);
			g.setXAxisText(file.xData.Text);
			YAxisData yAxis = file.getYAxisByName(function);
			g.setYAxisText(yAxis.Text);
			int index = yAxis.getData().getSeriesIndex(function);
			g.setPoints(yAxis.getData().getValues()[0], yAxis.getData().getValues()[index + 1]);
			addFunction(g);
		}
	}

	private class ChoosenRawFuncs {
		public boolean Min, Max, Med, Mean;
		public ChoosenRawFuncs(boolean min, boolean max, boolean med, boolean mean) {
			Min = min;
			Max = max;
			Med = med;
			Mean = mean;
		}
	}

	private void addFunctionsFromRawFile(BIGRawFile file, List<String> functions,
			ChoosenRawFuncs choosen) {
		for (String function : functions) {
			int funcIndex = file.getFunctionIndex(function);
			if (funcIndex < 0) {
				System.err.println("Did not find " + function + " in " + file.getFile().getName());
				continue;
			}
			if (choosen.Min)
				addFunction(file.getMin(funcIndex));
			if (choosen.Max)
				addFunction(file.getMax(funcIndex));
			if (choosen.Med)
				addFunction(file.getMedian(funcIndex));
			if (choosen.Mean)
				addFunction(file.getMean(funcIndex));
		}
	}

	/**
	 * gets the parsed commet. if not avail, use stdComment
	 * 
	 * @return String
	 */
	@Override
	public String getExtComment() {
		if (pComment != null)
			return pComment;
		return "";
	}

	public void addWholeFile(File f) throws FileNotFoundException {
		BIGOutputFile file = new BIGOutputFile(f);
		file.init();
		this.addWholeFile(file);
	}

	public void addWholeFile(BIGOutputFile file) {
		List<String> legends = file.getLegends();
		this.addFunctionsFromFile(file, legends);
	}

	@Override
	public void changeName(int whichDataSet, int index, String newName) {
		// System.out.println("BIGResultMixer.changeName( " + whichDataSet + ", " + index + ", " +
		// newName + ")" );
		List<String> oldNames = yAxis.get(displayedDataSets[whichDataSet]).getData().getNames();
		String oldName = oldNames.get(index);
		graphNameChanged(oldName, newName);
		oldNames.set(index, newName);
	}

	public void graphNameChanged(String oldGraphName, String newGraphName) {

		if (namesAndGraphs.containsKey(oldGraphName)) {
			Graph graph = namesAndGraphs.get(oldGraphName);
			graph.setGraphName(newGraphName);
			((DefaultTreeModel) resultTree.getModel()).reload(this.node);
			resultTree.showBRM(BIGResultMixer.this);
		}
	}

	public JTabbedPane getPlot() {
		if (yAxis.size() == 0) {
			plot = null;
			return new JTabbedPane();
		}
		if (plot == null) {
			plot = new BIGPlot(this);
		}
		if (plot != null) {
			if (plot.getChartPanel() == null) {
				plot = new BIGPlot(this);
			}
		}
		if (plot.getDisplayPanels() == null)
			return new JTabbedPane();
		return (JTabbedPane) plot.getDisplayPanels()[0];
	}

	private BIGPlot plot = null;

	/**
	 * starts a Dialog, that asks which Functions shall be combined to a new
	 * 
	 * @param jf the jframe, which calls and will be set to background when an messageBox appears
	 */
	public void combineFunctions(JFrame jf) {
		final JDialog jd = new JDialog(jf);
		final String functionName = JOptionPane.showInputDialog("set the Name for the new function");
		if (functionName == null)
			return;
		final String yAxisName = JOptionPane.showInputDialog("set the y-Axis-Name for the function",
				yAxis.get(displayedDataSets[0]).Text);
		if (yAxisName == null)
			return;
		JPanel jp = new JPanel();
		jp.setLayout(new java.awt.GridLayout(4, 1));
		final String[] jComboBox1And3Entries = new String[getTotalFunctions()];
		int nextEntry = 0;
		for (int i = 0; i < yAxis.size(); i++) {
			for (int j = 0; j < yAxis.get(i).getData().getSeriesCount(); j++) {
				jComboBox1And3Entries[nextEntry] = yAxis.get(i).getData().getSeriesName(j);
				nextEntry++;
			}
		}

		String[] jComboBox2Entries = new String[10];
		jComboBox2Entries[0] = "/";
		jComboBox2Entries[1] = "*";
		jComboBox2Entries[2] = "+";
		jComboBox2Entries[3] = "-";
		jComboBox2Entries[4] = "max(y1,y2)";
		jComboBox2Entries[5] = "min(y1,y2)";
		jComboBox2Entries[6] = "max(all)";
		jComboBox2Entries[7] = "min(all)";
		jComboBox2Entries[8] = "max and min(all-iRODS)";
		jComboBox2Entries[9] = "average and standard degression(all-iRODS)";

		final JComboBox<String> combo1 = new JComboBox<String>(jComboBox1And3Entries);
		final JComboBox<String> combo2 = new JComboBox<String>(jComboBox2Entries);
		final JComboBox<String> combo3 = new JComboBox<String>(jComboBox1And3Entries);
		combo1.setEditable(true);
		combo3.setEditable(true);
		JButton okayButton = new JButton("Okay");
		okayButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				double error = -1.0E99;
				try {
					error = BIGInterface.getInstance().getBIGConfigFileParser().intCheckOut("errorInt");
				} catch (Exception ex1) {}

				jd.setVisible(false);
				boolean combo1Acceptable = false;
				boolean combo1Number = true;
				boolean combo3Acceptable = false;
				boolean combo3Number = true;
				double combo1Double = -1.0;
				double combo3Double = -1.0;
				double[][] allYData = new double[combo1.getItemCount()][];
				int whichYData = 0;
				for (int i = 0; i < yAxis.size(); i++) {
					for (int j = 1; j < yAxis.get(i).getData().getValues().length; j++) {
						allYData[whichYData] = yAxis.get(i).getData().getValues()[j];
						whichYData++;
					}
				}
				if (combo2.getSelectedIndex() > 5) {
					// -----------------
					/*
					 * ArrayList originBitFiles = new ArrayList(); Graph tempGraph; for( Enumeration hashTableValues =
					 * namesAndGraphs.elements(); hashTableValues.hasMoreElements(); ) { tempGraph = (Graph)
					 * hashTableValues.nextElement(); originBitFiles.addAll( tempGraph.getOriginalBitFiles() ); } Graph g = new
					 * Graph( originBitFiles );
					 */
					Graph g = new Graph(functionName);
					// -----------------
					g.setXAxisText(xData.Text);
					g.setXValues(yAxis.get(0).getData().getValues()[0]);
					g.setYAxisText(yAxisName);
					double[] gYValues = new double[yAxis.get(0).getData().getValues()[0].length];
					if (combo2.getSelectedIndex() == 6) {
						double max;
						for (int j = 0; j < allYData[0].length; j++) {
							max = Double.NEGATIVE_INFINITY;
							for (int i = 0; i < allYData.length; i++) {
								if (allYData[i][j] < 0) {
									max = error;
									break;
								}
								if (max < allYData[i][j]) {
									max = allYData[i][j];
								}
							}
							gYValues[j] = max;
						}
					} else {
						if (combo2.getSelectedIndex() == 7) {
							double min = Double.POSITIVE_INFINITY;
							for (int j = 0; j < allYData[0].length; j++) {
								min = Double.POSITIVE_INFINITY;
								for (int i = 0; i < allYData.length; i++) {
									if (allYData[i][j] < 0) {
										min = error;
										break;
									}
									if (min > allYData[i][j]) {
										min = allYData[i][j];
									}
								}
								gYValues[j] = min;
							}

						}
						/** neu_anfang **/
						else if (combo2.getSelectedIndex() == 8) {
							Set<Double> x_value_set = new HashSet<Double>();
							double[][] my_values = new double[combo1.getItemCount() + 1][];
							int count = 0;
							for (int i = 0; i < yAxis.size(); i++) {
								for (int j = 0; j < yAxis.get(i).getData().getValues().length; j++) {
									my_values[count] = yAxis.get(i).getData().getValues()[j];
									count++;
								}
							}
							for (int j = 0; j < my_values[0].length; j++) {
								x_value_set.add(new Double(my_values[0][j]));
							}
							Double[] x_value = x_value_set.toArray(new Double[x_value_set.size()]);
							double[] y_max = new double[x_value.length];
							double[] y_min = new double[x_value.length];
							double max = Double.NEGATIVE_INFINITY;
							double min = Double.POSITIVE_INFINITY;
							for (int k = 0; k < x_value.length; k++) {
								max = Double.NEGATIVE_INFINITY;
								min = Double.POSITIVE_INFINITY;
								for (int j = 1; j < my_values.length; j++) {
									for (int i = 0; i < my_values[0].length; i++) {
										if (x_value[k].doubleValue() == my_values[0][i]) {
											if (max < my_values[j][i]) {
												max = my_values[j][i];
											}
											if (min > my_values[j][i]) {
												min = my_values[j][i];
											}
										}
									}
									for (int i = 0; i < my_values.length; i++) {
										if (my_values[j][i] < 0) {
											max = error;
											min = error;
										}
									}
								}
								y_max[k] = max;
								y_min[k] = min;
							}
							double[] gYValues_2 = new double[yAxis.get(0).getData().getValues()[0].length];
							for (int i = 0; i < my_values[0].length; i++) {
								for (int k = 0; k < x_value.length; k++) {
									if (x_value[k].doubleValue() == my_values[0][i]) {
										gYValues[i] = y_max[k];
										gYValues_2[i] = y_min[k];
										break;
									}
								}
							}
							g.setGraphName(functionName + "-max");
							Graph g_2 = new Graph(functionName + "-min");
							g_2.setXAxisText(xData.Text);
							g_2.setXValues(yAxis.get(0).getData().getValues()[0]);
							g_2.setYAxisText(yAxisName);
							g_2.setYValues(gYValues_2);
							addFunction(g_2);
						} else if (combo2.getSelectedIndex() == 9) {
							Set<Double> x_value_set = new HashSet<Double>();
							double[][] my_values = new double[combo1.getItemCount() + 1][];
							int count = 0;
							for (int i = 0; i < yAxis.size(); i++) {
								for (int j = 0; j < yAxis.get(i).getData().getValues().length; j++) {
									my_values[count] = yAxis.get(i).getData().getValues()[j];
									count++;
								}
							}
							for (int j = 0; j < my_values[0].length; j++) {
								x_value_set.add(new Double(my_values[0][j]));
							}
							Double[] x_value = x_value_set.toArray(new Double[x_value_set.size()]);
							double[] y_average = new double[x_value.length];
							double[] y_std_degression = new double[x_value.length];
							double sum = 0;
							double sum_2 = 0;
							double y_count = 0;
							int my_error = 0;
							for (int k = 0; k < x_value.length; k++) {
								sum = 0;
								sum_2 = 0;
								y_count = 0;
								my_error = 0;
								for (int j = 1; j < my_values.length; j++) {
									for (int i = 0; i < my_values[0].length; i++) {
										if (x_value[k].doubleValue() == my_values[0][i]) {
											sum += my_values[j][i];
											sum_2 += my_values[j][i] * my_values[j][i];
											y_count++;
										}
									}
									for (int i = 0; i < my_values.length; i++) {
										if (my_values[j][i] < 0) {
											my_error++;
										}
									}
								}
								if (my_error == 0) {
									if ((sum == 0) || (y_count == 0)) {
										y_average[k] = 0;
									} else {
										y_average[k] = sum / y_count;
									}
									if (y_average[k] > 0) {
										y_std_degression[k] = Math.sqrt((sum_2 / y_count)
												- (y_average[k] * y_average[k]));
									} else {
										y_std_degression[k] = 0;
									}
								} else {
									y_average[k] = error;
									y_std_degression[k] = error;
								}
							}
							double[] gYValues_2 = new double[yAxis.get(0).getData().getValues()[0].length];
							double[] gYValues_3 = new double[yAxis.get(0).getData().getValues()[0].length];
							for (int i = 0; i < my_values[0].length; i++) {
								for (int k = 0; k < x_value.length; k++) {
									if (x_value[k].doubleValue() == my_values[0][i]) {
										gYValues_2[i] = y_average[k];
										if (y_std_degression[k] != error) {
											if (y_std_degression[k] > y_average[k]) {
												gYValues[i] = 0;
											} else {
												gYValues[i] = y_average[k] - y_std_degression[k];
											}
											gYValues_3[i] = y_average[k] + y_std_degression[k];

										}
										break;
									}
								}
							}
							g.setGraphName(functionName + "-std-degression(negative)");
							Graph g_2 = new Graph(functionName + "-average");
							g_2.setXAxisText(xData.Text);
							g_2.setXValues(yAxis.get(0).getData().getValues()[0]);
							g_2.setYAxisText(yAxisName);
							g_2.setYValues(gYValues_2);
							addFunction(g_2);
							Graph g_3 = new Graph(functionName + "-std-degression(positive)");
							g_3.setXAxisText(xData.Text);
							g_3.setXValues(yAxis.get(0).getData().getValues()[0]);
							g_3.setYAxisText(yAxisName);
							g_3.setYValues(gYValues_3);
							addFunction(g_3);
						}

					}
					/** neu_ende **/
					g.setYValues(gYValues);
					addFunction(g);
					return;
				}
				try {
					combo1Double = (new Double((String) combo1.getSelectedItem())).doubleValue();
					combo1Acceptable = true;
				} catch (NumberFormatException ex) {
					combo1Number = false;
				}
				try {
					combo3Double = (new Double((String) combo3.getSelectedItem())).doubleValue();
					combo3Acceptable = true;
				} catch (NumberFormatException ex) {
					combo3Number = false;
				}
				if (!combo1Number) {
					for (int i = 0; i < jComboBox1And3Entries.length; i++)
						if (combo1.getSelectedItem().equals(jComboBox1And3Entries[i])) {
							combo1Acceptable = true;
						}
				}
				if (!combo3Number) {
					for (int i = 0; i < jComboBox1And3Entries.length; i++)
						if (combo3.getSelectedItem().equals(jComboBox1And3Entries[i])) {
							combo3Acceptable = true;
						}
				}
				if ((!combo1Acceptable) || (!combo3Acceptable)) {
					System.err.println("Inacceptable input");
					return;
				}
				// -----------------
				/*
				 * String str = (String) combo1.getSelectedItem(); ArrayList originBitFiles = new ArrayList();
				 * DefaultMutableTreeNode dmtn; Graph originGraph; if ( namesAndGraphs.containsKey( str ) ) { //
				 * System.err.println("Found Graph " + str); dmtn = (DefaultMutableTreeNode) namesAndGraphs.get( str );
				 * originGraph = (Graph) dmtn.getUserObject(); originBitFiles.addAll( originGraph.getOriginalBitFiles() ); }
				 * else { System.err.println("Graph " + str + " not found!!!!!!"); } str = (String) combo3.getSelectedItem(); if
				 * ( namesAndGraphs.containsKey( str ) ) { // System.err.println("Found Graph " + str); dmtn =
				 * (DefaultMutableTreeNode) namesAndGraphs.get( str ); originGraph = (Graph) dmtn.getUserObject();
				 * originBitFiles.addAll( originGraph.getOriginalBitFiles() ); } else { System.err.println("Graph " + str +
				 * " not found!!!!!!"); } Graph g = new Graph( originBitFiles );
				 */
				Graph g = new Graph(functionName);
				// -----------------

				g.setXAxisText(xData.Text);
				g.setXValues(yAxis.get(0).getData().getValues()[0]);
				g.setYAxisText(yAxisName);
				double[] combineGraph1 = new double[g.getXValues().length];
				double[] combineGraph2 = new double[g.getXValues().length];

				if (combo1Number) {
					for (int i = 0; i < combineGraph1.length; i++) {
						combineGraph1[i] = combo1Double;
					}
				} else {
					combineGraph1 = allYData[combo1.getSelectedIndex()];
				}
				if (combo3Number) {
					for (int i = 0; i < combineGraph2.length; i++) {
						combineGraph2[i] = combo3Double;
					}
				} else {
					combineGraph2 = allYData[combo3.getSelectedIndex()];
				}
				double[] gYValues = new double[combineGraph1.length];
				for (int i = 0; i < combineGraph1.length; i++) {
					if ((combineGraph1[i] < 0) || (combineGraph2[i] < 0)) {
						gYValues[i] = error;
					} else {
						switch (combo2.getSelectedIndex()) {
							case 0 :
								// System.err.println(combineGraph1[i]+"/"+combineGraph2[ i ]);
								gYValues[i] = combineGraph1[i] / combineGraph2[i];
								break;
							case 1 :
								gYValues[i] = combineGraph1[i] * combineGraph2[i];
								break;
							case 2 :
								gYValues[i] = combineGraph1[i] + combineGraph2[i];
								break;
							case 3 :
								gYValues[i] = combineGraph1[i] - combineGraph2[i];
								break;
							case 4 :
								gYValues[i] = combineGraph1[i];
								if (gYValues[i] < combineGraph2[i]) {
									gYValues[i] = combineGraph2[i];
								}
								break;
							case 5 :
								gYValues[i] = combineGraph1[i];
								if (gYValues[i] > combineGraph2[i]) {
									gYValues[i] = combineGraph2[i];
								}
								break;

							default :
								gYValues[i] = -1;
						}
					}
					// System.err.println(combineGraph1[i]+"\t"+combineGraph2[ i ]+"\t"+gYValues[ i ]);
				}
				g.setYValues(gYValues);
				addFunction(g);
			}
		});
		JButton cancelButton = new JButton("Cancel");
		cancelButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				jd.setVisible(false);
			}
		});
		jp.add(combo1);
		jp.add(combo2);
		jp.add(combo3);
		JPanel jp2 = new JPanel();
		jp2.add(okayButton, java.awt.BorderLayout.WEST);
		jp2.add(cancelButton, java.awt.BorderLayout.EAST);
		jp.add(jp2);
		jd.setContentPane(jp);
		jd.pack();
		Dimension frameSize = jd.getSize();
		if (frameSize.height > screenSize.height) {
			frameSize.height = screenSize.height;
		}
		if (frameSize.width > screenSize.width) {
			frameSize.width = screenSize.width;
		}
		jd.setLocation((screenSize.width - frameSize.width) / 2,
				(screenSize.height - frameSize.height) / 2);
		jd.setVisible(true);
	}

	/**
	 * show the dialog for adding a new function
	 * 
	 * @param jf JFrame on this the JDialog will be shown
	 */
	public void showAddDialog(JFrame jf) {
		if (lastPath == null)
			lastPath = BIGInterface.getInstance().getOutputPath();
		BIGFileChooser jfc = new BIGFileChooser(lastPath);
		jfc.addFileFilter("BenchIT ResultFiles", "bit");
		if (jfc.showOpenDialog(jf) == JFileChooser.APPROVE_OPTION) {
			BIGOutputFile file;
			try {
				file = new BIGOutputFile(jfc.getSelectedFile());
			} catch (FileNotFoundException e) {
				JOptionPane.showMessageDialog(null, "Selected file does not exist!", "Error",
						JOptionPane.ERROR_MESSAGE);
				return;
			}
			if (!file.init()) {
				return;
			}
			showSelectFunctionsDialog(file);
		}
	}

	public void showSelectFunctionsDialog(BIGOutputFile file) {
		lastPath = file.getFile().getPath();
		List<String> legends = file.getLegends();
		JList<String> jl = new JList<String>(legends.toArray(new String[legends.size()]));
		jl.setSelectedIndex(0);
		Checkbox cbOpt = new Checkbox("Optimum Value (From bit file)", true);
		Checkbox cbMin = new Checkbox("Minimum Value");
		Checkbox cbMax = new Checkbox("Maximum Value");
		Checkbox cbMean = new Checkbox("Mean Value");
		Checkbox cbMed = new Checkbox("Median Value");
		Object[] o = (file.getRawFile() == null) ? new Object[2] : new Object[8];
		o[0] = file.getFilenameWithoutExtension();
		o[1] = legends.size() > 1 ? jl : legends.get(0);
		if (file.getRawFile() != null) {
			o[2] = "\n\nWhich types?";
			o[3] = cbOpt;
			o[4] = cbMin;
			o[5] = cbMax;
			o[6] = cbMean;
			o[7] = cbMed;
		}
		int value = JOptionPane.showConfirmDialog(null, o, "Select functions",
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		if (value == JOptionPane.CANCEL_OPTION)
			return;
		else {
			if (cbOpt.getState())
				addFunctionsFromFile(file, jl.getSelectedValuesList());
			ChoosenRawFuncs choosen = new ChoosenRawFuncs(cbMin.getState(), cbMax.getState(),
					cbMed.getState(), cbMean.getState());
			addFunctionsFromRawFile(file.getRawFile(), jl.getSelectedValuesList(), choosen);
			if (resultTree != null) {
				resultTree.showBRM(this);
			}
		}

	}
	/**
	 * gets the root
	 * 
	 * @return DefaultMutableTreeNode the root node
	 */
	public DefaultMutableTreeNode getTreeNode() {
		if (node == null) {
			node = new DefaultMutableTreeNode(this);
		}
		return node;
	}

	/**
	 * sets this.resultTree
	 * 
	 * @param tree BIGResultTree
	 */
	public void setResultTree(BIGResultTree tree) {
		resultTree = tree;
	}

	/**
	 * gets the popupMenu for a BRM-node
	 */
	public JPopupMenu getBIGPopupMenu(final JFrame jf, final Object o) {
		// final BIGResultMixer mixer = this ;
		// o can be graph or this resultmixer
		// items for the popup menu
		JPopupMenu itemList = new JPopupMenu();
		if (o instanceof BIGResultMixer) {
			JMenuItem item = new JMenuItem("Add function from file");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					showAddDialog(jf);
				}
			});
			itemList.add(item);

			item = new JMenuItem("Combine functions");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					combineFunctions(jf);
				}
			});
			itemList.add(item);

			item = new JMenuItem("Change title");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					showTitleChangeDialog();
				}
			});
			itemList.add(item);

			item = new JMenuItem("Remove function");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					showRemoveDialog(jf);
				}
			});
			itemList.add(item);

			// -------------------
			// new entry for report generator
			item = new JMenuItem(BIGResultTree.createReportMenuText);
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					new BIGReportGeneratorWindow(resultTree.getSelectionPaths());
				}
			});
			itemList.add(item);
			// -------------------

			itemList.add(new JSeparator());
			item = new JMenuItem("Save");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					BIGFileChooser jfc = new BIGFileChooser();
					jfc.addFileFilter("BenchIT-GUI Mixer File", "bmf");
					if (jfc.showSaveDialog(null) != JFileChooser.APPROVE_OPTION)
						return;
					saveFile = jfc.getSelectedFile();
					save();
				}
			});
			itemList.add(item);
			item = new JMenuItem("Load");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					BIGFileChooser jfc = new BIGFileChooser();
					jfc.addFileFilter("BenchIT-GUI Mixer File", "bmf");
					if (jfc.showOpenDialog(null) != JFileChooser.APPROVE_OPTION)
						return;

					cleanUp();
					while ((yAxis.size() > 0) && (yAxis.get(0).getData().getSeriesCount() > 0)) {
						removeSubNode(yAxis.get(0).getData().getSeriesName(0));
					}
					Component com = getPlot();
					if (com != null) {
						plotPanel.add(com);
					}
					plotPanel.revalidate();
					init(jfc.getSelectedFile());
					if (resultTree != null) {
						resultTree.updateBRM(BIGResultMixer.this);
					}

					if (resultTree != null) {
						resultTree.updateBRM(BIGResultMixer.this);
					}

				}
			});
			itemList.add(item);

		} else {
			JMenuItem item = new JMenuItem("Remove");
			item.addActionListener(new ActionListener() {
				// BIGResultTree tree = null ;
				public void actionPerformed(ActionEvent ae) {
					// get the name of the leave
					removeSubNode(o.toString());
				}
			});
			itemList.add(item);
			item = new JMenuItem("Rename");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					String oldName = o.toString();
					String newName = JOptionPane.showInputDialog("Insert new Name", oldName);
					if (newName == null)
						return;
					// get the name of the leave
					rename(oldName, newName);
					plotPanel.removeAll();
					java.awt.Component com = getPlot();
					if (com != null) {
						plotPanel.add(com);
					}
					plotPanel.revalidate();
					if (resultTree != null) {
						resultTree.updateBRM(BIGResultMixer.this);
					}
				}
			});
			itemList.add(item);

		}
		return itemList;
	}

	private void showRemoveDialog(JFrame jf) {
		String[] s = new String[getTotalFunctions()];
		int where = 0;
		for (int i = 0; i < yAxis.size(); i++) {
			for (int j = 0; j < yAxis.get(i).getData().getSeriesCount(); j++) {
				s[where] = yAxis.get(i).getData().getSeriesName(j);
				where++;
			}
		}
		JList<String> jl = new JList<String>(s);
		Object o[] = new Object[2];
		o[0] = "Please select the functions you'd like to remove";
		o[1] = jl;
		int value = JOptionPane.showConfirmDialog(null, o, "Select functions to remove",
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		if (value == JOptionPane.CANCEL_OPTION)
			return;

		for (int i = jl.getSelectedValuesList().size() - 1; i >= 0; i--) {
			removeSubNode(s[jl.getSelectedIndices()[i]]);
		}
		System.gc();
		plotPanel.removeAll();
		Component com = getPlot();
		if (com != null) {
			plotPanel.add(com);
		}
		plotPanel.revalidate();
		if (resultTree != null) {
			resultTree.updateBRM(this);
		}
	}

	public void removeSubNode(String s) {
		for (int i = 0; i < node.getChildCount(); i++) {
			if (node.getChildAt(i).toString().equals(s)) {
				node.remove(i);
				if (resultTree != null) {
					((DefaultTreeModel) resultTree.getModel()).reload(node);
				}
				this.removeFunction(s);
				if (resultTree != null) {
					resultTree.showBRM(this);
				}
				return;
			}
		}
	}

	/**
	 * Opens an input dialog where you can change the title of a plot.
	 */
	public void showTitleChangeDialog() {
		// stores the result of the input dialog
		String newPlotTitle = null;
		if (plot != null) {
			// if the plot exists, display the actual plot tilte as default input
			newPlotTitle = JOptionPane.showInputDialog("Set the new Title", getTitle());
		} else {
			// if the plot doesn't exist show an "empty" input dialog
			newPlotTitle = JOptionPane.showInputDialog("Set the new Title");
		}

		if (newPlotTitle != null) {
			// if the input dialog wasn't canceled set the new title
			setTitle(newPlotTitle);
			if (plot != null) {

				// set the title
				setTitle(newPlotTitle);
				JFreeChart chart = plot.getChartPanel().getChart();
				// repaint the chart
				chart.getTitle().setFont(getFont("title"));
				// also fonts
				chart.getTitle().setText(getExtTitle());
				chart.titleChanged(new TitleChangeEvent(new TextTitle(getExtTitle())));
				plot.getChartPanel().chartChanged(new ChartChangeEvent(chart));

				plotPanel.revalidate();
			}
			if (resultTree != null) {
				resultTree.updateBRM(this);
			}
		}
	}

	/*
	 * public void load(int i) { File name=new File(BIGInterface.getInstance().getConfigPath()+ File.separator +
	 * "mixer_"+i); if (this.plot==null) this.plot=new BIGPlot(); this.plot.setOtherFile(name); this.plot.init();
	 * this.initGD(); } public void initGD() { if (this.plot==null) { } if (this.plot.getDataSetY1()==null) return; for
	 * (int i=0;i<this.plot.getDataSetY1().getNames().length;i++) {
	 * //System.err.println(i+"/"+this.plot.getDataSetY1().getNames().length); Graph g= new Graph();
	 * //System.err.println(this.plot.getDataSetY1().getNames()[i]);
	 * g.setGraphName(this.plot.getDataSetY1().getNames()[i]); g.setXAxisText(this.plot.xAxisTextField.getText());
	 * g.setYAxisText(this.plot.yAxisTextField.getText()); g.setXValues(this.plot.getDataSetY1().getValues()[0]);
	 * g.setYValues(this.plot.getDataSetY1().getValues()[i+1]); this.gd.addGraph(g); this.node.add(new
	 * DefaultMutableTreeNode(g)); } if (this.plot.getDataSetY2()!=null) { for (int
	 * i=0;i<this.plot.getDataSetY2().getNames().length;i++) { //System.err.println(i); Graph g= new Graph();
	 * g.setGraphName(this.plot.getDataSetY2().getNames()[i]); g.setXAxisText(this.plot.xAxisTextField.getText());
	 * g.setYAxisText(this.plot.yAxisTextField2.getText()); g.setXValues(this.plot.getDataSetY2().getValues()[0]);
	 * g.setYValues(this.plot.getDataSetY2().getValues()[i+1]); this.gd.addGraph(g); this.node.add(new
	 * DefaultMutableTreeNode(g)); } } //( ( DefaultTreeModel ) resultTree.getModel() ).reload( this.node ) ; } public
	 * void save(int i) { File name=new File(BIGInterface.getInstance().getConfigPath()+ File.separator + "mixer_"+i); if
	 * (this.gd.size()==0) { System.out.println( "mixer "+i+" not used" ) ; return; } if (plot!=null) {
	 * this.plot.setOtherFile( name ) ; this.plot.save(); } else { System.out.println( "mixer "+i+" not used" ) ; } }
	 */
	/**
	 * Returns the string representation of this mixer. The returned string consists of the string "Mixer " followed by
	 * the title of this mixer.
	 * 
	 * @return String string representation of this mixer
	 */
	@Override
	public String toString() {
		return "Mixer " + getExtTitle();
	}

	// NEW
	/**
	 * This methode stores all necessary data of this mixer in the specified file.
	 * 
	 * @param f - file for saving the informations of this mixer
	 */
	@Override
	public void save() {
		if (getSaveFile() == null)
			return;
		if (yAxis.size() == 0) {
			if (getSaveFile().exists()) {
				getSaveFile().delete();
			}
			return;
		}

		StringBuffer sb = new StringBuffer("#autoGenerated by BIGGUI\n");
		int whichFunction = 1;
		sb.append("title=" + getTitle() + "\n");

		sb.append("xaxistext=" + xData.Text + "\n");
		sb.append("xoutmin=" + xData.getMinAbs() + "\n");
		sb.append("xoutmax=" + xData.getMaxAbs() + "\n");
		sb.append("xaxislogbase=" + xData.Log + "\n");
		sb.append("xaxisticks=" + xData.Ticks + "\n");
		sb.append("xaxispre=" + xData.getPre() + "\n");
		sb.append("xaxisnumberformat=" + xData.NumberFormat + "\n");
		sb.append("xaxisscaletext=" + xData.scaleAxisText + "\n");

		for (int i = 0; i < yAxis.size(); i++) {
			YAxisDataMixed curAxis = (YAxisDataMixed) yAxis.get(i);
			BIGDataSet data = curAxis.getData();
			for (int j = 0; j < data.getSeriesCount(); j++) {
				sb.append("y" + whichFunction + "axistext=" + curAxis.Text + "\n");
				sb.append("y" + whichFunction + "outmin=" + curAxis.getMinAbs() + "\n");
				sb.append("y" + whichFunction + "outmax=" + curAxis.getMaxAbs() + "\n");
				sb.append("y" + whichFunction + "axislogbase=" + curAxis.Log + "\n");
				sb.append("y" + whichFunction + "axisnumberformat=" + curAxis.NumberFormat + "\n");
				sb.append("y" + whichFunction + "axisticks=" + curAxis.Ticks + "\n");
				sb.append("tlegendfunction" + whichFunction + "=" + data.getSeriesName(j) + "\n");
				// ------------------------------------------------------------------------
				// new: 08-04-29
				if (curAxis.originalBitFiles.size() > j) {
					sb.append("y" + whichFunction + "OriginalBitFiles=" + curAxis.originalBitFiles.get(j)
							+ "\n");
				}
				// ------------------------------------------------------------------------

				try {
					// TODO: Why is this undefined???
					sb.append("y" + whichFunction + "color=" + curAxis.Colors.get(j).getRGB() + "\n");
					int tempIndex = -1;
					for (int index = 0; index < BIGPlotRenderer.BIG_SHAPES.length; index++) {
						if (curAxis.Shapes.get(j) == BIGPlotRenderer.BIG_SHAPES[index]) {
							tempIndex = index;
							break;
						}
					}
					sb.append("y" + whichFunction + "shape=" + tempIndex + "\n");
				} catch (IndexOutOfBoundsException exc) {
					// fallback if something goes wrong (i.e. index out-of-bounds exception)
					Paint[] defaultPaints = BIGPlot.getDefaultPaintSequence();
					Color thisColor = (Color) defaultPaints[whichFunction % defaultPaints.length];
					sb.append("y" + whichFunction + "color=" + thisColor.getRGB() + "\n");

					sb.append("y" + whichFunction + "shape=-1\n");
				}
				sb.append("y" + whichFunction + "axispre=" + curAxis.getPre() + "\n");
				sb.append("y" + whichFunction + "scaletext=" + curAxis.scaleAxisText + "\n");
				sb.append("y" + whichFunction + "scalelegends=" + curAxis.getScaleLegends() + "\n");
				whichFunction++;
			}
		}
		sb.append("numfunctions=" + (whichFunction - 1) + "\n");
		super.appendFontInformation(sb);

		sb.append("drawAntiAliased=" + drawAntiAliased + "\n");
		sb.append("beginofdata\n");
		double[][] values = new double[whichFunction][yAxis.get(0).getData().getValues()[0].length];
		values[0] = yAxis.get(0).getData().getValues()[0];
		whichFunction = 1;
		for (int i = 0; i < yAxis.size(); i++) {
			for (int j = 0; j < yAxis.get(i).getData().getSeriesCount(); j++) {
				values[whichFunction] = yAxis.get(i).getData().getValues()[j + 1];
				whichFunction++;
			}
		}
		for (int j = 0; j < values[0].length; j++) {
			for (int i = 0; i < values.length; i++) {
				sb.append(values[i][j]);
				sb.append('\t');
			}
			sb.append('\n');
		}
		sb.append("endofdata\n");
		system.BIGFileHelper.saveToFile(sb.toString(), getSaveFile());
	}

	private File saveFile = null;
	private File loadedFile = null;

	/**
	 * This methode sets all informations stored in the specified file to this mixer.
	 * 
	 * @param f - file with all necessary informations about the mixer
	 */
	public void init(File f) {
		if (f == null)
			return;
		saveFile = f;
		loadedFile = null;
		init();
	}

	/**
	 * This methode sets all informations stored in the specified file to this mixer.
	 * 
	 * @param f - file with all necessary informations about the mixer
	 */
	@Override
	public boolean init() {
		if (saveFile == null)
			return false;
		if (saveFile.equals(loadedFile))
			return true;
		if (!saveFile.exists()) {
			super.initFonts(saveFile);
			// No file to load -->done
			loadedFile = saveFile;
			return true;
		}
		// First: parsing the output File
		// open the parser
		BIGOutputParser parser = null;
		try {
			parser = new BIGOutputParser(saveFile.getAbsolutePath());
		} catch (BIGParserException ex) {
			System.err.println("Error while parsing " + saveFile.getName() + ":" + ex.getMessage());
			return false;
		}
		super.initFonts(saveFile);
		// setting xValues (thank god, they are just once)
		xData.Text = parser.getValue("xaxistext");
		xData.TextDefault = xData.Text;
		xData.setMinAbs(parser.getDoubleValue("xoutmin"));
		xData.MinDefault = xData.getMinAbs();
		xData.setMaxAbs(parser.getDoubleValue("xoutmax"));
		xData.MaxDefault = xData.getMaxAbs();
		xData.Log = parser.getIntValue("xaxislogbase");
		xData.LogDefault = xData.Log;
		xData.Ticks = parser.getIntValue("xaxisticks");
		xData.TicksDefault = xData.Ticks;
		xData.NumberFormat = parser.getValue("xaxisnumberformat");
		if (xData.NumberFormat == null)
			xData.NumberFormat = "0.000";
		try {
			xData.setPre(parser.getIntValue("xaxispre"));
		} catch (NumberFormatException ig) {}
		xData.scaleAxisText = Boolean.valueOf(parser.getValue("xaxisscaletext"));
		titleDefault = parser.getValue("title");
		setTitle(titleDefault);
		String value = parser.getValue("drawAntiAliased");
		drawAntiAliased = (value != null) && !value.isEmpty() && !value.equals("0")
				&& !value.equalsIgnoreCase("false");
		// now the tough stuff the yaxis
		int numberOfFunctions = parser.getIntValue("numfunctions", -1);
		Vector<Vector<Integer>> whichFunctionsToWhichAxis = new Vector<Vector<Integer>>();
		for (int i = 1; i <= numberOfFunctions; i++) {
			boolean found = false;
			YAxisDataMixed curAxis;
			for (int j = 0; j < yAxis.size(); j++) {
				curAxis = (YAxisDataMixed) yAxis.get(j);
				if (curAxis.Text.equals(parser.getValue("y" + i + "axistext"))) {
					whichFunctionsToWhichAxis.get(j).add(i);
					curAxis.Colors.add(new Color(parser.getIntValue("y" + i + "color")));
					int shape;
					try {
						shape = parser.getIntValue("y" + i + "shape");
						curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
					} catch (Exception ex5) {
						shape = new Integer((i - 1) % BIGPlotRenderer.BIG_SHAPES.length);
						curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
					}
					// ------------------------------------------------------------------------
					// new: 08-04-29
					// if there are original bit files stored for this mixer, then we get a
					// comma seperated list with these file paths
					// else NULL will be returned by the parser
					curAxis.originalBitFiles.add(parser.getValue("y" + i + "OriginalBitFiles"));
					// ------------------------------------------------------------------------

					if (curAxis.getMinAbs() > parser.getDoubleValue("y" + i + "outmin", 0))
						curAxis.setMinAbs(parser.getDoubleValue("y" + i + "outmin", 0));
					if (curAxis.getMaxAbs() < parser.getDoubleValue("y" + i + "outmax"))
						curAxis.setMaxAbs(parser.getDoubleValue("y" + i + "outmax"));
					if (curAxis.Log > parser.getIntValue("y" + i + "axislogbase", 0))
						curAxis.Log = parser.getIntValue("y" + i + "axislogbase", 0);
					if (curAxis.Ticks < parser.getIntValue("y" + i + "axisticks", -1))
						curAxis.Ticks = parser.getIntValue("y" + i + "axisticks", -1);
					curAxis.scaleAxisText = curAxis.scaleAxisText
							&& Boolean.valueOf(parser.getValue("y" + i + "scaletext"));
					curAxis.setScaleLegends(curAxis.getScaleLegends()
							&& Boolean.valueOf(parser.getValue("y" + i + "scalelegends")));
					found = true;
				}
			}
			if (!found) {
				curAxis = new YAxisDataMixed(parser.getValue("y" + i + "axistext"), null);
				yAxis.add(curAxis);

				try {
					curAxis.setPre(parser.getIntValue("y" + i + "axispre"));
				} catch (Exception ex2) {
					System.err.println("Default");
				}
				Vector<Integer> functionsForThisAxis = new Vector<Integer>();
				functionsForThisAxis.add(i);
				whichFunctionsToWhichAxis.add(functionsForThisAxis);

				curAxis.Colors.add(new Color(parser.getIntValue("y" + i + "color")));
				int shape;
				try {
					shape = parser.getIntValue("y" + i + "shape");
					curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
				} catch (Exception ex5) {
					shape = new Integer((i - 1) % BIGPlotRenderer.BIG_SHAPES.length);
					curAxis.Shapes.add(BIGPlotRenderer.BIG_SHAPES[shape]);
				}
				curAxis.originalBitFiles.add(parser.getValue("y" + i + "OriginalBitFiles"));

				try {
					curAxis.setMinAbs(parser.getDoubleValue("y" + i + "outmin"));
				} catch (NumberFormatException ex3) {}
				try {
					curAxis.setMaxAbs(parser.getDoubleValue("y" + i + "outmax"));
				} catch (NumberFormatException ex4) {}
				curAxis.Log = parser.getIntValue("y" + i + "axislogbase", 0);
				curAxis.Ticks = parser.getIntValue("y" + i + "axisticks", -1);
				curAxis.NumberFormat = parser.getValue("y" + i + "axisnumberformat");
				if (curAxis.NumberFormat == null)
					curAxis.NumberFormat = "0.000";
				curAxis.scaleAxisText = Boolean.valueOf(parser.getValue("y" + i + "scaletext"));
				curAxis.setScaleLegends(Boolean.valueOf(parser.getValue("y" + i + "scalelegends")));
			}
		}
		for (YAxisData curAxis : yAxis) {
			curAxis.TextDefault = curAxis.Text;
			curAxis.MinDefault = curAxis.getMinAbs();
			curAxis.MaxDefault = curAxis.getMaxAbs();
			curAxis.LogDefault = curAxis.Log;
			curAxis.TicksDefault = curAxis.Ticks;
		}

		double[][] values = parser.getData();
		// Building the dataSets
		if (yAxis.size() > 1) {
			displayedDataSets = new int[2];
			displayedDataSets[0] = 0;
			displayedDataSets[1] = 1;
		} else {
			displayedDataSets = new int[1];
			displayedDataSets[0] = 0;
		}

		for (int i = 0; i < whichFunctionsToWhichAxis.size(); i++) {
			Vector<Integer> functionsForThisAxis = whichFunctionsToWhichAxis.get(i);
			double[][] thisAxissData = new double[functionsForThisAxis.size() + 1][values[0].length];
			// xData
			thisAxissData[0] = values[0];
			List<String> namesForThisAxis = new Vector<String>(functionsForThisAxis.size());
			// System.err.println(i+"/"+namesForThisAxis.length+"("+yNames.get(i)+")");
			for (int j = 0; j < functionsForThisAxis.size(); j++) {
				thisAxissData[j + 1] = values[functionsForThisAxis.get(j).intValue()];
				namesForThisAxis.add(parser.getValue("tlegendfunction" + functionsForThisAxis.get(j)));
			}
			BIGDataSet data = new BIGDataSet(thisAxissData, namesForThisAxis);
			YAxisData curAxis = yAxis.get(i);
			data.setDefaultMinsAndMaxs(xData.MinDefault, xData.MaxDefault, curAxis.MinDefault,
					curAxis.MaxDefault);
			curAxis.setXPre(xData.getPre());
			data.setMinimumDomainValue(xData.getMin());
			data.setMaximumDomainValue(xData.getMax());
			curAxis.setData(data);
		}
		loadedFile = saveFile;
		initNodes();
		return true;
	}
	private void initNodes() {
		for (YAxisData axis : yAxis) {
			for (int j = 0; j < axis.getData().getSeriesCount(); j++) {
				ArrayList<String> al = originalBitFilesToArrayList(axis, j);
				Graph g = new Graph(axis.getData().getSeriesName(j), al);
				g.setPoints(axis.getData().getValues()[0], axis.getData().getValues()[j + 1]);
				g.setYAxisText(axis.Text);
				g.setXAxisText(xData.Text);
				node.add(new DefaultMutableTreeNode(g));
				if (resultTree != null) {
					((DefaultTreeModel) resultTree.getModel()).reload(node);
				}
			}

		}
		getPlot();
	}

	private ArrayList<String> originalBitFilesToArrayList(YAxisData axis, int j) {
		if (!(axis instanceof YAxisDataMixed))
			return null;
		ArrayList<String> al = new ArrayList<String>();
		String str = ((YAxisDataMixed) axis).originalBitFiles.get(j);
		StringTokenizer strTok;

		if (str == null)
			return null;
		strTok = new StringTokenizer(str, ",");
		while (strTok.hasMoreElements()) {
			al.add(strTok.nextToken());
		}
		return al;
	}

	/**
	 * Removes a function from this mixer.
	 * 
	 * @param name - name of function that will be removed
	 */
	@Override
	public void removeFunction(String name) {
		int index = -1;
		YAxisData changedAxis = null;
		for (YAxisData axis : yAxis) {
			index = axis.getData().getSeriesIndex(name);
			if (index >= 0) {
				if (axis.getData().remove(name))
					changedAxis = axis;
				break;
			}
		}
		if (changedAxis != null) {
			if (changedAxis.getData().getSeriesCount() == 0) {
				removeDataSet(changedAxis);
				return;
			}
			changedAxis.Colors.remove(index);
			changedAxis.Shapes.remove(index);
			((YAxisDataMixed) changedAxis).originalBitFiles.remove(index);

			for (int i = 0; i < node.getChildCount(); i++)
				if (node.getChildAt(i).toString().equals(name)) {
					node.remove(i);
				}
		}
	}

	private boolean setXAchsisText(Graph g) {
		if (xData.Text == null || xData.Text.equals("") || xData.Text.equals("x") || yAxis.size() == 0) {
			xData.Text = g.getXAxisText();
			xData.TextDefault = xData.Text;
		} else if (!xData.Text.equals(g.getXAxisText())) {
			String tmpText = xData.Text + " / ";
			if (tmpText.indexOf(g.getXAxisText() + " / ") < 0) {
				String title = "Conflicting x-axis names!";
				EnterSomethingDialog esd = new EnterSomethingDialog(null, title,
						new String[]{title, "Current name: " + xData.Text, "New name: " + g.getXAxisText(),
								" ", "Enter new Name!"}, false);
				esd.setDefault(tmpText + g.getXAxisText());
				esd.setVisible(true);
				if (esd.answer == null)
					return false;
				xData.Text = esd.answer;
			}
		}
		return true;
	}

	@Override
	public boolean addFunction(Graph graph) {
		// System.err.println("Add Graph " + name + " to mixer");
		YAxisDataMixed curAxis = (YAxisDataMixed) getYAxis(graph.getYAxisText());
		if (curAxis == null) {
			// if we didn't found an appropriate y-axis and the mixer has just one y-axis
			// a second y-axis can be added
			if (yAxis.size() >= 2)
				return false;
			if (!setXAchsisText(graph))
				return false;
			curAxis = new YAxisDataMixed(graph.getYAxisText(), graph.getDataSet());
			yAxis.add(curAxis);
			if (yAxis.size() == 1) {
				// if mixer was empty, make some initial settings at first
				// System.err.println("Do some initial stuff");
				displayedDataSets = new int[]{0};

				xData.LogDefault = 0;
				xData.TicksDefault = -1;
				xData.MinDefault = curAxis.getData().getMinimumDomainValueAbs();
				xData.MaxDefault = curAxis.getData().getMaximumDomainValueAbs();
				xData.reset();
				xData.Text = graph.getXAxisText();// Do not reset text
			} else {
				xData.MinDefault = Math.min(xData.MinDefault, curAxis.getData().getMinimumDomainValueAbs());
				xData.MaxDefault = Math.min(xData.MaxDefault, curAxis.getData().getMaximumDomainValueAbs());
				// System.err.println("Add second y-axis");
				displayedDataSets = new int[]{0, 1};
			}
		} else {
			if (!setXAchsisText(graph))
				return false;
			// if correct y-axis was found, add the graph to this axis
			// System.err.println("y-axis " + yName + " found - add graph");
			if (!curAxis.getData().addFromGraph(graph))
				return false;
			// update all xValues
			double[][] xVals = new double[1][];
			xVals[0] = graph.getXValues();
			for (int j = 0; j < yAxis.size(); j++) {
				yAxis.get(j).getData().add(xVals, null);
			}
			xData.MinDefault = Math.min(xData.MinDefault, curAxis.getData().getMinimumDomainValueAbs());
			xData.MaxDefault = Math.min(xData.MaxDefault, curAxis.getData().getMaximumDomainValueAbs());
		}
		curAxis.setMinMax();
		curAxis.originalBitFiles.add(graph.getOriginalBitFilesAsString());
		curAxis.Colors.add(null);
		curAxis.Shapes.add(null);
		calculateScaleY(curAxis);
		calculateScaleX();

		DefaultMutableTreeNode dmtnode = new DefaultMutableTreeNode(graph);
		node.add(dmtnode);
		namesAndGraphs.put(graph.getGraphName(), graph);

		if (plot == null)
			plot = new BIGPlot(this);
		else
			plot.setupConfig();

		if (resultTree != null) {
			((DefaultTreeModel) resultTree.getModel()).reload(node);
			resultTree.showBRM(this);
		}
		return true;
	}

	public void removeDataSet(YAxisData axis) {
		yAxis.remove(axis);
		int[] newDisplays = null;
		if (yAxis.size() >= 2) {
			newDisplays = new int[]{0, 1};
		} else if (yAxis.size() == 1) {
			newDisplays = new int[]{0};
		} else {
			displayedDataSets = new int[0];
			plot = null;
			plotPanel = new JPanel();
			setTitle(titleDefault);
		}
		displayedDataSets = newDisplays;
		if (plot != null) {
			plot.setupConfig();
		}
	}

	@Override
	public void rename(String oldS, String newS) {
		super.rename(oldS, newS);
		if (namesAndGraphs.containsKey(oldS)) {
			Graph g = namesAndGraphs.get(oldS);
			namesAndGraphs.remove(oldS);
			g.setGraphName(newS);
			namesAndGraphs.put(newS, g);
			plot.getChartPanel().getChart()
					.legendChanged(new LegendChangeEvent(new BIGPlotLegend(plot.getChartPanel().getChart())));
		}
	}

	// ----------------------------------------------------------------------
	public void updateResultTree() {
		if (resultTree != null) {
			resultTree.updateUI();
		}
	}

	// ----------------------------------------------------------------------

	/**
	 * gets the save-file-name
	 */
	@Override
	public File getSaveFile() {
		return saveFile;
	}

	@Override
	public boolean IsInit() {
		return loadedFile != null;
	}

	// public List<String> getOriginalFiles(){ }
}
