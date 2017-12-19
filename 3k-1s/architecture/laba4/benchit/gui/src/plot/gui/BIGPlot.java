/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGPlot.java Author: SWTP
 * Nagel 1 Last change by: $Author: tschuet $ $Revision: 1.57 $ $Date: 2009/01/07 11:34:12 $
 ******************************************************************************/
package plot.gui;

import gui.*;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.*;

import org.freehep.graphicsio.pdf.PDFGraphics2D;
import org.freehep.graphicsio.ppm.PPMEncoder;
import org.freehep.graphicsio.ps.PSGraphics2D;
import org.freehep.graphicsio.svg.SVGGraphics2D;
import org.jfree.chart.*;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.event.*;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.*;

import plot.data.*;
import system.*;
import conn.Graph;

/**
 * a plotting class derived from ScatterPlotDemo.java <br>
 * we get the data in values, a double[][]<br>
 * the first dimension are the columns, the second the rows<br>
 * all columns have the same number of rows<br>
 * the first column are the x values of the points<br>
 * all other columns are the corresponding y values of the points<br>
 * each other row is called a series<br>
 * the name of the series are stored in names, a String[]<br>
 * we have also n series and n+1 columns and m rows
 * 
 * @author <a href="mailto:pisi@pisi.de">Christoph Mueller</a>
 */
public class BIGPlot {
	// shapeSize
	double shapeSize = 6.0;
	// for output of debug information
	final boolean debug = false;

	// that's the chart panel
	private ChartPanel chartPanel;
	// tabPane contains the chartPanel and a config panel
	// textScrollPane contains the result file's content
	private BIGEditor textScrollPane;
	// the display panels
	private JComponent[] displayPanels; // [0]->view, [1]->text

	// in the beginnig, when building the window, a lot is changed
	private boolean selfChanging = true;

	private final BIGPlotable plotable;
	private JFreeChart chart;

	private BIGPlotConfig config;

	/**
	 * creates new Panels, where the plotable is plotted, setup is available,...
	 * 
	 * @param newPlotable BIGPlotable
	 */
	public BIGPlot(BIGPlotable newPlotable) {
		plotable = newPlotable;
		plotable.init();
		setupConfig();
	}

	/**
	 * redraws the whole bigplot with this.plotable's settings
	 */
	public void setupConfig() {
		if (!plotable.IsInit()) {
			System.err.println("Cannot setup config for uninitialized plotables!");
			return;
		}
		// if axistext isn't found
		if (plotable.xData.Text == null) {
			plotable.xData.Text = "x";
		}
		if (plotable.xData.TextDefault == null) {
			plotable.xData.TextDefault = "x";
		}
		// get data for first y-axis
		YAxisData yAxis1 = plotable.yAxis.get(plotable.displayedDataSets[0]);
		YAxisData yAxis2 = null;
		if (plotable.displayedDataSets.length == 2)
			yAxis2 = plotable.yAxis.get(plotable.displayedDataSets[1]);
		if (debug) {
			System.err.println("build");
			System.err
					.println(plotable.getTitle() + " - " + plotable.xData.Text + " - " + plotable.yAxis);
		}
		// set the shape size (size of shapes in Plot)
		try {
			shapeSize = system.BIGInterface.getInstance().getBIGConfigFileParser()
					.intCheckOut("scaleShape");
		} catch (Exception ex) {}
		// create plot
		// attr.: title, xaxistext,1st yaxistext,dataset, display tooltips, display legends, create urls

		chart = ChartFactory.createScatterPlot(plotable.getExtTitle(), plotable.xData.Text,
				yAxis1.getAxisText(), yAxis1.getData(), true, true, false);
		// second yaxis?
		if (yAxis2 != null) {
			chart.getXYPlot().setSecondaryDataset(yAxis2.getData());
		}
		chart.getXYPlot().setInsets(plotable.insets);
		// build panel for chart with enabled offscreen buffer (double buffer???)
		chartPanel = new ChartPanel(chart);
		// this is how you could display, that this plot was made with benchit
		// dont use images, there's a problem with save them as eps/pdf...
		/*
		 * { public void paintComponent(Graphics g) { super.paintComponent(g); g.setColor(Color.LIGHT_GRAY);
		 * g.drawString("www.benchit.org",(int)getSize().getWidth()-120,20); } } ;
		 */
		// zoom is allowed
		chartPanel.setVerticalZoom(false);
		chartPanel.setHorizontalZoom(false);

		chartPanel.setLocation(100, 100);
		chartPanel.setVerticalZoom(true);
		chartPanel.setHorizontalZoom(true);
		// PlotListen l = new PlotListen(chart, chartPanel, plotable);
		// chartPanel.addMouseListener(l);
		// chartPanel.addMouseMotionListener(l);
		// chartPanel.addMouseWheelListener(l);

		// standard, but set it
		chart.setBackgroundPaint(Color.white);
		// own renderer for colors,
		chart.getXYPlot().setRenderer(
				new BIGPlotRenderer(yAxis1.Colors, yAxis1.Shapes, true, false, false));
		if (yAxis2 != null) {
			chart.getXYPlot().setSecondaryRenderer(
					new BIGPlotRenderer(yAxis2.Colors, yAxis2.Shapes, true, false, false));
		}

		// the drawing supplier gets shapes in the setted size (shapeSize)
		// and its colors (Paint[]) are synchronized (or should be :P)
		// with those from the BIGPlotRenderer
		DrawingSupplier ds = new BIGDrawingSupplier(this);
		// set the supplier
		chart.getXYPlot().getRenderer().setDrawingSupplier(ds);
		// the same supplier, so no colors are used twice
		if (yAxis2 != null) {
			chart.getXYPlot().getSecondaryRenderer().setDrawingSupplier(ds);
		}
		// set the plotable for legends
		BIGPlotLegend plotLegend = new BIGPlotLegend(chart);
		plotLegend.setPlotable(plotable);
		plotLegend.setDisplaySeriesShapes(true);
		chart.setLegend(plotLegend);
		// set comment
		plotable.setAnnComment(plotable.getAnnComment());
		// chartListener if we zoom or some internals say "Hey! Chart changed!"
		chart.addChangeListener(new ChartChangeListener() {
			public void chartChanged(ChartChangeEvent evt) {
				// only if others are changing (i.e. zoom)
				if (!selfChanging) {
					// we are changing the whole thing ourself
					selfChanging = true;
					// System.err.println("selfchanging");
					// set axis with self-finding ticks
					XYPlot xyPlot = chart.getXYPlot();
					if (xyPlot.getDomainAxis() instanceof BIGHorizontalNumberAxis) {
						((BIGHorizontalNumberAxis) xyPlot.getDomainAxis()).setTicks(null);
					}
					setMinMaxFromAxis(plotable.xData, xyPlot.getDomainAxis());
					plotable.xData.Ticks = -1;
					if (xyPlot.getRangeAxis() instanceof BIGVerticalNumberAxis) {
						((BIGVerticalNumberAxis) xyPlot.getRangeAxis()).setTicks(null);
					}
					setMinMaxFromAxis(plotable.yAxis.get(plotable.displayedDataSets[0]),
							xyPlot.getRangeAxis());
					plotable.yAxis.get(plotable.displayedDataSets[0]).Ticks = -1;
					if (plotable.displayedDataSets.length > 1) {
						if (xyPlot.getSecondaryRangeAxis() instanceof BIGVerticalNumberAxis) {
							((BIGVerticalNumberAxis) xyPlot.getSecondaryRangeAxis()).setTicks(null);
						}
						setMinMaxFromAxis(plotable.yAxis.get(plotable.displayedDataSets[1]),
								xyPlot.getSecondaryRangeAxis());
						plotable.yAxis.get(plotable.displayedDataSets[1]).Ticks = -1;
					}
					plotable.drawAntiAliased = chart.getAntiAlias();
					// to paint an annotation (comment in the plot)
					xyPlot.clearAnnotations();
					xyPlot.addAnnotation(plotable.getAnnotationComment());
					// done
					selfChanging = false;
				}
			}
		});

		// these panels are for an integrated GUI
		JTabbedPane displayTabPane = new JTabbedPane(SwingConstants.TOP);
		// Create the config tab
		config = new BIGPlotConfig(plotable);
		config.init(displayTabPane, chart);

		// overriding the "save as" menuitem
		overrideMenu();
		setChartProperties();
		// we dont change anymore
		selfChanging = false;

		displayTabPane.addTab("Plot", chartPanel);
		JScrollPane scPan = new JScrollPane(config.getPanel());
		// scPan.getVerticalScrollBar().setBlockIncrement(50);
		displayTabPane.addTab("Config", scPan);
		// when switching ...
		displayTabPane.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent evt) {
				// to the plot-tab
				if (((JTabbedPane) evt.getSource()).getSelectedIndex() == 0) {
					setConfigValues();

					chartPanel.revalidate();
					// then repaint it
					((JTabbedPane) evt.getSource()).revalidate();
					((JTabbedPane) evt.getSource()).repaint();
				} else {
					config.loadValues();
				}
			}
		});

		// view text from file
		textScrollPane = new BIGEditor(plotable.getText());
		// benchit outputfiles have shell syntax
		textScrollPane.setTokenMarker(new org.syntax.jedit.tokenmarker.ShellScriptTokenMarker());
		// and shall NOT be edited (yeah write your own results. fine. do it. looser...)
		textScrollPane.setEditable(false);

		displayPanels = new JComponent[2];
		displayPanels[0] = displayTabPane;
		displayPanels[1] = textScrollPane;
	}

	private void setMinMaxFromAxis(AxisData axis, ValueAxis vAxis) {
		// Avoid pointless values with more than 5 decimals
		axis.setMin(BIGUtility.round(vAxis.getMinimumAxisValue(), 6));
		axis.setMax(BIGUtility.round(vAxis.getMaximumAxisValue(), 6));
	}

	/**
	 * gets the display-panels ([0]plot, [1] text)
	 * 
	 * @return JComponent[] the panels
	 */
	public JComponent[] getDisplayPanels() {
		return displayPanels;
	}

	/**
	 * gets the plotted Panel
	 * 
	 * @return ChartPanel the chartPanel of this BIGPlot
	 */
	public ChartPanel getChartPanel() {
		return chartPanel;
	}

	private void setAxisTrace(boolean value) {
		chartPanel.setVerticalAxisTrace(value);
		chartPanel.setHorizontalAxisTrace(value);
	}

	private void setChartAxisData(ValueAxis vAxis, AxisData axis) {
		// first set it to +inf
		vAxis.setMaximumAxisValue(Double.MAX_VALUE);
		// then set the min
		vAxis.setMinimumAxisValue(axis.getMin());
		// then the max
		vAxis.setMaximumAxisValue(axis.getMax());
		// why did we do this (setting to inf)
		// just think: if we would set first min, then max, this could happen:
		// old min/max = 2/3 new min/max=4/5
		// the new min is larger then the old max. but both are set!
		// this can lead to problems (even if this is for a short time)

		// set also the ticks
		// First for x
		if (vAxis instanceof BIGHorizontalNumberAxis) {
			try {
				((BIGHorizontalNumberAxis) vAxis).setTicks(axis.Ticks);
			} catch (Exception ex1) {}
			// if its logarithmic select the displayed numbers:
			// display: n^m or the computed value of n^m
		} else if (vAxis instanceof BIGHorizontalLogarithmicAxis) {
			try {
				((BIGHorizontalLogarithmicAxis) vAxis).setLogTickLabelsFlag(axis.Ticks >= 0);
			} catch (Exception ex1) {}
		} else
		// Then for y
		if (vAxis instanceof BIGVerticalNumberAxis) {
			try {
				((BIGVerticalNumberAxis) vAxis).setTicks(axis.Ticks);
			} catch (Exception ex1) {}
		} else {
			try {
				((BIGVerticalLogarithmicAxis) vAxis).setLogTickLabelsFlag(axis.Ticks >= 0);
			} catch (Exception ex1) {}
		}
		vAxis.setLabel(axis.getAxisText());
	}

	private void setChartProperties() {
		selfChanging = true;
		YAxisData yAxis1 = plotable.yAxis.get(plotable.displayedDataSets[0]);
		YAxisData yAxis2 = (plotable.displayedDataSets.length > 1) ? plotable.yAxis
				.get(plotable.displayedDataSets[1]) : null;

		XYPlot xyPlot = chart.getXYPlot();
		XYPlot plot = xyPlot;
		((BIGPlotRenderer) plot.getRenderer()).setPlotLines(plotable.drawLines[0]);
		((BIGPlotRenderer) plot.getRenderer()).setDefaultShapeFilled(plotable.fillShapes[0]);
		ValueAxis vAxis;
		if (yAxis1.Log > 1)
			vAxis = new BIGVerticalLogarithmicAxis(yAxis1.Text, yAxis1.Log, plotable, yAxis1.NumberFormat);
		else
			vAxis = new BIGVerticalNumberAxis(yAxis1.Text, plotable, yAxis1.NumberFormat);
		plot.setRangeAxis(vAxis);
		setChartAxisData(vAxis, yAxis1);

		if (yAxis2 != null) {
			((BIGPlotRenderer) plot.getSecondaryRenderer()).setPlotLines(plotable.drawLines[1]);
			((BIGPlotRenderer) plot.getSecondaryRenderer()).setDefaultShapeFilled(plotable.fillShapes[1]);
			if (yAxis2.Log > 1)
				vAxis = new BIGVerticalLogarithmicAxis(yAxis2.Text, yAxis2.Log, plotable,
						yAxis2.NumberFormat);
			else
				vAxis = new BIGVerticalNumberAxis(yAxis2.Text, plotable, yAxis2.NumberFormat);
			plot.setSecondaryRangeAxis(vAxis);
			setChartAxisData(vAxis, yAxis2);
		}

		if (plotable.xData.Log > 1)
			vAxis = new BIGHorizontalLogarithmicAxis(plotable.xData.Text, plotable.xData.Log, plotable,
					plotable.xData.NumberFormat);
		else
			vAxis = new BIGHorizontalNumberAxis(plotable.xData.Text, plotable,
					plotable.xData.NumberFormat);
		plot.setDomainAxis(vAxis);
		setChartAxisData(vAxis, plotable.xData);

		xyPlot.setInsets(plotable.insets);
		setAxisTrace(config.currentAxisTrace);
		chart.setAntiAlias(plotable.drawAntiAliased);

		chart.getTitle().setFont(plotable.getFont("title"));
		// also fonts
		chart.getTitle().setText(plotable.getExtTitle());
		((BIGPlotLegend) (chart.getLegend())).setItemFont(plotable.getFont("legend"));
		// remove old annotation
		xyPlot.clearAnnotations();
		// and add new one
		xyPlot.addAnnotation(plotable.getAnnotationComment());
		chart.plotChanged(new PlotChangeEvent(xyPlot));
		selfChanging = false;
	}

	private void setConfigValues() {
		config.setValues();

		setChartProperties();

		if (plotable instanceof BIGResultMixer) {
			((BIGResultMixer) plotable).updateResultTree();
		}
	}

	/**
	 * overrides the menu of the chartPanel to change (and expand) save as
	 */
	private void overrideMenu() {
		// getting the entries of the popUpMenu
		MenuElement[] menuele = chartPanel.getPopupMenu().getSubElements();
		// getting the saveAs...
		JMenuItem saveAs = null;
		for (MenuElement ele : menuele) {
			JMenuItem item = (JMenuItem) ele.getComponent();
			if (item.getText().equals("Save as..."))
				saveAs = item;
		}
		// setting his new Action
		saveAs.removeActionListener(saveAs.getActionListeners()[0]);
		saveAs.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				SaveAs();
			}
		});

		// END overriding "Save as..."
		// detache means open a frame with this plot as content
		JMenuItem detach = new JMenuItem("Detach");
		detach.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				// new frame
				JFrame jf = new JFrame();
				// new plot
				plotable.save();
				BIGPlot bp = new BIGPlot(plotable);
				jf.getContentPane().add(bp.getDisplayPanels()[0], BorderLayout.CENTER);
				jf.setSize(512, 384);
				// show
				jf.setVisible(true);
			}
		});
		chartPanel.getPopupMenu().add(detach);

		JMenuItem showValueTable = new JMenuItem("Show value table");
		showValueTable.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				showValueTable();
			}
		});
		chartPanel.getPopupMenu().add(showValueTable);

		if (plotable instanceof BIGOutputFile && ((BIGOutputFile) plotable).getRawFile() != null) {
			JMenuItem exportBar = new JMenuItem("Export candlestick plot");
			exportBar.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					showUseAllValuesDlg("Export candlestick plot", new UseAllCallback() {
						public void exec(boolean useAll, int functionIndex) {
							exportCandlestickPlot(useAll, functionIndex);
						}
					});
				}
			});
			chartPanel.getPopupMenu().add(exportBar);

			JMenuItem showCS = new JMenuItem("Show candlestick plot");
			showCS.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					showUseAllValuesDlg("Show candlestick plot", new UseAllCallback() {
						public void exec(boolean useAll, int functionIndex) {
							showCandlestickPlot(useAll, functionIndex);
						}
					});
				}
			});
			chartPanel.getPopupMenu().add(showCS);
		}
	}

	private interface UseAllCallback {
		public void exec(boolean useAll, int functionIndex);
	}
	/**
	 * Shows a dialog to ask, if it should use all values or just current view and also asks for the function
	 * 
	 * @param title
	 * @param callback
	 */
	private void showUseAllValuesDlg(String title, UseAllCallback callback) {
		List<String> legends = plotable.getLegends();
		JComboBox<String> function = new JComboBox<String>(legends.toArray(new String[legends.size()]));
		function.setSelectedIndex(0);
		int inc = (legends.size() > 1) ? 3 : 0;
		Object[] msg = new Object[1 + inc];
		if (legends.size() > 1) {
			msg[0] = "Select function:";
			msg[1] = function;
			msg[2] = "\n";
		}
		String[] options = new String[]{"All", "Only current view", "Cancel"};
		msg[inc] = "Do you want to use all values or just values in current view?";
		int res = JOptionPane.showOptionDialog(chartPanel, msg, title, JOptionPane.DEFAULT_OPTION,
				JOptionPane.QUESTION_MESSAGE, null, options, options[0]);
		if (res == 0 || res == 1)
			callback.exec(res == 0, function.getSelectedIndex());
	}

	/**
	 * repaints the Status Label with the actual filesize until the thread is dead
	 * 
	 * @param f File file, which is written
	 * @param t Thread thread, wich writes the file
	 */
	private void startFileExporting(final File f, final Thread t) {
		final String filename = f.getName();
		// get the label
		final JLabel jl = BIGInterface.getInstance().getStatusLabel();
		(new Thread() {
			@Override
			public void run() {
				// while it is exporting
				while (t.isAlive()) {
					// write the actual size to the label
					jl.setText("Exporting " + filename + ":" + (int) (f.length() / 1024) + "kB");
					jl.validate();
					try {
						Thread.sleep(10);
					} catch (InterruptedException ex) {}
				}
				jl.setText("Done");
			}
		}).start();
	}

	/**
	 * gets the default sequence of colors. they are stored in cfg/stdColors.cfg
	 * 
	 * @return Paint[] the colors
	 */
	public static Paint[] getDefaultPaintSequence() {
		// standard colors with 35 possible entries
		Paint[] p = new Paint[35];
		if (!(new File(BIGInterface.getInstance().getConfigPath() + File.separator + "stdColors.cfg"))
				.exists()) {
			setDefaultPaintSequence(DefaultDrawingSupplier.DEFAULT_PAINT_SEQUENCE);
		}
		// read file content to string
		String s = BIGFileHelper.getFileContent(new File(BIGInterface.getInstance().getConfigPath()
				+ File.separator + "stdColors.cfg"));
		// separate by \ns
		StringTokenizer st = new StringTokenizer(s, "\n");
		// build colors from ints
		for (int i = 0; i < p.length; i++) {
			p[i] = new java.awt.Color(Integer.parseInt(st.nextToken()));
		}
		// return
		return p;
	}

	/**
	 * saves the default sequence of colors to a file (cfg/stdColors.cfg)
	 * 
	 * @param p Paint[] the colors to save
	 */
	public static void setDefaultPaintSequence(Paint[] p) {
		// creating filecontent
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < p.length; i++) {
			// adding rgb-value and a line-break
			sb.append(((Color) (p[i])).getRGB());
			sb.append('\n');
		}
		// deleting last linebreak
		sb.deleteCharAt(sb.length() - 1);
		// saving file
		BIGFileHelper.saveToFile(sb.toString(), new File(BIGInterface.getInstance().getConfigPath()
				+ File.separator + "stdColors.cfg"));
	}

	public boolean writePNGFile(String p, Dimension newDim) {
		File parent = new File(p);
		parent = parent.getParentFile();
		parent.mkdirs();
		File f = new File(p);
		Dimension oldDim = chartPanel.getSize();
		chartPanel.setSize(newDim);
		try {
			ChartUtilities.saveChartAsPNG(f, chart, chartPanel.getWidth(), chartPanel.getHeight());
		} catch (IOException e) {
			e.printStackTrace();
		}
		chartPanel.setSize(oldDim);
		if (f.exists())
			return true;
		// if png-file doesn't exist, then an error occured -> return false
		return false;
	}

	public boolean writeSVGFile(String p, Dimension newDim) {
		File parent = new File(p);
		parent = parent.getParentFile();
		parent.mkdirs();
		File f = new File(p);
		Dimension oldDim = chartPanel.getSize();
		chartPanel.setSize(newDim);
		try {
			FileOutputStream out = new FileOutputStream(f);
			try {
				ByteArrayOutputStream bas = new ByteArrayOutputStream();
				SVGGraphics2D gra = new SVGGraphics2D(bas, chartPanel) {
					@Override
					public void writeBackground() {}
				};
				gra.startExport();

				try {
					chartPanel.paintComponent(gra);
				} catch (Exception e) {
					System.err.println("Error while SVG export.");
					e.printStackTrace();
					return false;
				}
				gra.endExport();
				bas.writeTo(out);
				gra.closeStream();
				bas.close();
			} finally {
				out.close();
			}
		} catch (IOException ex1) {
			System.err.println("Error while SVG export.");
			ex1.printStackTrace();
			return false;
		}
		chartPanel.setSize(oldDim);
		if (f.exists())
			return true;
		// if svg-file doesn't exist, then an error occured -> return false
		return false;
	}

	public void exportCandlestickPlot(boolean useAll, int functionIndex) {
		if (!(plotable instanceof BIGOutputFile)) {
			JOptionPane.showMessageDialog(null, "This is limited to normal output files!", "Error",
					JOptionPane.ERROR_MESSAGE);
			return;
		}
		BIGRawFile raw = ((BIGOutputFile) plotable).getRawFile();
		if (raw == null) {
			JOptionPane.showMessageDialog(null, "This file does not have a raw file attached!", "Error",
					JOptionPane.ERROR_MESSAGE);
			return;
		}
		BIGFileChooser fch = new BIGFileChooser(raw.getFile().getParent());
		fch.addFileFilter("gp");
		if (fch.showSaveDialog(chartPanel) != JFileChooser.APPROVE_OPTION)
			return;
		File file = fch.getSelectedFile();
		doExportCandlestickPlot(functionIndex, useAll, file, new File(file.getName() + ".eps"), 0, 0);
		JOptionPane
				.showMessageDialog(chartPanel, "File saved", "Done", JOptionPane.INFORMATION_MESSAGE);
	}

	private double getSecondMinVal(double min, double[] values) {
		double res = Double.POSITIVE_INFINITY;
		boolean found = false;
		for (double val : values) {
			if (val < res && val > min) {
				res = val;
				found = true;
			}
		}
		if (!found)
			return min;
		return res;
	}

	private double getSecondMaxVal(double max, double[] values) {
		double res = Double.NEGATIVE_INFINITY;
		boolean found = false;
		for (double val : values) {
			if (val > res && val < max) {
				res = val;
				found = true;
			}
		}
		if (!found)
			return max;
		return res;
	}

	private void doExportCandlestickPlot(int functionIndex, boolean useAll, File outFile,
			File imgFile, int w, int h) {
		BIGRawFile raw = ((BIGOutputFile) plotable).getRawFile();
		if (raw == null) {
			JOptionPane.showMessageDialog(null, "This file does not have a raw file attached!", "Error",
					JOptionPane.ERROR_MESSAGE);
			return;
		}
		Graph[] graphs = new Graph[5];
		graphs[0] = raw.getMin(functionIndex);
		graphs[1] = raw.getFirstQuartile(functionIndex);
		graphs[2] = raw.getMedian(functionIndex);
		graphs[3] = raw.getThirdQuartile(functionIndex);
		graphs[4] = raw.getMax(functionIndex);
		if (!useAll) {
			for (Graph g : graphs)
				g.keepRange(plotable.xData.getMinAbs(), plotable.xData.getMaxAbs());
		}
		YAxisData yAxis = plotable.getYAxisByName(plotable.getLegends().get(functionIndex));
		double xMod = BIGDataSet.setModifiers[plotable.xData.getPre()];
		double yMod = BIGDataSet.setModifiers[yAxis.getPre()];
		int numPoints = graphs[0].getNumberOfPoints();

		BIGStrings text = new BIGStrings();
		text.add("#gnuplotfile Box-and-whiskers plot");
		text.add("set title '" + plotable.getExtTitle() + "'");
		text.add("set xlabel '" + plotable.xData.getAxisText() + "'");
		BIGDataSet minSet = graphs[0].getDataSet();
		BIGDataSet maxSet = graphs[4].getDataSet();
		double xMin = minSet.getMinimumDomainValue();
		double xMax = minSet.getMaximumDomainValue();
		double minDiff = (xMax - xMin) * 2.5 / 100;
		xMin -= Math.max(minDiff, getSecondMinVal(xMin, minSet.getValues()[0]) - xMin);
		xMax += Math.max(minDiff, xMax - getSecondMaxVal(xMax, minSet.getValues()[0]));
		text.add("set xrange [" + xMin * xMod + ":" + xMax * xMod + "]");
		double yMin = minSet.getMinimumRangeValue();
		double yMax = maxSet.getMaximumRangeValue();
		minDiff = (yMax - yMin) * 2.5 / 100;
		yMin -= Math.max(minDiff, getSecondMinVal(yMin, minSet.getValues()[1]) - yMin);
		yMax += Math.max(minDiff, yMax - getSecondMaxVal(yMax, maxSet.getValues()[1]));
		text.add("set yrange [" + yMin * yMod + ":" + yMax * yMod + "]");
		text.add("set ylabel '" + yAxis.getAxisText() + "'");
		String boxWidth = (numPoints > 1) ? " " + (xMax - xMin) / (numPoints + 2) : "";
		text.add("set boxwidth" + boxWidth);
		text.add("set style fill empty");
		String ext = imgFile.getName();
		String[] extParts = ext.split("\\.");
		ext = extParts[extParts.length - 1].toLowerCase();

		// calculate width and height
		// width=numPoints + 2 (additional space on left and right) * (4 (bar width) + 4 (bar spacing)) + 90 (legend etc.)
		if (w <= 0)
			w = Math.max((numPoints + 2) * (4 + 4) + 90, 640);
		if (h <= 0)
			h = Math.max(w * 9 / 16, 480);
		if (ext.equals("jpg") || ext.equals("jpeg"))
			text.add("set terminal jpeg enhanced size " + w + "," + h + " crop");
		else if (ext.equals("png"))
			text.add("set terminal png enhanced size " + w + "," + h + " crop");
		else if (ext.equals("gif"))
			text.add("set terminal gif enhanced size " + w + "," + h + " crop");
		else
			text.add("set terminal postscript eps color solid");

		text.add("set output '" + imgFile.getName() + "'");
		text.add("plot '" + outFile.getName() + "' using 1:3:2:6:5 with candlesticks title '"
				+ raw.getRawFunction(0).Legend + "' whiskerbars, \\");
		text.add("''         using 1:4:4:4:4 with candlesticks lt -1 notitle");
		text.add("exit");
		text.add("");
		text.add("#x\tMin\tQ1\tMed\tQ3\tMax");

		double[] x = graphs[0].getXValues();
		for (int i = 0; i < numPoints; i++) {
			String line = x[i] * xMod + "";
			for (Graph g : graphs)
				line += "\t" + g.getYValues()[i] * yMod;
			text.add(line);
		}
		text.saveToFile(outFile.getAbsolutePath());
	}

	public void SaveAs() {
		// a filechooser
		BIGFileChooser fch = new BIGFileChooser();
		// the standardname is the name of the bit-file without extension
		String tempName = plotable.getSaveFile().getName();
		if (tempName.endsWith(".bit.gui")) {
			tempName = tempName.substring(0, tempName.length() - 8);
		}
		String standardFile = fch.getCurrentDirectory().getAbsolutePath() + File.separator + tempName;
		fch.setSelectedFile(new File(standardFile));

		// the filter just accepts csv,jpg,png,emf,pdf,ppm
		fch.addFileFilter("csv");
		fch.addFileFilter("jpg");
		fch.addFileFilter("png");
		fch.addFileFilter("eps");
		fch.addFileFilter("pdf");
		fch.addFileFilter("ppm");
		fch.addFileFilter("emf");
		fch.addFileFilter("svg");

		// default
		fch.setFileFilter(fch.getChoosableFileFilters()[2]);
		if (fch.showSaveDialog(chartPanel) == JFileChooser.APPROVE_OPTION)
			saveAs(fch.getSelectedFile());
	}

	private void showCandlestickPlot(boolean useAll, int functionIndex) {
		File outFile = new File(BIGInterface.getInstance().getTempPath() + File.separator
				+ "tmpPlot.gp");
		File imgFile = new File(BIGInterface.getInstance().getTempPath() + File.separator
				+ "tmpPlot.jpg");
		doExportCandlestickPlot(functionIndex, useAll, outFile, imgFile, 0, 0);
		final String shellFile = BIGInterface.getInstance().getTempPath() + File.separator
				+ "creategp.sh";
		BIGStrings sh = new BIGStrings();
		sh.add("cd " + outFile.getParent());
		sh.add("gnuplot " + outFile.getName());
		sh.add("rm -f " + outFile.getName());
		sh.add("rm -f " + shellFile);
		sh.saveToFile(shellFile);
		try {
			if (BIGExecute.getInstance().execute("sh " + shellFile) != 0) {
				JOptionPane.showMessageDialog(chartPanel,
						"Error showing candlestick plot:\n Could not execute shell script.", "Error",
						JOptionPane.ERROR_MESSAGE);
			} else
				new BIGImageDlg(plotable.getExtTitle(), imgFile);
		} catch (Exception ex) {
			JOptionPane.showMessageDialog(chartPanel, "Error showing candlestick plot:\n" + ex, "Error",
					JOptionPane.ERROR_MESSAGE);
		}
	}

	private void showValueTable() {
		BIGDataSet data1 = plotable.yAxis.get(plotable.displayedDataSets[0]).getData();
		double[][] values1 = data1.getValues();

		BIGDataSet data2 = null;
		double[][] values2 = null;
		// check for second y axis
		if (plotable.displayedDataSets.length == 2) {
			data2 = plotable.yAxis.get(plotable.displayedDataSets[1]).getData();
			values2 = data2.getValues();
		}
		// get correct x axis name
		String xAxisName = plotable.xData.Text;
		if (xAxisName == null)
			xAxisName = plotable.xData.TextDefault;
		if (xAxisName == null)
			xAxisName = "x";

		// some vectors/ strings for table creation
		Vector<Double> singleRow;
		Vector<Vector<Double>> rowData = new Vector<Vector<Double>>();
		Vector<String> columnNames = new Vector<String>();

		for (int x = 0; x < values1[0].length; x++) {
			singleRow = new Vector<Double>();
			for (int y = 0; y < data1.getSeriesCount() + 1; y++) {
				singleRow.add(values1[y][x]);
			}
			if (data2 != null) {
				// we do not need first entry (because it contains the x-values
				// that should be the same as these for first y-axis)
				for (int z = 1; z < data2.getSeriesCount() + 1; z++) {
					singleRow.add(values2[z][x]);
				}
			}
			rowData.add(singleRow);
		}

		// at first the x-axis-name
		columnNames.add(xAxisName);
		// then names for series of first y-axis
		for (String name : data1.getNames()) {
			columnNames.add(name);
		}
		// finally names for series of second y-axis (if exists!!!)
		if (data2 != null) {
			for (String name : data2.getNames()) {
				columnNames.add(name);
			}
		}
		BIGValueTable table = new BIGValueTable(new BIGValueTableModel(rowData, columnNames));
		JFrame frm = new JFrame();
		frm.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		frm.getContentPane().add(new JScrollPane(table));
		frm.setTitle(plotable.getExtTitle());
		frm.pack();
		frm.setVisible(true);
	}

	/**
	 * @param f
	 * @param selectedFile
	 * @param extension
	 */
	void saveAs(final File selectedFile) {
		String name = selectedFile.getName();
		String[] nameParts = name.split("\\.");
		String extension = nameParts[nameParts.length - 1];
		// saving csv
		if (extension.equals("csv")) {
			(new Thread() {
				@Override
				public void run() {
					YAxisData yAxis1 = plotable.yAxis.get(plotable.displayedDataSets[0]);
					List<String> names1 = yAxis1.getData().getNames();
					double[][] values1 = yAxis1.getData().getValues();
					YAxisData yAxis2 = null;
					List<String> names2 = new Vector<String>();
					double[][] values2 = new double[0][];
					// save data
					// more then 1 yaxis
					if (plotable.displayedDataSets.length > 1) {
						yAxis2 = plotable.yAxis.get(plotable.displayedDataSets[1]);
						names2 = yAxis2.getData().getNames();
						values2 = yAxis2.getData().getValues();
					}
					// names of the xaxis and functions
					String[] theNames = new String[names1.size() + names2.size() + 1];
					theNames[0] = plotable.xData.Text;
					for (int i = 0; i < names1.size(); i++) {
						theNames[i + 1] = names1.get(i) + yAxis1.Text;
					}
					for (int i = 0; i < names2.size(); i++) {
						theNames[i + 1 + names1.size()] = names2.get(i) + yAxis2.Text;
					}
					// preparig the values
					double[][] theValues = new double[values1.length + values2.length - 1][values1[0].length];
					// write values (copy)
					// if you want to improve it use System.arrayCopy?
					for (int i = 0; i < values1.length; i++) {
						for (int j = 0; j < values1[0].length; j++) {
							theValues[i][j] = values1[i][j];
						}
					}
					for (int i = 1; i < values2.length; i++) {
						for (int j = 0; j < values2[0].length; j++) {
							theValues[i + values1.length - 1][j] = values2[i][j];
						}
					}
					// another info
					String[] additionalInfo = {plotable.getExtTitle()};
					// start export
					startFileExporting(selectedFile, this);
					system.BIGUtility.saveToCsv(selectedFile, theValues, theNames, additionalInfo);
				}
			}).start();
			return;
		}
		// used sth else then int for resolution
		boolean threwException = false;
		// now temp, not temp will be final
		int tempWidth = 0, tempHeight = 0;
		if ((extension.equals("jpg")) || (extension.equals("png"))) {
			do {
				threwException = false;
				// building option dialog
				Object[][] message = new Object[2][2];
				message[0][0] = new JLabel("Width");
				message[0][1] = new JTextField("" + BIGInterface.getInstance().getSaveWidth());
				message[1][0] = new JLabel("Height");
				message[1][1] = new JTextField("" + BIGInterface.getInstance().getSaveHeight());
				String[] options = new String[2];
				options[0] = "Okay";
				options[1] = "Cancel";
				int result = JOptionPane.showOptionDialog(chartPanel, message, "Please insert your data",
						JOptionPane.DEFAULT_OPTION, JOptionPane.INFORMATION_MESSAGE, null, options, options[0]);
				// if cancel was pressed
				if (result == 1)
					return;
				// okay was pressed
				// get height
				try {
					tempHeight = (new Integer(((JTextField) message[1][1]).getText())).intValue();
				}
				// not parseable
				catch (NumberFormatException ex) {
					threwException = true;
				}
				// get width
				try {
					tempWidth = (new Integer(((JTextField) message[0][1]).getText())).intValue();
				}
				// not parseable
				catch (NumberFormatException ex) {
					threwException = true;
				}
				// should have the values, which are expected
				if (threwException) {
					JOptionPane
							.showMessageDialog(chartPanel, "Please select the image-size in integer value");
				}
			} while (threwException);
		}
		// final height and width
		final int height = tempHeight;
		final int width = tempWidth;
		// will be stored and used next time when saving
		BIGInterface.getInstance().setSaveHeight(height);
		BIGInterface.getInstance().setSaveWidth(width);
		// writing jpg-file
		if (extension.equals("jpg")) {
			(new Thread() {
				@Override
				public void run() {
					// display in status label
					startFileExporting(selectedFile, this);
					boolean cme = false;
					try {
						do {
							cme = false;
							// save
							try {
								org.jfree.chart.ChartUtilities.saveChartAsJPEG(selectedFile, chart, width, height);
							} catch (java.util.ConcurrentModificationException exc) {
								cme = true;
							}

						} while (cme);
					} catch (IOException ex1) {
						System.err.println(ex1);
					}
				}
			}).start();
			return;
		}
		// writing png-file
		if (extension.equals("png")) {
			(new Thread() {
				@Override
				public void run() {
					// display in status label
					startFileExporting(selectedFile, this);
					boolean cme = false;
					try {
						do {
							cme = false;
							// save
							try {
								org.jfree.chart.ChartUtilities.saveChartAsPNG(selectedFile, chart, width, height);
							} catch (java.util.ConcurrentModificationException exc) {
								cme = true;
							}
						} while (cme);
					} catch (IOException ex1) {
						System.err.println(ex1);
					}

				}
			}).start();
			return;

		} // writing eps-file (can be very large when using large numbers of shapes)
		if (extension.equals("eps")) {
			(new Thread() {
				@Override
				public void run() {
					// status label
					startFileExporting(selectedFile, this);

					boolean notDone = true;
					while (notDone) {
						try {
							// write to...
							FileOutputStream out = new FileOutputStream(selectedFile);
							// freehep export
							PSGraphics2D eg = new PSGraphics2D(out, chartPanel.getSize());
							eg.writeHeader();

							chartPanel.paintComponent(eg);
							eg.writeTrailer();
							eg.closeStream();
							out.close();
							notDone = false;

						} catch (Exception e) {
							// concurrent modification :/
							// can happen when painted to screen and file
							// at the same time
						}
					}
				}
			}).start();
			return;
		}
		// writing pdf-file
		if (extension.equals("pdf")) {
			(new Thread() {
				@Override
				public void run() {
					// status label
					startFileExporting(selectedFile, this);
					boolean notDone = true;
					while (notDone) {
						try {
							// open file
							FileOutputStream out = new FileOutputStream(selectedFile);
							// freehep export
							PDFGraphics2D eg = new PDFGraphics2D(out, chartPanel.getSize());
							/*
							 * eg.writeHeader() ; eg.writeTrailer() ; eg.closeStream() ; out.close() ;
							 */
							Properties p = new Properties();
							p.setProperty("PageSize", "A4");
							eg.setProperties(p);

							eg.startExport();
							// eg.openPage( chartPanel );
							// eg.writeHeader();
							// chartPanel.paintComponent( eg );
							chartPanel.print(eg);
							// eg.writeTrailer();
							// eg.closePage();
							eg.endExport();
							out.close();

							// chartPanel.paintComponent( eg ) ;

							notDone = false;
						} catch (Exception e) {
							e.printStackTrace();
						}
					}
				}
			}).start();
			return;
		}

		// writing ppm-file
		if (extension.equals("ppm")) {
			(new Thread() {
				@Override
				public void run() {
					startFileExporting(selectedFile, this);
					boolean notDone = true;
					while (notDone) {
						try {
							FileOutputStream out = new FileOutputStream(selectedFile);
							BufferedImage i = new BufferedImage((int) chartPanel.getSize().getWidth(),
									(int) chartPanel.getSize().getHeight(), BufferedImage.TYPE_3BYTE_BGR);

							chartPanel.paintComponent(i.createGraphics());
							PPMEncoder pme = new PPMEncoder(i, out);
							pme.encode();
							out.close();

							notDone = false;
						} catch (Exception e) {

						}
					}
				}
			}).start();
			return;
		}

		// writing emf-file
		if (extension.equals("emf")) {
			(new Thread() {
				@Override
				public void run() {
					try {
						FileOutputStream out = new FileOutputStream(selectedFile);
						ByteArrayOutputStream bas = new ByteArrayOutputStream();
						startFileExporting(selectedFile, this);
						org.freehep.graphicsio.emf.EMFGraphics2D gra = new org.freehep.graphicsio.emf.EMFGraphics2D(
								bas, chartPanel) {
							@Override
							public void writeBackground() {}

						};
						gra.startExport();

						boolean notDone = true;
						while (notDone) {
							try {
								chartPanel.paintComponent(gra);
								notDone = false;
							} catch (Exception e) {

							}
						}
						gra.endExport();
						bas.writeTo(out);
						gra.closeStream();
						bas.close();
						out.close();

					} catch (IOException ex1) {
						System.err.println(ex1);
					}
				}
			}).start();
			return;
		}

		// writing svg-file
		if (extension.equals("svg")) {
			(new Thread() {
				@Override
				public void run() {
					try {
						FileOutputStream out = new FileOutputStream(selectedFile);
						ByteArrayOutputStream bas = new ByteArrayOutputStream();
						startFileExporting(selectedFile, this);
						SVGGraphics2D gra = new SVGGraphics2D(bas, chartPanel) {
							@Override
							public void writeBackground() {}
						};
						gra.startExport();

						boolean notDone = true;
						while (notDone) {
							try {
								chartPanel.paintComponent(gra);
								notDone = false;
							} catch (Exception e) {}
						}
						gra.endExport();
						bas.writeTo(out);
						gra.closeStream();
						bas.close();
						out.close();

					} catch (IOException ex1) {
						System.err.println(ex1);
					}
				}
			}).start();
			return;
		}
	}

}

/*****************************************************************************
 * Log-History
 */
