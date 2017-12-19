package plot.gui;

import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.*;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.DatasetChangeEvent;

import plot.data.*;
import system.BIGUtility;

public class BIGColorDialog {
	private final BIGPlotable plotable;
	private final JTabbedPane displayTabPane;
	private final JFreeChart chart;
	private int lastIndex = 0;

	public BIGColorDialog(BIGPlotable plotable, JTabbedPane displayTabPane, JFreeChart chart) {
		this.plotable = plotable;
		this.displayTabPane = displayTabPane;
		this.chart = chart;
	}

	public void show() {
		build();
	}

	private void build() {
		// get the names of the legends for 1st yaxis
		// this means its a refference! so if you change it here
		// it will also be changed in plotable
		final YAxisData yAxis1 = plotable.yAxis.get(plotable.displayedDataSets[0]);
		final YAxisData yAxis2 = (plotable.displayedDataSets.length > 1) ? plotable.yAxis
				.get(plotable.displayedDataSets[1]) : null;

		final List<String> names1 = yAxis1.getData().getNames();
		final List<String> names2 = (yAxis2 != null)
				? yAxis2.getData().getNames()
				: new Vector<String>();
		final List<String> names = new Vector<String>(names1);
		names.addAll(names2);

		// a colorchooser for choosing colors
		final JColorChooser jcc = new JColorChooser((Color) chart.getXYPlot().getRenderer()
				.getItemPaint(0, 0, 0));
		jcc.setToolTipText("Color for " + names.get(0));
		// set color
		jcc.getSelectionModel().addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent arg0) {
				Color color = jcc.getColor();
				XYPlot plot = chart.getXYPlot();
				if (lastIndex < names1.size()) {
					plot.getRenderer().setSeriesPaint(0, lastIndex, color);
					plot.datasetChanged(new DatasetChangeEvent(plot, yAxis1.getData()));
				} else {
					plot.getSecondaryRenderer().setSeriesPaint(0, lastIndex - names1.size(), color);
					plot.datasetChanged(new DatasetChangeEvent(plot, yAxis1.getData()));
				}
			}
		});

		// combobox for the shapes
		final ShapeComboBox shapeComboBox = new ShapeComboBox(BIGPlotRenderer.BIG_SHAPES);
		// a listener, that handles changes of the selected item
		shapeComboBox.addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent evt) {
				int index = ((ShapeComboBox) evt.getSource()).getSelectedIndex();
				Shape shape = ((ShapeComboBox) evt.getSource()).getShape(index);
				XYPlot plot = chart.getXYPlot();
				// draw the new shape
				if (lastIndex < names1.size()) {
					((BIGPlotRenderer) (plot.getRenderer())).setSeriesShape(lastIndex, shape);
					plot.datasetChanged(new DatasetChangeEvent(plot, yAxis1.getData()));
				} else {
					((BIGPlotRenderer) (chart.getXYPlot().getSecondaryRenderer())).setSeriesShape(lastIndex
							- names1.size(), shape);
					plot.datasetChanged(new DatasetChangeEvent(plot, yAxis2.getData()));
				}
			}
		});

		final JComboBox<String> jcbSelect = new JComboBox<String>(
				names.toArray(new String[names.size()]));
		jcbSelect.setEditable(true);
		jcbSelect.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				// which is the selected legend/function
				int j = jcbSelect.getSelectedIndex();
				if (j < 0) {
					j = lastIndex;
					names.set(j, (String) jcbSelect.getSelectedItem());
					if (j < names1.size()) {
						plotable.changeName(0, j, names.get(j));
					} else {
						plotable.changeName(1, j - names1.size(), names.get(j));
					}
					chart.getXYPlot().datasetChanged(
							new DatasetChangeEvent(chart.getXYPlot(), yAxis1.getData()));
					jcbSelect.removeAllItems();
					for (String name : names) {
						jcbSelect.addItem(name);
					}
					jcbSelect.setSelectedIndex(j);
					System.out.println("Name changed");
					return;
				}
				lastIndex = j;

				Shape actualShape;
				Color color;
				if (j < names1.size()) {
					color = yAxis1.Colors.get(j);
					actualShape = yAxis1.Shapes.get(j);
				} else {
					color = yAxis2.Colors.get(j - names1.size());
					actualShape = yAxis2.Shapes.get(j - names1.size());
				}
				// set the color for this function
				jcc.setColor(color);
				jcc.setToolTipText("Color for" + names.get(j));
				// rebuild the shape selection
				for (int index = 0; index < BIGPlotRenderer.BIG_SHAPES.length; index++) {
					if (BIGPlotRenderer.BIG_SHAPES[index].equals(actualShape)) {
						shapeComboBox.setSelectedIndex(index);
						break;
					}
				}
			}
		});

		JPanel scbPanel = new JPanel();
		scbPanel.setLayout(new FlowLayout());
		scbPanel.add(new JLabel("select a shape for actual function:"));
		scbPanel.add(shapeComboBox);

		JPanel buttonPanel = new JPanel();
		buttonPanel.setLayout(new FlowLayout());
		buttonPanel.add(new JLabel("select actual Function:"));
		buttonPanel.add(jcbSelect);

		// build a new frame, where the stuff can be selected
		JFrame jf = new JFrame("Color");
		JScrollPane content = new JScrollPane();
		JPanel main = new JPanel();

		GridBagLayout gridbag = new GridBagLayout();
		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.BOTH;
		c.weightx = 1.0;
		main.setLayout(gridbag);

		int row = 0;

		BIGUtility.setConstraints(c, 1, row, GridBagConstraints.REMAINDER, 1);
		BIGUtility.addComponent(main, buttonPanel, gridbag, c);
		row++;

		BIGUtility.setConstraints(c, 1, row, GridBagConstraints.REMAINDER, 1);
		BIGUtility.addComponent(main, jcc, gridbag, c);
		row++;

		// add the panel to main panel
		BIGUtility.setConstraints(c, 1, row, GridBagConstraints.REMAINDER, 1);
		BIGUtility.addComponent(main, scbPanel, gridbag, c);
		row++;

		content.setViewportView(main);

		jf.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		jf.setContentPane(content);
		jf.pack();
		jf.setVisible(true);
		jf.toFront();

		// show the plot
		displayTabPane.setSelectedIndex(0);
		displayTabPane.revalidate();
	}
}
