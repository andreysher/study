package plot.gui;

import gui.BIGTextField;

import java.awt.event.*;
import java.util.*;

import javax.swing.*;

import plot.data.*;
import system.BIGUtility;

/**
 * Config for one axis
 * 
 * @author alex
 */
public class BIGPlotAxisConfig {
	// number of columns for a small textfield (e.g. inXmin, inYmax ...)
	protected static final int SMALL_TEXTFIELD_COLUMNS = 8;
	// number of columns for a smaller textfield (e.g. inXticks, inYticks ...)
	protected static final int SMALLER_TEXTFIELD_COLUMNS = 4;

	protected final BIGPlotable plotable;
	protected final int axisNum;
	protected final String axisName;

	protected BIGTextField tfMin;
	protected BIGTextField tfMax;
	protected BIGTextField tfTicks;
	protected JTextField tfAxisText;
	protected BIGTextField tfLogarithmic;
	protected JComboBox<String> cbPre;
	protected JCheckBox cbScaleAxisText;
	protected JTextField tfNumberFormat;

	/**
	 * @param axisNum -1 for xAxis, 0 for y1, 1 for y2...
	 */
	public BIGPlotAxisConfig(BIGPlotable plotable, int axisNum) {
		this.plotable = plotable;
		this.axisNum = axisNum;
		axisName = (axisNum < 0) ? "x" : "y" + (axisNum + 1);
	}

	protected AxisData getAxis() {
		if (axisNum < 0)
			return plotable.xData;
		if (axisNum < plotable.displayedDataSets.length)
			return plotable.yAxis.get(plotable.displayedDataSets[axisNum]);
		return null;
	}

	protected final class PreListener implements ItemListener {
		public void itemStateChanged(ItemEvent evt) {
			AxisData axis = getAxis();
			if (axis == null)
				return;
			int pre = ((JComboBox<?>) evt.getSource()).getSelectedIndex();
			if (axis.getPre() == pre)
				return;// Nothing to change
			axis.setPre(pre);
			if (axisNum < 0) {
				// X Axis
				plotable.calculateNumberFormatX();
			} else {
				plotable.calculateNumberFormatY((YAxisData) axis);
			}

			tfMin.setValue(BIGUtility.round(axis.getMin(), 6));
			tfMax.setValue(BIGUtility.round(axis.getMax(), 6));

			tfAxisText.setText(axis.getAxisText());
			tfNumberFormat.setText(axis.NumberFormat);
		}
	}

	protected final class ScaleAxisTextListener implements ActionListener {
		public void actionPerformed(ActionEvent evt) {
			AxisData axis = getAxis();
			if (axis == null)
				return;
			axis.scaleAxisText = ((JCheckBox) evt.getSource()).isSelected();
			// this solution is not very pretty
			// a better solution should use a listener, who reacts on a specific event
			tfAxisText.setText(axis.getAxisText());
		}
	}

	public void createElements() {
		tfMin = new BIGTextField(0, BIGTextField.DOUBLE);
		tfMax = new BIGTextField(0, BIGTextField.DOUBLE);
		tfTicks = new BIGTextField(0, BIGTextField.INTEGER);
		tfAxisText = new JTextField();
		tfLogarithmic = new BIGTextField(0, BIGTextField.INTEGER);
		cbPre = new JComboBox<String>(BIGDataSet.sets);
		cbPre.addItemListener(new PreListener());
		cbScaleAxisText = new JCheckBox("scale " + axisName + " axis text");
		cbScaleAxisText.addActionListener(new ScaleAxisTextListener());
		tfNumberFormat = new JTextField();

		tfMin.setColumns(SMALL_TEXTFIELD_COLUMNS);
		tfMax.setColumns(SMALL_TEXTFIELD_COLUMNS);
		tfTicks.setColumns(SMALLER_TEXTFIELD_COLUMNS);
	}

	public List<LinkedHashMap<JComponent, Integer>> getElementLayout() {
		List<LinkedHashMap<JComponent, Integer>> layout = new ArrayList<LinkedHashMap<JComponent, Integer>>();
		LinkedHashMap<JComponent, Integer> row;

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel(axisName + " min:"), 1);
		row.put(tfMin, 1);
		row.put(new JLabel(axisName + " max:"), 1);
		row.put(tfMax, 1);
		row.put(new JLabel(axisName + " ticks:"), 1);
		row.put(tfTicks, 1);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel(axisName + " axis text:"), 1);
		row.put(tfAxisText, 3);
		row.put(new JLabel(axisName + " axis logbase:"), 1);
		row.put(tfLogarithmic, 1);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel(axisName + " scaling factor:"), 1);
		row.put(cbPre, 1);
		row.put(cbScaleAxisText, 2);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel(axisName + " number format:"), 1);
		row.put(tfNumberFormat, 1);
		layout.add(row);

		return layout;
	}
	public void reset() {
		AxisData axis = getAxis();
		if (axis == null)
			return;
		axis.reset();
	}

	public void loadValues() {
		AxisData axis = getAxis();
		if (axis == null)
			return;

		tfMin.setValue(axis.getMin());
		tfMax.setValue(axis.getMax());
		tfTicks.setValue(axis.Ticks);
		tfAxisText.setText(axis.getAxisText());
		tfLogarithmic.setValue(axis.Log);
		cbPre.setSelectedIndex(axis.getPre());
		cbScaleAxisText.setSelected(axis.scaleAxisText);
		tfNumberFormat.setText(axis.NumberFormat);
	}

	public void setValues() {
		AxisData axis = getAxis();
		if (axis == null)
			return;

		// Set pre first as min/max are relative!
		axis.setPre(cbPre.getSelectedIndex());
		axis.setMin(tfMin.getDoubleValue());
		axis.setMax(tfMax.getDoubleValue());
		axis.Ticks = tfTicks.getIntegerValue();
		axis.scaleAxisText = cbScaleAxisText.isSelected();
		axis.Text = tfAxisText.getText();
		if (axis.scaleAxisText && axis.Text.startsWith(axis.getPrefix())) {
			// axis text is scaled
			axis.Text = tfAxisText.getText().substring(axis.getPrefix().length());
		}
		axis.Log = tfLogarithmic.getIntegerValue();
		axis.NumberFormat = tfNumberFormat.getText();
	}
}
