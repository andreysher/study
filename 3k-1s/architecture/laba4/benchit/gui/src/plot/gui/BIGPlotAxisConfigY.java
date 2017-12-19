package plot.gui;

import java.awt.GridBagConstraints;
import java.awt.event.*;
import java.util.*;

import javax.swing.*;

import plot.data.*;

public class BIGPlotAxisConfigY extends BIGPlotAxisConfig {
	private JCheckBox cbScaleLegends;
	private JCheckBox cbFillShapes;
	private JCheckBox cbShowLines;

	protected final class ScaleLegendsListener implements ActionListener {
		public void actionPerformed(ActionEvent evt) {
			YAxisData yAxis = (YAxisData) getAxis();
			if (yAxis == null)
				return;

			yAxis.getData().setPreNamesSelected(((JCheckBox) evt.getSource()).isSelected());
		}
	}

	public BIGPlotAxisConfigY(BIGPlotable plotable, int axisNum) {
		super(plotable, axisNum);
	}

	@Override
	public void createElements() {
		super.createElements();
		cbScaleLegends = new JCheckBox("scale legends");
		cbScaleLegends.addActionListener(new ScaleLegendsListener());
		cbFillShapes = new JCheckBox("fill points for functions on " + axisName + " axis");
		cbShowLines = new JCheckBox("show lines for functions on " + axisName + " axis");
	}

	@Override
	public List<LinkedHashMap<JComponent, Integer>> getElementLayout() {
		List<LinkedHashMap<JComponent, Integer>> layout = super.getElementLayout();
		LinkedHashMap<JComponent, Integer> row = null;
		for (LinkedHashMap<JComponent, Integer> r : layout)
			for (JComponent c : r.keySet())
				if (c.equals(cbScaleAxisText)) {
					row = r;
					break;
				}
		if (row == null) {
			row = new LinkedHashMap<JComponent, Integer>();
			layout.add(row);
		}
		row.put(cbScaleLegends, GridBagConstraints.REMAINDER);

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(cbFillShapes, 2);
		row.put(cbShowLines, GridBagConstraints.REMAINDER);
		layout.add(row);

		return layout;
	}

	@Override
	public void reset() {
		super.reset();
		YAxisData yAxis = (YAxisData) getAxis();
		if (yAxis == null)
			return;

		yAxis.getData().resetAchsisValues();
		yAxis.getData().setPreNamesSelected(false);
		plotable.fillShapes[axisNum] = false;
		plotable.drawLines[axisNum] = false;
	}

	@Override
	public void loadValues() {
		super.loadValues();
		YAxisData yAxis = (YAxisData) getAxis();
		if (yAxis == null)
			return;

		cbScaleLegends.setSelected(yAxis.getData().getPreNamesSelected());
		cbFillShapes.setSelected(plotable.fillShapes[axisNum]);
		cbShowLines.setSelected(plotable.drawLines[axisNum]);
	}

	@Override
	public void setValues() {
		super.setValues();
		AxisData axis = getAxis();
		if (axis == null)
			return;
		plotable.fillShapes[axisNum] = cbFillShapes.isSelected();
		plotable.drawLines[axisNum] = cbShowLines.isSelected();
	}

}
