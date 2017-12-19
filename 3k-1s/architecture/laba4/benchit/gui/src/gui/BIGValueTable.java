package gui;

import java.awt.*;

import javax.swing.JTable;
import javax.swing.table.DefaultTableCellRenderer;

import system.BIGValueTableModel;

public class BIGValueTable extends JTable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor of table. Sets appropriate cell renderer.
	 * 
	 * @param vtm
	 */
	public BIGValueTable(BIGValueTableModel vtm) {
		super(vtm);
		setDefaultRenderer(String.class, new BIGValueTableCellRenderer());
		setDefaultRenderer(Double.class, new BIGValueTableCellRenderer());
	}

	/**
	 * <p>
	 * Überschrift: BenchIT
	 * </p>
	 * <p>
	 * Beschreibung: Implementaion of table cell renderer
	 * </p>
	 * <p>
	 * Copyright: Copyright (c) 2008
	 * </p>
	 * <p>
	 * Organisation: ZIH TU Dresden
	 * </p>
	 * 
	 * @author Ronny Tschueter
	 * @version 1.0
	 */
	public class BIGValueTableCellRenderer extends DefaultTableCellRenderer {
		private static final long serialVersionUID = 1L;

		/**
		 * Adapted implementation of cell rendering. Every second row gets a blue background color.
		 * 
		 * @param table - the JTable that is asking the renderer to draw
		 * @param value - the value of the cell to be rendered
		 * @param isSelected - true if the cell is to be rendered with the selection highlighted; otherwise false
		 * @param hasFocus - if true, render cell appropriately
		 * @param row - the row index of the cell being drawn. When drawing the header, the value of row is -1
		 * @param column - the column index of the cell being drawn
		 * @return Component - display component of table cell
		 */
		@Override
		public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
				boolean hasFocus, int row, int column) {
			Component comp;
			Double d = (Double) value;

			if (d.doubleValue() < 0.0) {
				value = "N/A";
			}
			comp = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
			if (row % 2 == 0) {
				comp.setBackground(new Color(0, 0, 255, 20));
			} else {
				comp.setBackground(new Color(250, 250, 250, 20));
			}
			return comp;
		}
	}

}
