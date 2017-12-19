package system;

import java.util.Vector;

import javax.swing.table.AbstractTableModel;

public class BIGValueTableModel extends AbstractTableModel {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public BIGValueTableModel(Vector<?> rowData, Vector<?> columnNames) {
		this.rowData = rowData;
		this.columnNames = columnNames;
	}

	/**
	 * Returns the number of columns in table
	 * 
	 * @return int - number of columns
	 */
	public int getColumnCount() {
		return columnNames.size();
	}

	/**
	 * Returns number of rows in table
	 * 
	 * @return int - number of rows
	 */
	public int getRowCount() {
		return rowData.size();
	}

	/**
	 * Returns the value of table cell indexed by <code>row</code> and <code>col</code>
	 * 
	 * @return Object - value of indexed table cell
	 */
	public Object getValueAt(int row, int col) {
		Double dbl = (Double) ((Vector<?>) rowData.elementAt(row)).elementAt(col);
		if (dbl.doubleValue() < 0.0)
			// illegal measurement -> return NaN (Not a Number)
			return new Double(Double.NaN);
		// otherwise return this value
		return dbl;
	}

	/**
	 * Returns header name of specified column
	 * 
	 * @return String - name of column
	 */
	@Override
	public String getColumnName(int columnIndex) {
		return columnNames.get(columnIndex).toString();
	}

	/**
	 * Returns class of values in specified column
	 * 
	 * @return Class - class of values in column
	 */
	@Override
	public Class<Double> getColumnClass(int columnIndex) {
		return Double.class;
	}

	// table values and table headers
	Vector<?> rowData, columnNames;

}
