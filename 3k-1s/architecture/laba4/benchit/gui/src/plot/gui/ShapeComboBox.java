package plot.gui;

import java.awt.*;

import javax.swing.*;

/**
 * <p>
 * Überschrift: BenchIT
 * </p>
 * <p>
 * Beschreibung: Maybe someday, you can select shapes with this
 * </p>
 * <p>
 * Copyright: Copyright (c) 2004
 * </p>
 * <p>
 * Organisation: ZHR TU Dresden
 * </p>
 * 
 * @author Robert Schoene
 * @version 1.0
 */
class ShapeComboBox extends JComboBox<String> {
	private static final long serialVersionUID = 1L;
	final Shape[] s;

	// public ShapeComboBox()
	public ShapeComboBox(Shape[] shapes) {
		s = shapes;
		// ------------------------------------------------------------------------------------
		for (int i = 0; i < s.length; i++) {
			addItem(i + ":");
		}
		setSelectedIndex(0);
		setRenderer(new ListCellRenderer<Object>() {
			public Component getListCellRendererComponent(final JList<?> list, Object value,
					final int index, final boolean isSelected, final boolean cellHasFocus) {
				JLabel lab = new JLabel();
				if (index >= 0) {
					lab = new JLabel(s[index].toString()) {
						private static final long serialVersionUID = 1L;

						@Override
						public void paintComponent(Graphics g) {
							int realindex = index;
							g.translate(10, 10);
							((Graphics2D) g).scale(2, 2);
							if (index == -1) {
								realindex = 0;
							}
							if (realindex >= 0) {
								((Graphics2D) g).setBackground(isSelected ? Color.red : Color.white);
								((Graphics2D) g).setColor(isSelected ? Color.white : Color.black);
								((Graphics2D) g).draw(s[realindex]);
								// ( ( Graphics2D ) g ).fill( s[ realindex ] ) ;
							}
						}
					};
				}
				// width ,height
				lab.setPreferredSize(new Dimension(20, 20));
				return lab;
			}

		});
	}

	@Override
	public void paint(Graphics g) {
		super.paint(g);
		g.translate(10, 10);
		((Graphics2D) g).scale(2, 2);
		((Graphics2D) g).draw(s[getSelectedIndex()]);
		// ( ( Graphics2D ) g ).fill( s[ getSelectedIndex() ] ) ;

	}

	// -----------------------------------------------------------------------------------
	/**
	 * Returns the standard shape sequence
	 * 
	 * @return - the shape sequence
	 */
	public Shape[] getShapeSequence() {
		return s;
	}

	/**
	 * Returns the shape at the given index
	 * 
	 * @param index - the position of the requested shape
	 * @return - the shape at the given position
	 */
	public Shape getShape(int index) {
		return s[index];
	}
	// -----------------------------------------------------------------------------------
}