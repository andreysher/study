package gui;

import java.awt.*;

import javax.swing.*;

/**
 * <p>
 * Überschrift: BenchIT
 * </p>
 * <p>
 * Beschreibung:
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

public class BIGInsetsPanel extends JPanel {
	private static final long serialVersionUID = 1L;
	Insets i = null;
	JSpinner spinLeft;
	JSpinner spinTop;
	JSpinner spinBottom;
	JSpinner spinRight;

	public BIGInsetsPanel() {
		this.i = new Insets(0, 0, 0, 0);
		init();
	}

	public BIGInsetsPanel(Insets i) {
		this.i = i;
		init();
	}

	private void init() {
		setLayout(new GridLayout(1, 8));
		this.add(new JLabel("top:"));
		spinTop = new JSpinner(new SpinnerNumberModel(i.top, 0, 500, 1));
		this.add(spinTop);
		this.add(new JLabel("left:"));
		spinLeft = new JSpinner(new SpinnerNumberModel(i.left, 0, 500, 1));
		this.add(spinLeft);
		this.add(new JLabel("bottom:"));
		spinBottom = new JSpinner(new SpinnerNumberModel(i.bottom, 0, 500, 1));
		this.add(spinBottom);
		this.add(new JLabel("right:"));
		spinRight = new JSpinner(new SpinnerNumberModel(i.right, 0, 500, 1));
		this.add(spinRight);
	}

	public void setValues(Insets i) {
		spinTop.setValue(new Integer(i.top));
		spinLeft.setValue(new Integer(i.left));
		spinBottom.setValue(new Integer(i.bottom));
		spinRight.setValue(new Integer(i.right));
		revalidate();
	}

	public Insets getValues() {
		i.top = (Integer) spinTop.getValue();
		i.left = (Integer) spinLeft.getValue();
		i.right = (Integer) spinRight.getValue();
		i.bottom = (Integer) spinBottom.getValue();
		return i;
	}

}
