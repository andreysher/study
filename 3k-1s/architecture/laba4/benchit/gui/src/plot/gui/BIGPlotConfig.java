package plot.gui;

import gui.*;

import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

import javax.swing.*;

import org.jfree.chart.JFreeChart;

import plot.data.*;

/**
 * Config dialog for BIGPlot
 * 
 * @author alex
 */
public class BIGPlotConfig {
	private final class Spacer extends JPanel {
		private static final long serialVersionUID = 1L;
		@Override
		public Dimension getMinimumSize() {
			return getPreferredSize();
		}
		@Override
		public Dimension getPreferredSize() {
			return new Dimension(super.getPreferredSize().width, 10);
		}
		@Override
		public Dimension getMaximumSize() {
			return getPreferredSize();
		}
	}

	/**
	 * A button with fixed width
	 * 
	 * @author alex
	 */
	private final class BIGFixedButton extends JButton {
		private static final long serialVersionUID = 1L;
		private BIGFixedButton(Action a) {
			super(a);
		}
		private BIGFixedButton(String text) {
			super(text);
		}
		@Override
		public Dimension getMinimumSize() {
			return getPreferredSize();
		}
		@Override
		public Dimension getPreferredSize() {
			return new Dimension(BIGPlotConfig.BUTTON_SIZE, super.getPreferredSize().height);
		}
		@Override
		public Dimension getMaximumSize() {
			return getPreferredSize();
		}
	}

	private final class BIGFixedToggleButton extends JToggleButton {
		private static final long serialVersionUID = 1L;
		private BIGFixedToggleButton(Action a) {
			super(a);
		}
		@Override
		public Dimension getMinimumSize() {
			return getPreferredSize();
		}
		@Override
		public Dimension getPreferredSize() {
			return new Dimension(BIGPlotConfig.BUTTON_SIZE, super.getPreferredSize().height);
		}
		@Override
		public Dimension getMaximumSize() {
			return getPreferredSize();
		}
	}

	// button size
	protected static final int BUTTON_SIZE = 500;

	private final BIGPlotable plotable;

	private JTextField commentTextField;
	private BIGTextField commentX;
	private BIGTextField commentY;
	private BIGInsetsPanel insetsPanel;
	private JTextField titleTextField;
	private JToggleButton antiAliasedButton;
	// for drawing the cross hair lines following the mouse cursor
	private JToggleButton crosshairButton;
	public boolean currentAxisTrace = false;
	private BIGPlotAxisConfig[] axisConfig;
	private JPanel configPanel;
	private JButton resetButton;
	private JButton colorButton;
	private JButton fontButton;
	private BIGFontDialog fontDlg;
	private BIGColorDialog colorDlg;

	public BIGPlotConfig(BIGPlotable plotable) {
		this.plotable = plotable;
	}

	public JPanel getPanel() {
		return configPanel;
	}

	public void init(JTabbedPane parent, JFreeChart chart) {
		axisConfig = new BIGPlotAxisConfig[1 + plotable.displayedDataSets.length];
		axisConfig[0] = new BIGPlotAxisConfig(plotable, -1);
		for (int i = 1; i < axisConfig.length; i++)
			axisConfig[i] = new BIGPlotAxisConfigY(plotable, i - 1);
		createElements(parent, chart);
		doLayout();
	}

	protected void addComponent(JComponent container, JComponent component, GridBagLayout gridbag,
			GridBagConstraints c) {
		gridbag.setConstraints(component, c);
		container.add(component);
	}

	protected void setConstraints(GridBagConstraints c, int x, int y, int gw, int gh) {
		c.gridx = x;
		c.gridy = y;
		c.gridwidth = gw;
		c.gridheight = gh;
	}

	private int applyLayout(List<LinkedHashMap<JComponent, Integer>> layout, int row,
			GridBagLayout gridbag, GridBagConstraints c) {
		for (LinkedHashMap<JComponent, Integer> r : layout) {
			int col = 0;
			for (JComponent comp : r.keySet()) {
				setConstraints(c, col, row, r.get(comp), 1);
				addComponent(configPanel, comp, gridbag, c);
				col += r.get(comp);
			}
			row++;
		}
		return row;
	}

	public List<LinkedHashMap<JComponent, Integer>> getElementLayout() {
		List<LinkedHashMap<JComponent, Integer>> layout = new ArrayList<LinkedHashMap<JComponent, Integer>>();
		LinkedHashMap<JComponent, Integer> row;
		JPanel tmpPanel;

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel("title:"), 1);
		row.put(titleTextField, GridBagConstraints.REMAINDER);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel("comment:"), 1);
		row.put(commentTextField, GridBagConstraints.REMAINDER);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel("x-pos of comment (%):"), 1);
		row.put(commentX, 1);
		row.put(new JLabel("y-pos of comment (%):"), 1);
		row.put(commentY, 1);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		row.put(new JLabel("insets:"), 1);
		row.put(insetsPanel, GridBagConstraints.REMAINDER);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		tmpPanel = new JPanel();
		tmpPanel.add(resetButton, BorderLayout.CENTER);
		row.put(tmpPanel, GridBagConstraints.REMAINDER);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		tmpPanel = new JPanel();
		tmpPanel.add(crosshairButton, BorderLayout.CENTER);
		row.put(tmpPanel, GridBagConstraints.REMAINDER);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		tmpPanel = new JPanel();
		tmpPanel.add(antiAliasedButton, BorderLayout.CENTER);
		row.put(tmpPanel, GridBagConstraints.REMAINDER);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		tmpPanel = new JPanel();
		tmpPanel.add(colorButton, BorderLayout.CENTER);
		row.put(tmpPanel, GridBagConstraints.REMAINDER);
		layout.add(row);

		row = new LinkedHashMap<JComponent, Integer>();
		tmpPanel = new JPanel();
		tmpPanel.add(fontButton, BorderLayout.CENTER);
		row.put(tmpPanel, GridBagConstraints.REMAINDER);
		layout.add(row);

		return layout;
	}

	private void doLayout() {
		// layout of configtab
		GridBagLayout gridbag = new GridBagLayout();
		GridBagConstraints c = new GridBagConstraints();
		Insets insetsForGridbag = new Insets(5, 5, 5, 5);
		c.insets = insetsForGridbag;
		configPanel.setLayout(gridbag);
		c.fill = GridBagConstraints.BOTH;
		int row = 0;

		// building the config tab...
		for (BIGPlotAxisConfig cfg : axisConfig) {
			List<LinkedHashMap<JComponent, Integer>> layout = cfg.getElementLayout();
			row = applyLayout(layout, row, gridbag, c);
			// add a spacer between the sections
			setConstraints(c, 0, row, GridBagConstraints.REMAINDER, 1);
			addComponent(configPanel, new Spacer(), gridbag, c);
			row++;
		}

		applyLayout(getElementLayout(), row, gridbag, c);
	}

	private void createElements(JTabbedPane parent, JFreeChart chart) {
		configPanel = new JPanel();

		for (int i = 0; i < axisConfig.length; i++)
			axisConfig[i].createElements();

		titleTextField = new JTextField(0);

		commentTextField = new JTextField(0);
		commentX = new BIGTextField(1, BIGTextField.INTEGER);
		commentY = new BIGTextField(1, BIGTextField.INTEGER);

		insetsPanel = new BIGInsetsPanel();
		resetButton = new BIGFixedButton(new AbstractAction("Reset") {
			private static final long serialVersionUID = 1L;

			public void actionPerformed(ActionEvent e) {
				reset();
			}
		});

		// a button for disabling and enabling the 2 blue lines following the mouse
		crosshairButton = new BIGFixedToggleButton(new AbstractAction("Show Crosshair") {
			private static final long serialVersionUID = 1L;

			public void actionPerformed(ActionEvent e) {
				currentAxisTrace = ((JToggleButton) e.getSource()).isSelected();
			}
		});
		antiAliasedButton = new BIGFixedToggleButton(new AbstractAction("Draw Anti-Aliased") {
			private static final long serialVersionUID = 1L;

			public void actionPerformed(ActionEvent e) {
				plotable.drawAntiAliased = ((JToggleButton) e.getSource()).isSelected();
			}
		});

		colorButton = new BIGFixedButton("Change Colors, Names and Shapes for Single Functions");
		colorButton.setToolTipText("Set the color, names and shapes for the shown functions.");
		colorDlg = new BIGColorDialog(plotable, parent, chart);
		colorButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				setValues();
				colorDlg.show();
			}
		});
		// END COLOR - Select
		// start font

		fontButton = new BIGFixedButton("Set Plot Fonts");
		fontButton.setToolTipText("Set the fonts for this plot.");
		fontDlg = new BIGFontDialog(plotable, parent, new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				// setConfigValues();
			}
		});
		// when the button is pressed a frame will be opened, where fonts can be selected
		fontButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent aevt) {
				setValues();
				fontDlg.setVisible(true);
			}
		});
	}

	public void loadValues() {
		for (BIGPlotAxisConfig cfg : axisConfig)
			cfg.loadValues();
		commentTextField.setText(plotable.getAnnComment());
		commentX.setValue(plotable.getCommentXPercent());
		commentY.setValue(plotable.getCommentYPercent());

		titleTextField.setText(plotable.getTitle());
		insetsPanel.setValues(plotable.insets);
		insetsPanel.revalidate();
		crosshairButton.setSelected(currentAxisTrace);
		antiAliasedButton.setSelected(plotable.drawAntiAliased);
	}

	public void setValues() {
		for (BIGPlotAxisConfig cfg : axisConfig)
			cfg.setValues();
		// and the title
		plotable.setTitle(titleTextField.getText());
		// and for sure the annotation
		plotable.setAnnComment(commentTextField.getText());
		plotable.setCommentPos(commentX.getDoubleValue(), commentY.getDoubleValue());
		plotable.insets = insetsPanel.getValues();
	}

	private void reset() {
		for (BIGPlotAxisConfig cfg : axisConfig)
			cfg.reset();
		plotable.calculateScaleX();
		plotable.calculateNumberFormatX();
		for (YAxisData axis : plotable.yAxis) {
			plotable.calculateScaleY(axis);
			plotable.calculateNumberFormatY(axis);
		}
		for (BIGPlotAxisConfig cfg : axisConfig)
			cfg.loadValues();
		plotable.insets = plotable.getDefaultInsets();
		plotable.setTitle(plotable.getDefaultTitle());
		loadValues();
	}

}
