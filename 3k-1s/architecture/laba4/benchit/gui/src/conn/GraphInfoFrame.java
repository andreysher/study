package conn;

import java.awt.*;

import javax.swing.*;

import plot.data.BIGResultMixer;

public class GraphInfoFrame extends JFrame {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor
	 * 
	 * @param g the Graph to show Infos about in this frame
	 * @todo could be done with a JTable
	 */
	public GraphInfoFrame(Graph g) {
		// identifiers
		String ident[] = g.settings[0];
		JScrollPane content = new JScrollPane();
		JPanel panel = new JPanel();
		// layout
		GridBagLayout gridbag = new GridBagLayout();
		panel.setLayout(gridbag);
		GridBagConstraints gc = new GridBagConstraints();
		gc.anchor = GridBagConstraints.WEST;
		int i;
		gc.gridx = 0;
		gc.gridy = 0;
		// first line: name
		panel.add(new JLabel(g.graphName), gc);
		gc.gridy = 1;
		// following lines:
		// for each setting
		// identifier setting-for-identifier
		panel.add(new JLabel("" + g.graphID), gc);
		for (i = 0; i < ident.length; i++) {
			gc.gridx = 0;
			gc.gridy = i + 1;
			panel.add(new JLabel(ident[i]), gc);
			gc.gridx = 1;
			panel.add(new JLabel(g.getSetting(ident[i])), gc);
		}
		BIGResultMixer m = new BIGResultMixer(null, g.getGraphName());
		m.addFunction(g);
		gc.gridy = i + 1;
		gc.gridx = 0;
		gc.gridwidth = 2;
		panel.add(m.getPlot().getComponentAt(0), gc);
		content.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		content.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
		content.getVerticalScrollBar().setUnitIncrement(15);
		double[] temp = {.5, .5};
		// +3: one because an array starts with 0 and one because of the buttons and one because of
		// (name)(id)
		double[] temp2 = new double[i + 4];
		for (int j = 0; j < temp2.length; j++) {
			temp2[i] = 1 / temp2.length;
		}
		gridbag.columnWeights = temp;
		gridbag.rowWeights = temp2;
		content.setViewportView(panel);
		setContentPane(content);
		pack();
		setVisible(true);
	}

	/**
	 * Constructor
	 * 
	 * @param g the Graph to show Infos about in this frame
	 * @todo could be done with a JTable
	 */
	public GraphInfoFrame(Graph[] g) {
		if (g == null)
			return;
		if (g.length == 0)
			return;
		// identifiers (see above)
		String ident[] = g[0].settings[0];
		JScrollPane content = new JScrollPane();
		JPanel panel = new JPanel();
		// layout
		GridBagLayout gridbag = new GridBagLayout();
		panel.setLayout(gridbag);
		GridBagConstraints gc = new GridBagConstraints();
		gc.anchor = GridBagConstraints.WEST;
		int id;
		gc.gridx = 1;
		gc.gridy = 0;
		// first line all functions/graphs (first row free for identifiers)
		for (int k = 0; k < g.length; k++) {
			gc.gridx = k + 1;
			panel.add(new JLabel(g[k].graphName), gc);
		}
		gc.gridy = 1;
		// following lines: identifiers and settings
		for (id = 0; id < ident.length; id++) {
			gc.gridx = 0;
			gc.gridy = id + 1;
			panel.add(new JLabel(ident[id]), gc);
			// difference means: if there are different settings for same identifiers
			// it is shown with a blue foreground
			boolean difference = false;
			String setting = g[0].getSetting(ident[id]);
			// System.err.println("Processing identifier " + ident[id] + "\nSetting of Graph 0 is " +
			// setting + "\n");
			for (int k = 1; k < g.length; k++) {
				// System.err.println("Setting of Graph " + k + " is " + g[ k ].getSetting( ident[id] ) +
				// "\n");
				if (!g[k].getSetting(ident[id]).equals(setting)) {
					difference = true;
				}
			}

			for (int index = 0; index < g.length; index++) {
				gc.gridx = index + 1;
				JLabel lab = new JLabel(g[index].getSetting(ident[id]));
				if (difference) {
					lab.setForeground(java.awt.Color.BLUE);
				}
				panel.add(lab, gc);
			}
		}
		BIGResultMixer m = new BIGResultMixer(null, "Comparison");

		for (int j = 0; j < g.length; j++) {
			m.addFunction(g[j]);
		}
		gc.gridy = id + 1;
		gc.gridx = 0;
		gc.gridwidth = g.length + 1;

		/*
		 * JTabbedPane jtp = m.getPlot(); int tabcnt = jtp.getTabCount(); System.err.println("JTabbebPane contains " +
		 * tabcnt + " tabs" ); for (int index=0; index<tabcnt; index++) { System.err.println("Tab Title == " +
		 * jtp.getTitleAt(index)); } Component comp = m.getPlot().getComponentAt(0);
		 * System.err.println("JTabbebPane contains " + jtp.getTabCount() + " tabs after getComponent()" );
		 * panel.add(comp,gc);
		 */

		panel.add(m.getPlot().getComponentAt(0), gc);

		content.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		content.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		content.getVerticalScrollBar().setUnitIncrement(15);
		double[] temp = {.5, .5};
		// +3: one because an array starts with 0 and one because of the buttons and one because of
		// (name)(id)
		double[] temp2 = new double[id + 4];
		for (int j = 0; j < temp2.length; j++) {
			temp2[id] = 1 / temp2.length;
		}
		gridbag.columnWeights = temp;
		gridbag.rowWeights = temp2;
		content.setViewportView(panel);
		panel.setBackground(java.awt.Color.WHITE);
		setContentPane(content);
		pack();
		setVisible(true);
	}

}
