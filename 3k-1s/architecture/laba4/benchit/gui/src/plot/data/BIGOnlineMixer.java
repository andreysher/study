package plot.data;

import gui.BIGResultTree;

import javax.swing.*;
import javax.swing.tree.DefaultMutableTreeNode;

import conn.Graph;

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
public class BIGOnlineMixer extends BIGResultMixer {
	/**
	 * in principle this is a resultMixer with additional features
	 * 
	 * @param plotPanel JPanel plotPanel for this mixer
	 * @param textPanel JPanel textPanel for this mixer
	 * @param title String title for plot
	 */
	public BIGOnlineMixer(JPanel plotPanel, String title) {
		super(plotPanel, title);
		node = new DefaultMutableTreeNode(this);
	}

	/**
	 * use this to add a new graph received from somewhere (db or local)
	 * 
	 * @param g Graph
	 */
	@Override
	public boolean addFunction(Graph g) {
		// name
		String newFunctionRenamed = g.getGraphName();
		// if there is a function with the same name
		// add a number (starting with 1)
		int number = 1;
		while (namesAndGraphs.containsKey(newFunctionRenamed)) {
			newFunctionRenamed = g.getGraphName() + number;
			number++;
		}
		g.setGraphName(newFunctionRenamed);
		// add it
		return super.addFunction(g);
	}

	/**
	 * gets the popupMenu for a BRM-node
	 */
	@Override
	public JPopupMenu getBIGPopupMenu(final JFrame jf, final Object o) {
		// use the standard popup
		JPopupMenu menu = super.getBIGPopupMenu(jf, o);
		// and add compare if this rootNode is used
		if ((o instanceof BIGOnlineMixer)) {
			menu.addSeparator();
			JMenuItem i = new JMenuItem("Compare online measurements");
			i.addActionListener(new java.awt.event.ActionListener() {
				public void actionPerformed(java.awt.event.ActionEvent ae) {
					showCompareDialog();
				}
			});
			menu.add(i);

			// get all elements of pop-up menu
			MenuElement[] elements = menu.getSubElements();
			// search for the "create report" item
			for (int index = 0; index < elements.length; index++) {
				if (((JMenuItem) elements[index].getComponent()).getText().equals(
						BIGResultTree.createReportMenuText)) {
					menu.remove(index);
				}
			}

		}
		// or view info, if a leaf was clicked
		else {
			menu.addSeparator();
			JMenuItem i = new JMenuItem("View info");
			i.addActionListener(new java.awt.event.ActionListener() {
				public void actionPerformed(java.awt.event.ActionEvent ae) {
					if (namesAndGraphs.containsKey(o.toString())) {
						new conn.GraphInfoFrame(namesAndGraphs.get(o.toString()));
					}
				}
			});
			menu.add(i);
		}
		return menu;
	}

	/**
	 * compare measurements (same options are black, differences blue)
	 */
	private void showCompareDialog() {
		// names of functions
		String[] s = new String[getTotalFunctions()];
		int where = 0;
		for (int i = 0; i < yAxis.size(); i++) {
			for (int j = 0; j < yAxis.get(i).getData().getSeriesCount(); j++) {
				s[where] = yAxis.get(i).getData().getSeriesName(j);
				where++;
			}
		}
		// select which to compare
		JList<String> jl = new JList<String>(s);
		Object o[] = new Object[2];
		o[0] = "Please select the functions you'd like to compare";
		o[1] = jl;
		int value = JOptionPane.showConfirmDialog(null, o, "select functions to compare",
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		if (value == JOptionPane.CANCEL_OPTION)
			return;
		// which were selected
		Graph[] selected = new Graph[jl.getSelectedIndices().length];
		for (int i = 0; i < jl.getSelectedIndices().length; i++) {
			selected[i] = namesAndGraphs.get(jl.getSelectedValuesList().get(i));
		}
		// show it in new frame
		new conn.GraphInfoFrame(selected);
	}
}
