package gui;

import javax.swing.*;

import plot.data.BIGOutputFile;
import plot.gui.BIGPlot;

/**
 * This class builds a frame which analyses and plots the output files.
 */
class BIGQuickViewWindow {
	// the textual and graphical display panels
	private JComponent[] displayPanels;
	private final BIGGUI gui;
	private BIGPlot plot = null;

	// private JList resultList;
	// private JProgressBar progressBar;
	// private Thread runner;
	private final BIGOutputFile file;

	public BIGQuickViewWindow(BIGGUI gui, BIGOutputFile file) {
		this.gui = gui;
		this.file = file;
		// initialize the display panels
		initComponents();
	}

	/**
	 * init all comps.
	 */
	private void initComponents() {
		// System.err.println("init components "+this.file);
		// get the selected result file from the tree
		if (file == null) {
			System.out.println("No result file selected for quick view!");
			return;
		}
		// final BIGOutputFile resultFile = resultFileTemp;
		// resultFile.setProgress(statusProgress,statusLabel);
		file.setProgress(gui.getStatusProgress(), gui.getStatusLabel());
		// no result file is selected post message
		// plotWindow=new BIGPlot(resultFile);
		plot = new BIGPlot(file);
		displayPanels = plot.getDisplayPanels();
		if (displayPanels == null) {
			System.out.println("Please select result file.");
			// inititalize the display panel field size
			displayPanels = new JComponent[2];
			displayPanels[0] = new JPanel();
			displayPanels[1] = new JPanel();
		}
	}

	public JComponent[] getDisplayPanels() {
		return displayPanels;
	}
} // end of BIGQuickViewWindow