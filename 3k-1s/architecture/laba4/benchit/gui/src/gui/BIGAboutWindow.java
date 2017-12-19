/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGAboutWindow.java Author:
 * SWTP Nagel 1 Last change by: $Author: rschoene $ $Revision: 1.2 $ $Date: 2006/09/28 04:09:37 $
 ******************************************************************************/
package gui;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.io.File;

import javax.swing.*;

import system.*;

/**
 * This class is the About Dialog of BenchIT-GUI.
 * 
 * @author Carsten Luxig <a href="mailto:c.luxig@lmcsoft.com">c.luxig@lmcsoft.com</a>
 **/
public class BIGAboutWindow extends JDialog {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	// Attributes
	private final BIGInterface bigInterface = BIGInterface.getInstance();
	private final BIGConsole console = bigInterface.getConsole();
	private final Dimension dim = new Dimension(580, 310);

	private final String heading = "<html><b>BenchIT - GUI 3.0, June, 20th 2013</b></html>";
	private final String[] tabNames = {"BenchIT", "About", "Members", "Thanks to", "License"};
	private final String pathToFiles = bigInterface.getHelpPath() + File.separator;
	private static final String aboutFile = "about.html";
	private static final String membersFile = "members.html";
	private static final String thankstoFile = "thanksto.html";
	private static final String licenseFile = "bigui-license.html";

	/**
	 * Creates the About window.
	 * 
	 * @param parent the the frame, that calls this window
	 **/
	public BIGAboutWindow(JFrame parent) {
		super(parent);
		initComponents();
		setDefaultCloseOperation(WindowConstants.HIDE_ON_CLOSE);
		setTitle("About BenchIT-GUI");
		pack();
		setVisible(true);
		// calc top left corner to fit in the middle of the screen
		DisplayMode dm = getGraphicsConfiguration().getDevice().getDisplayMode();
		int x = (dm.getWidth() / 2) - 290;
		int y = (dm.getHeight() / 2) - 155;
		setLocation(x, y);
	}

	/**
	 * Initializes the components.
	 **/
	private void initComponents() {
		JPanel contentPane = new JPanel(new BorderLayout());
		// compose heading
		JPanel head = new JPanel(new FlowLayout(FlowLayout.CENTER));
		JLabel label = new JLabel(heading);
		head.add(label);
		// compose tabbed pane
		JTabbedPane tabbedPane = new JTabbedPane();
		// start with tabNames[0] (BenchIT)
		ImageIcon icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "splash.jpg");
		label = new JLabel(icon);
		tabbedPane.addTab(tabNames[0], label);
		// next tab (About)
		BIGStrings content = new BIGStrings(17);
		try {
			content.readFromFile(pathToFiles + aboutFile);
		} catch (Exception e) {
			console.postMessage("The about file couldn't be read (" + pathToFiles + aboutFile
					+ ")! (BIGAboutWindow.java)", BIGConsole.WARNING);
		}
		label = new JLabel(content.toString());
		tabbedPane.addTab(tabNames[1], new JScrollPane(label));
		// next tab (Members)
		content = new BIGStrings(50);
		try {
			content.readFromFile(pathToFiles + membersFile);
		} catch (Exception e) {
			console.postMessage("The members file couldn't be read (" + pathToFiles + membersFile
					+ ")! (BIGAboutWindow.java)", BIGConsole.WARNING);
		}
		label = new JLabel(content.toString());
		tabbedPane.addTab(tabNames[2], new JScrollPane(label));
		// next tab (Thanks To)
		content = new BIGStrings(10);
		try {
			content.readFromFile(pathToFiles + thankstoFile);
		} catch (Exception e) {
			console.postMessage("The thanks to file couldn't be read (" + pathToFiles + thankstoFile
					+ ")! (BIGAboutWindow.java)", BIGConsole.WARNING);
		}
		label = new JLabel(content.toString());
		tabbedPane.addTab(tabNames[3], new JScrollPane(label));
		// next tab (License)
		content = new BIGStrings(355);
		try {
			content.readFromFile(pathToFiles + licenseFile);
		} catch (Exception e) {
			console.postMessage("The license file couldn't be read (" + pathToFiles + licenseFile
					+ ")! (BIGAboutWindow.java)", BIGConsole.WARNING);
		}
		label = new JLabel(content.toString());
		tabbedPane.addTab(tabNames[4], new JScrollPane(label));

		// now finish with the okay button to close the dialog
		JButton button = new JButton(new AbstractAction("Okay") {
			/**
		 * 
		 */
			private static final long serialVersionUID = 1L;

			public void actionPerformed(ActionEvent ae) {
				setVisible(false);
			}
		});
		JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
		buttonPanel.add(button);

		// add the components to the content pane
		contentPane.add(head, BorderLayout.NORTH);
		contentPane.add(tabbedPane, BorderLayout.CENTER);
		contentPane.add(buttonPanel, BorderLayout.SOUTH);
		contentPane.setMaximumSize(dim);
		contentPane.setMinimumSize(dim);
		contentPane.setPreferredSize(dim);
		setContentPane(contentPane);

	}
}
/*****************************************************************************
 * Log-History
 *****************************************************************************/
