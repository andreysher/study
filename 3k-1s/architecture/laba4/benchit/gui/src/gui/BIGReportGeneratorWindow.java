package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.tree.*;
import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.jfree.chart.LegendItemCollection;
import org.w3c.dom.*;

import plot.data.*;
import plot.gui.BIGPlot;
import system.*;
import conn.Graph;

public class BIGReportGeneratorWindow extends JDialog {
	private final BIGConsole console = BIGInterface.getInstance().getConsole();

	/**
	 * Constructor creates and shows the dialog with all components
	 * 
	 * @param selection - selected items in BIGResultTree
	 */
	public BIGReportGeneratorWindow(TreePath[] selection) {
		super();
		allSelectedPlotables = getAllSelectedPlotables(selection);
		initComponents();
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		setTitle("Create your report");
		DisplayMode dm = getGraphicsConfiguration().getDevice().getDisplayMode();
		int x = (dm.getWidth() / 2) - dialogDimension.width / 2;
		int y = (dm.getHeight() / 2) - dialogDimension.height / 2;
		setLocation(x, y);
		pack();
		setVisible(true);
	}

	/**
	 * Iterates over <code>selection</code> and adds every BIGPlotable to a list. Finally this list, that contains all
	 * BIGPlotables from the tree selection, is returned.
	 * 
	 * @param selection - TreePaths of all selected BIGResultTree items
	 * @return a list of all BIGPlotables from the selection
	 */
	private List<BIGPlotable> getAllSelectedPlotables(TreePath[] selection) {
		List<BIGPlotable> list = new ArrayList<BIGPlotable>();
		DefaultMutableTreeNode tnode;

		if (selection == null)
			return null;
		for (int i = 0; i < selection.length; i++) {
			tnode = (DefaultMutableTreeNode) selection[i].getLastPathComponent();
			getAllSelectedPlotablesFromTreeNode(tnode, list);
		}
		for (int i = 0; i < list.size(); i++) {
			BIGPlotable plotable = list.get(i);
			if (plotable instanceof BIGOutputFile) {
				if (!((BIGOutputFile) plotable).getFile().exists()) {
					console.postErrorMessage("File '" + ((BIGOutputFile) plotable).getFile().getName()
							+ "' doesn't exist. Skipping...");
					list.remove(i);
					i--;
				}
			} else if (plotable instanceof BIGResultMixer) {
				BIGResultMixer mixer = (BIGResultMixer) plotable;
				// List<String> origFiles=((BIGResultMixer) plotable).getO
			}
		}
		return list;
	}
	/**
	 * If <code>object</code> is an instance of BIGOutputFile or BIGResultMixer it will added to <code>list</code>.
	 * Otherwise all children (note that <code>object</code> is a tree node) will be analysed.
	 * 
	 * @param tnode - selected node of BIGResultTree
	 * @param list - contains all in BIGResultTree selected BIGPlotables
	 */
	private void getAllSelectedPlotablesFromTreeNode(DefaultMutableTreeNode tnode,
			List<BIGPlotable> list) {
		Enumeration<?> children;

		if (tnode.getUserObject() instanceof BIGPlotable) {
			list.add((BIGPlotable) tnode.getUserObject());
		} else {
			// node has an other type -> look at children
			children = tnode.children();
			while (children.hasMoreElements()) {
				getAllSelectedPlotablesFromTreeNode((DefaultMutableTreeNode) children.nextElement(), list);
			}
		}
	}

	/**
	 * Initialize all GUI-components of the BIGReportGeneratorWindow. The BIGReportGeneratorWindow consists of a
	 * JTabbedPane with two tabs (a "General"-tab and a "PDF Settings"-tab) and a button panel.
	 */
	private void initComponents() {
		JTabbedPane jtp = new JTabbedPane();
		JPanel mainPanel, buttonPanel, generalPanel, PDFPanel, rightsPanel, titlesPanel, archInfoPanel, measurementInfoPanel;
		JButton saveBtn, loadBtn, resetBtn, createBtn;
		JTextField tf;
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
		GridBagLayout gridbag = new GridBagLayout();
		GridBagConstraints gbc = new GridBagConstraints();
		Insets insetsForGridbag = new Insets(5, 5, 5, 5);
		List<String> allArchInfoItems;
		List<String> allMeasurementInfoItems;
		JCheckBox cb;
		gbc.insets = insetsForGridbag;
		int i, row = 0;

		gbc.fill = GridBagConstraints.BOTH;
		gbc.weightx = 1.0;

		// ------------------------------------------------//
		// build the first tabpane (general settings) //
		// ------------------------------------------------//
		generalPanel = new JPanel();
		generalPanel.setLayout(gridbag);

		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(generalPanel, new JLabel(authorLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, GridBagConstraints.REMAINDER, 1);
		BIGUtility.addComponent(generalPanel, authorTF, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(generalPanel, new JLabel(titleLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, GridBagConstraints.REMAINDER, 1);
		BIGUtility.addComponent(generalPanel, titleTF, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(generalPanel, new JLabel(dateLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, GridBagConstraints.REMAINDER, 1);
		// set dateTF to actual date
		dateTF.setText(sdf.format(new Date()));
		BIGUtility.addComponent(generalPanel, dateTF, gridbag, gbc);
		// row++ ;
		jtp.addTab(tabNames[0], generalPanel);

		// ------------------------------------------------//
		// build the second tabpane (PDF settings) //
		// ------------------------------------------------//
		PDFPanel = new JPanel();
		PDFPanel.setLayout(gridbag);

		row = 0;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(docSizeLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 2, 1);
		BIGUtility.addComponent(PDFPanel, docSizeCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(headingLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(fontLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 3, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, headFontCB, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 4, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(sizeLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 5, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, headSizeCB, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 6, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, headBold, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 7, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, headItal, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 8, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, headUnder, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(textLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(fontLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 3, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, textFontCB, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 4, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(sizeLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 5, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, textSizeCB, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 6, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, textBold, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 7, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, textItal, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 8, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, textUnder, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(PDFPanel, new JLabel(marginsLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, GridBagConstraints.REMAINDER, 3);
		BIGUtility.addComponent(PDFPanel, buildMarginPanel(), gridbag, gbc);
		// row++ ;
		jtp.addTab(tabNames[1], PDFPanel);

		// -------------------------------------------------------//
		// build the third tabpane (rights management settings) //
		// -------------------------------------------------------//
		rightsPanel = new JPanel();
		rightsPanel.setLayout(gridbag);

		// CheckBox for activating/deactivating rights management gui-components
		useRightsManagementCB.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				JCheckBox cb;
				// toggle TextFields for password from enabled to disabled or vice versa
				firstUserPasswdTF.setEnabled(!firstUserPasswdTF.isEnabled());
				confirmUserPasswdTF.setEnabled(!confirmUserPasswdTF.isEnabled());
				firstOwnerPasswdTF.setEnabled(!firstOwnerPasswdTF.isEnabled());
				confirmOwnerPasswdTF.setEnabled(!confirmOwnerPasswdTF.isEnabled());
				for (Iterator<JCheckBox> iter = allRightsCBs.iterator(); iter.hasNext();) {
					cb = iter.next();
					// toggle CheckBox from enabled to disabled or vice versa
					cb.setEnabled(!cb.isEnabled());
				}
			}
		});
		row = 0;
		BIGUtility.setConstraints(gbc, 1, row, GridBagConstraints.REMAINDER, 1);
		BIGUtility.addComponent(rightsPanel, useRightsManagementCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, new JLabel(firstUserPasswdLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, firstUserPasswdTF, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, new JLabel(confirmUserPasswdLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, confirmUserPasswdTF, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, new JLabel(firstOwnerPasswdLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, firstOwnerPasswdTF, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, new JLabel(confirmOwnerPasswdLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, confirmOwnerPasswdTF, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, assemblyCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, copyCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, degradedPrintCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, modAnnotationsCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, modContentsCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, printingCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(rightsPanel, screenreadersCB, gridbag, gbc);
		// row++;
		jtp.addTab(tabNames[2], new JScrollPane(rightsPanel));

		// ------------------------------------------------//
		// build fourth tabpane (titles) //
		// ------------------------------------------------//
		titlesPanel = new JPanel();
		titlesPanel.setLayout(gridbag);

		row = 0;
		if (allSelectedPlotables != null) {
			// do following for every single selected tree item
			for (BIGPlotable plotable : allSelectedPlotables) {
				tf = new JTextField(plotable.getExtTitle());
				allTitleTFs.add(tf);
				BIGUtility.setConstraints(gbc, 1, row, 1, 1);
				if (plotable instanceof BIGOutputFile) {
					BIGUtility.addComponent(titlesPanel,
							new JLabel(((BIGOutputFile) plotable).getFilenameWithoutExtension()), gridbag, gbc);
				} else {
					BIGUtility.addComponent(titlesPanel,
							new JLabel("Mixer '" + plotable.getExtTitle() + "'"), gridbag, gbc);
				}
				BIGUtility.setConstraints(gbc, 2, row, 1, 1);
				BIGUtility.addComponent(titlesPanel, tf, gridbag, gbc);
				row++;
			}
		}
		jtp.addTab(tabNames[3], new JScrollPane(titlesPanel));

		// ------------------------------------------------//
		// build fifth tabpane (arch info) //
		// ------------------------------------------------//
		archInfoPanel = new JPanel();
		archInfoPanel.setLayout(gridbag);

		row = 0;
		allArchInfoItems = fillInfoItemsFromSelectedPlotables(SearchType.ARCHITECTURE);
		for (i = 0; i < allArchInfoItems.size(); i++) {
			cb = new JCheckBox(allArchInfoItems.get(i).toString());
			allArchInfoCBs.add(cb);
			BIGUtility.setConstraints(gbc, 1, row, 1, 1);
			BIGUtility.addComponent(archInfoPanel, cb, gridbag, gbc);
			row++;
		}
		jtp.addTab(tabNames[4], new JScrollPane(archInfoPanel));

		// ------------------------------------------------//
		// build sixth tabpane (measurement info) //
		// ------------------------------------------------//
		measurementInfoPanel = new JPanel();
		measurementInfoPanel.setLayout(gridbag);

		row = 0;
		allMeasurementInfoItems = fillInfoItemsFromSelectedPlotables(SearchType.MEASUREMENT);
		for (i = 0; i < allMeasurementInfoItems.size(); i++) {
			cb = new JCheckBox(allMeasurementInfoItems.get(i).toString());
			allMeasurementInfoCBs.add(cb);
			BIGUtility.setConstraints(gbc, 1, row, 1, 1);
			BIGUtility.addComponent(measurementInfoPanel, cb, gridbag, gbc);
			row++;
		}
		jtp.addTab(tabNames[5], new JScrollPane(measurementInfoPanel));

		// ------------------------------------------------//
		// build buttonPanel //
		// ------------------------------------------------//
		buttonPanel = new JPanel();
		saveBtn = new JButton(btnNames[0]);
		saveBtn.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				BIGFileChooser jfc = new BIGFileChooser(BIGInterface.getInstance().getGUIPath());
				jfc.addFileFilter("bpf");
				if (jfc.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
					profParser = new BIGConfigFileParser(jfc.getSelectedFile().getAbsolutePath());
					saveProfile();
				}
			}
		});
		loadBtn = new JButton(btnNames[1]);
		loadBtn.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				BIGFileChooser jfc = new BIGFileChooser(BIGInterface.getInstance().getGUIPath());
				jfc.addFileFilter("bpf");
				if (jfc.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
					setInitialValuesToComponents(jfc.getSelectedFile());
				}
			}
		});
		resetBtn = new JButton(btnNames[2]);
		resetBtn.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				resetValuesToDefault();
			}
		});
		createBtn = new JButton(btnNames[3]);
		createBtn.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				if (profParser.getFile().getAbsolutePath().equals(stdProfile)) {
					saveProfile();
				}
				File xmlFile = generateReportXMLFile();
				reportgen.ReportGenerator.generateReport(xmlFile);
				closeWindow();
			}
		});
		buttonPanel.add(saveBtn);
		buttonPanel.add(loadBtn);
		buttonPanel.add(resetBtn);
		buttonPanel.add(createBtn);

		// ------------------------------------------------//
		// putting all together //
		// ------------------------------------------------//
		mainPanel = new JPanel(new BorderLayout());
		mainPanel.add(jtp, BorderLayout.CENTER);
		mainPanel.add(buttonPanel, BorderLayout.SOUTH);
		mainPanel.setMaximumSize(dialogDimension);
		mainPanel.setMinimumSize(dialogDimension);
		mainPanel.setPreferredSize(dialogDimension);
		setContentPane(mainPanel);

		// initialize values of components
		setInitialValuesToComponents();

		// add all CheckBoxes for rigths management to allRightsCBs
		allRightsCBs.add(assemblyCB);
		allRightsCBs.add(copyCB);
		allRightsCBs.add(degradedPrintCB);
		allRightsCBs.add(modAnnotationsCB);
		allRightsCBs.add(modContentsCB);
		allRightsCBs.add(printingCB);
		allRightsCBs.add(screenreadersCB);
	}

	/**
	 * Builds the margin panel. Components for top margin are located in the north, for left margin in the west, for right
	 * margin in the east and for bottom margin in the south of the margin panel.
	 * 
	 * @return JPanel - the margin panel
	 */
	private JComponent buildMarginPanel() {
		JPanel marginPanel = new JPanel();
		GridBagLayout gridbag = new GridBagLayout();
		GridBagConstraints gbc = new GridBagConstraints();
		Insets insetsForGridbag = new Insets(5, 5, 5, 5);
		gbc.insets = insetsForGridbag;
		marginPanel.setLayout(gridbag);
		int row = 0;

		BIGUtility.setConstraints(gbc, 4, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(topLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 5, row, 1, 1);
		BIGUtility.addComponent(marginPanel, topJS, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 6, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(unitLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 7, row, 1, 1);
		BIGUtility.addComponent(marginPanel, topUnitCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 1, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(leftLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 2, row, 1, 1);
		BIGUtility.addComponent(marginPanel, leftJS, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 3, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(unitLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 4, row, 1, 1);
		BIGUtility.addComponent(marginPanel, leftUnitCB, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 7, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(rightLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 8, row, 1, 1);
		BIGUtility.addComponent(marginPanel, rightJS, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 9, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(unitLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 10, row, 1, 1);
		BIGUtility.addComponent(marginPanel, rightUnitCB, gridbag, gbc);
		row++;
		BIGUtility.setConstraints(gbc, 4, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(bottomLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 5, row, 1, 1);
		BIGUtility.addComponent(marginPanel, bottomJS, gridbag, gbc);
		BIGUtility.setConstraints(gbc, 6, row, 1, 1);
		BIGUtility.addComponent(marginPanel, new JLabel(unitLB), gridbag, gbc);
		BIGUtility.setConstraints(gbc, 7, row, 1, 1);
		BIGUtility.addComponent(marginPanel, bottomUnitCB, gridbag, gbc);
		// row++ ;

		return marginPanel;
	}

	/**
	 * Closes this window
	 */
	private void closeWindow() {
		setVisible(false);
		System.gc();
	}

	/**
	 * Iterates over <code>allSelectedPlotables</code> and checks every contained plotable for architecture info element.
	 * If a new architecture element is found (that means, it is not already contained in list
	 * <code>allArchInfoItems</code>), it will be added to <code>allArchInfoItems</code>. In the end the list
	 * <code>allArchInfoItems</code> is returned.
	 * 
	 * @return ArrayList - list of all architecture info elements contained in selected plotables
	 */
	private List<String> fillInfoItemsFromSelectedPlotables(SearchType searchtype) {
		List<String> tempList, allInfoItems = new ArrayList<String>();

		if (allSelectedPlotables != null) {
			for (BIGPlotable plotable : allSelectedPlotables) {
				if (plotable instanceof BIGOutputFile) {
					switch (searchtype) {
						case ARCHITECTURE :
							// get all arch info from corresponding bit file
							tempList = getArchInfoItems(((BIGOutputFile) plotable).getFile());
							break;
						case MEASUREMENT :
							// get all arch measurement info from corresponding bit file
							tempList = getMeasurementInfoItems(((BIGOutputFile) plotable).getFile());
							break;
						default :
							tempList = new ArrayList<String>();
					}
					for (String str : tempList) {
						// check every arch info item
						if (!allInfoItems.contains(str))
							allInfoItems.add(str);
					}
				} else if (plotable instanceof BIGResultMixer) {
					BIGResultMixer mixer = (BIGResultMixer) plotable;
					// get corresponding tree node to actual mixer and declare a temporal variable
					DefaultMutableTreeNode mixerNode = mixer.getTreeNode(), tempNode;
					Enumeration<?> children = mixerNode.children();
					while (children.hasMoreElements()) {
						// iterate over all children, children are of type Graph
						tempNode = (DefaultMutableTreeNode) children.nextElement();
						Graph graph = (Graph) tempNode.getUserObject();
						// each Graph has a list with bit files, where the data is from
						List<String> origBitFiles = graph.getOriginalBitFiles();
						try {
							for (String bitFile : origBitFiles) {
								switch (searchtype) {
									case ARCHITECTURE :
										tempList = getArchInfoItems(new File(bitFile));
										break;
									case MEASUREMENT :
										tempList = getMeasurementInfoItems(new File(bitFile));
										break;
									default :
										tempList = new ArrayList<String>();
								}
								for (String str : tempList) {
									// iterate over all associated bit files
									if (!allInfoItems.contains(str))
										allInfoItems.add(str);
								}

							}
						} catch (Exception e) {
							System.err.println("An error occured while reading bit files of a graph.\n");
							// e.printStackTrace();
						}
					}
				}
			}
		}
		return allInfoItems;
	}

	/**
	 * Sets all values of gui-components to built-in standard values.
	 */
	private void resetValuesToDefault() {
		// set author
		authorTF.setText(stdAuthor);
		// set title
		titleTF.setText(stdTitle);
		// skip dateTF here
		// set document size
		docSizeCB.setSelectedIndex(stdDocSizeIndex);
		// set all units to stdUnitIndex
		topUnitCB.setSelectedIndex(stdUnitIndex);
		bottomUnitCB.setSelectedIndex(stdUnitIndex);
		leftUnitCB.setSelectedIndex(stdUnitIndex);
		rightUnitCB.setSelectedIndex(stdUnitIndex);
		// set all margins to stdMarginIndex
		topJS.setValue(new Double(stdMargin));
		bottomJS.setValue(new Double(stdMargin));
		leftJS.setValue(new Double(stdMargin));
		rightJS.setValue(new Double(stdMargin));
		// set heading and text font to standard value
		headFontCB.setSelectedIndex(stdFontIndex);
		textFontCB.setSelectedIndex(stdFontIndex);
		// set heading and text font size to standard value
		headSizeCB.setSelectedIndex(stdFontSizeIndex);
		textSizeCB.setSelectedIndex(stdFontSizeIndex);
		// disable all bold, italic and underline checkboxes
		headBold.setSelected(false);
		headItal.setSelected(false);
		headUnder.setSelected(false);
		textBold.setSelected(false);
		textItal.setSelected(false);
		textUnder.setSelected(false);
		// handle rigths management settings
		useRightsManagementCB.setSelected(false);
		assemblyCB.setSelected(false);
		copyCB.setSelected(false);
		degradedPrintCB.setSelected(false);
		modAnnotationsCB.setSelected(false);
		modContentsCB.setSelected(false);
		printingCB.setSelected(false);
		screenreadersCB.setSelected(false);
		// disable rights management gui components
		firstUserPasswdTF.setEnabled(false);
		confirmUserPasswdTF.setEnabled(false);
		firstOwnerPasswdTF.setEnabled(false);
		confirmOwnerPasswdTF.setEnabled(false);
		assemblyCB.setEnabled(false);
		copyCB.setEnabled(false);
		degradedPrintCB.setEnabled(false);
		modAnnotationsCB.setEnabled(false);
		modContentsCB.setEnabled(false);
		printingCB.setEnabled(false);
		screenreadersCB.setEnabled(false);

	}

	/**
	 * This method reads the profile in file f and sets values of the gui-components (textfiels, combo boxes...) according
	 * to this profile. Unless a value in the profile exists, built-in standard values are used instead of.
	 * 
	 * @param f File which contains the profile
	 */
	private void setInitialValuesToComponents(File f) {
		profParser = new BIGConfigFileParser(f.getAbsolutePath());
		// set author
		try {
			authorTF.setText(profParser.stringCheckOut(profAuthor));
		} catch (Exception e) {
			authorTF.setText(stdAuthor);
			profParser.set(profAuthor, stdAuthor);
		}
		// set title
		try {
			titleTF.setText(profParser.stringCheckOut(profTitle));
		} catch (Exception e) {
			titleTF.setText(stdTitle);
			profParser.set(profTitle, stdTitle);
		}
		// dateTF was already set to actual date in method initComponents(), so we can skip it here
		// set document size
		try {
			docSizeCB.setSelectedIndex(profParser.intCheckOut(profDocSize));
		} catch (Exception e) {
			docSizeCB.setSelectedIndex(stdDocSizeIndex);
			profParser.set(profDocSize, new Integer(stdDocSizeIndex).toString());
		}
		// set all units to stdUnitIndex
		try {
			topUnitCB.setSelectedIndex(profParser.intCheckOut(profTopUnit));
		} catch (Exception e) {
			topUnitCB.setSelectedIndex(stdUnitIndex);
			profParser.set(profTopUnit, new Integer(stdUnitIndex).toString());
		}
		try {
			bottomUnitCB.setSelectedIndex(profParser.intCheckOut(profBottomUnit));
		} catch (Exception e) {
			bottomUnitCB.setSelectedIndex(stdUnitIndex);
			profParser.set(profBottomUnit, new Integer(stdUnitIndex).toString());
		}
		try {
			leftUnitCB.setSelectedIndex(profParser.intCheckOut(profLeftUnit));
		} catch (Exception e) {
			leftUnitCB.setSelectedIndex(stdUnitIndex);
			profParser.set(profLeftUnit, new Integer(stdUnitIndex).toString());
		}
		try {
			rightUnitCB.setSelectedIndex(profParser.intCheckOut(profRightUnit));
		} catch (Exception e) {
			rightUnitCB.setSelectedIndex(stdUnitIndex);
			profParser.set(profRightUnit, new Integer(stdUnitIndex).toString());
		}
		// set all margins to stdMarginIndex
		try {
			topJS.setValue(new Double(profParser.doubleCheckOut(profTopMargin)));
		} catch (Exception e) {
			topJS.setValue(new Double(stdMargin));
			profParser.set(profTopMargin, new Double(stdMargin).toString());
		}
		try {
			bottomJS.setValue(new Double(profParser.doubleCheckOut(profBottomMargin)));
		} catch (Exception e) {
			bottomJS.setValue(new Double(stdMargin));
			profParser.set(profBottomMargin, new Double(stdMargin).toString());
		}
		try {
			leftJS.setValue(new Double(profParser.doubleCheckOut(profLeftMargin)));
		} catch (Exception e) {
			leftJS.setValue(new Double(stdMargin));
			profParser.set(profLeftMargin, new Double(stdMargin).toString());
		}
		try {
			rightJS.setValue(new Double(profParser.doubleCheckOut(profRightMargin)));
		} catch (Exception e) {
			rightJS.setValue(new Double(stdMargin));
			profParser.set(profRightMargin, new Double(stdMargin).toString());
		}
		// set heading and text font to standard value
		try {
			headFontCB.setSelectedIndex(profParser.intCheckOut(profHeadingFont));
		} catch (Exception e) {
			headFontCB.setSelectedIndex(stdFontIndex);
			profParser.set(profHeadingFont, new Integer(stdFontIndex).toString());
		}
		try {
			textFontCB.setSelectedIndex(profParser.intCheckOut(profTextFont));
		} catch (Exception e) {
			textFontCB.setSelectedIndex(stdFontIndex);
			profParser.set(profTextFont, new Integer(stdFontIndex).toString());
		}
		// set heading and text font size to standard value
		try {
			headSizeCB.setSelectedIndex(profParser.intCheckOut(profHeadingSize));
		} catch (Exception e) {
			headSizeCB.setSelectedIndex(stdFontSizeIndex);
			profParser.set(profHeadingSize, new Integer(stdFontSizeIndex).toString());
		}
		try {
			textSizeCB.setSelectedIndex(profParser.intCheckOut(profTextSize));
		} catch (Exception e) {
			textSizeCB.setSelectedIndex(stdFontSizeIndex);
			profParser.set(profTextSize, new Integer(stdFontSizeIndex).toString());
		}
		// settings for all bold, italic and underline checkboxes
		try {
			headBold.setSelected(profParser.boolCheckOut(profHeadingBold));
		} catch (Exception e) {
			headBold.setSelected(false);
			profParser.set(profHeadingBold, booleanToString(false));
		}
		try {
			headItal.setSelected(profParser.boolCheckOut(profHeadingItalic));
		} catch (Exception e) {
			headItal.setSelected(false);
			profParser.set(profHeadingItalic, booleanToString(false));
		}
		try {
			headUnder.setSelected(profParser.boolCheckOut(profHeadingUnderline));
		} catch (Exception e) {
			headUnder.setSelected(false);
			profParser.set(profHeadingUnderline, booleanToString(false));
		}

		try {
			textBold.setSelected(profParser.boolCheckOut(profTextBold));
		} catch (Exception e) {
			textBold.setSelected(false);
			profParser.set(profTextBold, booleanToString(false));
		}
		try {
			textItal.setSelected(profParser.boolCheckOut(profTextItalic));
		} catch (Exception e) {
			textItal.setSelected(false);
			profParser.set(profTextItalic, booleanToString(false));
		}
		try {
			textUnder.setSelected(profParser.boolCheckOut(profTextUnderline));
		} catch (Exception e) {
			textUnder.setSelected(false);
			profParser.set(profTextUnderline, booleanToString(false));
		}

		// handle rigths management settings
		try {
			useRightsManagementCB.setSelected(profParser.boolCheckOut(profUseRightsManagement));
		} catch (Exception e) {
			useRightsManagementCB.setSelected(false);
			profParser.set(profUseRightsManagement, booleanToString(false));
		}
		try {
			assemblyCB.setSelected(profParser.boolCheckOut(profAllowAssembly));
		} catch (Exception e) {
			assemblyCB.setSelected(false);
			profParser.set(profAllowAssembly, booleanToString(false));
		}
		try {
			copyCB.setSelected(profParser.boolCheckOut(profAllowCopy));
		} catch (Exception e) {
			copyCB.setSelected(false);
			profParser.set(profAllowCopy, booleanToString(false));
		}
		try {
			degradedPrintCB.setSelected(profParser.boolCheckOut(profAllowDegradedPrint));
		} catch (Exception e) {
			degradedPrintCB.setSelected(false);
			profParser.set(profAllowDegradedPrint, booleanToString(false));
		}
		try {
			modAnnotationsCB.setSelected(profParser.boolCheckOut(profAllowModAnnotations));
		} catch (Exception e) {
			modAnnotationsCB.setSelected(false);
			profParser.set(profAllowModAnnotations, booleanToString(false));
		}
		try {
			modContentsCB.setSelected(profParser.boolCheckOut(profAllowModContents));
		} catch (Exception e) {
			modContentsCB.setSelected(false);
			profParser.set(profAllowModContents, booleanToString(false));
		}
		try {
			printingCB.setSelected(profParser.boolCheckOut(profAllowPrinting));
		} catch (Exception e) {
			printingCB.setSelected(false);
			profParser.set(profAllowPrinting, booleanToString(false));
		}
		try {
			screenreadersCB.setSelected(profParser.boolCheckOut(profAllowScreenreaders));
		} catch (Exception e) {
			screenreadersCB.setSelected(false);
			profParser.set(profAllowScreenreaders, booleanToString(false));
		}
		if (!useRightsManagementCB.isSelected()) {
			// disable rights management gui components
			firstUserPasswdTF.setEnabled(false);
			confirmUserPasswdTF.setEnabled(false);
			firstOwnerPasswdTF.setEnabled(false);
			confirmOwnerPasswdTF.setEnabled(false);
			assemblyCB.setEnabled(false);
			copyCB.setEnabled(false);
			degradedPrintCB.setEnabled(false);
			modAnnotationsCB.setEnabled(false);
			modContentsCB.setEnabled(false);
			printingCB.setEnabled(false);
			screenreadersCB.setEnabled(false);
		}
	}

	/**
	 * This method reads the standard profile and sets values of the gui-components (textfiels, combo boxes...) according
	 * to this standard profile. Unless the standard profile exists, built-in standard values are used instead of profile
	 * values.
	 */
	private void setInitialValuesToComponents() {
		// Does standard profile exist?
		File profFile = new File(stdProfile);
		if (!profFile.exists()) {
			try {
				profFile.createNewFile();
				// createNewFile() atomically creates a new, empty file named
				// by the abstract pathname if and only if a file with this
				// name does not yet exist
				// so the if-statement is not really necessary, but to be on the safe side...
			} catch (IOException ioe) {
				// file could not be created (e.g. we don't have necessary rights...)
				System.err.println("File " + stdProfile + "could not be created.");
				// ioe.printStackTrace();
			}
		}
		setInitialValuesToComponents(profFile);
	}

	/**
	 * Checks all values of gui-components and saves the profile.
	 */
	private void saveProfile() {
		// check author
		profParser.set(profAuthor, authorTF.getText());
		// check title
		profParser.set(profTitle, titleTF.getText());
		// check document size
		profParser.set(profDocSize, new Integer(docSizeCB.getSelectedIndex()).toString());
		// check margins
		profParser.set(profTopMargin, ((JSpinner.NumberEditor) topJS.getEditor()).getTextField()
				.getText());
		profParser.set(profBottomMargin, ((JSpinner.NumberEditor) bottomJS.getEditor()).getTextField()
				.getText());
		profParser.set(profLeftMargin, ((JSpinner.NumberEditor) leftJS.getEditor()).getTextField()
				.getText());
		profParser.set(profRightMargin, ((JSpinner.NumberEditor) rightJS.getEditor()).getTextField()
				.getText());
		// check margin units
		profParser.set(profTopUnit, new Integer(topUnitCB.getSelectedIndex()).toString());
		profParser.set(profBottomUnit, new Integer(bottomUnitCB.getSelectedIndex()).toString());
		profParser.set(profLeftUnit, new Integer(leftUnitCB.getSelectedIndex()).toString());
		profParser.set(profRightUnit, new Integer(rightUnitCB.getSelectedIndex()).toString());
		// check setting for heading
		profParser.set(profHeadingFont, new Integer(headFontCB.getSelectedIndex()).toString());
		profParser.set(profHeadingSize, new Integer(headSizeCB.getSelectedIndex()).toString());
		profParser.set(profHeadingBold, booleanToString(headBold.isSelected()));
		profParser.set(profHeadingItalic, booleanToString(headItal.isSelected()));
		profParser.set(profHeadingUnderline, booleanToString(headUnder.isSelected()));
		// check setting for text
		profParser.set(profTextFont, new Integer(textFontCB.getSelectedIndex()).toString());
		profParser.set(profTextSize, new Integer(textSizeCB.getSelectedIndex()).toString());
		profParser.set(profTextBold, booleanToString(textBold.isSelected()));
		profParser.set(profTextItalic, booleanToString(textItal.isSelected()));
		profParser.set(profTextUnderline, booleanToString(textUnder.isSelected()));
		// check rights management settings
		profParser.set(profUseRightsManagement, booleanToString(useRightsManagementCB.isSelected()));
		profParser.set(profAllowAssembly, booleanToString(assemblyCB.isSelected()));
		profParser.set(profAllowCopy, booleanToString(copyCB.isSelected()));
		profParser.set(profAllowDegradedPrint, booleanToString(degradedPrintCB.isSelected()));
		profParser.set(profAllowModAnnotations, booleanToString(modAnnotationsCB.isSelected()));
		profParser.set(profAllowModContents, booleanToString(modContentsCB.isSelected()));
		profParser.set(profAllowPrinting, booleanToString(printingCB.isSelected()));
		profParser.set(profAllowScreenreaders, booleanToString(screenreadersCB.isSelected()));
		// and now save the profile
		profParser.save();
	}

	/**
	 * Converts a given boolean value into a string. The string is 1 if b equals true, otherwise the string is 0.
	 * 
	 * @param b - boolean to convert
	 * @return String - 1 if b equals true, 0 otherwise
	 */
	private String booleanToString(boolean b) {
		if (b)
			return "1";
		else
			return "0";
	}

	/**
	 * Generates the xml file that contains all information used by the report generator back end utility.
	 */
	private File generateReportXMLFile() {
		// list of files in directory 'report'
		// File[] fileList;
		// stuff for gerating the xml file
		DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
		// if something goes wrong we set this variables to true
		boolean errorOccured = false, missingBitFile = false;

		// if rights management should be used,
		// we test that first password and confirm password are equal
		if (useRightsManagementCB.isSelected()) {
			if (!firstUserPasswdTF.getText().equals(confirmUserPasswdTF.getText())) {
				JOptionPane.showMessageDialog(null, "User password and confirmation of "
						+ "user password are not equal.\nPlease correct your inputs.",
						"Wrong user password inputs", JOptionPane.ERROR_MESSAGE);
				errorOccured = true;
			}
			if (!firstOwnerPasswdTF.getText().equals(confirmOwnerPasswdTF.getText())) {
				JOptionPane.showMessageDialog(null, "Owner password and confirmation of "
						+ "owner password are not equal.\nPlease correct your inputs.",
						"Wrong owner password inputs", JOptionPane.ERROR_MESSAGE);
				errorOccured = true;
			}
			if (errorOccured)
				// error occured -> abort xml generation
				return null;
		}
		// check and create paths at first
		File f = new File(stdXMLPath);
		if (f.exists()) {
			// create a backup folder and save old reports
			String temp = f.getAbsolutePath() + "_" + (new Date(f.lastModified())).toString();
			f.renameTo(new File(temp));

			/*
			 * // directory report already exists -> clear it at first fileList = f.listFiles(); for(int i = 0; i <
			 * fileList.length; i++ ) { fileList[ i ].delete(); } // now delete the directory f.delete();
			 */
		}
		f.mkdir();
		DocumentBuilder builder;
		try {
			builder = factory.newDocumentBuilder();
		} catch (ParserConfigurationException e) {
			System.err.println("DocumentBuilder could not be instanciated.\n"
					+ "Generation of xml-file for BenchIT report was aborted.");
			return null;
		}
		// create a new document
		Document document = builder.newDocument();
		String bitNS = "http://reportgen_xsd/reportgen.xsd";
		// create root node
		Element root = document.createElementNS(bitNS, "benchit:ReportGenerator");
		root.setAttribute("xmlns:benchit", bitNS);
		// root.setAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance");
		root.setAttribute("benchit:schemaLocation", "http://reportgen_xsd/reportgen.xsd reportgen.xsd ");
		// and add it to the document
		document.appendChild(root);
		// the same for PDFInfo part of the xml document
		Element pdfInfo = document.createElementNS(bitNS, "benchit:PDFInfo");
		root.appendChild(pdfInfo);

		// create author node
		Element author = document.createElementNS(bitNS, "benchit:Author");
		// set the node value (in this case it is the author name)
		Text nodeText = document.createTextNode(authorTF.getText());
		// adjust hierachy
		author.appendChild(nodeText);
		pdfInfo.appendChild(author);

		Element title = document.createElementNS(bitNS, "benchit:Title");
		nodeText = document.createTextNode(titleTF.getText());
		title.appendChild(nodeText);
		pdfInfo.appendChild(title);

		Element date = document.createElementNS(bitNS, "benchit:Date");
		nodeText = document.createTextNode(dateTF.getText());
		date.appendChild(nodeText);
		pdfInfo.appendChild(date);

		Element rights = document.createElementNS(bitNS, "benchit:Rights");
		if (useRightsManagementCB.isSelected()) {
			rights.setAttribute("userpassword", firstUserPasswdTF.getText());
			rights.setAttribute("ownerpassword", firstOwnerPasswdTF.getText());
			Element right;
			JCheckBox cb;
			String s;
			for (int i = 0; i < allRightsCBs.size(); i++) {
				cb = (allRightsCBs.get(i));
				if (cb.isSelected()) {
					right = document.createElementNS(bitNS, "benchit:Right");
					s = (cb.getText()).toUpperCase();
					s = s.replaceAll(" ", "_");
					nodeText = document.createTextNode(s);
					right.appendChild(nodeText);
					rights.appendChild(right);
				}
			}
		}
		pdfInfo.appendChild(rights);

		Element format = document.createElementNS(bitNS, "benchit:Format");
		pdfInfo.appendChild(format);
		Element docSize = document.createElementNS(bitNS, "benchit:DocumentSize");
		nodeText = document.createTextNode(docSizeCB.getSelectedItem().toString());
		docSize.appendChild(nodeText);
		format.appendChild(docSize);
		Element margins = document.createElementNS(bitNS, "benchit:Margins");
		format.appendChild(margins);
		Element leftMargin = document.createElementNS(bitNS, "benchit:LeftMargin");
		leftMargin.setAttribute("unit", leftUnitCB.getSelectedItem().toString());
		nodeText = document.createTextNode(((JSpinner.NumberEditor) leftJS.getEditor()).getTextField()
				.getText());
		leftMargin.appendChild(nodeText);
		margins.appendChild(leftMargin);
		Element rightMargin = document.createElementNS(bitNS, "benchit:RightMargin");
		rightMargin.setAttribute("unit", rightUnitCB.getSelectedItem().toString());
		nodeText = document.createTextNode(((JSpinner.NumberEditor) rightJS.getEditor()).getTextField()
				.getText());
		rightMargin.appendChild(nodeText);
		margins.appendChild(rightMargin);
		Element topMargin = document.createElementNS(bitNS, "benchit:TopMargin");
		topMargin.setAttribute("unit", topUnitCB.getSelectedItem().toString());
		nodeText = document.createTextNode(((JSpinner.NumberEditor) topJS.getEditor()).getTextField()
				.getText());
		topMargin.appendChild(nodeText);
		margins.appendChild(topMargin);
		Element bottomMargin = document.createElementNS(bitNS, "benchit:BottomMargin");
		bottomMargin.setAttribute("unit", bottomUnitCB.getSelectedItem().toString());
		nodeText = document.createTextNode(((JSpinner.NumberEditor) bottomJS.getEditor())
				.getTextField().getText());
		bottomMargin.appendChild(nodeText);
		margins.appendChild(bottomMargin);

		Element text = document.createElementNS(bitNS, "benchit:Text");
		pdfInfo.appendChild(text);
		Element textFont = document.createElementNS(bitNS, "benchit:FontType");
		if (textBold.isSelected()) {
			textFont.setAttribute("bold", "true");
		} else {
			textFont.setAttribute("bold", "false");
		}
		if (textItal.isSelected()) {
			textFont.setAttribute("italic", "true");
		} else {
			textFont.setAttribute("italic", "false");
		}
		if (textUnder.isSelected()) {
			textFont.setAttribute("underline", "true");
		} else {
			textFont.setAttribute("underline", "false");
		}
		nodeText = document.createTextNode(textFontCB.getSelectedItem().toString());
		textFont.appendChild(nodeText);
		text.appendChild(textFont);
		Element textSize = document.createElementNS(bitNS, "benchit:FontSize");
		nodeText = document.createTextNode(textSizeCB.getSelectedItem().toString());
		textSize.appendChild(nodeText);
		text.appendChild(textSize);
		Element heading = document.createElementNS(bitNS, "benchit:Heading");
		pdfInfo.appendChild(heading);
		Element headSize = document.createElementNS(bitNS, "benchit:FontSize");
		nodeText = document.createTextNode(headSizeCB.getSelectedItem().toString());
		headSize.appendChild(nodeText);
		heading.appendChild(headSize);
		Element headFont = document.createElementNS(bitNS, "benchit:FontType");
		if (headBold.isSelected()) {
			headFont.setAttribute("bold", "true");
		} else {
			headFont.setAttribute("bold", "false");
		}
		if (headItal.isSelected()) {
			headFont.setAttribute("italic", "true");
		} else {
			headFont.setAttribute("italic", "false");
		}
		if (headUnder.isSelected()) {
			headFont.setAttribute("underline", "true");
		} else {
			headFont.setAttribute("underline", "false");
		}
		nodeText = document.createTextNode(headFontCB.getSelectedItem().toString());
		headFont.appendChild(nodeText);
		heading.appendChild(headFont);

		Element measurements = document.createElementNS(bitNS, "benchit:Measurements");
		root.appendChild(measurements);

		Element measurement;
		BIGResultMixer mixer;
		DefaultMutableTreeNode mixerNode;
		Graph actualGraph;
		List<String> bitFileList;
		Color seriesColor;
		String description;
		int index;
		// if there are selected tree items
		if (allSelectedPlotables != null) {
			// do following for every single selected tree item
			for (BIGPlotable plotable : allSelectedPlotables) {
				if (plotable instanceof BIGOutputFile) {
					BIGOutputFile bigFile = (BIGOutputFile) plotable;
					// get only file name not hte whole path
					String fileName = bigFile.getFile().getName();
					// replace file extension
					if (fileName.endsWith(".bit")) {
						fileName = fileName.substring(0, fileName.length() - 3) + "png";
					} else {
						System.out.println("File " + bigFile.getFile().getAbsolutePath()
								+ " was selected, but a bit-file was expected.");
						fileName = fileName + ".png";
					}

					BIGPlot plot = new BIGPlot(bigFile);

					if (plot.writePNGFile(stdXMLPath + fileName, svgDimension)) {
						measurement = document.createElementNS(bitNS, ("benchit:Measurement"));
						measurement.setAttribute("origin", "plot");
						// graphic element
						Element graphic = document.createElementNS(bitNS, "benchit:Graphic");
						nodeText = document.createTextNode(fileName);
						graphic.appendChild(nodeText);
						measurement.appendChild(graphic);
						// title element
						Element mesTitle = document.createElementNS(bitNS, "benchit:Title");
						// nodeText = document.createTextNode( bigFile.getExtTitle() );
						index = allSelectedPlotables.indexOf(plotable);
						nodeText = document.createTextNode(allTitleTFs.get(index).getText());
						mesTitle.appendChild(nodeText);
						measurement.appendChild(mesTitle);
						// archInfos element
						Element archInfos = document.createElementNS(bitNS, "benchit:ArchInfos");
						Element info;
						// add architecture info that should be displayed
						for (int j = 0; j < allArchInfoCBs.size(); j++) {
							if (allArchInfoCBs.get(j).isSelected()) {
								info = document.createElementNS(bitNS, "benchit:Info");
								nodeText = document.createTextNode(allArchInfoCBs.get(j).getText());
								info.appendChild(nodeText);
								archInfos.appendChild(info);
							}
						}
						// add measurement info that should be displayed
						for (int k = 0; k < allMeasurementInfoCBs.size(); k++) {
							if (allMeasurementInfoCBs.get(k).isSelected()) {
								info = document.createElementNS(bitNS, "benchit:Info");
								nodeText = document.createTextNode(allMeasurementInfoCBs.get(k).getText());
								info.appendChild(nodeText);
								archInfos.appendChild(info);
							}
						}
						measurement.appendChild(archInfos);
						// bitFiles element
						Element bitFiles = document.createElementNS(bitNS, "benchit:BitFiles");
						Element bitFile = document.createElementNS(bitNS, "benchit:BitFile");

						Element color = document.createElementNS(bitNS, "benchit:Color");
						nodeText = document.createTextNode(new Integer((Color.BLACK).getRGB()).toString());
						color.appendChild(nodeText);
						bitFile.appendChild(color);

						Element desc = document.createElementNS(bitNS, "benchit:Description");
						nodeText = document.createTextNode("none");
						desc.appendChild(nodeText);
						bitFile.appendChild(desc);

						Element dump = document.createElementNS(bitNS, "benchit:Dump");
						nodeText = document.createTextNode(makeDumpOfBitFile(bigFile.getFile()));
						dump.appendChild(nodeText);
						bitFile.appendChild(dump);

						bitFiles.appendChild(bitFile);
						measurement.appendChild(bitFiles);
						measurements.appendChild(measurement);
					}
					plot = null;
				} else if (plotable instanceof BIGResultMixer) {
					mixer = (BIGResultMixer) plotable;
					String fileName = mixer.getExtTitle() + ".png";
					BIGPlot plot = new BIGPlot(mixer);
					if (plot.writePNGFile(stdXMLPath + fileName, svgDimension)) {
						measurement = document.createElementNS(bitNS, ("benchit:Measurement"));
						measurement.setAttribute("origin", "gui");
						// graphic element
						Element graphic = document.createElementNS(bitNS, "benchit:Graphic");
						nodeText = document.createTextNode(fileName);
						graphic.appendChild(nodeText);
						measurement.appendChild(graphic);
						// title element
						Element mesTitle = document.createElementNS(bitNS, "benchit:Title");
						index = allSelectedPlotables.indexOf(plotable);
						nodeText = document.createTextNode(allTitleTFs.get(index).getText());
						mesTitle.appendChild(nodeText);
						measurement.appendChild(mesTitle);
						// archInfos element
						Element archInfos = document.createElementNS(bitNS, "benchit:ArchInfos");
						Element info;
						// add architecture info that should be displayed
						for (int j = 0; j < allArchInfoCBs.size(); j++) {
							if (allArchInfoCBs.get(j).isSelected()) {
								info = document.createElementNS(bitNS, "benchit:Info");
								nodeText = document.createTextNode(allArchInfoCBs.get(j).getText());
								info.appendChild(nodeText);
								archInfos.appendChild(info);
							}
						}
						// add measurement info that should be displayed
						for (int k = 0; k < allMeasurementInfoCBs.size(); k++) {
							if (allMeasurementInfoCBs.get(k).isSelected()) {
								info = document.createElementNS(bitNS, "benchit:Info");
								nodeText = document.createTextNode(allMeasurementInfoCBs.get(k).getText());
								info.appendChild(nodeText);
								archInfos.appendChild(info);
							}
						}
						measurement.appendChild(archInfos);
						// bitFiles element
						Element bitFiles = document.createElementNS(bitNS, "benchit:BitFiles");
						mixerNode = mixer.getTreeNode();
						LegendItemCollection legendItems = plot.getChartPanel().getChart().getPlot()
								.getLegendItems();
						for (Enumeration<?> e = mixerNode.children(); e.hasMoreElements();) {
							// iterate over children-nodes of mixer (not that they are of type graph)
							actualGraph = (Graph) ((DefaultMutableTreeNode) e.nextElement()).getUserObject();
							// get legend via actualGraph.getGraphName();
							description = actualGraph.getGraphName();
							seriesColor = Color.WHITE;
							// find legend item (with the same name as graph name)
							// and read/set the corresponding color
							for (int k = 0; k < legendItems.getItemCount(); k++) {
								if (legendItems.get(k).getLabel().equals(description)) {
									seriesColor = (Color) legendItems.get(k).getPaint();
								}
							}
							bitFileList = actualGraph.getOriginalBitFiles();
							if ((bitFileList != null) && (bitFileList.size() > 0)) {
								// there are elements in the list (the original bit files of a graph)
								// -> iterate over the list
								for (String bitFile : bitFileList) {
									Element elBitFile = document.createElementNS(bitNS, "benchit:BitFile");

									Element color = document.createElementNS(bitNS, "benchit:Color");
									nodeText = document.createTextNode(new Integer(seriesColor.getRGB()).toString());
									color.appendChild(nodeText);
									elBitFile.appendChild(color);

									Element desc = document.createElementNS(bitNS, "benchit:Description");
									nodeText = document.createTextNode(description);
									desc.appendChild(nodeText);
									elBitFile.appendChild(desc);

									Element dump = document.createElementNS(bitNS, "benchit:Dump");
									nodeText = document.createTextNode(makeDumpOfBitFile(new File(bitFile)));
									dump.appendChild(nodeText);
									elBitFile.appendChild(dump);
									bitFiles.appendChild(elBitFile);
								}
							} else {
								// actual mixer is an old mixer with no stored bit-file reference
								// -> so we can not process this mixer
								System.err.println("There is no associated bit-file for graph "
										+ actualGraph.getGraphName() + ".\nGraph " + actualGraph.getGraphName()
										+ " was skipped.");
								System.err.println("Mixer " + mixer.getExtTitle() + "was skipped for"
										+ "report generation.");
								missingBitFile = true;
							}
						}
						// add all bit-file related nodes
						measurement.appendChild(bitFiles);
						if (!missingBitFile) {
							// if we had all information about bit files,
							// then this measurement-node is valid
							// -> so we can add the node here
							measurements.appendChild(measurement);
						}
					}
				}
			}
			if (!measurements.hasChildNodes()) {
				// xml-tree doesn't contain at least one valid measurements
				// -> abort report generation
				System.err.println("No valid measurement contained. Report generation was aborted.");
				return null;
			}
		}

		// now we write the xml file
		f = new File(stdXMLPath + "report.xml");
		try {
			f.createNewFile();
		} catch (IOException e) {
			System.err.println("File " + f.getAbsolutePath() + " could not be created.\n"
					+ "BenchIT wasn't able to generate the xml file for your report.");
			return null;
		}
		try {
			BIGFileHelper.saveToFile(documentToString(document), f);
		} catch (Exception e) {
			System.err.println("Error while writting the xml file.\n"
					+ "BenchIT wasn't able to generate the xml file for your report.");
			return null;
		}
		return f;
	}

	/**
	 * Reads the given bitfile and collects all architectureinfo items. Finally a list with all contained items is
	 * returned.
	 * 
	 * @param bitFile - file to read
	 * @return list of contained items
	 */
	private List<String> getArchInfoItems(File bitFile) {
		List<String> items = new ArrayList<String>();
		BufferedReader in = null;
		String line;
		int index;
		boolean NotFound = true;
		try {
			in = new BufferedReader(new FileReader(bitFile));
			// search for beginning of architecture info
			do {
				line = in.readLine();
				if ((line != null) && line.startsWith("beginofarchitecture")) {
					// we found beginning of architecture info
					NotFound = false;
				}
			} while (NotFound || (line == null));
			// now we will search for end of architecture info and use also the NotFound variable
			NotFound = true;
			// now we collect all arch info items
			while (((line = in.readLine()) != null) && NotFound) {
				if (!line.startsWith("#")) {
					// this line is not a comment, so we must look at it
					if (line.startsWith("endofarchitecture")) {
						// we found end of architecture info
						NotFound = false;
					} else {
						// System.err.println("Original line: " + line );
						index = line.indexOf("=");
						if (index >= 0) {
							// this line contains the character "="
							line = line.substring(0, index).trim();
							items.add(line);
							// System.err.println("extracted arch info: " + line );
						}
					}
				}
			}

		} catch (Exception e) {
			System.err.println("Could not read bit-file " + bitFile.getAbsolutePath() + ".");
		} finally {
			try {
				in.close();
			} catch (Exception e) {}
		}
		return items;
	}

	/**
	 * Reads the given bitfile and collects all measurementinfo items. Finally a list with all contained items is
	 * returned.
	 * 
	 * @param bitFile - file to read
	 * @return list of contained items
	 */
	private List<String> getMeasurementInfoItems(File bitFile) {
		List<String> items = new ArrayList<String>();
		BufferedReader in = null;
		String line;
		int index;
		boolean NotFound = true;

		try {
			in = new BufferedReader(new FileReader(bitFile));
			// search for beginning of measurement info
			do {
				line = in.readLine();
				if ((line != null) && line.startsWith("beginofmeasurementinfos")) {
					// we found beginning of architecture info
					NotFound = false;
				}
			} while (NotFound || (line == null));
			// now we will search for end of measurement info and use also the NotFound variable
			NotFound = true;
			// now we collect all measurement info items
			while (((line = in.readLine()) != null) && NotFound) {
				if (!line.startsWith("#")) {
					// this line is not a comment, so we must look at it
					if (line.startsWith("endofmeasurementinfos")) {
						// we found end of measurement info
						NotFound = false;
					} else {
						index = line.indexOf("=");
						if (index >= 0) {
							// this line contains the character "="
							line = line.substring(0, index).trim();
							items.add(line);
						}
					}
				}
			}

		} catch (Exception e) {
			System.err.println("Could not read bit-file " + bitFile.getAbsolutePath() + ".");
		} finally {
			try {
				in.close();
			} catch (Exception e) {}
		}
		return items;
	}

	/**
	 * Reads content of file <CODE>file</CODE> and replace characters like 'less than' or 'greater than' with '&lt' or
	 * '&gt'. Finally the file content is returned as string.
	 * 
	 * @param file File to read
	 * @return file content as string
	 */
	private String makeDumpOfBitFile(File file) {
		BufferedReader in = null;
		String line, content = "\n";

		try {
			in = new BufferedReader(new FileReader(file));
			while ((line = in.readLine()) != null) {
				line = line.replaceAll("<", "&lt");
				line = line.replaceAll(">", "&gt");
				content = content + line + "\n";
			}
		} catch (Exception e) {
			System.err.println("Could not read bit-file " + file.getAbsolutePath() + ".");
		} finally {
			try {
				in.close();
			} catch (Exception e) {}
		}
		return content;
	}

	/**
	 * Converts the XML document doc into a String.
	 * 
	 * @param doc the XML document, which is to be converted into a String
	 * @return the String-representation of the XML document doc
	 * @throws Exception
	 */
	private String documentToString(Document document) throws Exception {
		DOMSource ds = new DOMSource(document);
		ByteArrayOutputStream os = new ByteArrayOutputStream();
		StreamResult sr = new StreamResult(os);
		TransformerFactory tf = TransformerFactory.newInstance();
		Transformer t = tf.newTransformer();
		t.transform(ds, sr);
		return new String(os.toByteArray(), "UTF-8");
	}

	private final Dimension dialogDimension = new Dimension(700, 310);
	private final Dimension svgDimension = new Dimension(800, 600);

	// BIGConfigFileParser
	BIGConfigFileParser profParser;

	// Strings
	private final String[] tabNames = {"General", "PDF Settings", "Rights Management", "Titles",
			"Architecture Infos", "Measurement Info"};
	private final String[] btnNames = {"Save Profile", "Load Profile", "Reset Defaults",
			"Create Report"};
	private final String[] units = {"cm", "pt"};
	private final String[] fonts = {"COURIER", "HELVETICA", "TIMES"};
	private final String[] sizes = {"A3", "A4", "A5", "LETTER", "TABLOID"};
	private final String[] fontSizes = {"8", "9", "10", "11", "12", "14", "16", "18", "20", "22",
			"24"};

	// standard values
	private final static String stdAuthor = "BenchIT";
	private final static String stdTitle = "BenchIT-Report";
	private final static int stdDocSizeIndex = 1;
	private final static double stdMargin = 2.0;
	private final static int stdUnitIndex = 0;
	private final static int stdFontIndex = 0;
	private final static int stdFontSizeIndex = 4;

	private enum SearchType {
		ARCHITECTURE, MEASUREMENT;
	}

	// standard xml file path
	private final static String stdXMLPath = BIGInterface.getInstance().getBenchItPath()
			+ File.separator + "report" + File.separator;

	// path to standard profile
	private final static String stdProfile = BIGInterface.getInstance().getConfigPath()
			+ File.separator + "stdProfile.bpf";

	// attributes of the profile
	private final static String profAuthor = "Author";
	private final static String profTitle = "Title";
	private final static String profDocSize = "DocumentSize";
	private final static String profTopMargin = "TopMargin";
	private final static String profBottomMargin = "BottomMargin";
	private final static String profLeftMargin = "LeftMargin";
	private final static String profRightMargin = "RightMargin";
	private final static String profTopUnit = "TopUnit";
	private final static String profBottomUnit = "BottomUnit";
	private final static String profLeftUnit = "LeftUnit";
	private final static String profRightUnit = "RightUnit";
	private final static String profHeadingFont = "HeadingFont";
	private final static String profHeadingSize = "HeadingSize";
	private final static String profHeadingBold = "HeadingBold";
	private final static String profHeadingItalic = "HeadingItalic";
	private final static String profHeadingUnderline = "HeadingUnderline";
	private final static String profTextFont = "TextFont";
	private final static String profTextSize = "TextSize";
	private final static String profTextBold = "TextBold";
	private final static String profTextItalic = "TextItalic";
	private final static String profTextUnderline = "TextUnderline";
	private final static String profUseRightsManagement = "UseRightsManagement";
	private final static String profAllowAssembly = "AllowAssembly";
	private final static String profAllowCopy = "AllowCopy";
	private final static String profAllowDegradedPrint = "AllowDegradedPrinting";
	private final static String profAllowModAnnotations = "AllowModifyAnnotations";
	private final static String profAllowModContents = "AllowModifyContents";
	private final static String profAllowPrinting = "AllowPrinting";
	private final static String profAllowScreenreaders = "AllowScrennreaders";

	// captions of labels
	private final static String authorLB = "Author:";
	private final static String titleLB = "Title:";
	private final static String dateLB = "Date:";
	private final static String docSizeLB = "Document Size:";
	private final static String marginsLB = "Margins:";
	private final static String topLB = "Top:";
	private final static String bottomLB = "Bottom:";
	private final static String leftLB = "Left:";
	private final static String rightLB = "Right:";
	private final static String unitLB = "Unit:";
	private final static String headingLB = "Heading:";
	private final static String textLB = "Text:";
	private final static String fontLB = "Font:";
	private final static String sizeLB = "Size:";
	private final static String bold = "bold";
	private final static String italic = "italic";
	private final static String underline = "underline";
	private final static String firstUserPasswdLB = "Enter user password";
	private final static String confirmUserPasswdLB = "Confirm user password";
	private final static String firstOwnerPasswdLB = "Enter owner password";
	private final static String confirmOwnerPasswdLB = "Confirm owner password";
	private final static String allow = "allow ";
	private final static String assembly = "assembly";
	private final static String copy = "copy";
	private final static String degradedPrint = "degraded printing";
	private final static String modAnnotations = "modify annotations";
	private final static String modContents = "modify contents";
	private final static String printing = "printing";
	private final static String screenreaders = "screenreaders";

	// JTextFields
	private final JTextField authorTF = new JTextField();
	private final JTextField titleTF = new JTextField();
	private final JTextField dateTF = new JTextField();
	// for passwords (needed for rights management)
	private final JTextField firstUserPasswdTF = new JTextField();
	private final JTextField confirmUserPasswdTF = new JTextField();
	private final JTextField firstOwnerPasswdTF = new JTextField();
	private final JTextField confirmOwnerPasswdTF = new JTextField();

	// JSpinner
	private final JSpinner topJS = new JSpinner(new SpinnerNumberModel(0.0, 0.0, 10.0, 0.1));
	private final JSpinner bottomJS = new JSpinner(new SpinnerNumberModel(0.0, 0.0, 10.0, 0.1));
	private final JSpinner leftJS = new JSpinner(new SpinnerNumberModel(0.0, 0.0, 10.0, 0.1));
	private final JSpinner rightJS = new JSpinner(new SpinnerNumberModel(0.0, 0.0, 10.0, 0.1));

	// JComboBoxes
	private final JComboBox<?> docSizeCB = new JComboBox<Object>(sizes);
	private final JComboBox<?> headFontCB = new JComboBox<Object>(fonts);
	private final JComboBox<?> headSizeCB = new JComboBox<Object>(fontSizes);
	private final JComboBox<?> textFontCB = new JComboBox<Object>(fonts);
	private final JComboBox<?> textSizeCB = new JComboBox<Object>(fontSizes);
	private final JComboBox<?> topUnitCB = new JComboBox<Object>(units);
	private final JComboBox<?> leftUnitCB = new JComboBox<Object>(units);
	private final JComboBox<?> rightUnitCB = new JComboBox<Object>(units);
	private final JComboBox<?> bottomUnitCB = new JComboBox<Object>(units);

	// JCheckBoxes
	// for font styles
	private final JCheckBox headBold = new JCheckBox(bold);
	private final JCheckBox headItal = new JCheckBox(italic);
	private final JCheckBox headUnder = new JCheckBox(underline);
	private final JCheckBox textBold = new JCheckBox(bold);
	private final JCheckBox textItal = new JCheckBox(italic);
	private final JCheckBox textUnder = new JCheckBox(underline);
	// for rights management
	private final JCheckBox useRightsManagementCB = new JCheckBox("Use rights management");
	private final JCheckBox assemblyCB = new JCheckBox(allow + assembly);
	private final JCheckBox copyCB = new JCheckBox(allow + copy);
	private final JCheckBox degradedPrintCB = new JCheckBox(allow + degradedPrint);
	private final JCheckBox modAnnotationsCB = new JCheckBox(allow + modAnnotations);
	private final JCheckBox modContentsCB = new JCheckBox(allow + modContents);
	private final JCheckBox printingCB = new JCheckBox(allow + printing);
	private final JCheckBox screenreadersCB = new JCheckBox(allow + screenreaders);

	// ArrayList of all selected Plotables in BIGResultTree
	private final List<BIGPlotable> allSelectedPlotables;
	// ArrayList of all CheckBoxes for rights management
	private final List<JCheckBox> allRightsCBs = new ArrayList<JCheckBox>();
	// ArrayList of all TextFields for titles
	private final List<JTextField> allTitleTFs = new ArrayList<JTextField>();
	// ArrayList of all ArchInfos, which should be displayed
	private final List<JCheckBox> allArchInfoCBs = new ArrayList<JCheckBox>();
	// ArrayList of all MeasurementInfos, which should be displayed
	private final List<JCheckBox> allMeasurementInfoCBs = new ArrayList<JCheckBox>();

	private static final long serialVersionUID = 1L;
}
