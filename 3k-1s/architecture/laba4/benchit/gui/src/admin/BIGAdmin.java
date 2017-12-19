/******************************************************************************
 *
 *  B e n c h I T - Performance Measurement for Scientific Applications
 *
 *  BIGAdmin.java
 *
 *  Author: SWTP Nagel 1
 *  Last change by: $Author: tschuet $
 *  $Revision: 1.6 $
 *  $Date: 2008/06/02 10:00:19 $
 *
 ******************************************************************************/
package admin;

import gui.*;

import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.table.DefaultTableModel;

import system.*;

/**
 * The Admintool provides you to edit, install and delete parameters with their definitions, which are used by the
 * Parser. The parameters were written in the global config XML-file. Read the <a href="../../help/index.html">Help</a>
 * for more information.
 * 
 * @author Christina Stoitchkova
 * @author Carsten Luxig <a href="mailto:c.luxig@lmcsoft.com">c.luxig@lmcsoft.com</a>
 */
public class BIGAdmin extends JFrame implements BIGGUIProgressObservable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	// variables
	/** the mainPanel / contentPane of this JFrame */
	private JPanel mainPanel;
	/** points to BIGInterface.getInstance() */
	private BIGInterface bigInterface = BIGInterface.getInstance();
	/** points to this.bigInterface.getConsole() */
	private final BIGConsole console = bigInterface.getConsole();
	/** name of the selected File */
	private String selectedFile = null;

	/** layoutComponent */
	private final GridBagLayout gbLayout = new GridBagLayout();
	/** layoutComponent */
	private final GridBagConstraints gbConst = new GridBagConstraints();

	/** Label, that contains the String "File" */
	private final JLabel fileLabel = new JLabel("File: ");
	/** here you can select the file you edit */
	private JComboBox<String> fileJCB;
	/** contains some buttons (e.g. this.addFileButton) */
	private JPanel windowButtonPanel;
	/** buttons within this.windowButtonPanel */
	private JButton addFileButton, delFileButton, newButton;
	/** buttons for changing entries of a file */
	private JButton okButton, resetButton, deleteButton;
	/** saves the xml-file, when clicked */
	private JButton saveButton /* , closeButton */;
	/** last selected row */
	public int selectedRow;
	/** last selected condition */
	public boolean selectedCondition;

	/** contains the table with content from this.selectedFile */
	private JTable paramTable;
	/** model for this.paramTable */
	private DefaultTableModel paramModel;
	/** column-names for this.paramTable */
	private final String[] columnNames = {"Name", "Type", "Default", "Position", "Detail Level"};
	/** types, a setting can have */
	/** labels for the editing-area (on the right) */
	private final JLabel[] editLabel = new JLabel[13];
	/** names for this.editLabel */
	private final String[] editLabelText = {"Name", "View Name", "Type", "Default Value",
			"Multiple Choice", "Multiple Choice Count", "Position", "Tooltip Text", "Help", "Action",
			"Detail Level", "Necessity", "Active Status"};
	/** shows progressInformation */
	private JLabel statusLabel;

	/** textArea for editing help */
	private final JTextArea textAreaHelp = new JTextArea(3, 40);
	/** textArea for editing tool-tip-text */
	private final JTextArea textAreaToolTipText = new JTextArea(3, 40);
	/** textArea for editing the multiple choices */
	private final JTextArea textAreaMultipleChoice = new JTextArea(3, 40);
	/** textArea for editing action */
	private final JTextArea textAreaAction = new JTextArea(3, 40);
	/** textArea for editing position in the field (LOCALDEFs graphical edit) */
	private final BIGTextField textFieldPos = new BIGTextField("0", 5, BIGTextField.INTEGER);
	/** textArea for editing name */
	private final BIGTextField textFieldName = new BIGTextField(40, BIGTextField.TEXT);
	/** textArea for editing viewname */
	private final BIGTextField textFieldViewName = new BIGTextField(40, BIGTextField.TEXT);
	/** textArea for editing priority (for view-level) */
	private final JComboBox<String> comboBoxDetail = new JComboBox<String>(
			BIGInterface.enumNameToStringArray(DetailLevel.values()));
	/** textArea for editing numbers of multiple choices */
	private final BIGTextField textFieldMultipleChoiceCount = new BIGTextField("0", 5,
			BIGTextField.INTEGER);
	/** textArea for setting default value */
	private final BIGTextField textFieldDefaultValue = new BIGTextField(40, BIGTextField.TEXT);
	/** checkBox: is this item active? */
	private final JCheckBox checkBoxActiveStatus = new JCheckBox();
	/** checkBox: is this item necessary? */
	private final JCheckBox checkBoxNecessity = new JCheckBox();
	/** comboBox: what type does the item have? */
	private final JComboBox<String> comboBoxType = new JComboBox<String>(
			BIGInterface.enumNameToStringArray(BIGType.values()));

	/** currently edited entry */
	private BIGEntry editEntry = null;
	/** currently selected file in combobox */
	private String editFile = null;

	/** addNewFiles to config file? shoul be no. the 3 files are known */
	private boolean addNew = false;
	/** addNewEntries to config file? */
	private boolean addEntries = false;
	protected boolean isModifingEntries;

	/**
	 * creates a new BIGAdminFrame afterwards do the getDisplayPanel() to work with this
	 */
	public BIGAdmin() {
		// check whether it should be possible to add new files to the config
		// information is placed in the config file{
		addNew = bigInterface.getBIGConfigFileParser().boolCheckOut("adminNewLOCALDEFFile", false);
		mainPanel = null;
		// initalize all panels and stuff
		initComponents();
	}
	/**
	 * initializes the Components within the mainPanel, and setting it to this.contentPane
	 */
	private void initComponents() {
		// contains buttons for new files and so on (on the lower left side of the admin-panel)
		windowButtonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 0));

		if (bigInterface == null) {
			bigInterface = BIGInterface.getInstance();
		}
		// if it is allowed to add new files
		if (addNew) {
			// button to add a file-description from LOCALDEF-xml
			addFileButton = new JButton(new AbstractAction("Add File") {
				private static final long serialVersionUID = 1L;

				public void actionPerformed(ActionEvent e) {
					String file = JOptionPane
							.showInputDialog("Type in the (abstract) name of the new file you want "
									+ "to add.\nThe file must have the format " + "\"<hostname>_your_file\".");
					if ((file != null) && !file.equals("")) {
						bigInterface.addFile(file);
						fileJCB.addItem(file);
					}
				}
			});
			// button to delete a file-description from LOCALDEF-xml
			delFileButton = new JButton(new AbstractAction("Delete File") {
				private static final long serialVersionUID = 1L;

				public void actionPerformed(ActionEvent e) {
					String file = fileJCB.getSelectedItem().toString();
					int check = JOptionPane.showConfirmDialog(null,
							"Are you sure to delete the whole file \"" + file + "\" with all parameters?",
							"Delete file", JOptionPane.YES_NO_OPTION);
					if (check == JOptionPane.YES_OPTION) {
						bigInterface.removeFile(file);
						fileJCB.removeItem(file);
					}
				}
			});
		}
		// check if we are allowed to add new entries for files
		try {
			addEntries = bigInterface.getBIGConfigFileParser().boolCheckOut("adminNewDefinitions");
		} catch (Exception ignored) {}
		// if it is allowed (std. is yes)
		if (addEntries) {
			// button to create a new entry
			newButton = new JButton(new AbstractAction("New") {
				private static final long serialVersionUID = 1L;

				public void actionPerformed(ActionEvent e) {
					BIGEntry bigEntryDefault = new BIGEntry("NEW BIGENTRY");
					editEntry = null;
					statusLabel.setText("create new parameter");
					loadValues(bigEntryDefault);
				}
			});
			// delete button
			deleteButton = new JButton(new AbstractAction("Delete") {
				private static final long serialVersionUID = 1L;

				public void actionPerformed(ActionEvent e) {
					int check = JOptionPane.showConfirmDialog(null,
							"Are you sure to delete the selected parameter?", "Delete parameter",
							JOptionPane.YES_NO_OPTION);
					if (check == JOptionPane.YES_OPTION) {
						// if the entry shall be deleted
						int sel = paramTable.getSelectedRow();
						// get the filename
						String file = fileJCB.getSelectedItem().toString();
						// get the entryname
						String entryName = paramTable.getValueAt(sel, 0).toString();
						// remove the row in the jtable
						isModifingEntries = true;
						paramTable.removeRowSelectionInterval(sel, sel);
						isModifingEntries = false;
						bigInterface.removeEntry(file, entryName);
						statusLabel.setText("parameter deleted");
						// reload
						loadEntries(file);
					}
				}
			});
		}

		// save button. saves actual settings
		saveButton = new JButton(new AbstractAction("Save") {
			private static final long serialVersionUID = 1L;

			public void actionPerformed(ActionEvent e) {
				save();
			}
		});

		// OK Button to confirm the changes already made
		okButton = new JButton(new AbstractAction("OK") {
			private static final long serialVersionUID = 1L;

			public void actionPerformed(ActionEvent e) {
				acceptChanges();
			}
		});

		// reset button to undo the changes made
		resetButton = new JButton(new AbstractAction("Reset") {
			private static final long serialVersionUID = 1L;

			public void actionPerformed(ActionEvent e) {
				if (editEntry != null) {
					loadValues(editEntry);
				}
			}
		});

		// tabel model. edit in fields on the left side. not in the table itself
		paramModel = new DefaultTableModel(0, 5) {
			private static final long serialVersionUID = 1L;

			@Override
			public boolean isCellEditable(int row, int col) {
				return false;
			}
		};
		// set some defaults
		paramModel.setColumnIdentifiers(columnNames);
		paramTable = new JTable(paramModel);
		paramTable.setRowSelectionAllowed(true);
		// only one selection. you cannot handle two entries at once
		paramTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		paramTable.getSelectionModel().addListSelectionListener(new ListSelectionListener() {
			public void valueChanged(ListSelectionEvent e) {
				if (e.getValueIsAdjusting() || isModifingEntries)
					return;
				BIGEntry entry = null;
				int edtRow = paramTable.getSelectedRow();
				if (edtRow >= 0) {
					// now load new selected entry
					String entryName = paramTable.getValueAt(edtRow, 0).toString();
					entry = bigInterface.getEntry(fileJCB.getSelectedItem().toString(), entryName);
				}
				if (editEntry != null && !editEntry.equals(entry))
					acceptChanges();
				if (entry == null || entry.equals(editEntry)) {
					return;
				}
				// load the values from the chosen entry
				statusLabel.setText("editing parameter");
				loadValues(entry);
				editEntry = entry;
				// need to save selected file to this entry to accept changes
				// properly after another file was selected
				editFile = fileJCB.getSelectedItem().toString();
			}
		});
		// scrollpane around table
		JScrollPane tableJSP = new JScrollPane(paramTable);
		// allfiles means all filenames
		BIGStrings allFiles = bigInterface.getAllFilenames();
		editFile = allFiles.toArray()[0];
		// select the specifing file here
		fileJCB = new JComboBox<String>(allFiles.toArray());
		fileJCB.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				editFile = fileJCB.getSelectedItem().toString();
				loadEntries(fileJCB.getSelectedItem().toString());
			}
		});
		// if more then a file is specified (which surely is), set the selected
		if (allFiles.size() > 0) {
			selectedFile = allFiles.get(0);
		}
		// load its entries
		loadEntries(selectedFile);

		// init layout
		mainPanel = new JPanel();
		mainPanel.setLayout(gbLayout);

		int ypos = 0;

		addComponent(fileLabel, ypos++, mainPanel, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHWEST, 70, 0);

		addComponent(fileJCB, ypos++, mainPanel, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHWEST, 70, 0);
		createStrut(ypos++, 50, 20, mainPanel);

		addComponent(tableJSP, ypos++, mainPanel, GridBagConstraints.BOTH,
				GridBagConstraints.NORTHWEST, 50, 100);
		if (addNew) {
			windowButtonPanel.add(addFileButton);
			windowButtonPanel.add(delFileButton);
		}
		if (addEntries) {
			windowButtonPanel.add(newButton);
			windowButtonPanel.add(deleteButton);

		}
		windowButtonPanel.add(Box.createHorizontalStrut(20));
		windowButtonPanel.add(saveButton);
		addComponent(windowButtonPanel, ypos++, mainPanel, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHWEST, 70, 0);
		statusLabel = bigInterface.getStatusLabel();
		statusLabel.setText("AdminTool loaded");
		JPanel editPanel = initEditPanel();

		addComponent(editPanel, 0, 1, ypos, 1, mainPanel, GridBagConstraints.BOTH,
				GridBagConstraints.NORTHEAST, 50, 100);

		getContentPane().add(mainPanel, BorderLayout.CENTER);

	}

	/**
	 * Stores the changes made to entries.
	 **/
	private void acceptChanges() {
		try {
			// loading selected entry
			BIGEntry entry = new BIGEntry(textFieldName.getText(),
					BIGType.values()[comboBoxType.getSelectedIndex()]);
			entry.setPos(textFieldPos.getIntegerValue().intValue());
			entry.setActiveStatus(checkBoxActiveStatus.isSelected());
			entry.setViewName(textFieldViewName.getText());
			entry.setToolTipText(textAreaToolTipText.getText());
			entry.setAction(textAreaAction.getText());
			entry.setHelp(textAreaHelp.getText());
			entry.setDetailLevel(DetailLevel.values()[comboBoxDetail.getSelectedIndex()]);
			entry.setNecessity(checkBoxNecessity.isSelected());
			Object defaultObj = null;
			BIGStrings multiple = null;
			// check the type
			try {
				switch (entry.getType()) {
					case None :
						defaultObj = null;
						break;
					case Boolean :
						defaultObj = Boolean.valueOf(textFieldDefaultValue.getText());
						break;
					case Float :
						defaultObj = Float.valueOf(textFieldDefaultValue.getText());
						break;
					case Integer :
						defaultObj = Integer.valueOf(textFieldDefaultValue.getText());
						break;
					case List :
						defaultObj = new BIGStrings(textFieldDefaultValue.getText().split("\n"));
						break;
					case Multiple :
						multiple = new BIGStrings(textAreaMultipleChoice.getText().split("\n"));
						defaultObj = textFieldDefaultValue.getText();
						break;
					case String :
						defaultObj = textFieldDefaultValue.getText();
						break;
					default :
						defaultObj = null;
						break;
				}
			}
			// catched if no number is set in the textfield
			catch (NumberFormatException nfe) {}
			if (entry.getType() != BIGType.None)
				entry.setDefaultValue(defaultObj);
			entry.setMultipleChoice(multiple);
			entry.setMultipleChoiceCount(textFieldMultipleChoiceCount.getIntegerValue().intValue());
			if (editEntry != null) {
				bigInterface.removeEntry(editFile, editEntry.getName());
			}
			try {
				bigInterface.addEntry(editFile, entry);
			} catch (BIGDoubleIdentifierException bdie) {
				JOptionPane.showMessageDialog(null, bdie.getMessage());
				System.out.println(bdie.getMessage());
			}
			editEntry = entry;
			statusLabel
					.setText("changes accepted - click \"Save\" button " + "to make changes permanent");
		} catch (BIGAccessViolationException bave) {
			console.postMessage("An error occured during setting the BIGEntry:\n" + bave,
					BIGConsole.ERROR);
		}
		loadEntries(editFile);
	}

	/**
	 * This method is called by the save button to save the config file.
	 */
	public synchronized void save() {
		saveButton.setEnabled(false);
		if (addNew) {
			newButton.setEnabled(false);
			deleteButton.setEnabled(false);
		}
		okButton.setEnabled(false);
		resetButton.setEnabled(false);
		try {
			statusLabel.setText("Saving...");
			BIGGUIObserverProgress prog = (BIGGUIObserverProgress) bigInterface.getStatusProgress();
			// set this to force the progress bar calling this
			// class' progressFinished() method when the progress
			// is done so the buttons can be reactivated
			prog.setProgressObservable(this);
			bigInterface.save(true); // admin is calling

		} catch (BIGParserException bpe) {
			console.postMessage("An error occured during " + "saving the files:\n" + bpe);
			// progressFinished() ;
		}
	}

	/**
	 * Returns the display panel (content pane) for an integration in another GUI.
	 * 
	 * @return JComponent the content pane
	 */
	public JComponent getDisplayPanel() {
		return mainPanel;
	}

	/** Implementation of the BIGGUIProgressObservable interface. */
	public void progressFinished() {
		loadEntries(fileJCB.getSelectedItem().toString());
		saveButton.setEnabled(true);
		if (addNew) {
			newButton.setEnabled(true);
			deleteButton.setEnabled(true);
		}
		okButton.setEnabled(true);
		resetButton.setEnabled(true);
	}

	private void addComponent(Component comp, int pos, JPanel panel, int fill, int anchor, int wx,
			int wy) {
		addComponent(comp, pos, 0, 1, 1, panel, fill, anchor, wx, wy);
	}

	private void addComponent(Component comp, int ypos, int xpos, int height, int width,
			JPanel panel, int fill, int anchor, int wx, int wy) {
		gbConst.gridx = xpos;
		gbConst.gridy = ypos;
		gbConst.gridheight = height;
		gbConst.gridwidth = width;
		gbConst.fill = fill;
		gbConst.anchor = anchor;
		gbConst.insets = new Insets(1, 3, 1, 3);
		gbConst.weightx = wx;
		gbConst.weighty = wy;
		gbLayout.setConstraints(comp, gbConst);
		panel.add(comp);
	}

	private void createStrut(int pos, int width, int height, JPanel panel) {
		createStrut(pos, 0, 1, 1, width, height, panel);
	}

	private void createStrut(int ypos, int xpos, int gh, int gw, int width, int height, JPanel panel) {
		gbConst.gridx = xpos;
		gbConst.gridy = ypos;
		gbConst.gridheight = gh;
		// align over to cells
		gbConst.gridwidth = gw;
		// resize to max
		gbConst.weightx = 0;
		gbConst.weighty = 0;
		// in both directions
		gbConst.fill = GridBagConstraints.BOTH;
		// always add on NORTH
		gbConst.anchor = GridBagConstraints.NORTH;
		gbConst.insets = new Insets(0, 0, 0, 0);
		// create and add empty panel
		JPanel glue = new JPanel();
		glue.setPreferredSize(new Dimension(width, height));
		gbLayout.setConstraints(glue, gbConst);
		panel.add(glue);
	}
	/**
	 * initiallizes the editPanel (seen on the right side)
	 * 
	 * @return JPanel the initiallized Panel
	 */
	private JPanel initEditPanel() {
		JPanel ret = new JPanel();
		ret.setLayout(gbLayout);
		ret.setBorder(new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED), "Edit Parameter"));

		int ypos = 0;
		for (int i = 0; i < editLabel.length; i++) {
			editLabel[i] = new JLabel(editLabelText[i]);
			addComponent(editLabel[i], ypos++, ret, GridBagConstraints.HORIZONTAL,
					GridBagConstraints.NORTHWEST, 0, 0);
		}

		JPanel editButtonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 0));
		editButtonPanel.add(okButton);
		editButtonPanel.add(resetButton);

		addComponent(editButtonPanel, ypos++, 0, 1, 2, ret, GridBagConstraints.BOTH,
				GridBagConstraints.SOUTHEAST, 0, 10);

		textAreaMultipleChoice.setRows(3);
		textAreaToolTipText.setRows(3);
		textAreaHelp.setRows(3);
		textAreaAction.setRows(3);

		JScrollPane jsMultipleChoice = new JScrollPane(textAreaMultipleChoice);
		JScrollBar horiBar = jsMultipleChoice.getHorizontalScrollBar();
		horiBar.setPreferredSize(new Dimension(horiBar.getPreferredSize().width, 8));
		JScrollBar vertiBar = jsMultipleChoice.getVerticalScrollBar();
		vertiBar.setPreferredSize(new Dimension(8, vertiBar.getPreferredSize().height));

		JScrollPane jsToolTipText = new JScrollPane(textAreaToolTipText);
		horiBar = jsToolTipText.getHorizontalScrollBar();
		horiBar.setPreferredSize(new Dimension(horiBar.getPreferredSize().width, 8));
		vertiBar = jsToolTipText.getVerticalScrollBar();
		vertiBar.setPreferredSize(new Dimension(8, vertiBar.getPreferredSize().height));

		JScrollPane jsHelp = new JScrollPane(textAreaHelp);
		horiBar = jsHelp.getHorizontalScrollBar();
		horiBar.setPreferredSize(new Dimension(horiBar.getPreferredSize().width, 8));
		vertiBar = jsHelp.getVerticalScrollBar();
		vertiBar.setPreferredSize(new Dimension(8, vertiBar.getPreferredSize().height));

		JScrollPane jsAction = new JScrollPane(textAreaAction);
		horiBar = jsAction.getHorizontalScrollBar();
		horiBar.setPreferredSize(new Dimension(horiBar.getPreferredSize().width, 8));
		vertiBar = jsAction.getVerticalScrollBar();
		vertiBar.setPreferredSize(new Dimension(8, vertiBar.getPreferredSize().height));

		ypos = 0;
		addComponent(textFieldName, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(textFieldViewName, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(comboBoxType, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(textFieldDefaultValue, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(jsMultipleChoice, ypos++, 1, 1, 1, ret, GridBagConstraints.BOTH,
				GridBagConstraints.NORTHEAST, 100, 10);
		addComponent(textFieldMultipleChoiceCount, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(textFieldPos, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(jsToolTipText, ypos++, 1, 1, 1, ret, GridBagConstraints.BOTH,
				GridBagConstraints.NORTHEAST, 100, 10);
		addComponent(jsHelp, ypos++, 1, 1, 1, ret, GridBagConstraints.BOTH,
				GridBagConstraints.NORTHEAST, 100, 10);
		addComponent(jsAction, ypos++, 1, 1, 1, ret, GridBagConstraints.BOTH,
				GridBagConstraints.NORTHEAST, 100, 10);
		addComponent(comboBoxDetail, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(checkBoxNecessity, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);
		addComponent(checkBoxActiveStatus, ypos++, 1, 1, 1, ret, GridBagConstraints.HORIZONTAL,
				GridBagConstraints.NORTHEAST, 100, 0);

		// createStrut( ypos++, 250, 20, ret );

		return ret;
	}

	/**
	 * loads entry for the given LOCALDEF-filename
	 * 
	 * @param filename String filenam, for file, that will be loaded
	 */
	private void loadEntries(String filename) {
		if (filename == null || isModifingEntries)
			return;

		isModifingEntries = true;
		String s = null;
		if (paramTable.getSelectedRow() >= 0) {
			s = (String) paramTable.getValueAt(paramTable.getSelectedRow(), 0);
		}
		if (statusLabel == null)
			statusLabel = bigInterface.getStatusLabel();
		statusLabel.setText("Parameter for file \"" + filename + "\" loaded");
		// clear the table first
		paramModel.setRowCount(0);
		ArrayList<BIGEntry> entries = bigInterface.getAllEntries(filename);
		for (int i = 0; i < entries.size(); i++) {
			BIGEntry entry = entries.get(i);
			// if (entry.getType() != BIGType.None) {
			Object[] row = new Object[5];
			row[0] = entry.getName();
			row[1] = entry.getType().name();
			row[2] = entry.getDefaultValue();
			row[3] = new Integer(entry.getPos());
			row[4] = entry.getDetailLevel().name();
			paramModel.addRow(row);
			// }
		}
		for (int i = 0; i < paramTable.getRowCount(); i++)
			if (paramTable.getValueAt(i, 0).equals(s)) {
				paramTable.setRowSelectionInterval(i, i);
			}
		isModifingEntries = false;
	}

	// load the values of the entry
	private void loadValues(BIGEntry entryToEdit) {
		textFieldPos.setValue("" + entryToEdit.getPos());
		textFieldPos.setCaretPosition(0);
		textFieldName.setText(entryToEdit.getName());
		textFieldName.setCaretPosition(0);
		comboBoxType.setSelectedItem(entryToEdit.getType().name());
		checkBoxActiveStatus.setSelected(entryToEdit.getActiveStatus());
		textFieldViewName.setText(entryToEdit.getViewName());
		textFieldViewName.setCaretPosition(0);
		textAreaHelp.setText(entryToEdit.getHelp());
		textAreaHelp.setCaretPosition(0);
		textAreaToolTipText.setText(entryToEdit.getToolTipText());
		textAreaToolTipText.setCaretPosition(0);
		comboBoxDetail.setSelectedItem(entryToEdit.getDetailLevel().name());
		checkBoxNecessity.setSelected(entryToEdit.getNecessity());

		if (entryToEdit.getType() == BIGType.List) {
			String[] all = ((BIGStrings) entryToEdit.getDefaultValue()).toArray();
			String str = "";
			for (int i = 0; i < all.length; i++) {
				str += all[i] + ((i < all.length - 1) ? "\n" : "");
			}
			textFieldDefaultValue.setText(str);
		} else if (entryToEdit.getDefaultValue() == null) {
			textFieldDefaultValue.setText("");
		} else {
			textFieldDefaultValue.setText("" + entryToEdit.getDefaultValue());
		}

		if ((entryToEdit.getMultipleChoice() != null)
				&& ((entryToEdit.getType() == BIGType.Multiple) || (entryToEdit.getType() == BIGType.List))) {
			String[] all = entryToEdit.getMultipleChoice().toArray();
			String str = "";
			for (int i = 0; i < all.length; i++) {
				str += all[i] + ((i < all.length - 1) ? "\n" : "");
			}
			textAreaMultipleChoice.setText(str);
		} else {
			textAreaMultipleChoice.setText("");
		}

		textFieldMultipleChoiceCount.setText("" + entryToEdit.getMultipleChoiceCount());
		textAreaAction.setText("" + ((entryToEdit.getAction() == null) ? "" : entryToEdit.getAction()));
	}
}