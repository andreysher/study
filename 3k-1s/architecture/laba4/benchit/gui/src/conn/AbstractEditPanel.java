package conn;

import java.awt.*;
import java.awt.event.*;
import java.io.IOException;
import java.util.Vector;

import javax.swing.*;

import conn.SocketCon.ComboBoxContent;

/**
 * <p>
 * Überschrift: The AbstractEditPanel
 * </p>
 * <p>
 * Beschreibung: Here you can select your Graph with the settings you search for
 * </p>
 * <p>
 * Copyright: Copyright (c) 2004
 * </p>
 * <p>
 * Organisation: ZHR
 * </p>
 * 
 * @author rschoene
 * @version 1.0
 */

public class AbstractEditPanel extends JScrollPane implements ItemListener, ActionListener {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/** needed for no doublefiring a changeEvent */

	private String lastSettedString = new String();
	/** needed for no doublefiring a changeEvent */
	private JComboBox<?> lastChangedComboBox = new JComboBox<Object>();

	/** the panel, where all editpanels can be selected and so on */
	private final MainEditPanel root;

	// this will contain
	// vector v: element at 0:String[] identifiers,elementAt 1:JComboBox[] entries

	private final class ComboBoxes {
		public String[] identifiers;
		public Vector<JComboBox<String>> entries;
	}

	private final ComboBoxes selectedComboBoxes = new ComboBoxes();

	/** communicate over this */
	private final SocketCon socketCon;

	/** clear selection */
	private JButton clearButton;
	/** add graph */
	private JButton addButton;
	/** not used */
	private JButton previewButton;

	/**
	 * constructor
	 * 
	 * @param socketCon the socketCon of the program
	 * @param root the MainEditPanel of the program
	 * @param identifiers identifiers (if you have loaded them from the db)
	 */
	public AbstractEditPanel(SocketCon socketCon, MainEditPanel root, String[] identifiers) {
		this.socketCon = socketCon;
		this.root = root;
		getVerticalScrollBar().setUnitIncrement(15);
		selectedComboBoxes.identifiers = identifiers;
		selectedComboBoxes.entries = new Vector<JComboBox<String>>(identifiers.length);
		fillAllChoices();
	}

	/**
	 * fills the jComboBoxes with selected <any>
	 */
	private void fillAllChoices() {
		// start a thread, so the gui can run on...
		(new Thread() {
			@Override
			public void run() {
				// if the length of comboboxes is zero (none are selected)
				if (selectedComboBoxes.identifiers.length == 0) {
					initLayout();
					enableItemListener();
					return;
				}
				// else: we add anys
				// get name of comboboxes and the number of comboboxes
				String[] anys = new String[selectedComboBoxes.identifiers.length];
				// for all comboboxes
				for (int i = 0; i < anys.length; i++) {
					// if the setting for combobox i is not null
					if (selectedComboBoxes.entries.get(i) != null) {
						// it is disabled
						selectedComboBoxes.entries.get(i).setEnabled(false);
					}
					// and the name of the combobox is setted to any
					anys[i] = "<any>";
				}
				// get from the server
				ComboBoxContent v = null;
				try {
					// send him the combobox names (what is asked),
					// the combobox names (what is defined)
					// and the setting (definitions for 2nd argument)
					v = socketCon.getComboBoxContent(selectedComboBoxes.identifiers,
							selectedComboBoxes.identifiers, anys);
				} catch (IOException ex) {
					// we hope no exception occures
				}
				selectedComboBoxes.entries.clear();
				// for all comboboxes
				for (int i = 0; i < selectedComboBoxes.entries.size(); i++) {
					// set content of combobox i
					selectedComboBoxes.entries.add(new JComboBox<String>(v.comboContent.get(i)));
					// enable combobox i
					selectedComboBoxes.entries.get(i).setEnabled(true);
				}
				// reinit layout
				initLayout();
				// enable change and item listener
				enableItemListener();
			}
		}).start();

	}

	/**
	 * fills the jComboBoxes and lets selected what is
	 */
	private void fillChoices() {
		// disable all comboboxes
		for (int i = 0; i < selectedComboBoxes.entries.size(); i++) {
			selectedComboBoxes.entries.get(i).setEnabled(false);
		}
		// disable buttons
		try {
			addButton.setEnabled(false);
			clearButton.setEnabled(false);
			previewButton.setEnabled(false);
		}
		// what should go wrong???
		catch (Exception ignored) {}
		// to what is setted
		final String[] settedSettings = new String[selectedComboBoxes.identifiers.length];
		// which comboboxes are set
		String[] settedIdentifiers = new String[selectedComboBoxes.identifiers.length];
		// what is interesting to get
		String[] toGetIdentifiers = new String[selectedComboBoxes.identifiers.length];
		// set setted identifiers
		settedIdentifiers = selectedComboBoxes.identifiers;
		// set which identifiers should be get
		toGetIdentifiers = selectedComboBoxes.identifiers;
		// set them to final, so an inner thread can access them
		final String[] finalSettedIdentifiers = settedIdentifiers;
		final String[] finalToGetIdentifiers = toGetIdentifiers;
		// set what is selected in the comboboxes
		for (int i = 0; i < settedSettings.length; i++) {
			settedSettings[i] = (String) (selectedComboBoxes.entries.get(i)).getSelectedItem();
		}
		// the inner thread
		Thread t1 = new Thread() {
			@Override
			public void run() {
				try {
					// get what the server sends back
					ComboBoxContent nameAndEntries = null;
					try {
						// send him which comboboxes are interesting (shall be get)
						// what is set and to what it is set
						nameAndEntries = socketCon.getComboBoxContent(finalToGetIdentifiers,
								finalSettedIdentifiers, settedSettings);
					} catch (NullPointerException ex) {
						// should not occur, but the server is still on :)
						ex.printStackTrace();
						JOptionPane.showMessageDialog(null,
								"The selected item caused an internal error. Please choose another one.");
						// enabling everything
						for (int i = 0; i < selectedComboBoxes.entries.size(); i++) {
							selectedComboBoxes.entries.get(i).setEnabled(true);
						}
						addButton.setEnabled(true);
						clearButton.setEnabled(true);
						return;

					}
					// the new comboboxes
					Vector<JComboBox<String>> jcbs = new Vector<JComboBox<String>>();
					// set the contents of the new boxes
					for (int i = 0; i < nameAndEntries.comboContent.size(); i++) {
						jcbs.add(new JComboBox<String>(nameAndEntries.comboContent.get(i)));
					}
					// remove the old boxes
					selectedComboBoxes.entries = jcbs;
				} catch (java.io.IOException e) {
					// AAAH an error! Which one?
					System.err.println(e);
				}
				// no event shall be occure
				disableItemListener();
				// find the corresponding received comboboxes
				// and set the selected value
				for (int i = 0; i < finalSettedIdentifiers.length; i++) {
					for (int j = 0; j < selectedComboBoxes.identifiers.length; j++) {
						if (finalSettedIdentifiers[i].equals(selectedComboBoxes.identifiers[j])) {
							selectedComboBoxes.entries.get(j).setSelectedItem(settedSettings[i]);
							selectedComboBoxes.entries.get(j).setToolTipText(settedSettings[i]);
						}
					}
				}
				initLayout();
				// now events are welcome again :)
				enableItemListener();
			}
		};
		// start the thread above
		t1.start();

	}

	/**
	 * repaints the panel / resets the panel
	 */
	public void initLayout() {
		int i = 0;
		// where everything is added
		JPanel mainPanel = new JPanel();
		// this is just the scrollpane around the mainPanel
		setViewportView(mainPanel);
		// layout
		GridBagLayout gridbag = new GridBagLayout();
		mainPanel.setLayout(gridbag);
		GridBagConstraints gc = new GridBagConstraints();
		gc.anchor = GridBagConstraints.WEST;
		// for: add all comboboxes to the mainpanel
		for (i = 0; i < selectedComboBoxes.identifiers.length; i++) {

			// Layout:
			// 0 identifier[0] jComboBox(0)
			// 1 identifier[1] jComboBox(1)
			// ...
			// n identifier[n] jComboBox(n)

			// activated ComboBox Vector
			// elementAt(n)=v1
			// v1.elementAt(0)=identifier
			// v1.elementAt(1)=jComboBox
			gc.gridx = 0;
			gc.gridy = i;
			mainPanel.add(new JLabel(selectedComboBoxes.identifiers[i]), gc);
			gc.gridx = 1;
			mainPanel.add(selectedComboBoxes.entries.get(i), gc);
			selectedComboBoxes.entries.get(i).addItemListener(this);
			// this.selectedComboBoxes.add(actComBoxVector);
		}
		// add the buttonpanel to the mainpanel
		// first init
		JPanel buttonPanel = new JPanel();
		GridBagLayout buttonGridbag = new GridBagLayout();
		buttonPanel.setLayout(buttonGridbag);
		GridBagConstraints buttonGc = new GridBagConstraints();
		buttonGc.anchor = GridBagConstraints.CENTER;
		// second create buttons
		clearButton = new JButton("Clear");
		clearButton.setToolTipText("Click here to set all ComboBoxes to \"<any>\"");
		clearButton.addActionListener(this);
		addButton = new JButton("Add");
		addButton.setToolTipText("Click here add the Graph that is defined by your settings");
		addButton.addActionListener(this);
		previewButton = new JButton("Preview");
		previewButton.addActionListener(this);
		// third add all
		buttonGc.gridx = 0;
		buttonGc.gridy = 0;
		buttonPanel.add(clearButton);
		buttonGc.gridx = 1;
		buttonPanel.add(new JPanel());
		buttonGc.gridx = 2;
		buttonPanel.add(addButton, buttonGc);
		buttonGc.gridx = 3;
		gc.weightx = 2;
		gc.gridy = i + 1;
		// last add buttonPanel to mainPanel
		mainPanel.add(buttonPanel, gc);
		// always display scrollbars
		setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
		// more layout
		double[] temp = {.05, .95};
		// +2: one because an array starts with 0 and one because of the buttons
		double[] temp2 = new double[i + 2];
		for (int j = 0; j < temp2.length; j++) {
			temp2[i] = 1 / temp2.length;
		}
		gridbag.columnWeights = temp;
		gridbag.rowWeights = temp2;
	}

	/**
	 * this enables the ItemListeners of all JComboBoxes
	 */
	private void enableItemListener() {
		// for all comboboxes add itemlistener
		for (int i = 0; i < selectedComboBoxes.identifiers.length; i++) {
			selectedComboBoxes.entries.get(i).addItemListener(this);
		}
	}

	/**
	 * this disables the ItemListeners of all JComboBoxes
	 */
	private void disableItemListener() {
		// for all comboboxes remove itemlistener
		for (int i = 0; i < selectedComboBoxes.identifiers.length; i++) {
			selectedComboBoxes.entries.get(i).removeItemListener(this);
		}
	}

	/**
	 * tis is what is called if we select s.th. in a comboBox
	 * 
	 * @param ie the Event, that happens, when s.th. was selected in the combobox
	 */
	public void itemStateChanged(ItemEvent ie) {
		// it may be, that one changed comboBox fires more then one itemEvent
		synchronized (this) {
			// it still may be!!! (do you know why?)
			if (!((lastSettedString.equals(((JComboBox<?>) ie.getSource()).getSelectedItem())) && (lastChangedComboBox == ie
					.getSource()))) {
				// a really interesting event ;)
				// disable following events
				disableItemListener();
				// set the new selected item as last setted string
				lastSettedString = (String) ((JComboBox<?>) ie.getSource()).getSelectedItem();
				// and also save the setted box als last setted
				lastChangedComboBox = (JComboBox<?>) ie.getSource();
				// rebuild (get from server ...)
				fillChoices();
				// enable itemlistener again
				enableItemListener();
			}
		}
	}

	/**
	 * performs some action from the buttons
	 * 
	 * @param evt the Event, which occured
	 */
	public void actionPerformed(ActionEvent evt) {
		synchronized (this) {
			// for all comboboxes: disable!!
			for (int i = 0; i < selectedComboBoxes.identifiers.length; i++) {
				selectedComboBoxes.entries.get(i).setEnabled(false);
			}
			// disable buttons
			addButton.setEnabled(false);
			clearButton.setEnabled(false);
			previewButton.setEnabled(false);
			// which button was pressed?
			String source = ((JButton) evt.getSource()).getText();
			if (source.equals("Clear")) {
				fillAllChoices();
			}
			if (source.equals("Add")) {
				addGraph();
			}
			if (source.equals("Preview")) {
				// could be added one day. COULD!
				// startPreviewWork();
			}
		}
	}

	/**
	 * adds the selected Graph if possible if not it prints s.th on System.err THIS should be used ;)
	 */
	private void addGraph() {
		(new Thread() {
			@Override
			public void run() {
				// settings
				String[] comboBoxSettings = new String[selectedComboBoxes.identifiers.length];
				// setted identifiers
				String[] comboBoxIdentifiers = new String[selectedComboBoxes.identifiers.length];
				// set 'em
				for (int i = 0; i < selectedComboBoxes.identifiers.length; i++) {
					comboBoxIdentifiers[i] = selectedComboBoxes.identifiers[i];
					comboBoxSettings[i] = (String) ((selectedComboBoxes.entries.get(i)).getSelectedItem());
				}
				// add the graph(s)
				root.addGraph(comboBoxIdentifiers, comboBoxSettings);
				// enabling buttons
				addButton.setEnabled(true);
				clearButton.setEnabled(true);
				previewButton.setEnabled(true);
				// enabling comboboxes
				for (int i = 0; i < selectedComboBoxes.entries.size(); i++) {
					selectedComboBoxes.entries.get(i).setEnabled(true);
				}
			}
		}).start();
	}

}
