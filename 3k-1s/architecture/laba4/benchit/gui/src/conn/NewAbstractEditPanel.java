package conn;

/**
 * <p>
 * Überschrift: BenchIT
 * </p>
 * <p>
 * Beschreibung: A Panel, which let you select graphs from the server. Very advanced stuff.
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
import java.awt.*;
import java.awt.dnd.*;
import java.awt.event.*;
import java.io.IOException;
import java.util.*;

import javax.swing.*;
import javax.swing.event.*;

import conn.SocketCon.ComboBoxContent;

public class NewAbstractEditPanel extends JSplitPane {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * which is the last item to divergate from eg. you've selected filename as bottomselecta, it won't appear as
	 * combobox, but all filenames, which fulfill upper selections are displayed in the JList at the middle
	 */
	public String bottomSelecta = "Filename";
	/**
	 * communication with the server
	 */
	private SocketCon conn = null;
	/**
	 * whether this panel is actualized
	 */
	private boolean isActualising = false;
	/**
	 * Panel at the bottom containing buttons
	 */
	private BottomPanel bottomPanel = null;
	/**
	 * referrence to call different methods
	 */
	private MainEditPanel root = null;

	/**
	 * Constructor, sets up layout and so on
	 * 
	 * @param socketCon SocketCon for communication with the server
	 * @param root MainEditPanel for some methods
	 * @param identifiers String[] all available identifiers
	 * @param bottomSel String the last difference, used for the list, must be one of the identifiers
	 */
	public NewAbstractEditPanel(SocketCon socketCon, MainEditPanel root, String[] identifiers,
			String bottomSel) {
		// setting variables
		this.root = root;
		conn = socketCon;
		bottomSelecta = bottomSel;
		// top left
		ActiveItemsPanel left = new ActiveItemsPanel(getListForActive(identifiers));
		// top right
		NonActiveItemsPanel right = new NonActiveItemsPanel(getListForInactive(identifiers));
		left.nonActiveItemViewPanel = right;
		right.activePanel = left;
		// panel containing buttons and jlist
		bottomPanel = new BottomPanel(left, right);

		if (!left.hs.isEmpty()) {
			left.updateCombos(null);
		}

		JSplitPane up = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, left, right);
		up.setDividerLocation(400);
		setOrientation(JSplitPane.VERTICAL_SPLIT);
		setTopComponent(up);
		this.setDividerLocation(200);
		setBottomComponent(bottomPanel);
	}

	class ActiveItemsPanel extends JScrollPane {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;
		private NonActiveItemsPanel nonActiveItemViewPanel = null;
		private ActionListener checkBoxListener = null;
		private ActionListener comboBoxListener = null;
		// hashSet which contains vectors for the comboBoxes
		private final HashSet<JComboBox<String>> hs = new HashSet<JComboBox<String>>();

		/**
		 * with this function (and this function only) it is possible to add graphs selected in this NewAbstractPanel
		 * 
		 * @param whichSelectInBottom String which is the bottomSelecta
		 * @param selectedInBottom String[] which selections had been made in JList
		 */
		public void addGraph(final String whichSelectInBottom, final String[] selectedInBottom) {
			(new Thread() {
				@Override
				public void run() {
					// items, which are selected
					// the number of comboboxes here and an additional for the
					// incoming "whichSelectInBottom"
					String[] items = new String[hs.size() + 1];
					String[] settings = new String[hs.size() + 1];
					// used for remembering position in array
					int where = 0;
					Iterator<JComboBox<String>> it = hs.iterator();
					// set the arrays items and settings
					while (it.hasNext()) {
						JComboBox<?> actualItem = (it.next());
						items[where] = actualItem.getName();
						settings[where] = (String) actualItem.getSelectedItem();
						where++;
					}
					// writing last item
					items[where] = whichSelectInBottom;
					// for all selected strings in JList add the graphs
					for (int i = 0; i < selectedInBottom.length; i++) {
						settings[where] = selectedInBottom[i];
						root.addGraph(items, settings);
					}
				}
			}).start();

		}

		public String[][] getAllSettedInformation(String otherItem, String otherSetting) {
			// items, which are selected
			// the number of comboboxes here and an additional for the
			// incoming "whichSelectInBottom"
			String[] items = new String[hs.size() + 1];
			String[] settings = new String[hs.size() + 1];
			// used for remembering position in array
			int where = 0;
			Iterator<JComboBox<String>> it = hs.iterator();
			// set the arrays items and settings
			while (it.hasNext()) {
				JComboBox<?> actualItem = (it.next());
				items[where] = actualItem.getName();
				settings[where] = (String) actualItem.getSelectedItem();
				where++;
			}
			// writing last item
			items[where] = otherItem;
			settings[where] = otherSetting;
			String[][] retStrings = new String[2][];
			retStrings[0] = items;
			retStrings[1] = settings;
			return retStrings;
		}

		/**
		 * add an additional CheckBox. Is called from NonActiveItemsPanel, when a checkbox was selected
		 * 
		 * @param cb JCheckBox the selected Checkbox, which should be added cb must have a name of an item!
		 */
		public void addItem(JCheckBox cb) {
			String saveToCfgStr = new String();
			int scrollBarValue = getVerticalScrollBar().getValue();
			// rebuild layout
			JPanel viewPanel = (JPanel) getViewport().getView();
			JPanel newViewPanel = new JPanel();
			newViewPanel.setBorder(new javax.swing.border.TitledBorder("active select items"));
			GridBagLayout gbl = new GridBagLayout();
			GridBagConstraints c = new GridBagConstraints();
			newViewPanel.setLayout(gbl);
			int x = 0;
			int y = 0;
			// all known checkboxes are re-added here
			while (viewPanel.getComponentCount() > 0) {
				Component com = viewPanel.getComponent(0);
				viewPanel.remove(com);
				if (com instanceof JComboBox) {
					x = 1;
					saveToCfgStr = saveToCfgStr + com.getName() + ",";
				} else {
					x = 0;
				}
				setConstraints(c, x, y, 1, 1);
				addComponent(newViewPanel, com, gbl, c);
				if (com instanceof JComboBox) {
					y++;
				}

			}
			// and here, the new one is added
			if (cb != null) {
				Vector<String> v = new Vector<String>();
				v.add("<any>");
				JComboBox<String> combo = new JComboBox<String>(v);
				combo.setSelectedIndex(0);
				cb.addActionListener(checkBoxListener);
				combo.setName(cb.getText());
				combo.setMaximumRowCount(20);
				combo.setMaximumSize(new Dimension(10, 20));
				y++;
				setConstraints(c, 0, y, 1, 1);
				addComponent(newViewPanel, cb, gbl, c);
				setConstraints(c, 1, y, 1, 1);
				addComponent(newViewPanel, combo, gbl, c);
				hs.add(combo);
				setViewportView(newViewPanel);
				// get information from server is implied
				updateCombos(null);
				combo.addActionListener(comboBoxListener);
				saveToCfgStr = saveToCfgStr + combo.getName() + ",";
			}
			if (saveToCfgStr.length() > 0) {
				saveToCfgStr = saveToCfgStr.substring(0, saveToCfgStr.length() - 1);
			}
			system.BIGInterface.getInstance().getBIGConfigFileParser().set("freeSelect", saveToCfgStr);

			// finalizing new layout
			setViewportView(newViewPanel);
			getVerticalScrollBar().setValue(scrollBarValue);
			revalidate();
		}

		/**
		 * update the content of containing comboboxes
		 * 
		 * @param actuator JComboBox the combobox, which has caused the update not used yet, but hopefully soon. the
		 *          changing combobox don't have to update its content
		 */
		public void updateCombos(JComboBox<?> actuator) {
			// one actualizing at the time others are ignored
			if (!isActualising) {
				// yah i know race condition blahblah
				isActualising = true;
				JPanel viewPanel = (JPanel) viewport.getView();
				// vectors for comboboxes, which have identifiers as names
				// comboboxes which have sth important selected
				Vector<JComboBox<String>> boxes = new Vector<JComboBox<String>>();
				// comboboxes where <any> is selected
				// Vector emptyBoxes = new Vector() ;
				// dividing boxes to filled and empty
				for (int i = 0; i < viewPanel.getComponentCount(); i++) {
					if (viewPanel.getComponent(i) instanceof JComboBox) {
						@SuppressWarnings("unchecked") JComboBox<String> box = (JComboBox<String>) viewPanel
								.getComponent(i);
						boxes.add(box);
					}
				}
				// write settings for filled boxes
				String[] filledBoxSettings = new String[boxes.size()];
				for (int i = 0; i < filledBoxSettings.length; i++) {
					filledBoxSettings[i] = (String) ((JComboBox<?>) boxes.get(i)).getSelectedItem();
				}
				// which comboboxes must be set? (+1 for JList bottomSelecta)
				String[] comboBoxesToSet = new String[boxes.size()];
				if (actuator == null) {
					comboBoxesToSet = new String[boxes.size() + 1];
				}
				// what is setted ...
				String[] settedIdentifiers = new String[boxes.size()];
				// ... to what value
				String[] settedSettings = new String[boxes.size()];
				// where are we in the arrays
				int index = 0;
				// write settings for boxes
				for (int i = 0; i < boxes.size(); i++) {
					if (boxes.get(i) != actuator) {
						comboBoxesToSet[index] = ((JComboBox<?>) boxes.get(i)).getName();
						settedSettings[index] = (String) ((JComboBox<?>) boxes.get(i)).getSelectedItem();
						settedIdentifiers[index++] = ((JComboBox<?>) boxes.get(i)).getName();
					}
				}
				// the bottom must be got to
				comboBoxesToSet[index] = bottomSelecta;
				// if its a new combobox or everything is deselected
				// actuator might be null
				if (actuator != null) {
					settedSettings[index] = (String) actuator.getSelectedItem();
					settedIdentifiers[index++] = actuator.getName();
				}
				ComboBoxContent returnV;
				// get the stuff from server
				try {
					returnV = conn.getComboBoxContent(comboBoxesToSet, settedIdentifiers, settedSettings);
				} catch (IOException ex) {
					return;
				}
				// combobox and bottomselecta names and
				String[] recvdIdentifiers = returnV.identifiers;
				// Vector of String[], content for comboboxes
				Vector<Vector<String>> recvdSettings = returnV.comboContent;
				// write the content for the comboboxes
				for (int i = 0; i < recvdIdentifiers.length - 1; i++) {
					// first the comboboxes with <any>
					for (int j = 0; j < boxes.size(); j++) {
						// if it is the combobox
						if (recvdIdentifiers[i].equals(((JComboBox<?>) boxes.get(j)).getName())) {
							// get the box
							JComboBox<String> box = (boxes.get(j));
							// remove all old
							box.removeAllItems();
							// write new
							Vector<String> content = recvdSettings.get(i);
							for (int k = 0; k < content.size(); k++) {
								box.addItem(content.get(k));
							}
							// select the old selected items
							for (int k = 0; k < content.size(); k++) {
								if (content.get(k).equals(filledBoxSettings[j])) {
									box.setSelectedItem(content.get(k));
									k = content.size();
								}
							}
							box.setMaximumRowCount(20);
							box.setMaximumSize(new Dimension(10, 20));
							box.revalidate();

						}
					}
				}
				// update bottom panel - write new settings to jlist...
				bottomPanel.update(recvdSettings.lastElement());
				revalidate();
				isActualising = false;
				// resetting size
				((JPanel) getViewport().getView()).setMaximumSize(new Dimension(getWidth(), (hs.size() + 1)
						* ((new JComboBox<Object>()).getHeight())));
			}
		}

		/**
		 * Constructor, create an active items panel, which will contain some comboBox in the beginning, later more can be
		 * added or removed
		 * 
		 * @param identifiers String[] identifiers from the server, which shall be start-comboboxes
		 */
		public ActiveItemsPanel(String[] identifiers) {
			// the panel for this.viewport
			final JPanel pan = new JPanel(new GridLayout(identifiers.length, 1));
			pan.setBorder(new javax.swing.border.TitledBorder("active select items"));
			// if a checkbox is selected, it must swap to the NonActivePanel
			checkBoxListener = new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					// which checkbox was clicked
					JPanel pan = (JPanel) getViewport().getView();
					JCheckBox source = (JCheckBox) evt.getSource();
					Component[] comps = pan.getComponents();
					// remove combobox from panel and hashset
					for (int i = 0; i < comps.length; i++) {
						if (source.getText().equals(comps[i].getName())) {
							pan.remove(comps[i]);
							hs.remove(comps[i]);
						}
					}
					// remove al
					source.removeActionListener(source.getActionListeners()[0]);
					// remove it here
					pan.remove(source);
					addItem(null);
					// add it to nonactivepanel
					if (nonActiveItemViewPanel != null) {
						nonActiveItemViewPanel.addItem(source);
					}
				}
			};
			// what if sth else is selected in a combobox update the panel
			comboBoxListener = new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					JComboBox<?> source = (JComboBox<?>) evt.getSource();
					updateCombos(source);
				}
			};
			// start-comboboxes
			GridBagLayout gbl = new GridBagLayout();
			GridBagConstraints c = new GridBagConstraints();
			pan.setLayout(gbl);
			int y = 0;

			for (int i = 0; i < identifiers.length; i++) {
				JCheckBox newCheckBox = new JCheckBox(identifiers[i], true);
				newCheckBox.addActionListener(checkBoxListener);
				Vector<String> v = new Vector<String>();
				v.add("<any>");
				JComboBox<String> combo = new JComboBox<String>(v);
				combo.setSelectedIndex(0);
				combo.setName(newCheckBox.getText());
				combo.setMaximumRowCount(20);
				combo.setMaximumSize(new Dimension(10, 20));
				setConstraints(c, 0, y, 1, 1);
				addComponent(pan, newCheckBox, gbl, c);
				setConstraints(c, 1, y, 1, 1);
				addComponent(pan, combo, gbl, c);
				y++;
				hs.add(combo);
				combo.addActionListener(comboBoxListener);
			}
			setViewportView(pan);
			getVerticalScrollBar().setUnitIncrement(15);
			pan.revalidate();
			revalidate();
		}

		public ActionListener getCheckBoxListener() {
			return checkBoxListener;
		}

		public JPanel getViewportView() {
			return (JPanel) viewport.getView();
		}
	}

	class NonActiveItemsPanel extends JScrollPane {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;
		/**
		 * reference to the active items panel on the left
		 */
		private ActiveItemsPanel activePanel = null;
		/**
		 * listens for change on all checkboxes
		 */
		private ActionListener checkBoxListener = null;

		public NonActiveItemsPanel(String[] identifiers) {
			// the viewport, it will be a panel containing a lot of checkboxes
			final JPanel pan = new JPanel(new GridLayout(identifiers.length, 1));
			pan.setBorder(new javax.swing.border.TitledBorder("inactive select items"));
			// if a checkbox is selected:
			// swap it to activeItemsPanel
			checkBoxListener = new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					JPanel pan = (JPanel) getViewport().getView();
					JCheckBox source = (JCheckBox) evt.getSource();
					// remove it here
					source.removeActionListener(this);

					pan.remove(source);
					// update
					addItem(null);
					// add it there
					if (activePanel != null) {
						activePanel.addItem(source);
					}
				}
			};
			// adding starting comboboxes
			for (int i = 0; i < identifiers.length; i++) {
				JCheckBox newCheckBox = new JCheckBox(identifiers[i]);
				newCheckBox.addActionListener(checkBoxListener);
				pan.add(newCheckBox);
			}
			getVerticalScrollBar().setUnitIncrement(15);
			setViewportView(pan);
		}

		public ActionListener getCheckBoxListener() {
			return checkBoxListener;
		}

		/**
		 * an item is swapped to this panel, or it shall be updated (cb==null)
		 * 
		 * @param cb JCheckBox which checkBox shall be added here
		 */
		public void addItem(JCheckBox cb) {
			int scrollBarValue = getVerticalScrollBar().getValue();
			// get panels
			JPanel viewPanel = (JPanel) getViewport().getView();
			JPanel newViewPanel = new JPanel();
			newViewPanel.setLayout(new GridLayout(viewPanel.getComponentCount() + 1, 1));
			newViewPanel.setBorder(new javax.swing.border.TitledBorder("inactive select items"));
			// GridBagLayout gbl = new GridBagLayout() ;
			// GridBagConstraints c = new GridBagConstraints() ;
			// newViewPanel.setLayout( gbl ) ;
			// int y = 0;
			// new one is on top
			if (cb != null) {
				cb.addActionListener(checkBoxListener);
				// y++;
				// setConstraints( c , 0 , y , 1 , 1 ) ;
				// addComponent( newViewPanel , cb , gbl , c ) ;
				newViewPanel.add(cb);
				// y++;
			}
			// old ones below
			while (viewPanel.getComponentCount() > 0) {
				Component com = viewPanel.getComponent(0);
				viewPanel.remove(com);
				// setConstraints( c , 0 , y , 1 , 1 ) ;
				// addComponent( newViewPanel , com , gbl , c ) ;
				newViewPanel.add(com);
				// y++;

			}

			setViewportView(newViewPanel);
			getVerticalScrollBar().setValue(scrollBarValue);
			revalidate();
		}

		public JPanel getViewportView() {
			return (JPanel) viewport.getView();
		}
	}

	class BottomPanel extends JPanel {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;
		/**
		 * JList to select the last differences
		 */
		private final JList<String> fileNameList = new JList<String>();
		/**
		 * button: add selected function(s)
		 */
		private final JButton addAllButton = new JButton("Add selected function(s)");
		/**
		 * button unselect all in active item panel and list
		 */
		private final JButton resetAllInformation = new JButton("unselect all settings");

		private final InformationThread lastThread = null;
		private final JTextPane informationArea = new JTextPane();

		/**
		 * Constructor creates a Panel with a JList at the top and a button-line below
		 * 
		 * @param a ActiveItemsPanel the activeItemsPanel from top of this
		 * @param na NonActiveItemsPanel the nonActiveItemsPanel from top of this
		 */
		public BottomPanel(final ActiveItemsPanel a, final NonActiveItemsPanel na) {
			fileNameList.setBorder(new javax.swing.border.TitledBorder("remaining results"));
			// get info when sth is selected
			fileNameList.addListSelectionListener(new ListSelectionListener() {
				public void valueChanged(ListSelectionEvent evt) {
					if (((JList<?>) evt.getSource()).getSelectedValue() == null) {
						informationArea.setText("select a specific file to get information");
						informationArea.revalidate();
						return;
					}
					((JList<?>) evt.getSource()).getSelectedIndex();
					if (lastThread != null) {
						synchronized (informationArea) {
							lastThread.setTextArea(null);
						}
					}
					InformationThread t = new InformationThread(informationArea, ((JList<?>) evt.getSource())
							.getSelectedValue().toString());
					// any selected -> dont start
					if (!((JList<?>) evt.getSource()).getSelectedValue().toString().equals("<any>")) {
						t.start();
					} else {
						informationArea.setText("select a specific file to get information");

						informationArea.revalidate();

					}
				}
			});
			new DropTarget(fileNameList, conn.gui.getResultTree().getDropListener());
			DragSource.getDefaultDragSource().createDefaultDragGestureRecognizer(fileNameList,
					DnDConstants.ACTION_COPY_OR_MOVE, getDragGestureListener(a));
			JPanel buttonPan = new JPanel(new FlowLayout(FlowLayout.CENTER));
			resetAllInformation.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					JPanel pan = a.getViewportView();
					Component[] c = pan.getComponents();
					for (int i = 0; i < c.length; i++) {
						if (c[i] instanceof JComboBox) {
							((JComboBox<?>) c[i]).setSelectedIndex(0);
						}
					}
					// reload combobox content
					a.updateCombos(null);
				}
			});
			// add selected function(s)
			addAllButton.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					// storing selected values
					String[] selectedItems = new String[fileNameList.getSelectedIndices().length];
					if (selectedItems.length == 0)
						return;
					for (int i = 0; i < selectedItems.length; i++) {
						selectedItems[i] = fileNameList.getSelectedValuesList().get(i);
					}
					// adding graphs
					a.addGraph(bottomSelecta, selectedItems);
				}
			});
			// building buttonPanel
			buttonPan.add(addAllButton);
			buttonPan.add(resetAllInformation);
			informationArea.setEditable(false);
			GridBagLayout layout = new GridBagLayout();
			setLayout(layout);
			GridBagConstraints constraints = new GridBagConstraints();
			setConstraints(constraints, 0, 0, 1, 1);
			constraints.weightx = 0.7;
			constraints.weighty = 0.8;
			constraints.fill = GridBagConstraints.BOTH;
			addComponent(this, new JScrollPane(fileNameList), layout, constraints);
			setConstraints(constraints, 1, 0, 1, 1);
			constraints.weightx = 0.3;
			addComponent(this, new JScrollPane(informationArea), layout, constraints);
			setConstraints(constraints, 0, 1, 2, 1);
			constraints.weightx = 1.0;
			constraints.weighty = 0.2;
			addComponent(this, buttonPan, layout, constraints);
			revalidate();
		}

		/**
		 * updtaes List with data in fileNames
		 * 
		 * @param fileNames Vector the new data for the JList
		 */
		public void update(Vector<String> fileNames) {
			fileNameList.setListData(fileNames);
			fileNameList.setSelectedIndex(0);
			fileNameList.revalidate();
		}

		private DragGestureListener getDragGestureListener(final ActiveItemsPanel a) {
			DragGestureListener dgl = new DragGestureListener() {
				public void dragGestureRecognized(DragGestureEvent e) {
					if (e.getComponent() == fileNameList) {
						// storing selected values
						String[] selectedItems = new String[fileNameList.getSelectedValuesList().size()];
						if (selectedItems.length == 0)
							return;
						Graph[][] gs = new Graph[selectedItems.length][];
						for (int i = 0; i < selectedItems.length; i++) {
							selectedItems[i] = fileNameList.getSelectedValuesList().get(i);
							String[][] strings = a.getAllSettedInformation(bottomSelecta, selectedItems[i]);
							try {
								gs[i] = conn.receiveGraphsFromServer(strings[0], strings[1]);
							} catch (IOException ex) {
								ex.printStackTrace();
							}
						}
						for (int i = 0; i < selectedItems.length; i++) {
							e.startDrag(null, gs[i][0]);
						}

					} else {}
				}
			};
			return dgl;
		}

	}

	/**
	 * gets the list of items for the active-panel
	 * 
	 * @param identifiers String[] all identifiers
	 * @return String[] identifiers to use
	 */

	private String[] getListForActive(String[] identifiers) {
		String[] newOnes = null;
		system.BIGConfigFileParser p = system.BIGInterface.getInstance().getBIGConfigFileParser();
		String toSet = null;
		try {
			toSet = p.stringCheckOut("freeSelect");
		} catch (Exception ex) {
			p.set("freeSelect", "");
			return new String[0];
		}
		StringTokenizer st = new StringTokenizer(toSet, ",");
		newOnes = new String[st.countTokens()];
		int count = 0;
		for (int i = 0; i < identifiers.length; i++) {
			st = new StringTokenizer(toSet, ",");
			while (st.hasMoreTokens())

				if (identifiers[i].equals(st.nextToken())) {
					newOnes[count++] = identifiers[i];
				}
		}
		return newOnes;

	}

	/**
	 * gets the list of items for the inactive-panel
	 * 
	 * @param identifiers String[] all identifiers
	 * @return String[] identifiers to use
	 */
	private String[] getListForInactive(String[] identifiers) {
		String[] newOnes = null;
		String[] active = getListForActive(identifiers);
		newOnes = new String[identifiers.length - active.length];
		int count = 0;
		for (int i = 0; i < identifiers.length; i++) {
			boolean found = false;
			for (int j = 0; j < active.length; j++) {
				if (identifiers[i].equals(active[j])) {
					found = true;
				}
			}
			if (!found) {
				newOnes[count++] = identifiers[i];
			}
		}
		return newOnes;
	}

	/**
	 * A Thread, which is used to show information for a selected file
	 */
	public class InformationThread extends Thread {
		// where shall the info be shown
		private JTextPane text = null;
		// file to get information for
		private String filename = null;

		/**
		 * constructs a thread to get information for a specific file
		 * 
		 * @param text JTextPane where the information shall be painted to
		 * @param fileName String name of the file to get information for
		 */
		public InformationThread(JTextPane text, String fileName) {
			this.text = text;
			filename = fileName;

		}

		/**
		 * use start() to run the getInformation as a Thread
		 */
		@Override
		public void run() {
			// what we want to have
			String[] toGet = {"Kernelname", "Processor name", "Processor Clock Rate", "Compiler",
					"Compilerflags", "Description"};
			// what we know
			String[] setted = {"Filename"};
			// what is set
			String[] settedName = {filename};
			// receive information
			ComboBoxContent received = null;
			try {
				received = conn.getComboBoxContent(toGet, setted, settedName);
			} catch (IOException ex) {
				System.out.println("There was an error while contacting server.");
				return;
			}
			// build information text
			final StringBuffer textForArea = new StringBuffer();
			textForArea.append("<html>\n");
			textForArea.append("<font size=" + ((new JList<Object>()).getFont().getSize() - 9) + ">\n");

			for (int i = 0; i < setted.length; i++) {
				textForArea.append("<b>" + setted[i] + "</b><br>\n");
				textForArea.append(settedName[i] + "<br>\n");
			}
			toGet = received.identifiers;
			Vector<Vector<String>> entries = received.comboContent;
			for (int i = 0; i < toGet.length; i++) {
				textForArea.append("<b>" + toGet[i] + "</b><br>\n");
				// 1 because no any
				for (int j = 1; j < entries.get(i).size(); j++) {
					textForArea.append(entries.get(i).get(j) + ",");
				}
				// remove last comma
				textForArea.deleteCharAt(textForArea.length() - 1);
				textForArea.append("<br>\n");
			}

			textForArea.append("</html>");
			SwingUtilities.invokeLater(new Thread() {
				@Override
				public void run() {

					if (text != null) {
						synchronized (text) {
							// set and repaint
							text.setEditorKit(new javax.swing.text.html.HTMLEditorKit());
							text.setText(textForArea.toString());
							text.setCaretPosition(0);
							text.revalidate();
						}
					}
				}
			});
		}/*
			 * public void finalize() { System.err.println( "Finalizing " + this.getClass().getName() + "@" + this.hashCode()
			 * + "(" + toString() + ")" ) ; }
			 */

		/**
		 * set the area, where the text shall be set to
		 * 
		 * @param pane JTextPane the textarea to use
		 */
		public void setTextArea(JTextPane pane) {
			text = pane;
		}

	}

	/**
	 * used for layout
	 * 
	 * @param container JComponent container, where component is added
	 * @param component Component component to add
	 * @param gridbag GridBagLayout layout of container
	 * @param c GridBagConstraints constraints of gridbag
	 */
	private static final void addComponent(JComponent container, Component component,
			GridBagLayout gridbag, GridBagConstraints c) {
		gridbag.setConstraints(component, c);
		container.add(component);
	}

	/**
	 * used for Layout (fill and anchor northwest)
	 * 
	 * @param c GridBagConstraints constraints to set
	 * @param x int x-position in c
	 * @param y int y-position in c
	 * @param gw int gridwidth for c
	 * @param gh int gridheight for c
	 */
	private static final void setConstraints(GridBagConstraints c, int x, int y, int gw, int gh) {
		// c.fill = GridBagConstraints.NONE ;
		c.anchor = GridBagConstraints.NORTHWEST;
		c.gridx = x;
		c.gridy = y;
		c.gridwidth = gw;
		c.gridheight = gh;
	}
}
