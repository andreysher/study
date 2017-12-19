/*********************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGKernelScriptWindow.java
 * Author: Robert Wloch Last change by: $Author: tschuet $ $Revision: 1.12 $ $Date: 2007/05/08
 * 09:41:53 $
 *********************************************************************/
package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.tree.*;

import system.*;

/**
 * The BIGKernelScriptWindow provides the content of the kernel script files COMPILE.SH and RUN.SH in an editor panel so
 * it can be displayed by some GUI.
 * <p>
 * This class requires a JTree, that provides the data it needs to operate.<br>
 * If the selected TreePath is a kernel base name instead of a kernel name, this class will provide a CardLayout
 * containing JTabbedPanes with script files. The cards can be choosen via a JComboBox displayed at the top.<br>
 * If the selected TreePath is a kernel name only the JTabbedPane will be displayed.
 * <p>
 * This class works for all JTrees with the following structure:<br>
 * "root node"<br>
 * "basename1"->{"basename1_bla_bla_bla", "basename1_[...]"}<br>
 * "basename2"->{"basename1_[...]"}<br>
 * and so on. In words: For any JTree, whose leaf nodes are the child nodes of nodes that are directly child of the root
 * node with the condition that the names of the leaf nodes of one root child start with the name of that root child.
 * Got it? OK!
 * 
 * @author Robert Wloch <a href="mailto:wloch@zhr.tu-dresden.de">wloch@zhr.tu-dresden.de</a>
 */
public class BIGKernelScriptWindow extends Thread implements Runnable {
	// The referrence to BIGInterface.
	private final BIGInterface bigInterface = BIGInterface.getInstance();

	// The referrence to the console panel.
	private final BIGConsole console = bigInterface.getConsole();

	// The debug level.
	private final int debug = bigInterface.getDebug("BIGGUI");

	// This TreeMap will contain kernels as keys.
	// For each key there will be a TreeMap of the following
	// structure:<br>
	// keys are values of allFiles, values are
	// BIGEditors representing the content of the script files.
	private TreeMap<String, List<BIGEditor>> kernels = null;

	// This list will hold the script file names that will be loaded
	// for each selected kernel. The values will actually not change
	// during runtime.
	private final String[] allFiles = {"PARAMETERS", "COMPILE.SH"/* , "RUN.SH" */};
	// first tab
	private static final String leadingFile = "PARAMETERS";
	// A variable holding the number of kernels.
	private int cardCount = -1;

	// The combo box for choosing the cards of the CardLayout.
	private JComboBox<?> cardComboBox = null;

	// The panel with a cardLayout.
	private JPanel cardPanel = null;

	// The panel with the allFiles elements' contents.
	private JTabbedPane[] tabbedPanes = null;

	// The display panel for a GUI.
	private JComponent labelPart = null;
	private JComponent displayPart = null;
	// this is the above two in one panel for old gui style
	private JComponent displayPanel = null;

	// support source-code-edit?
	private boolean sourceEdit = false;

	// reference to the GUI
	private BIGGUI gui = null;

	private final TreePath[] selection;

	// the observable object.
	private BIGObservable progress;

	private volatile boolean wasInterupted = false;

	/**
	 * The constructor of this class.
	 * 
	 * @param tree the JTree containing the selected TreePath.
	 * @param selection the selected TreePath.
	 * @param sourceEdit whether you want to edit the sourceFiles
	 */

	public BIGKernelScriptWindow(TreePath[] selection, BIGGUI bgui) {
		this(selection, false, bgui);
	}

	public BIGKernelScriptWindow(TreePath[] sel, boolean sourceEdit, BIGGUI bgui) {
		super();
		this.sourceEdit = sourceEdit;
		gui = bgui;
		selection = sel;
	}

	@Override
	public boolean isInterrupted() {
		return wasInterupted || super.isInterrupted();
	}

	@Override
	public void interrupt() {
		wasInterupted = true;
		super.interrupt();
	}

	@Override
	public void run() {
		progress = new BIGObservable();
		progress.addObserver(gui.getStatusProgress());

		gui.getStatusLabel().setText("Preparing file view...");
		progress.setProgress(0);
		// creating the kernels TreeMap
		kernels = new TreeMap<String, List<BIGEditor>>();
		// determining cardCount and filling the kernels TreeMap

		DefaultMutableTreeNode[] node = new DefaultMutableTreeNode[selection.length];
		for (int i = 0; i < selection.length; i++) {
			if (selection[i].getLastPathComponent() instanceof DefaultMutableTreeNode) {
				node[i] = (DefaultMutableTreeNode) selection[i].getLastPathComponent();
			} else {
				System.err.println("Internal Error (Tree with no DMTN)");
				displayPanel = new JPanel();
				return;
			}
		}

		if (isInterrupted())
			return;

		gui.getStatusLabel().setText("Loading kernel files...");
		// fill kernels TreeMap
		if ((node.length == 1) && node[0].isLeaf() && (node[0].getChildCount() == 0)) {
			// only one kernel that's selected
			BIGKernel nodeObj = (BIGKernel) node[0].getUserObject();
			if (debug > 0) {
				System.err.println("NodeUserObj:" + nodeObj);
			}
			List<BIGEditor> editorList = getKernelEditors(nodeObj, 0, 0);
			if (editorList != null) {
				kernels.put(nodeObj.getNameAfterSorting(0), editorList);
				cardCount = 1;
			}
		} else {
			// a subdir is selected -> add all kernels under this subdir
			cardCount = 0;
			// addKernels increases cardCount for each selected kernel
			addKernels(kernels, node);
		}

		if (isInterrupted())
			return;

		gui.getStatusLabel().setText("Initializing components for the display panel...");
		// initialize components for the display panel
		if (cardCount > 0) {
			// if a kernel base name is selected, initialize
			// the cardComboBox and cardPanel
			if (cardCount > 1) {
				// first, get all kernel names
				int numKernels = kernels.size();
				if (numKernels < 2) {
					console.postMessage("Whoops! numKernels is " + "smaller than cardCount!",
							BIGConsole.WARNING);
					displayPanel = new JPanel();
					labelPart = new JLabel();
					displayPart = new JPanel();
				} else {
					String[] kernelNames = new String[numKernels];
					Iterator<String> it = kernels.keySet().iterator();
					int n = 0;
					gui.getStatusLabel().setText("Getting kernel names...");
					while (it.hasNext() && (n < numKernels)) {
						kernelNames[n++] = it.next();
					}

					if (isInterrupted())
						return;

					gui.getStatusLabel().setText("Creating combo box...");
					// second create the cardComboBox
					cardComboBox = new JComboBox<Object>(kernelNames);
					cardComboBox.setToolTipText("choose the kernel out of your selection");
					cardComboBox.addActionListener(new ActionListener() {
						public void actionPerformed(ActionEvent e) {
							JComboBox<?> cb = (JComboBox<?>) e.getSource();
							String kernelName = (String) cb.getSelectedItem();
							((CardLayout) (cardPanel.getLayout())).show(cardPanel, kernelName.toString());
						}
					});
					// third create the cardPanel
					cardPanel = new JPanel(new CardLayout());
					// create the JTabbedPanes
					tabbedPanes = new JTabbedPane[numKernels];
					// forth add the JTabbedPanes to the cardPanel
					for (int i = 0; i < numKernels && !isInterrupted(); i++) {
						gui.getStatusLabel().setText("Loading script files for kernel " + kernelNames[i]);
						tabbedPanes[i] = new JTabbedPane();
						final int actualI = i;
						tabbedPanes[i].addMouseListener(new MouseAdapter() {
							@Override
							public void mouseClicked(MouseEvent evt) {
								if (evt.getClickCount() == 2) {
									String editor = BIGInterface.getInstance().getBIGConfigFileParser()
											.stringCheckOut("fileEditor", "gedit");
									String file = ((BIGEditor) tabbedPanes[actualI].getSelectedComponent())
											.getEditingFile().getAbsolutePath();
									BIGExecute.getInstance().execute(editor + " " + file, false);
								}
							}
						});
						tabbedPanes[i].setName(kernelNames[i]);
						// add a tab for each script file to the JTabbedPane
						progress.setProgress(0);
						String kernelName = kernelNames[i];
						// get the collection of a kernel's script files
						List<BIGEditor> editorList = kernels.get(kernelName);
						int pmax = editorList.size();
						int p = 0;
						for (BIGEditor editor : editorList) {
							if (isInterrupted())
								break;
							progress.setProgress(100 * p++ / pmax);
							// fileName should be used by the tabs in the JTabbedPane
							String fileName = editor.getFileName();
							editor.setCaretPosition(0);
							if (leadingFile.equals(fileName)) {
								tabbedPanes[i].insertTab(fileName, null, editor, null, 0);
							} else {
								tabbedPanes[i].addTab(fileName, editor);
							}
						}
						progress.setProgress(100);
						tabbedPanes[i].setSelectedIndex(0);

						// add the JTabbedPane to the cardPanel
						cardPanel.add(tabbedPanes[i], kernelNames[i].toString());
					}

					if (isInterrupted())
						return;

					progress.setProgress(100);
					gui.getStatusLabel().setText("Done. Composing display panel...");
					// fifth compose the displayPanel
					GridBagConstraints c = new GridBagConstraints();
					JPanel panel = new JPanel(new GridBagLayout());
					JPanel topPanel = new JPanel(new GridBagLayout());
					// java.util.List allFilesList =
					// ( java.util.List ) Arrays.asList( allFiles ) ;
					// in new GUI style no more save button
					/*
					 * saveAction = new BIGSaveAction( "save changes to " + allFilesList.toString() , KeyStroke.getKeyStroke(
					 * KeyEvent.VK_S , ActionEvent.CTRL_MASK ) , true ) ; JButton saveButton = new JButton( saveAction ) ;
					 * saveButton.setText( "Save Changes" ) ;
					 */
					// first add the save button
					c.fill = GridBagConstraints.HORIZONTAL;
					// c.anchor = GridBagConstraints.NORTHWEST ;
					c.gridx = 0;
					c.gridy = 0;
					// panel.add( saveButton , c ) ;
					// second add the cardComboBox
					// c.anchor = GridBagConstraints.NORTHEAST ;
					c.weightx = 1.0;
					// c.gridx = 1 ;
					topPanel.add(cardComboBox, c);
					// third add the cardPanel
					c.fill = GridBagConstraints.BOTH;
					// c.anchor = GridBagConstraints.NORTHWEST ;
					c.weighty = 1.0;
					// c.gridwidth = 2 ;
					// c.gridx = 0 ;
					// c.gridy = 1 ;
					panel.add(cardPanel, c);
					displayPanel = panel;
					labelPart = topPanel;
					displayPart = panel;
				}
			}
			// if a kernel name is selected, initialize
			// the JTabbedPane
			else { // this means: cardCount = 1
				String kernelName = kernels.firstKey();
				// create the JTabbedPane
				tabbedPanes = new JTabbedPane[1];
				tabbedPanes[0] = new JTabbedPane();
				tabbedPanes[0].setName(kernelName);
				tabbedPanes[0].addMouseListener(new MouseAdapter() {
					@Override
					public void mouseClicked(MouseEvent evt) {
						if (evt.getClickCount() == 2) {
							String editor = BIGInterface.getInstance().getBIGConfigFileParser()
									.stringCheckOut("fileEditor", "gedit");
							String file = ((BIGEditor) tabbedPanes[0].getSelectedComponent()).getEditingFile()
									.getAbsolutePath();
							BIGExecute.getInstance().execute(editor + " " + file, false);
						}
					}
				});

				// add a tab for each script file to the JTabbedPane
				gui.getStatusLabel().setText("Loading script files for kernel " + kernelName);
				progress.setProgress(0);
				if (debug > 0) {
					System.err.println("Searching for kernel " + kernelName);
				}
				List<BIGEditor> editorList = kernels.get(kernelName);
				if (debug > 0) {
					System.err.println(kernels);
					System.err.println(editorList);
				}
				int pmax = editorList.size();
				int p = 0;
				for (BIGEditor editor : editorList) {
					if (isInterrupted())
						break;
					progress.setProgress(100 * p++ / pmax);
					// fileName should be used by the tabs in the JTabbedPane
					String fileName = editor.getFileName();
					editor.setCaretPosition(0);
					boolean foundFile = false;
					if (leadingFile.equals(fileName)) {
						foundFile = true;
						tabbedPanes[0].insertTab(fileName, null, editor, null, 0);
					}
					if (!foundFile) {
						tabbedPanes[0].addTab(fileName, editor);
					}
				}
				tabbedPanes[0].setSelectedIndex(0);
				progress.setProgress(100);
				gui.getStatusLabel().setText("Done. Composing display panel...");

				if (isInterrupted())
					return;

				GridBagConstraints c = new GridBagConstraints();
				JPanel panel = new JPanel(new GridBagLayout());
				JPanel topPanel = new JPanel(new GridBagLayout());
				// java.util.List allFilesList =
				// ( java.util.List ) Arrays.asList( allFiles ) ;
				c.gridx = 0;
				c.gridy = 0;
				// first, add the file name as a label
				topPanel.add(new JLabel(kernelName.toString()), c);
				// second add the JTabbedPane
				c.fill = GridBagConstraints.BOTH;
				c.weightx = 1.0;
				c.weighty = 1.0;
				// c.gridy = 1 ;
				panel.add(tabbedPanes[0], c);
				displayPanel = panel;
				labelPart = topPanel;
				displayPart = panel;
			}
		} else {
			displayPanel = new JPanel();
			displayPart = displayPanel;
			labelPart = new JLabel();
		}

		if (isInterrupted())
			return;
		gui.setKernelScriptWindow(this, false);
		if (displayPanel != null) {
			displayPanel.revalidate();
		}
		if (progress != null) {
			progress.setProgress(100);
		}
		gui.getStatusLabel().setText("done");
	}

	/**
	 * For one kernel creates a list of BIGEditors of the script files to show as tabs and edit them.
	 */
	private List<BIGEditor> getKernelEditors(BIGKernel kernel, int progressSpan, int progressStart) {
		// create the script files map for the kernels TreeMap
		List<BIGEditor> editorList = new ArrayList<BIGEditor>();
		// Use the range of progressSpan percent points
		// in the progress bar starting at progressStart
		// percent. Take first half of that span for this
		// for loop, the other half for the next one.
		float progressConst = (float) (1.0 * progressSpan / (2.0 * allFiles.length));
		for (int i = 0; i < allFiles.length; i++) {
			if (progressSpan != 0) {
				progress.setProgress((int) (progressStart + i * progressConst));
				if (i == allFiles.length - 1) {
					progressStart = (int) (progressStart + i * progressConst);
				}
			}
			// construct the script filename
			String scriptFileName = bigInterface.getBenchItPath() + File.separator + "kernel"
					+ File.separator + kernel.getRelativePath() + File.separator + allFiles[i];
			File file = new File(scriptFileName);
			if (!file.exists()) {
				try {
					file.createNewFile();
				} catch (IOException ex) {
					BIGFileHelper.saveToFile("", file);
				}
				System.out.println("Warning: Could not find File " + scriptFileName
						+ ". The behaviour of running this benchmark is unpredictable.");
			}
			BIGEditor editor = new BIGEditor(file);
			editor.setToolTipText("Here you may change some "
					+ "kernel spezific variables, e.g. STEPSIZE or INCREMENT. "
					+ "This is very kernel dependent!");
			editorList.add(editor);
		}
		if (sourceEdit) {
			// source-Code files:
			// path:
			String path = bigInterface.getBenchItPath() + File.separator + "kernel" + File.separator
					+ kernel.getRelativePath() + File.separator;
			if (debug > 0) {
				System.err.println(path);
			}
			File filesWithInPath[] = (new File(path)).listFiles(new FileFilter() {
				public boolean accept(File f) {
					if (f.isDirectory())
						return false;
					for (int i = 0; i < allFiles.length; i++)
						if (allFiles[i].equals(f.getName()))
							return false;
					return BIGKernel.isFilenameWanted(f.getName());
				}
			});
			// Use the range of progressSpan percent points
			// in the progress bar starting at progressStart
			// percent. Take first half of that span for this
			// for loop, the other half for the next one.
			for (int i = 0; i < filesWithInPath.length; i++) {
				if (progressSpan != 0) {
					progress.setProgress((int) (progressStart + i * progressConst));
				}

				// create and set kernels TreeMap entries
				BIGStrings fileStrings = new BIGStrings();
				try {
					fileStrings.readFromFile(path + File.separator + filesWithInPath[i].getName());
				} catch (FileNotFoundException fnfe) {
					console.postMessage("File " + path + File.separator + filesWithInPath[i].getName()
							+ " was not found.", BIGConsole.WARNING);
					return null;
				} catch (IOException ioe) {
					console.postMessage("File " + path + File.separator + filesWithInPath[i].getName()
							+ " could not be read.", BIGConsole.WARNING);
					return null;
				}

				BIGEditor editor = new BIGEditor(filesWithInPath[i]);
				editor.setToolTipText("source to file " + filesWithInPath[i].getName());
				editorList.add(editor);
			}

		}
		return editorList;
	}

	/**
	 * Add all kernels under a subdirectory to a TreeMap. The TreeMap contains TreeMaps with BIGEditors for the scripts of
	 * that kernel.
	 */
	private void addKernels(TreeMap<String, List<BIGEditor>> kernels, DefaultMutableTreeNode[] parent) {
		int count = 0;
		for (int i = 0; i < parent.length; i++) {
			count = count + parent[i].getLeafCount();
		}
		progress.setProgress(0);
		for (int i = 0; i < parent.length; i++) {
			Enumeration<?> enu = parent[i].depthFirstEnumeration();
			while (enu.hasMoreElements()) {
				DefaultMutableTreeNode leaf = (DefaultMutableTreeNode) enu.nextElement();
				Object nodeObj = leaf.getUserObject();
				/* progress.setProgress( 100 * cardCount / count ); */
				// if nodeObj is instance of BIGKernel add that kernel name
				if (nodeObj instanceof BIGKernel) {
					int progSpan = 100 / count++;
					int progStart = 100 * cardCount / count;
					List<BIGEditor> editorList = getKernelEditors((BIGKernel) nodeObj, progSpan, progStart);
					if (editorList == null)
						continue;
					kernels.put(((BIGKernel) nodeObj).getNameAfterSorting(0), editorList);
					cardCount++;
				}
			}
		}
		progress.setProgress(100);
	}

	/**
	 * This method returns the display panels representing this class.<br>
	 * The first element of the array is the main display component. All following elements are JLabels or JComponents for
	 * navigational purposes and should be placed above or underneath the main display part.
	 * 
	 * @return The display panels array.
	 */
	public JComponent[] getDisplayPanels() {
		JComponent[] retval = new JComponent[2];
		retval[0] = displayPart;
		retval[1] = labelPart;
		return retval;
	}

	/**
	 * This method returns the display panel representing this class.<br>
	 * If <code>getCardCount()</code> returns 1 this method returns a JTabbedPane, if <code>getCardCount()</code> returns
	 * a value greater than 1 this method returns a JPanel containing a JComboBox for choosing the card of the CardLayout
	 * of the JPanel that's at <code>BorderLayout.CENTER</code><br>
	 * With <code>getSelectedCard()</code> and <code>getSelectedItem()</code> you can obtain the selection status.
	 * 
	 * @return The display panel.
	 */
	public JComponent getDisplayPanel() {
		return displayPanel;
	}

	/**
	 * This method behaves exactly like the above. However, the returned JComponent's preferred size will be the
	 * dimension's size.
	 * 
	 * @param dim The requested dimension of the display panel.
	 */
	public JComponent getDisplayPanel(Dimension dim) {
		displayPanel.setPreferredSize(dim);
		displayPanel.setMinimumSize(dim);
		return displayPanel;
	}

	/**
	 * This method returns the number of cards the the CardLayout handels. This number also represents the number of
	 * kernels of one kernel base name.<br>
	 * If the returned value equals 1, the display panel is only a JTabbedPane and <code>getSelectedCard()</code> will
	 * return <bold>null</bold>.<br>
	 * If this method returns 0, the no selected kernel could be found in the JTree.
	 * 
	 * @return The number of kernels displayable.
	 */
	public int getCardCount() {
		return cardCount;
	}

	/**
	 * This method returns a list of the script files supported by this class.
	 * 
	 * @return A list of editable files.
	 */
	public String[] getAllFiles() {
		return allFiles;
	}

	/**
	 * This method returns the name of the selected card.<br>
	 * The name maybe <bold>null</bold> if <code>getCardCount()</code> is smaller than 2.<br>
	 * The name of the selected card is the name of the kernel.
	 * 
	 * @return The name of the selected card, or <bold>null</bold>.
	 **/
	public String getSelectedCard() {
		String retval = null;
		if (getCardCount() > 2) {
			retval = (String) cardComboBox.getSelectedItem();
		}
		return retval;
	}

	/**
	 * gets the actual selected Editor which is displayed in this panel
	 * 
	 * @return BIGEditor
	 */
	public BIGEditor getActualEditor() {
		System.err.println(getSelectedCard());
		System.err.println(getSelectedTab());
		if (getSelectedCard() == null)
			return (BIGEditor) tabbedPanes[0].getSelectedComponent();
		else
			return (BIGEditor) tabbedPanes[cardComboBox.getSelectedIndex()].getSelectedComponent();

	}

	/**
	 * This method returns the name of the selected tab. That name is one of the elements of <code>getAllFiles()</code>,
	 * or <bold>null</code> if <code>getCardCount()</code> returns 0.
	 * 
	 * @return The name of the selected tab, or <bold>null</bold>.
	 */
	public String getSelectedTab() {
		String retval = null;
		if (getCardCount() == 1) {
			JTabbedPane tb = (tabbedPanes[0]);
			retval = tb.getTitleAt(tb.getSelectedIndex());
		} else if (getCardCount() > 1) {
			int index = cardComboBox.getSelectedIndex();
			JTabbedPane tb = tabbedPanes[index];
			retval = tb.getTitleAt(tb.getSelectedIndex());
		}
		return retval;
	}

	/**
	 * This method is called by the GUI, if the user chose "save" from the menu or hit the key combination CTRL+S while
	 * the GUI is in its KERNEL state.
	 **/
	public void saveRequest(boolean askForSave) {
		saveChanges(askForSave);
	}

	/**
	 * This method is called by the BIGSaveAction when a request to save the changes was made.
	 */
	private void saveChanges(boolean askForSave) {
		if (kernels == null)
			return;
		if (kernels.keySet() == null)
			return;
		// iterate over the editors and save the content if
		// the text was changed since the editor's creation or
		// the last call to BIGEdito.textSaved()
		Iterator<String> it = kernels.keySet().iterator();
		// whether to ask for saving files
		int result = 1;
		if (!askForSave) {
			result = 0;
		}

		while (it.hasNext()) {
			String kernelName = it.next();
			// get the collection of a kernel's script files
			List<BIGEditor> editorList = kernels.get(kernelName);
			for (BIGEditor editor : editorList) {
				// fileName should be used by the tabs in the JTabbedPane
				String fileName = editor.getFileName();
				// this is the file with it's whole path (File for removing double slashes)
				String pathFileName = editor.getEditingFile().getAbsolutePath();
				if (editor.textChanged()) {
					// that editor's text changed, so save it
					boolean saveSucceeded = true;
					if ((result > 0) && (result < 3)) {
						String[] options = {"Save All", "Save", "Dont save", "Save none"};
						result = JOptionPane.showOptionDialog(gui, "Save file " + fileName,
								"Save changed File", 1, JOptionPane.INFORMATION_MESSAGE, null, options, options[1]);
					}
					if (result < 2) {
						BIGStrings strings = editor.getAllLines();
						// save text to scriptFileName
						strings.saveToFile(pathFileName);
						if (saveSucceeded) {
							// reset the textChanged field
							editor.textSaved();
							console.postMessage("File " + pathFileName + " was successfully saved.",
									BIGConsole.WARNING);
						}
					}
				}
			}
		}
	}

	/**
	 * This class defines the Action done if the "Save Changes" button was clicked.
	 */
	class BIGSaveAction extends AbstractAction {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;

		public BIGSaveAction(String text, String desc, Integer mnemonic) {
			super(text);
			putValue(SHORT_DESCRIPTION, desc);
			putValue(MNEMONIC_KEY, mnemonic);
		}

		public BIGSaveAction(String desc, KeyStroke accelerator, boolean enabled) {
			super();
			putValue(SHORT_DESCRIPTION, desc);
			putValue(ACCELERATOR_KEY, accelerator);
			setEnabled(enabled);
		}

		public BIGSaveAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, icon);
			putValue(SHORT_DESCRIPTION, desc);
			putValue(ACCELERATOR_KEY, accelerator);
			putValue(SMALL_ICON, icon);
			setEnabled(enabled);
		}

		public void actionPerformed(ActionEvent e) {
			saveChanges(false);
		}
	}

	/** This class is used for progress monitoring. */
	private class BIGObservable extends Observable {
		private int oldValue = -1;

		public void setProgress(int value) {
			if (oldValue != value) {
				oldValue = value;
				setChanged();
				notifyObservers(new Integer(value));
				try {
					Thread.sleep(0, 1);
				} catch (InterruptedException ie) {}
			}
		}
	}
}
/*****************************************************************************
 * Log-History
 *****************************************************************************/

