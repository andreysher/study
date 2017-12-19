/*********************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGResultTree.java Author:
 * Robert Wloch Last change by: $Author: tschuet $ $Revision: 1.32 $ $Date: 2009/01/07 12:07:27 $
 *********************************************************************/
package gui;

import java.awt.*;
import java.awt.datatransfer.*;
import java.awt.dnd.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import java.util.zip.*;

import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

import org.benchit.bitconnect.BITConnectMain;

import plot.data.*;
import system.*;
import conn.Graph;

/**
 * The BIGResultTree provides neccessary functionality for the results an their tree view.
 **/
public class BIGResultTree extends BIGTree {
	private static final long serialVersionUID = 1L;

	public final static String createReportMenuText = "Create report";

	Cursor standard = null;
	// is used by inner class (popup), and
	private boolean text = true;
	/** Reference to the BIGInterface. */
	private final BIGInterface bigInterface = BIGInterface.getInstance();
	// sorting for
	// 0:type
	// 1:name
	// 2:language
	// 3:parallel libraries
	// 4:other libraries
	// 5:data type
	private int newSorting = 0;

	/**
	 * The constructor.
	 * 
	 * @param newModel the TreeModel used in the tree.
	 * @param gui the BIGGUI this tree is added to
	 **/
	public BIGResultTree(TreeModel newModel, BIGGUI gui) {
		super(newModel, gui);

		createToolTip();

		// get the initial sorting from BGUI.cfg
		try {
			newSorting = bigInterface.getBIGConfigFileParser().intCheckOut("lastResultSortOrder");
		} catch (Exception e) {
			newSorting = 0;
		}
		BIGResultMixer[] mixers = gui.getResultMixer();
		// System.err.println(mixers);
		for (int i = 0; i < mixers.length; i++) {
			// System.err.println("i:"+mixers[i]);
			mixers[i].setResultTree(this);
		}
		addDragNDrop();
	}

	/**
	 * remove a Node
	 * 
	 * @param lastPathComp DefaultMutableTreeNode the node to remove
	 * @param ask boolean ask whether to delete the files
	 * @return boolean succesfully removed?
	 */
	private boolean removeNode(DefaultMutableTreeNode lastPathComp, boolean ask) {
		if (lastPathComp.getUserObject() instanceof BIGOutputFile) {
			BIGOutputFile file = (BIGOutputFile) lastPathComp.getUserObject();
			// show last chance before deleting dialog
			if (ask) {
				if (JOptionPane.showConfirmDialog(gui,
						"Do you really want to delete the files of the selected measurements?",
						"Delete results", JOptionPane.YES_NO_OPTION) == JOptionPane.NO_OPTION)
					return false;
			}
			file.delete();
			// remove the result node in the tree
			DefaultMutableTreeNode parent = ((DefaultMutableTreeNode) (lastPathComp.getParent()));
			parent.remove(lastPathComp);
			updateResultTree(parent);
			while (parent.getChildCount() == 0) {
				lastPathComp = parent;
				parent = ((DefaultMutableTreeNode) (lastPathComp.getParent()));
				if (parent == null)
					return false;
				parent.remove(lastPathComp);
				updateResultTree(parent);
			}
			return true;
		}
		if (((DefaultMutableTreeNode) (lastPathComp.getParent())).getUserObject() instanceof BIGResultMixer) {
			BIGResultMixer brm = (BIGResultMixer) ((DefaultMutableTreeNode) (lastPathComp.getParent()))
					.getUserObject();
			brm.removeSubNode(lastPathComp.toString());
		}
		return true;
	}

	/**
	 * Sets the sorting field of the tree to the given value. Valid values are listed in the Fields section of this class'
	 * documentation and are subject to change. An invalid sorting flag will default to BIGResultTree.SORT_BY_NAME. If the
	 * new sorting is different than the current one the tree will be updated, otherwise nothing will happen.
	 * 
	 * @param newSorting The new sort order for the result tree.
	 **/
	private void setSorting(int newSorting) {
		this.newSorting = newSorting;
		bigInterface.getBIGConfigFileParser().set("lastResultSortOrder", "" + newSorting);
		fillTree();
	}

	public int getSorting() {
		return newSorting;
	}

	/**
	 * This is called by the BIGKernelTree to tell the result tree a change of sorting state. The BIGResultTree will
	 * ignore the kernel tree's new sorting as long as it's not in the SORT_BY_KERNEL_TREE sorting state.
	 **/
	public void kernelSortingChanged() {
		updateResultTree(null);
	}

	/**
	 * Initital setup of the kernel tree. Should be called only once. Use updateKernelTree for later updates and reloads.
	 **/
	public void setupTree() {
		// get sort order before last GUI exit
		try {
			newSorting = bigInterface.getBIGConfigFileParser().intCheckOut("lastResultSortOrder");
		} catch (Exception e) {
			newSorting = 0;
		}
		setRootVisible(false);
		setEditable(false);
		setShowsRootHandles(true);
		setCellRenderer(new ResultTreeCellRenderer(gui));
		// setExpandsSelectedPaths( true );
		DefaultTreeSelectionModel selectionModel = new DefaultTreeSelectionModel();
		selectionModel.setSelectionMode(TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);
		setSelectionModel(selectionModel);
		// fills the tree and returns the path to the root node
		fillTree();
		updateUI();
	}

	/**
	 * Selects the BRM, updates the tree and forces display of BRM.
	 **/
	public void showBRM(BIGResultMixer brm) {
		// updateResultTree() ;
		gui.setResultMixer(brm);
	}

	public void removePlotInformations() {
		Vector<BIGOutputFile> v = new Vector<BIGOutputFile>();
		readOutputFiles(v, new File(BIGInterface.getInstance().getOutputPath()));
		for (int i = 0; i < v.size(); i++) {
			v.get(i).removeSavedInformations();
		}
	}

	/**
	 * Updates the result tree.
	 * 
	 * @param node the node to update
	 **/
	public void updateResultTree(TreeNode node) {
		DefaultTreeModel model = ((DefaultTreeModel) getModel());
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) (model.getRoot());

		Vector<BIGResultMixer> mixer = new Vector<BIGResultMixer>();
		if (node == null || node == rootNode) {
			for (int i = 0; i < rootNode.getChildCount(); i++) {
				if (((DefaultMutableTreeNode) rootNode.getChildAt(i)).getUserObject() instanceof BIGResultMixer) {
					mixer.add((BIGResultMixer) ((DefaultMutableTreeNode) rootNode.getChildAt(i))
							.getUserObject());
					rootNode.remove(i);
					i--;
				}
			}

			if (node == null) {
				fillTree();
			}
			for (int i = 0; i < mixer.size(); i++) {
				rootNode.add(mixer.get(i).getTreeNode());
			}
		}

		if (node != null)
			model.reload(node);
		updateUI();
	}

	/**
	 * Fills the tree's model with content.
	 * 
	 * @return the TreePath to the root node.
	 **/
	private TreePath fillTree() {
		DefaultTreeModel model = (DefaultTreeModel) getModel();
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) (model.getRoot());

		TreePath[] oldSelection = getSelectionPaths();
		Enumeration<TreePath> oldExp = getExpandedDescendants(new TreePath(rootNode));

		rootNode.removeAllChildren();
		File outputDir = new File(bigInterface.getOutputPath());

		buildTree(rootNode, outputDir);
		model.setRoot(rootNode);

		expand(oldExp);
		setSelectionPathsMutable(oldSelection);

		return new TreePath(model.getPathToRoot(rootNode));
	}

	/**
	 * Sets up a TreeSelectionListener and a MouseListener and adds them to the tree.
	 **/
	@Override
	public void setListeners() {
		super.setListeners();
		// adds tree selection listener for view display change
		addTreeSelectionListener(new TreeSelectionListener() {
			public void valueChanged(TreeSelectionEvent tse) { // called on selection change only
				gui.setGUIState(GUIState.RESULTS);
				TreePath path = null;
				if (tse.isAddedPath()) {
					// path was added to the selection
					path = tse.getPath();
				}
				if (path == null)
					return;
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) path.getLastPathComponent();
				final Object obj = node.getUserObject();
				// !!System.err.println("UserObject:"+obj.toString());
				if (obj instanceof String) {
					// node
				} else {
					// leaf
					if (obj instanceof BIGOutputFile) {
						gui.loadResult((BIGOutputFile) obj, true);
					} else { // BRM stuff
						if (obj instanceof BIGResultMixer) {
							gui.setResultMixer((BIGResultMixer) obj);
							// BRM node
						} else {
							if (obj instanceof conn.Graph) {
								gui.setResultMixer((BIGResultMixer) ((DefaultMutableTreeNode) node.getParent())
										.getUserObject());
								// BRM subnode
							} else {
								System.err.println("Unknown node in Resulttree:\n" + obj);
							}

						}
					}
				}
			}
		}); // end of addTreeSelectionListener()

		// JTree MUST be registered manually to display tool tips
		ToolTipManager.sharedInstance().registerComponent(this);
	}

	public void updateBRM(BIGResultMixer brm) {
		DefaultTreeModel model = (DefaultTreeModel) getModel();
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) model.getRoot();

		model.nodeChanged(brm.getTreeNode());
		model.nodeChanged(rootNode);
		((DefaultTreeModel) getModel()).reload(brm.getTreeNode());
		gui.setResultMixer(brm);
	}

	/**
	 * Searches the model of the tree for the TreeNode specified by the TreePath.
	 * 
	 * @param selectedPath The TreePath to find the node for.
	 * @return the DefaultMutableTreeNode or null, if node not found.
	 **/
	public DefaultMutableTreeNode findNode(TreePath selectedPath) {
		DefaultMutableTreeNode result = null;
		String nodeName = selectedPath.getLastPathComponent().toString();
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) getModel().getRoot();
		// find tree node of selected node in the tree
		boolean nodeFound = false;
		DefaultMutableTreeNode parent = rootNode;
		for (int k = 1; k < selectedPath.getPathCount(); k++) {
			Object pathObj = selectedPath.getPathComponent(k);
			if (pathObj == null) {
				break;
			}
			int end = parent.getChildCount();
			boolean childFound = false;
			for (int l = 0; l < end; l++) {
				DefaultMutableTreeNode childNode = (DefaultMutableTreeNode) parent.getChildAt(l);
				if (pathObj.toString().equals(childNode.toString())) {
					childFound = true;
					parent = childNode;
					if (nodeName.equals(childNode.toString())) {
						result = childNode;
						nodeFound = true;
					}
					break;
				}
			}
			if (!childFound || nodeFound) {
				break;
			}
		}
		return result;
	}

	/**
	 * This method should return true if a popup menu should be shown for a certain TreeNode or component.
	 * 
	 * @param me The MouseEvent trying to invoke the popup menu.
	 * @param name The identifier name of the BIGPopupMenu trying to popup.
	 * @return a popup is shown
	 **/
	public boolean showPopup(MouseEvent me, String name) {
		boolean retval = false;
		if (name.compareTo("ResultPopup") == 0) {
			TreePath selPath = getPathForLocation(me.getX(), me.getY());
			// if the path expander is clicked, no path is selected
			if (selPath != null) {
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) selPath.getLastPathComponent();
				if (node.getUserObject() instanceof BIGOutputFile) {
					retval = true;
				}
			}
		} else if (name.compareTo("BRMPopup") == 0) {
			TreePath selPath = getPathForLocation(me.getX(), me.getY());
			// if the path expander is clicked, no path is selected
			if (selPath != null) {
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) selPath.getLastPathComponent();
				if (node.getUserObject() instanceof BIGResultMixer) {
					retval = true;
				}
			}
		} else if (name.compareTo("BRMSubPopup") == 0) {
			TreePath selPath = getPathForLocation(me.getX(), me.getY());
			// if the path expander is clicked, no path is selected
			if (selPath != null) {
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) selPath.getLastPathComponent();
				if (node.getUserObject() instanceof conn.Graph) {
					retval = true;
				}

			}
		} else if (name.compareTo("SortPopup") == 0) {
			TreePath selPath = getPathForLocation(me.getX(), me.getY());
			// the sort only popup is only shown, if all the others can't be shown.
			if (selPath != null) {
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) selPath.getLastPathComponent();
				if (node.getUserObject() instanceof String) {
					retval = true;
				}
				/*
				 * Object obj = selPath.getLastPathComponent(); Object parent = selPath.getPathComponent( selPath.getPathCount()
				 * - 2 ); if ( obj == null ) { retval = true; } else if ( !this.gui.getResults().contains( obj.toString() ) &&
				 * !obj.toString().startsWith( "BRM" ) ) { if ( parent == null ) { retval = true; } else if (
				 * !parent.toString().startsWith( "BRM" ) ) { retval = true; } }
				 */
			}
		}
		return retval;
	}

	/**
	 * The result tree's cell renderer. This class chooses the appropriate icons for tree nodes and displays tool tip
	 * texts if the mouse cursor is above a tree node.
	 **/
	class ResultTreeCellRenderer extends DefaultTreeCellRenderer {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		ImageIcon resultIcon;
		ImageIcon resultBaseOpenedIcon;
		ImageIcon resultBaseClosedIcon;
		ImageIcon onlineMixerIcon;
		BIGGUI big;

		/**
		 * Constructs a new tree cell renderer.
		 * 
		 * @param gui the gui where this renderers tree is added to
		 **/
		public ResultTreeCellRenderer(BIGGUI gui) {
			big = gui;
			resultIcon = new ImageIcon(bigInterface.getImgPath() + File.separator + "result16.png");
			resultBaseOpenedIcon = new ImageIcon(bigInterface.getImgPath() + File.separator
					+ "result_opened16.png");
			resultBaseClosedIcon = new ImageIcon(bigInterface.getImgPath() + File.separator
					+ "result_closed16.png");
			onlineMixerIcon = new ImageIcon(bigInterface.getImgPath() + File.separator + "webmixer.png");

		}

		/**
		 * Overriding super class' method.
		 **/
		@Override
		public java.awt.Component getTreeCellRendererComponent(JTree tree, Object value, boolean sel,
				boolean expanded, boolean leaf, int row, boolean hasFocus) {
			super.getTreeCellRendererComponent(tree, value, sel, expanded, leaf, row, hasFocus);
			DefaultMutableTreeNode node = (DefaultMutableTreeNode) value;
			if (node.getUserObject() instanceof BIGOnlineMixer) {
				setIcon(onlineMixerIcon);
				return this;
			}
			setIcon(resultIcon); // default node icon
			if (node.equals(node.getRoot())) {
				// set icon for root node although it's not visible
				setIcon(resultBaseOpenedIcon);
			} else if (!node.isLeaf()) {
				// set icons for parent nodes
				if (expanded) {
					setIcon(resultBaseOpenedIcon);
				} else {
					setIcon(resultBaseClosedIcon);
				}
			}
			return this;
		}

		/*
		 * private int countLeaves( TreeNode node ) { int retval = 0; int count = node.getChildCount(); if ( count > 0 ) {
		 * for ( int i = 0; i < count; i++ ) { TreeNode child = node.getChildAt( i ); if ( child.isLeaf() ) { if (
		 * ((DefaultMutableTreeNode)child).getUserObject().toString().endsWith( ".bit" ) ) { retval++; } } else { retval +=
		 * countLeaves( child ); } } } return retval; }
		 */
	}

	/**
	 * (re)builds this' content
	 * 
	 * @param rootNode DefaultMutableTreeNode the node, where to add the results from the directory
	 * @param directory File the directory which is checked for result
	 */
	public void buildTree(final DefaultMutableTreeNode rootNode, final File directory) {
		Vector<BIGOutputFile> outputFilesVec = new Vector<BIGOutputFile>();
		readOutputFiles(outputFilesVec, directory);
		// sort and insert kernels
		Vector<BIGOutputFile> sortedVec = new Vector<BIGOutputFile>();
		// get min
		while (outputFilesVec.size() > 0) {
			int position = 0;
			String lastkernelString = null;

			for (int i = 0; i < outputFilesVec.size(); i++) {
				if (lastkernelString == null) {
					position = i;
					lastkernelString = outputFilesVec.get(i).getFile().getAbsolutePath();
				} else if (outputFilesVec.get(i).getFile().getAbsolutePath().compareTo(lastkernelString) < 0) {
					position = i;
					lastkernelString = outputFilesVec.get(i).getFile().getAbsolutePath();
				}
			}
			sortedVec.add(outputFilesVec.get(position));
			outputFilesVec.removeElementAt(position);
		}
		for (int i = 0; i < sortedVec.size(); i++) {
			addOutputFile(rootNode, sortedVec.get(i));
		}

		// gui.getResultMixer().setResultTree( this );
		BIGResultMixer[] mixers = gui.getResultMixer();
		for (int i = 0; i < mixers.length; i++) {
			rootNode.add(mixers[i].getTreeNode());
		}
	}

	/**
	 * builds a menu for this tree for sorting
	 * 
	 * @return JMenu the sorting menu
	 */
	private JMenu getSortMenu() {
		JMenu sortMenu = new JMenu("Sort by");
		JMenuItem item = new JMenuItem("type");
		ActionListener sortListener = new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				JMenuItem source = (JMenuItem) evt.getSource();
				// !!System.err.println(source.getText());
				for (int i = 0; i < getSortMenu().getItemCount(); i++) {
					// !!System.err.println( ( ( JMenuItem ) getSortMenu().getItem( i ) ).
					// !! getText()) ;
					if (getSortMenu().getItem(i).getText().equals(source.getText())) {
						// !!System.err.println("found");
						// !!System.err.println("set sorting to "+i);
						setSorting(i);
						break;

					}
				}
			}
		};
		item.addActionListener(sortListener);
		sortMenu.add(item);
		item = new JMenuItem("name");
		item.addActionListener(sortListener);
		sortMenu.add(item);
		item = new JMenuItem("language");
		item.addActionListener(sortListener);
		sortMenu.add(item);
		item = new JMenuItem("parallelization libraries");
		item.addActionListener(sortListener);
		sortMenu.add(item);
		item = new JMenuItem("other libraries");
		item.addActionListener(sortListener);
		sortMenu.add(item);
		item = new JMenuItem("data type");
		item.addActionListener(sortListener);
		sortMenu.add(item);
		return sortMenu;
	}

	/**
	 * adds an outputfile to a node by adding all the nodes, which are directories as subnodes between and the file as
	 * leaf
	 * 
	 * @param rootNode DefaultMutableTreeNode here the output file (with its directories) is added (should be the
	 *          rootnode)
	 * @param outFile BIGOutputFile the file, you want to add
	 */
	private void addOutputFile(DefaultMutableTreeNode rootNode, BIGOutputFile outFile) {
		// System.err.println("New Sorting:"+this.newSorting);
		String name = outFile.getNameAfterSorting(newSorting) + "."
				+ outFile.getFilenameWithoutExtension();
		// System.err.println("Output-file:\n"+name);
		DefaultMutableTreeNode tempNode = rootNode;
		StringTokenizer stok = new StringTokenizer(name, ".");
		String part;
		boolean found;
		// first the folders
		for (int j = 0; j < outFile.getSortedNames(0).length; j++) {
			part = stok.nextToken();
			found = false;

			for (int i = 0; i < tempNode.getChildCount(); i++) {
				if (((DefaultMutableTreeNode) tempNode.getChildAt(i)).toString().equals(part)) {
					tempNode = (DefaultMutableTreeNode) tempNode.getChildAt(i);
					found = true;
					break;
				}
			}
			if (!found) {
				DefaultMutableTreeNode newNode = new DefaultMutableTreeNode(part);
				tempNode.insert(newNode, tempNode.getChildCount());
				updateResultTree(tempNode);
				tempNode = newNode;
			}
		}
		// then the filename
		part = stok.nextToken() + '.';
		while (stok.hasMoreTokens()) {
			part = part + stok.nextToken() + '.';
		}
		// remove last '.'
		part = part.substring(0, part.length() - 1);
		// part=part.substring(0,part.lastIndexOf(".bit"));
		found = false;

		for (int i = 0; i < tempNode.getChildCount(); i++) {
			if (((DefaultMutableTreeNode) tempNode.getChildAt(i)).toString().equals(part)) {
				found = true;
				break;
			}
		}
		if (!found) {
			DefaultMutableTreeNode newNode = new DefaultMutableTreeNode(outFile);
			tempNode.insert(newNode, tempNode.getChildCount());
			updateResultTree(tempNode);
		}

	}

	/**
	 * reads all the output-files within a directory and adds the BIGOutputFile to a Vector
	 * 
	 * @param v Vector will contain the outputfiles within a directory
	 * @param directory File the directory where you want to search for output files
	 */
	public void readOutputFiles(Vector<BIGOutputFile> v, File directory) {
		File[] subFiles = directory.listFiles(new FileFilter() {
			public boolean accept(File f) {
				if (f.isDirectory())
					if (!f.getName().equals("CVS"))
						return true;
				if (f.isFile())
					if (f.getName().endsWith(".bit"))
						return true;
				return false;
			}
		});
		if (subFiles != null) {
			for (int i = 0; i < subFiles.length; i++) {
				if (subFiles[i].isDirectory()) {
					readOutputFiles(v, subFiles[i]);
				} else {
					try {
						v.add(new BIGOutputFile(subFiles[i]));
					} catch (FileNotFoundException e) {}
				}
			}
		}

	}

	/**
	 * getDragGestureListener
	 * 
	 * @return DragGestureListener
	 */
	private DragGestureListener getDragGestureListener() {
		DragGestureListener dgl = new DragGestureListener() {
			public void dragGestureRecognized(DragGestureEvent e) {
				if (e.getComponent() == BIGResultTree.this) {
					TreePath tp = ((BIGResultTree) e.getComponent()).getPathForLocation((int) e
							.getDragOrigin().getX(), (int) e.getDragOrigin().getY());
					if (tp != null)
						if (tp.getLastPathComponent() != null) {
							Object o = ((DefaultMutableTreeNode) tp.getLastPathComponent()).getUserObject();
							if (o != null)
								if (o instanceof BIGOutputFile) {
									e.startDrag(null, (BIGOutputFile) o);
								} else {
									if (o instanceof conn.Graph) {
										e.startDrag(null, ((conn.Graph) o).getCopyOfGraph());
									}
								}
						}
				}
			}
		};
		return dgl;
	}

	DropTargetListener dtl = null;

	public DropTargetListener getDropListener() {
		if (dtl == null) {
			dtl = new DropTargetAdapter() {
				private TreePath getTreePath(DropTargetEvent evt) {
					if (!(evt.getSource() instanceof DropTarget))
						return null;
					Object com = ((DropTarget) evt.getSource()).getComponent();
					if (!(com instanceof BIGResultTree))
						return null;
					BIGResultTree tree = (BIGResultTree) com;
					Point pos = (evt instanceof DropTargetDropEvent) ? ((DropTargetDropEvent) evt)
							.getLocation() : ((DropTargetDragEvent) evt).getLocation();
					return tree.getPathForLocation((int) pos.getX(), (int) pos.getY());
				}
				private BIGResultMixer getTarget(TreePath tp) {
					if (tp == null || tp.getLastPathComponent() == null)
						return null;
					Object o = ((DefaultMutableTreeNode) tp.getLastPathComponent()).getUserObject();
					if (!(o instanceof BIGResultMixer)) {
						return null;
					}
					return (BIGResultMixer) o;
				}

				public void drop(DropTargetDropEvent evt) {
					TreePath tp = getTreePath(evt);
					BIGResultMixer brm = getTarget(tp);
					if (brm == null)
						evt.rejectDrop();
					DataFlavor[] df = evt.getCurrentDataFlavors();
					for (int i = 0; i < df.length; i++) {
						Object o = null;
						try {
							o = evt.getTransferable().getTransferData(df[0]);
						} catch (IOException ex) {
							System.err.println("Internal Error while DragnDrop (1)");
						} catch (UnsupportedFlavorException ex) {
							System.err.println("Internal Error while DragnDrop (2)");
						}
						if (o != null) {
							if (o instanceof BIGOutputFile) {
								brm.showSelectFunctionsDialog((BIGOutputFile) o);
							} else if (o instanceof Graph) {
								brm.addFunction((Graph) o);
							} else {
								evt.rejectDrop();
							}
							setSelectionPath(tp);
						}
					}
				}

				@Override
				public void dragEnter(DropTargetDragEvent evt) {
					evt.acceptDrag(DnDConstants.ACTION_MOVE);
				}
			};
		}
		return dtl;
	}

	private void addDragNDrop() {
		DropTarget dt = new DropTarget(this, getDropListener());
		setDropTarget(dt);
		DragSource.getDefaultDragSource().createDefaultDragGestureRecognizer(this,
				DnDConstants.ACTION_COPY, getDragGestureListener());
		setDragEnabled(true);
		addKeyListener(new KeyAdapter() {
			@Override
			public void keyTyped(KeyEvent ke) {
				if (ke.getKeyChar() == KeyEvent.VK_DELETE) {
					delete();
				}
			}

		});
	}

	// -------------------------------------------------------------------------------
	/**
	 * This method packs all selected BIGOutputFiles in the ResultTree in one zip-file
	 */
	private void createZipFile() {
		String OUTPUTFOLDER = new File(bigInterface.getOutputPath()).getName();

		String fileName;

		TreePath[] selectedPaths = getSelectionPaths();
		DefaultMutableTreeNode node;

		FileInputStream in;
		FileOutputStream out;
		File file = getNewOutFile(BIGInterface.getInstance().getLastWorkPath().getAbsolutePath());
		BIGOutputFile bigFile;
		ZipOutputStream zipOut;
		ZipEntry entry;

		int read = 0;
		int indexOfOccurence;
		byte[] data = new byte[1024];

		if (file != null) {
			BIGInterface.getInstance().setLastWorkPath(file);
			try {
				out = new FileOutputStream(file);
				zipOut = new ZipOutputStream(out);
				// set compression method for entries
				zipOut.setMethod(ZipOutputStream.DEFLATED);
				// if there are selected tree items
				if (selectedPaths != null) {
					// do the following for every selected tree item
					for (int i = 0; i < selectedPaths.length; i++) {
						node = (DefaultMutableTreeNode) selectedPaths[i].getLastPathComponent();
						if (node.getUserObject() instanceof BIGOutputFile) {
							bigFile = (BIGOutputFile) node.getUserObject();
							fileName = bigFile.getFile().getAbsolutePath();
							// search for "output" in the path of the BIGOutputFile
							indexOfOccurence = fileName.indexOf(OUTPUTFOLDER);
							// Was "output" found in the path?
							if (indexOfOccurence != -1) {
								// if output was found, then pack the BIGOutputFile relative to
								// the "output"-location into the zip-file
								fileName = fileName.substring(indexOfOccurence);
							}
							// System.out.println(fileName);

							entry = new ZipEntry(fileName);
							in = new FileInputStream(bigFile.getFile());

							zipOut.putNextEntry(entry);
							while ((read = in.read(data)) > 0) {
								zipOut.write(data, 0, read);
							}
							zipOut.closeEntry();
							in.close();

							setSelectionPath(null);
						}
					}
				}
				zipOut.close();
				out.close();
			} catch (Exception e) {
				System.err.println("Error while zipping: abort zipping");
			}
		}
	}

	/**
	 * Opens a JFileChooser to select the zip file
	 * 
	 * @return newOutFile file the new output file (used to create a zip file later)
	 */
	private File getNewOutFile(String path) {
		BIGFileChooser jfc = new BIGFileChooser(path);
		jfc.addFileFilter("zip");
		if (jfc.showSaveDialog(null) != JFileChooser.APPROVE_OPTION)
			return null;
		return jfc.getSelectedFile();
	}

	// -------------------------------------------------------------------------------

	private void delete() {

		TreePath[] selectedPaths = getSelectionPaths();
		if (selectedPaths != null) {
			boolean delete = false;
			// here ask
			DefaultMutableTreeNode node = (DefaultMutableTreeNode) selectedPaths[0]
					.getLastPathComponent();
			if ((node.getUserObject() instanceof BIGOutputFile)
					|| (node.getUserObject() instanceof conn.Graph)) {
				delete = removeNode(node, true);
				setSelectionPath(null);
			}
			// now don't ask again
			if ((selectedPaths.length > 1) && (delete)) {
				for (int i = 1; i < selectedPaths.length; i++) {
					node = (DefaultMutableTreeNode) selectedPaths[i].getLastPathComponent();
					if ((node.getUserObject() instanceof BIGOutputFile)
							|| (node.getUserObject() instanceof conn.Graph)) {
						removeNode(node, false);
						setSelectionPath(null);
					}
				}
			}
		}

	}

	@Override
	protected JPopupMenu computePopup(MouseEvent evt) {
		final TreePath path = getClosestPathForLocation(evt.getX(), evt.getY());
		final DefaultMutableTreeNode selectedNode = ((DefaultMutableTreeNode) path
				.getLastPathComponent());
		JMenuItem sortMenu = getSortMenu();
		if (selectedNode.getUserObject() instanceof BIGOutputFile) {
			final JPopupMenu pop = new JPopupMenu();
			JMenuItem item = new JMenuItem("Display");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					gui.loadResult(null, true);
				}
			});
			pop.add(item);

			item = new JMenuItem(createReportMenuText);
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					new BIGReportGeneratorWindow(getSelectionPaths());
				}
			});
			pop.add(item);

			item = new JMenuItem("Upload results");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					BITConnectMain bitconn = BITConnectMain.getInstance(gui);

					TreePath[] selection = getSelectionPaths();
					ArrayList<String> list = new ArrayList<String>();
					DefaultMutableTreeNode tnode;
					File f;

					if (selection == null)
						return;
					for (int i = 0; i < selection.length; i++) {
						tnode = (DefaultMutableTreeNode) selection[i].getLastPathComponent();
						if (tnode.getUserObject() instanceof BIGOutputFile) {
							// selected node is a BIGOutputFile
							f = ((BIGOutputFile) tnode.getUserObject()).getFile();
							list.add(f.getAbsolutePath());
						}
					}

					try {
						bitconn.uploadFilesFromPathStrings(list);
					} catch (Exception e) {
						System.err.println("Connection to server failed");
						e.printStackTrace();
					}
				}
			});
			pop.add(item);

			item = new JMenuItem("Zip");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent evt) {
					createZipFile();
				}
			});
			pop.add(item);
			// ---------------------------------------------------------

			// !! wird immer wieder neu erzeugt!!!!!
			item = new JMenuItem("View file as text");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					// System.err.println("view");
					gui.loadResult(null, true);
					if (text) {
						// System.err.println("viewtxt1"+text);
						gui.setTextViewMode(true);
						((JMenuItem) ae.getSource()).setText("View file as chart");
					} else {
						// System.err.println("viewtxt2"+text);
						gui.setTextViewMode(false);
						((JMenuItem) ae.getSource()).setText("View file as text");
					}
					((JMenuItem) ae.getSource()).revalidate();
					text = !text;
				}
			});
			if (!text) {
				item.setText("View file as chart");
			}
			pop.add(item);

			item = new JMenuItem("Start QUICKVIEW.SH");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					String fileName = ((BIGOutputFile) selectedNode.getUserObject()).getFile()
							.getAbsolutePath();
					try {
						Process p = Runtime.getRuntime().exec(
								bigInterface.getBenchItPath() + File.separator + "tools" + File.separator
										+ "QUICKVIEW.SH " + fileName);
						bigInterface.getConsole().addStream(p.getErrorStream(), BIGConsole.ERROR);
						bigInterface.getConsole().addStream(p.getInputStream(), BIGConsole.DEBUG);

					} catch (IOException ex) {}
				}
			});
			pop.add(item);
			pop.addSeparator();

			item = new JMenuItem("Fix invalid values");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					// instance for cleaning bit files
					system.BIGUtility util = new system.BIGUtility();
					if (gui.getStatusProgress() != null) {
						util.getObservable().addObserver(gui.getStatusProgress());
					}
					String fileName = ((BIGOutputFile) selectedNode.getUserObject()).getFile()
							.getAbsolutePath();
					// clean the file
					util.removeInfinitiesFromFile(fileName);
				}
			});
			pop.add(item);

			item = new JMenuItem("Change architecture/display infos");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					TreePath[] paths = getSelectionPaths();
					String[] availFiles = (new File(BIGInterface.getInstance().getBenchItPath()
							+ File.separator + "LOCALDEFS")).list(new FilenameFilter() {
						public boolean accept(File f, String name) {
							if ((new File(f.getAbsolutePath() + File.separator + name + "_input_architecture"))
									.exists()) {
								if ((new File(f.getAbsolutePath() + File.separator + name + "_input_display"))
										.exists())
									return true;
							}
							return false;
						}
					});
					JList<?> jl = new JList<Object>(availFiles);
					jl.setPreferredSize(new Dimension(300, 450));
					int select = JOptionPane.showConfirmDialog(gui, jl, "Select the architecture file",
							JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
					if (select != JOptionPane.OK_OPTION)
						return;
					if (jl.getSelectedValue() == null)
						return;
					for (int i = 0; i < paths.length; i++) {
						DefaultMutableTreeNode actualNode = ((DefaultMutableTreeNode) paths[i]
								.getLastPathComponent());
						String fileName = ((BIGOutputFile) actualNode.getUserObject()).getFile()
								.getAbsolutePath();
						// get file content
						File outputfile = null;
						try {
							outputfile = new File(fileName);
						} catch (Exception ex) {
							System.out.println("There's no file selected");
							return;
						}
						String content = null;
						try {
							content = BIGFileHelper.getFileContent(outputfile);
						} catch (Exception ex1) {
							System.out.println("Could not load " + outputfile);
							return;
						}
						// will contain everything before the architecture
						String firstPart = "";
						// will contain everything after the architecture
						String lastPart = "";
						// everything before hostname
						try {
							firstPart = firstPart + content.substring(0, content.indexOf("\nhostname=\"") + 1);
							// find hostname in resultfile and go to the character after the
							// linebreak to hostname
							content = content.substring(content.indexOf("\nhostname=\"") + 1);
							firstPart = firstPart + content.substring(0, content.indexOf("\n") + 1);
							content = content.substring(content.indexOf("\n") + 1);
							firstPart = firstPart + content.substring(0, content.indexOf("\n") + 1);
							content = content.substring(content.indexOf("\n") + 1);

							// goto the line after hostname
							// find end
							content = content.substring(content.indexOf("\nbeginofenvironmentvariables\n"));
							lastPart = content;
						} catch (Exception ex2) {
							ex2.printStackTrace();
							System.out.println("Error while parsing outputfile " + outputfile);
							return;
						}
						// content will now be the architectur info
						try {
							content = BIGFileHelper.getFileContent(new File(BIGInterface.getInstance()
									.getBenchItPath()
									+ File.separator
									+ "LOCALDEFS"
									+ File.separator
									+ jl.getSelectedValue() + "_input_architecture"));
						} catch (Exception ex3) {
							System.out.println("Error while loading " + jl.getSelectedValue());
							return;
						}
						// now the input_display
						// search for last fontsizelegendfunction
						firstPart = firstPart + content;
						int index = 0;
						int oldIndex = 0;
						while (true) {
							index = lastPart.indexOf("\nfontsizelegendfunction", oldIndex) + 1;
							if (index > 0) {
								oldIndex = index;
							} else {
								index = lastPart.indexOf("\n", oldIndex) + 1;
								firstPart = firstPart + lastPart.substring(0, index);
								lastPart = lastPart.substring(index);
								break;
							}
						}
						// goto data (which is after display infos)
						lastPart = lastPart.substring(lastPart.indexOf("beginofdata"));
						// content will now be the display info
						try {
							content = BIGFileHelper.getFileContent(new File(BIGInterface.getInstance()
									.getBenchItPath()
									+ File.separator
									+ "LOCALDEFS"
									+ File.separator
									+ jl.getSelectedValue() + "_input_display"));
						} catch (Exception ex3) {
							System.out.println("Error while loading " + jl.getSelectedValue());
							return;
						}
						// build all together
						content = firstPart + content + lastPart;
						// save
						try {
							BIGFileHelper.saveToFile(content, outputfile);
						} catch (Exception ex4) {
							System.out.println("Error while saving " + outputfile);
							return;
						}
						System.out.println("Succesfully changed architecture/display infos for "
								+ outputfile.getName());
					}
				}
			});
			pop.add(item);
			pop.addSeparator();

			item = new JMenuItem("Delete");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					delete();
				}

			});
			pop.add(item);
			pop.addSeparator();
			pop.add(sortMenu);
			return pop;
		} else {
			if (selectedNode.getUserObject() instanceof String
					&& (!(selectedNode.getParent() == null))
					&& (!((((DefaultMutableTreeNode) (selectedNode.getParent())).getUserObject()) instanceof BIGResultMixer))) {
				final JPopupMenu pop = new JPopupMenu();
				pop.add(sortMenu);
				return pop;

			} else {

				JPopupMenu pop = new JPopupMenu();
				if ((selectedNode.getUserObject() instanceof BIGResultMixer)
						|| ((!(selectedNode.getParent() == null)) && (((((DefaultMutableTreeNode) (selectedNode
								.getParent())).getUserObject()) instanceof BIGResultMixer)))) {
					if ((selectedNode.getUserObject() instanceof BIGResultMixer)) {
						pop = ((BIGResultMixer) (selectedNode.getUserObject())).getBIGPopupMenu(gui,
								selectedNode.getUserObject());
					} else {
						pop = ((BIGResultMixer) (((DefaultMutableTreeNode) (selectedNode.getParent()))
								.getUserObject())).getBIGPopupMenu(gui, selectedNode.getUserObject());

					}

					pop.addSeparator();
					pop.add(sortMenu);
					return pop;
				}
			}
		}
		return null;
	}

	@Override
	public String getToolTipText(MouseEvent evt) {
		return computeToolTipText(evt);
	}

	public String computeToolTipText(MouseEvent evt) {
		final TreePath path = getClosestPathForLocation(evt.getX(), evt.getY());
		final DefaultMutableTreeNode node = ((DefaultMutableTreeNode) path.getLastPathComponent());

		// set tool tip text
		Object o = node.getUserObject();
		if (o instanceof BIGOutputFile) {
			String comment = ((BIGOutputFile) o).getComment();
			return o.toString() + ((comment != null && !comment.isEmpty()) ? " (" + comment + ")" : "");
		} else if (node.isLeaf()) {
			// a leaf tool tip is needed
			String tip = o.toString();
			setToolTipText(tip);
		} else if ((!node.equals(node.getRoot())) && (!node.isLeaf())) {
			// a kernel base name tool tip is needed
			int count = 0;
			// for the BRM
			if (node.getUserObject() instanceof BIGResultMixer) {
				count = ((BIGResultMixer) node.getUserObject()).getNumberOfFunctions();

			} else {
				count = getNumberOfLeaves(node);
			}
			String tip = "There ";
			if (count == 1) {
				tip += "is 1 result ";
			} else {
				tip += "are " + count + " results ";
			}
			tip += "in this node.";
			return tip;
		}
		return "";
	}

	private BIGOnlineMixer online = null;

	public void setOnline(boolean on) {
		if ((online == null) && on) {
			addOnlineMixer();
		}
		if ((online != null) && !on) {
			removeOnlineMixer();
		}
	}

	public boolean isOnline() {
		if (online == null)
			return false;
		return true;
	}

	public BIGOnlineMixer getOnlineMixer() {
		return online;
	}

	/**
	 * removeOnlineMixer
	 */
	private void removeOnlineMixer() {
		DefaultTreeModel model = ((DefaultTreeModel) getModel());
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) (model.getRoot());
		for (int i = 0; i < rootNode.getChildCount(); i++)
			if (rootNode.getChildAt(i) instanceof BIGOnlineMixer) {
				rootNode.remove(i);
			}
		updateResultTree(rootNode);
	}

	private void addOnlineMixer() {
		DefaultTreeModel model = ((DefaultTreeModel) getModel());
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) (model.getRoot());
		online = new BIGOnlineMixer(gui.getViewPanel(), "Online-Mixer");
		online.setResultTree(this);
		rootNode.add(online.getTreeNode());
		updateResultTree(rootNode);
	}

	public void selectKernelResultFile(BIGKernel kernel) {
		String name = kernel.getNameAfterSorting(newSorting);
		StringTokenizer stok = new StringTokenizer(name, ".");
		TreeNode parent = (TreeNode) getModel().getRoot();
		TreePath tp = new TreePath(parent);
		while (stok.hasMoreTokens()) {
			String curName = stok.nextToken();
			boolean found = false;
			for (Enumeration<?> childs = parent.children(); childs.hasMoreElements();) {
				TreeNode curChild = (TreeNode) childs.nextElement();
				if (curChild.toString().equals(curName)) {
					parent = curChild;
					tp = tp.pathByAddingChild(curChild);
					found = true;
					break;
				}
			}
			if (!found)
				break;
		}
		setSelectionPath(tp);
		gui.setGUIState(GUIState.RESULTS);
	}
}
/******************************************************************************
 * Log-History $Log: BIGResultTree.java,v $ Revision 1.32 2009/01/07 12:07:27 tschuet report generator included Revision
 * 1.31 2008/12/17 08:58:43 dreiche build.xml: added target/version/bootstrap options to compile task, use JAVA_HOME to
 * specify java environment. lib/jedit.jar replaced FortranTokenMarker.class with Java1.4 compatible version
 * BIGRemoteMenu.java: removed unneeded try-catch because of related exception not in method signature any more.
 * BIGResultTree.java: popup-menu entry for BITConnect upload service BenchIT.jar: complete clean rebuild with latest
 * files Revision 1.30 2008/06/02 10:10:37 tschuet report generator gui front end implemented Revision 1.29 2007/10/22
 * 08:54:36 tschuet fixed bug of disappearring nodes in the result-tree after measurements (in case of ceating a new
 * main node like "numerical") Revision 1.28 2007/09/14 08:53:13 tschuet attempt to solve the memory consumption problem
 * of plots Revision 1.27 2007/07/10 12:25:52 tschuet it is possible to cancel the save-dialog of the zip feature
 * Revision 1.26 2007/07/10 11:39:26 tschuet zip feature use the last path as default Revision 1.25 2007/07/03 11:29:14
 * tschuet correct actualizing of plot titles Revision 1.24 2007/06/05 10:17:27 tschuet changes on handling the
 * resulttree-popupmenu Revision 1.23 2007/05/22 13:13:19 tschuet addition of an attempt for context menus on
 * mac-systems Revision 1.22 2007/05/08 09:43:30 tschuet reorganize imports Revision 1.21 2007/05/08 09:31:05 tschuet in
 * the context menu of BIGOutputFiles a new function to pack selelcted files in a ZIP-file was added Revision 1.20
 * 2007/04/10 10:11:14 tschuet if you move a graph from a mixer to another mixer (drag'n'drop) a copy of this graph will
 * be produced so that changes only affect the actual graph Revision 1.19 2007/02/27 12:37:49 tschuet different bugfixes
 * (mixer, remote folder -> see meeting at february, 27th) Revision 1.18 2006/12/07 14:54:22 rschoene removed bug in new
 * functionality change architecture of result Revision 1.17 2006/12/07 14:26:00 rschoene change architecture Revision
 * 1.16 2006/11/20 14:15:34 rschoene Drag n drop is what its meant to be like Revision 1.15 2006/08/04 09:32:07 rschoene
 * some updates and debugs Revision 1.14 2006/07/05 09:28:38 rschoene debugged reset all plot-information Revision 1.13
 * 2006/07/04 08:30:26 rschoene debugged update Revision 1.12 2006/07/01 10:47:14 rschoene again... removed bug Revision
 * 1.11 2006/07/01 10:28:40 rschoene removed bug in update() Revision 1.10 2006/06/28 11:39:05 rschoene removed bug
 * Revision 1.9 2006/06/26 10:47:54 rschoene integrated online- in result-mode Revision 1.8 2006/05/11 10:30:45 rschoene
 * changes for reset plot-info, save height/width for save/change default colors Revision 1.7 2006/02/07 16:36:40
 * rschoene all new plotting Revision 1.6 2006/01/31 21:11:33 rschoene shows comment as tooltip Revision 1.5 2006/01/09
 * 16:35:23 rschoene some advances / debugging for saveAndLadPlotting Revision 1.4 2005/12/07 10:24:34 rschoene ask once
 * for deleting Revision 1.3 2005/11/02 14:15:15 rschoene removed bugs and added multiple mixer support Revision 1.2
 * 2005/10/21 11:25:01 rschoene removed fills in RemoteMenu and delete-bug in ResultTree Revision 1.1 2005/10/20
 * 13:13:57 rschoene src3 add Revision 1.5 2005/04/26 12:33:49 wloch fixed disapearance of certain results when
 * changeing result sorting Revision 1.4 2005/04/19 15:22:19 wloch implemented sorting in result tree Revision 1.3
 * 2005/02/22 17:01:54 wloch added subnode to BRM and instant loading Revision 1.2 2005/02/21 19:13:16 wloch major
 * changes to menu structure and popup menus Revision 1.1 2005/02/18 09:45:19 wloch implemented a better popup menu API
 ******************************************************************************/
