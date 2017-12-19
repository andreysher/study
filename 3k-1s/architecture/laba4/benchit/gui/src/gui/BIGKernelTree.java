/*********************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGKernelTree.java Author:
 * Robert Wloch Last change by: $Author: rschoene $ $Revision: 1.8 $ $Date: 2006/12/06 06:50:59 $
 *********************************************************************/
package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;

import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

import system.*;

/**
 * The BIGKernelTree provides neccessary functionality for the kernels an their tree view.
 **/
public class BIGKernelTree extends BIGTree {
	private static final long serialVersionUID = 1L;
	private final static boolean debug = false;
	/** Sorting of the elements (which element of kernel is first) */
	private int sorting = 0;

	/**
	 * The constructor.
	 * 
	 * @param newModel the TreeModel used in the tree.
	 **/
	public BIGKernelTree(TreeModel newModel, BIGGUI gui) {
		super(newModel, gui);

		sorting = BIGInterface.getInstance().getBIGConfigFileParser()
				.intCheckOut("lastKernelSortOrder", 0);
	}

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

	private void buildTree(DefaultMutableTreeNode rootNode, File directory) {
		if (debug) {
			System.err.println("-------------buildTree(KernelTree)-------------");
		}
		Vector<BIGKernel> kernelVec = new Vector<BIGKernel>();
		// readKernels
		readKernels(kernelVec, directory);
		// sort and insert kernels
		Vector<BIGKernel> sortedVec = new Vector<BIGKernel>();
		// get min
		while (kernelVec.size() > 0) {
			int position = 0;
			String lastkernelString = null;

			for (int i = 0; i < kernelVec.size(); i++) {
				if (lastkernelString == null) {
					position = i;
					lastkernelString = kernelVec.get(i).getNameAfterSorting(sorting);
				} else if (kernelVec.get(i).getNameAfterSorting(sorting).compareTo(lastkernelString) < 0) {
					position = i;
					lastkernelString = kernelVec.get(i).getNameAfterSorting(sorting);
				}
			}
			sortedVec.add(kernelVec.get(position));
			kernelVec.removeElementAt(position);
		}
		for (int i = 0; i < sortedVec.size(); i++) {
			// insert Kernels
			addKernel(rootNode, sortedVec.get(i));
		}

	}

	private void addKernel(DefaultMutableTreeNode rootNode, BIGKernel kernel) {
		if (debug) {
			System.err.println("Sorting:" + sorting);
		}
		kernel.setSorting(sorting);
		String name = kernel.getNameAfterSorting(sorting);
		if (debug) {
			System.err.println("Output-file:\n" + name);
		}
		DefaultMutableTreeNode tempNode = rootNode;
		StringTokenizer stok = new StringTokenizer(name, ".");
		String part;
		boolean found;
		// first the folders
		for (int j = 0; j < kernel.getSortedNames().length - 1; j++) {
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
				if (debug) {
					System.err.println("Adding " + newNode + " to " + tempNode);
				}
				tempNode.add(newNode);
				tempNode = newNode;
			}
		}
		// then the filename
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
			DefaultMutableTreeNode newNode = new DefaultMutableTreeNode(kernel);
			if (debug) {
				System.err.println("Adding " + newNode + " to " + tempNode);
			}
			tempNode.add(newNode);
			tempNode = newNode;
		}

	}

	public void readKernels(java.util.Vector<BIGKernel> v, File directory) {
		File[] subFiles = directory.listFiles(new FileFilter() {
			public boolean accept(File f) {
				if (f.isDirectory())
					if (!f.getName().equals("CVS"))
						return true;
				return false;
			}
		});
		if (subFiles == null)
			return;
		for (int i = 0; i < subFiles.length; i++) {
			if (!(new File(subFiles[i].getAbsolutePath() + File.separator + "COMPILE.SH")).exists()) {
				readKernels(v, subFiles[i]);
			} else {
				try {
					v.add(new BIGKernel(subFiles[i]));
				} catch (StringIndexOutOfBoundsException ex) {
					System.err.println(subFiles[i] + " is not a valid kernel");
				}
				if (debug) {
					System.err.println("Found new kernel " + subFiles[i]);
				}
			}
		}

	}

	/**
	 * Sets the sorting field of the tree to the given value. For valid values look at the documentation of BIGExecute. An
	 * invalid sorting flag will default to BIGExecute.KERNEL_DIRECTORY. If the new sorting is different than the current
	 * one the kernel tree will be updated, otherwise nothing will happen.
	 * 
	 * @param newSorting The new sort order for the kernel tree.
	 **/
	private void setSorting(int newSorting) {
		if (sorting == newSorting)
			return;

		sorting = newSorting;

		BIGInterface.getInstance().getBIGConfigFileParser().set("lastKernelSortOrder", "" + sorting);

		Thread t = new Thread() {
			@Override
			public void run() {
				updateKernelTree(true);
				gui.getResultTree().kernelSortingChanged();
			}
		};
		SwingUtilities.invokeLater(t);
	}

	/**
	 * Initital setup of the kernel tree. Should be called only once. Use updateKernelTree for later updates and reloads.
	 **/
	public void setupTree() {
		setRootVisible(true);
		setEditable(false);
		setShowsRootHandles(true);
		setCellRenderer(new KernelTreeCellRenderer(gui));
		setExpandsSelectedPaths(true);
		DefaultTreeModel model = (DefaultTreeModel) getModel();
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) (model.getRoot());
		buildTree(rootNode, new File(BIGInterface.getInstance().getBenchItPath() + File.separator
				+ "kernel" + File.separator));
		updateUI();
	}

	/**
	 * Updates the kernel tree.
	 **/
	public void updateKernelTree() {
		updateKernelTree(true);
	}

	public void updateKernelTree(final boolean selectOldPath) {
		Thread t = new Thread() {
			@Override
			public void run() {
				TreePath[] oldSelection = getSelectionPaths();
				Enumeration<TreePath> oldExp = getExpandedDescendants(new TreePath(getModel().getRoot()));
				DefaultTreeModel model = (DefaultTreeModel) getModel();
				DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) (model.getRoot());
				if (debug) {
					System.err.println("Childs:" + rootNode.getChildCount());
				}
				rootNode.removeAllChildren();
				if (debug) {
					System.err.println("Childs:" + rootNode.getChildCount());
				}

				buildTree(rootNode, new File(BIGInterface.getInstance().getBenchItPath() + File.separator
						+ "kernel" + File.separator));
				model.setRoot(rootNode);

				if (selectOldPath) {
					expand(oldExp);
					setSelectionPathsMutable(oldSelection);
				}
			}
		};
		SwingUtilities.invokeLater(t);
	}

	/*
	 * private void setSelectedTreePath(TreePath[] path) { TreeSelectionListener[] listeners =
	 * getTreeSelectionListeners(); for (int i = 0; i < listeners.length; i++) removeTreeSelectionListener(listeners[i]);
	 * for (int i = 0; i < path.length; i++) { DefaultMutableTreeNode dmtn = (DefaultMutableTreeNode)
	 * path[i].getLastPathComponent(); TreePath[] subpaths = new TreePath[dmtn.getChildCount()]; for (int j = 0; j <
	 * dmtn.getChildCount(); j++) subpaths[j] = path[i].pathByAddingChild(dmtn.getChildAt(j));
	 * setSelectedTreePath(subpaths); } this.addSelectionPaths(path); for (int i = 0; i < listeners.length; i++)
	 * addTreeSelectionListener(listeners[i]); }
	 */
	/*
	 * private void removeSelectedTreePath(TreePath[] path) { TreeSelectionListener[] listeners =
	 * getTreeSelectionListeners(); for (int i = 0; i < listeners.length; i++) removeTreeSelectionListener(listeners[i]);
	 * for (int j = 0; j < path.length; j++) { DefaultMutableTreeNode dmtn = (DefaultMutableTreeNode)
	 * path[j].getLastPathComponent(); if (!dmtn.isRoot()) subRemoveSelectedTreePath(path[j].getParentPath()); TreePath[]
	 * paths = new TreePath[dmtn.getChildCount()]; for (int i = 0; i < dmtn.getChildCount(); i++) { paths[i] =
	 * path[j].pathByAddingChild(dmtn.getChildAt(i)); } removeSelectedTreePath(paths); } for (int i = 0; i <
	 * listeners.length; i++) { addTreeSelectionListener(listeners[i]); listeners[i].valueChanged(null); } } private void
	 * subRemoveSelectedTreePath(TreePath path) { TreeSelectionListener[] listeners = getTreeSelectionListeners(); for
	 * (int i = 0; i < listeners.length; i++) removeTreeSelectionListener(listeners[i]); DefaultMutableTreeNode dmtn =
	 * (DefaultMutableTreeNode) path.getLastPathComponent(); if (!dmtn.isRoot())
	 * subRemoveSelectedTreePath(path.getParentPath()); for (int i = 0; i < listeners.length; i++) {
	 * addTreeSelectionListener(listeners[i]); listeners[i].valueChanged(null); } }
	 */

	/**
	 * Sets up a TreeSelectionListener and a MouseListener and adds them to the tree.
	 **/
	@Override
	public void setListeners() {
		super.setListeners();
		// adds tree selection listener for view display change
		addTreeSelectionListener(new TreeSelectionListener() {
			Thread thread = null;

			public void valueChanged(TreeSelectionEvent tse) { // called on selection change only
				gui.setGUIState(GUIState.KERNELS);

				final TreePath path[] = getSelectionPaths();
				// add children
				// maybe implemented later (add shold work, but remove?)
				if (tse.isAddedPath()) {
					// setSelectedTreePath(path);
				} else {
					// removeSelectedTreePath(path);
				}
				if (path != null) {
					if (thread != null) {
						// kill old load thread
						thread.interrupt();
						thread = null;
					}
					if (gui.getKernelScriptWindow() != null) {
						// save changed files
						if (gui.getKernelScriptWindow() != null) {
							gui.getKernelScriptWindow().saveRequest(true);
						}

						gui.setKernelScriptWindow(null, true);
					}
					boolean sourceEdit;
					try {
						sourceEdit = BIGInterface.getInstance().getBIGConfigFileParser()
								.boolCheckOut("kernelSourceEdit");
					} catch (Exception e) {
						sourceEdit = false;
					}
					thread = new BIGKernelScriptWindow(path, sourceEdit, gui);
					thread.start();
				}
			} // end of valueChanged()
		}); // end of addTreeSelectionListener()

		// adds mouse listener for pop up menu
		addMouseListener(new MouseAdapter() {
			@Override
			public void mousePressed(MouseEvent e) {
				gui.setGUIState(GUIState.KERNELS);
			}
		});
	}

	/**
	 * Recursive method to add all kernel names under a node to a BIGStrings object.
	 * 
	 * @param result The BIGStrings object to fill.
	 * @param parentNode The "root node" of the sub tree to search in.
	 **/
	public static void addKernelsToBIGString(BIGStrings result, DefaultMutableTreeNode parentNode) {
		int end = parentNode.getChildCount();
		// traverse all children of parentNode to add all kernels
		for (int i = 0; i < end; i++) {
			DefaultMutableTreeNode childNode = (DefaultMutableTreeNode) parentNode.getChildAt(i);
			// get the node's name
			Object nodeObj = childNode.getUserObject();
			// if nodeObj is instance of BIGKernel add that kernel name
			if (nodeObj instanceof BIGKernel) {
				result.add(((BIGKernel) nodeObj).getAbsolutePath());
			} else {
				// it's another subdir name -> recurse
				// the root node falls also into this branch
				addKernelsToBIGString(result, childNode);
			}
		}
	}

	/**
	 * Searches the model of the tree for the TreeNode specified by the TreePath.
	 * 
	 * @param selectedPath The TreePath to find the node for.
	 * @return the DefaultMutableTreeNode or null, if node not found.
	 **/
	public DefaultMutableTreeNode findNode(TreePath selectedPath) {

		DefaultMutableTreeNode result = null;

		String kernelName = selectedPath.getLastPathComponent().toString();
		DefaultTreeModel model = (DefaultTreeModel) getModel();
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) model.getRoot();
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
				if (pathObj.toString().compareTo(childNode.toString()) == 0) {
					childFound = true;
					parent = childNode;
					if (kernelName.compareTo(childNode.toString()) == 0) {
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

	public BIGKernel findKernel(String s) {
		BIGKernel retKernel = null;
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) getModel().getRoot();
		Stack<TreeNode> nodeStack = new Stack<TreeNode>();
		nodeStack.add(rootNode);
		while (!nodeStack.isEmpty()) {
			DefaultMutableTreeNode dmtn = (DefaultMutableTreeNode) nodeStack.pop();
			if (dmtn.getUserObject() instanceof BIGKernel) {
				if (((BIGKernel) dmtn.getUserObject()).getAbsolutePath().equals(s)) {
					retKernel = (BIGKernel) dmtn.getUserObject();
					return retKernel;
				}

			} else {
				for (int i = 0; i < dmtn.getChildCount(); i++) {
					nodeStack.push(dmtn.getChildAt(i));
				}
			}

		}
		return retKernel;
	}

	public void reloadKernelsExecutables() {
		DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) getModel().getRoot();
		Stack<BIGKernel> kernelStack = new Stack<BIGKernel>();
		Stack<TreeNode> nodeStack = new Stack<TreeNode>();
		nodeStack.push(rootNode);
		while (!nodeStack.isEmpty()) {
			DefaultMutableTreeNode dmtn = (DefaultMutableTreeNode) nodeStack.pop();
			if (dmtn.getUserObject() instanceof BIGKernel) {
				kernelStack.push((BIGKernel) dmtn.getUserObject());
			} else {
				for (int i = 0; i < dmtn.getChildCount(); i++) {
					nodeStack.push(dmtn.getChildAt(i));
				}
			}
		}
		// second reload all Kernels executables
		while (!kernelStack.isEmpty()) {
			kernelStack.pop().reloadExecutables();
		}

	}

	@Override
	protected JPopupMenu computePopup(MouseEvent evt) {
		JPopupMenu pop = new JPopupMenu();
		final TreePath path = getClosestPathForLocation(evt.getX(), evt.getY());
		DefaultMutableTreeNode selectedNode = ((DefaultMutableTreeNode) path.getLastPathComponent());
		JMenuItem sortMenu = getSortMenu();
		JMenuItem item;
		item = new JMenuItem("Execute local");
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae) {
				gui.executeSelected();
			}
		});
		pop.add(item);

		pop.add(gui.getRemoteMenu().execRemote);
		pop.add(gui.getRemoteMenu().copyRemote);
		// set a flag for which radio button is active at first
		/*
		 * boolean sourceSorting = false; if ( sorting == BIGExecute.getInstance().SOURCE_LANGUAGE ) { sourceSorting = true;
		 * } // create the radio button button group ButtonGroup group = new ButtonGroup();
		 */
		pop.addSeparator();
		pop.add(sortMenu);
		// if it is a node (a kernel)
		if (selectedNode.getUserObject() instanceof system.BIGKernel) {
			final BIGKernel kernel = (BIGKernel) selectedNode.getUserObject();
			item = new JMenuItem("Duplicate");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					cloneKernel(kernel);
				}
			});

			pop.addSeparator();
			pop.add(item);
			item = new JMenuItem("Remove");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					if (JOptionPane.showConfirmDialog(BIGKernelTree.this,
							"Do you really want to delete the kernel \"" + kernel.getNameAfterSorting(0) + "\"?",
							"Deleting kernel", JOptionPane.YES_NO_OPTION) == JOptionPane.OK_OPTION) {
						BIGFileHelper.remove(new File(kernel.getAbsolutePath()));
						updateKernelTree();
					}
				}
			});
			pop.add(item);

			pop.addSeparator();
			item = new JMenuItem("create new File");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					String name = JOptionPane.showInputDialog(BIGKernelTree.this,
							"Please type the name of the new file.",
							"Creating new File for kernel " + kernel.getNameAfterSorting(0),
							JOptionPane.OK_CANCEL_OPTION);
					if (name == null || name.isEmpty()) {
						System.err.println("Won't create File without name");
						return;
					}
					File f = new File(kernel.getAbsolutePath() + File.separator + name);
					if (f.exists()) {
						System.err.println("File " + name + " already exists");
						return;
					}
					try {
						f.createNewFile();
					} catch (IOException ex) {
						System.err.println("Couldn't create file " + name);
					}

					TreeSelectionListener lis = getTreeSelectionListeners()[0];
					TreePath[] paths = new TreePath[1];
					paths[0] = path;
					boolean[] isNew = new boolean[1];
					isNew[0] = false;
					lis.valueChanged(new TreeSelectionEvent(this, paths, isNew, null, null));

				}
			});
			pop.add(item);
			item = new JMenuItem("remove single File");
			item.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					String[] options = new String[kernel.getFiles().length];
					for (int i = 0; i < options.length; i++) {
						options[i] = kernel.getFiles()[i].getName();
					}
					int retVal = JOptionPane.showOptionDialog(BIGKernelTree.this,
							"select the file you'd like to remove", "Remove single file",
							JOptionPane.OK_CANCEL_OPTION, 0, null, options, options[0]);
					if (retVal > -1) {
						File f = kernel.getFiles()[retVal];
						f.delete();
					}
					TreeSelectionListener lis = getTreeSelectionListeners()[0];
					TreePath[] paths = new TreePath[1];
					paths[0] = path;
					boolean[] isNew = new boolean[1];
					isNew[0] = false;

					lis.valueChanged(new TreeSelectionEvent(this, paths, isNew, null, null));

				}
			});
			pop.add(item);

		}
		return pop;
	}

	private void cloneKernel(BIGKernel kernel) {
		// return value of JOptionPane
		int returnValue;

		JPanel main = new JPanel(new GridLayout(2, 5));
		JTextField genre = new JTextField(kernel.getType(), kernel.getType().length());
		JTextField algorithm = new JTextField(kernel.getName(), kernel.getName().length());
		JTextField language = new JTextField(kernel.getSourceLanguage(), kernel.getSourceLanguage()
				.length());
		JTextField parLib = new JTextField(kernel.getParallelLibraries(), kernel.getParallelLibraries()
				.length());
		JTextField lib = new JTextField(kernel.getLibraries(), kernel.getLibraries().length());
		JTextField datatype = new JTextField(kernel.getDataType(), kernel.getDataType().length());

		main.add(new JLabel(" Genre "));
		main.add(new JLabel(" Algorithm "));
		main.add(new JLabel(" Language "));
		main.add(new JLabel(" parallel Libraries "));
		main.add(new JLabel(" other Libraries "));
		main.add(new JLabel(" Type/Datatype "));

		main.add(genre);
		main.add(algorithm);
		main.add(language);
		main.add(parLib);
		main.add(lib);
		main.add(datatype);

		while (true) {
			returnValue = JOptionPane.showConfirmDialog(this, main, "Insert the new name of the kernel",
					JOptionPane.OK_CANCEL_OPTION);
			if (returnValue == JOptionPane.OK_OPTION) {
				String dot = ".";
				boolean kernelNameContainsDot = false;

				// check if genre contains dots
				kernelNameContainsDot = (kernelNameContainsDot || BIGUtility.stringContains(
						genre.getText(), dot));
				// check if algorithm contains dots
				kernelNameContainsDot = (kernelNameContainsDot || BIGUtility.stringContains(
						algorithm.getText(), dot));
				// check if language contains dots
				kernelNameContainsDot = (kernelNameContainsDot || BIGUtility.stringContains(
						language.getText(), dot));
				// check if parLib contains dots
				kernelNameContainsDot = (kernelNameContainsDot || BIGUtility.stringContains(
						parLib.getText(), dot));
				// check if lib contains dots
				kernelNameContainsDot = (kernelNameContainsDot || BIGUtility.stringContains(lib.getText(),
						dot));
				// check if datatype contains dots
				kernelNameContainsDot = (kernelNameContainsDot || BIGUtility.stringContains(
						datatype.getText(), dot));

				if (kernelNameContainsDot) {
					// show error message, because actual kernel name contains dots
					JOptionPane.showMessageDialog(null,
							"Kernel name must not contain dots. Please correct your input.",
							"Incorrect kernel name", JOptionPane.ERROR_MESSAGE);
				} else {
					String kernelpath = BIGInterface.getInstance().getBenchItPath() + File.separator
							+ "kernel" + File.separator;
					File newKernelDir = new File(kernelpath + genre.getText() + File.separator
							+ algorithm.getText() + File.separator + language.getText() + File.separator
							+ parLib.getText() + File.separator + lib.getText() + File.separator
							+ datatype.getText());
					if (newKernelDir.exists()) {
						System.out.println("This kernel already exists!");
						return;
					}
					newKernelDir.mkdirs();
					while (!newKernelDir.exists()) {
						;
					}
					// copy
					File[] oldFiles = kernel.getFiles();
					for (int i = 0; i < oldFiles.length; i++) {
						BIGFileHelper.copyToFolder(oldFiles[i], newKernelDir, true);
					}
					DefaultTreeModel model = (DefaultTreeModel) getModel();
					DefaultMutableTreeNode rootNode = (DefaultMutableTreeNode) (model.getRoot());
					BIGKernel newKernel = null;
					try {
						newKernel = new BIGKernel(newKernelDir);
					} catch (StringIndexOutOfBoundsException ex) {
						System.err.println(newKernelDir + " is not a valid kernel");
						return;
					}
					addKernel(rootNode, newKernel);
					updateKernelTree();
					// everything ok -> end while-loop
					return;
				}
			} else
				// any other button than OK was clicked
				// end while-loop
				return;
		}
	}

	public BIGStrings getSelected() {
		BIGStrings selected = new BIGStrings();
		if (isSelectionEmpty()) {
			return selected;
		}
		int selectionCount = getSelectionCount();
		TreePath[] selectedPaths = getSelectionPaths();
		// traverse selected nodes
		for (int i = 0; i < selectionCount; i++) {
			TreePath selectedPath = selectedPaths[i];
			DefaultMutableTreeNode lastNode = (DefaultMutableTreeNode) selectedPath
					.getLastPathComponent();
			if (lastNode.getUserObject() instanceof BIGKernel) {
				// leaf
				selected.add(((BIGKernel) lastNode.getUserObject()).getAbsolutePath());
			} else {
				// a subdir is selected -> execute all kernel under that subdir
				addKernelsToBIGString(selected, lastNode);
			}
		}
		return selected;
	}

	/**
	 * The kernel tree's cell renderer. This class chooses the appropriate icons for tree nodes and displays tool tip
	 * texts if the mouse cursor is above a tree node.
	 **/
	class KernelTreeCellRenderer extends DefaultTreeCellRenderer {
		private static final long serialVersionUID = 1L;
		ImageIcon kernelIcon;
		ImageIcon kernelBaseOpenedIcon;
		ImageIcon kernelBaseClosedIcon;
		BIGGUI big;
		BIGInterface bigInterface;

		/**
		 * Constructs a new tree cell renderer.
		 **/
		public KernelTreeCellRenderer(BIGGUI gui) {
			big = gui;
			bigInterface = BIGInterface.getInstance();
			String imgPath = bigInterface.getImgPath() + File.separator;
			kernelIcon = new ImageIcon(imgPath + "kernel16.png");
			kernelBaseOpenedIcon = new ImageIcon(imgPath + "kernel_opened16.png");
			kernelBaseClosedIcon = new ImageIcon(imgPath + "kernel_closed16.png");
		}

		/**
		 * Overriding super class' method.
		 **/
		@Override
		public Component getTreeCellRendererComponent(JTree tree, Object value, boolean sel,
				boolean expanded, boolean leaf, int row, boolean hasFocus) {
			super.getTreeCellRendererComponent(tree, value, sel, expanded, leaf, row, hasFocus);
			DefaultMutableTreeNode node = (DefaultMutableTreeNode) value;
			setIcon(kernelIcon); // default node icon
			if (node.equals(node.getRoot())) {
				/* set icon for root node */
				setIcon(kernelBaseOpenedIcon);
			} else if (!node.isLeaf()) {
				/* set icons for parent nodes */
				if (expanded) {
					setIcon(kernelBaseOpenedIcon);
				} else {
					setIcon(kernelBaseClosedIcon);
				}
			}
			Object o = node.getUserObject();
			if (o instanceof BIGKernel) {
				// a kernel tool tip is needed
				String tip = ((BIGKernel) o).getRelativePath();
				setToolTipText(tip);
			} else {
				// a subdir tool tip is needed
				int count = getNumberOfLeaves(node);
				String tip = "The subdirectory  \"" + o.toString() + "\" ";
				if (node.equals(node.getRoot())) {
					tip = "The kernel directory ";
				}
				tip = tip + "contains " + count + " kernel";
				if (count > 1) {
					tip = tip + "s";
				}
				tip = tip + ".";
				setToolTipText(tip);
			}
			return this;
		}
	}
}
/******************************************************************************
 * Log-History $Log: BIGKernelTree.java,v $ Revision 1.8 2006/12/06 06:50:59 rschoene surrounded new BIGKernel with
 * try-catch Revision 1.7 2006/09/28 04:09:37 rschoene commenting Revision 1.6 2006/03/06 09:28:54 rschoene added some
 * single file options Revision 1.5 2005/12/15 11:27:02 rschoene changes in duplicate kernel Revision 1.4 2005/12/15
 * 11:11:09 rschoene duplicate-dialog has larger textfields Revision 1.3 2005/12/14 16:01:18 rschoene added duplicate
 * und delete Revision 1.2 2005/11/11 13:43:45 rschoene now helps KernelScriptWindow showing all selected kernels
 * Revision 1.1 2005/10/20 13:13:56 rschoene src3 add Revision 1.17 2005/06/16 10:46:04 rschoene execute is shown only
 * in unix-systems Revision 1.16 2005/05/23 08:07:46 rschoene check whether to save files when selecting another kernel
 * Revision 1.15 2005/05/10 11:01:32 wloch fixed spacing error in tooltip Revision 1.14 2005/04/21 06:49:26 rschoene
 * removed bug, that set the first selected sorting in the Menu to Language in every case Revision 1.13 2005/04/19
 * 15:22:19 wloch implemented sorting in result tree Revision 1.12 2005/04/14 16:11:57 wloch implemented sorting of the
 * kernel tree Revision 1.11 2005/02/22 11:41:01 wloch corrected remote execute menu entry in menu and popup Revision
 * 1.10 2005/02/21 19:13:16 wloch major changes to menu structure and popup menus Revision 1.9 2005/02/18 09:45:19 wloch
 * implemented a better popup menu API
 ******************************************************************************/
