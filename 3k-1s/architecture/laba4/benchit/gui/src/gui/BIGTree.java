package gui;

import java.awt.event.*;
import java.util.*;

import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

public abstract class BIGTree extends JTree {
	private static final long serialVersionUID = 1L;

	/** Reference to the GUI. */
	protected final BIGGUI gui;

	public BIGTree(TreeModel newModel, BIGGUI gui) {
		super(newModel);
		this.gui = gui;
		setExpandsSelectedPaths(true);
	}

	protected int getNumberOfLeaves(DefaultMutableTreeNode node) {
		int files = 0;
		for (int i = 0; i < node.getChildCount(); i++) {
			if (node.getChildAt(i).isLeaf()) {
				files++;
			} else {
				files = files + getNumberOfLeaves((DefaultMutableTreeNode) node.getChildAt(i));
			}
		}
		return files;
	}

	protected int getNumberOfFinalNodes(DefaultMutableTreeNode node) {
		int files = 0;
		boolean added = false;
		for (int i = 0; i < node.getChildCount(); i++) {
			if (node.getChildAt(i).isLeaf()) {
				if (!added) {
					files++;
					added = true;
				}
			} else {
				files = files + getNumberOfFinalNodes((DefaultMutableTreeNode) node.getChildAt(i));
			}
		}
		return files;
	}

	protected abstract JPopupMenu computePopup(MouseEvent evt);

	private boolean isMyPopupTrigger(MouseEvent evt) {
		int macOnMask = InputEvent.META_MASK | InputEvent.BUTTON1_MASK | InputEvent.BUTTON3_MASK;
		int winOnMask = InputEvent.META_MASK | InputEvent.BUTTON3_MASK;
		if ((evt.getModifiers() & macOnMask) == macOnMask)
			return true;
		else if ((evt.getModifiers() & winOnMask) == winOnMask)
			return true;
		else
			return false;
	}

	/**
	 * Sets up a TreeExpansionListener and MouseListener for popup.
	 **/
	public void setListeners() {
		addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent evt) {
				if (isMyPopupTrigger(evt)) {
					JPopupMenu popUp = computePopup(evt);
					popUp.setInvoker(BIGTree.this);
					popUp.setRequestFocusEnabled(true);
					popUp.setLocation(evt.getX() + (int) getLocationOnScreen().getX(), evt.getY()
							+ (int) getLocationOnScreen().getY());
					popUp.setVisible(true);
					popUp.revalidate();
				}
			}
		});

		addTreeExpansionListener(new TreeExpansionListener() {
			private void expandAll(JTree tree, TreePath parent, boolean expand) {
				// Traverse children
				TreeNode node = (TreeNode) parent.getLastPathComponent();
				if (node.getChildCount() >= 0) {
					for (Enumeration<?> e = node.children(); e.hasMoreElements();) {
						TreeNode n = (TreeNode) e.nextElement();
						TreePath path = parent.pathByAddingChild(n);
						expandAll(tree, path, expand);
					}
				}

				// Expansion or collapse must be done bottom-up
				if (expand) {
					tree.expandPath(parent);
				} else {
					tree.collapsePath(parent);
				}
			}

			public void treeExpanded(TreeExpansionEvent tEvt) {
				TreePath path = tEvt.getPath();
				DefaultMutableTreeNode lastNode = (DefaultMutableTreeNode) path.getLastPathComponent();
				if (getNumberOfLeaves(lastNode) < 8 || getNumberOfFinalNodes(lastNode) <= 2) {
					expandAll(BIGTree.this, path, true);
				}
			}

			public void treeCollapsed(TreeExpansionEvent tEvt) {}
		});
		// JTree MUST be registered manually to display tool tips
		ToolTipManager.sharedInstance().registerComponent(this);
	}

	public void setSelectionPathsMutable(TreePath[] paths) {
		if (paths == null)
			return;
		List<TreePath> list = new ArrayList<TreePath>();
		for (TreePath path : paths) {
			TreePath newPath = getPathByName(path.toString());
			if (newPath != null)
				list.add(newPath);
		}
		TreePath[] newPaths = list.toArray(new TreePath[list.size()]);
		setSelectionPaths(newPaths);
	}

	public void expand(Enumeration<TreePath> toExpand) {
		if (toExpand == null)
			return;
		Vector<TreePath> oldPaths = sort(toExpand);
		for (TreePath path : oldPaths) {
			TreePath newPath = getPathByName(path.toString());
			if (newPath != null)
				expandPath(newPath);
		}
	}

	public TreePath getPathByName(String name) {
		TreePath[] tp = getPathBetweenRows(0, getRowCount());

		for (int i = 0; i < tp.length; ++i) {
			if (tp[i].toString().equals(name)) {
				return tp[i];
			}
		}
		return null;
	}

	public Vector<TreePath> sort(Enumeration<TreePath> expanded) {
		Vector<TreePath> paths = new Vector<TreePath>();
		Vector<Integer> levels = new Vector<Integer>();

		int maxLevel = -1;
		int actualLevel = 0;
		int cur = 0;

		while (expanded.hasMoreElements()) {
			TreePath tp = expanded.nextElement();
			paths.add(tp);
			levels.add(cur, tp.getPathCount());
			cur++;
			if (tp.getPathCount() > maxLevel)
				maxLevel = tp.getPathCount();
		}

		Vector<TreePath> sorted = new Vector<TreePath>();

		while (actualLevel <= maxLevel) {
			for (int i = 0; i < levels.size(); ++i) {
				if (levels.get(i) == actualLevel) {
					sorted.add(paths.get(i));
				}
			}
			actualLevel++;
		}
		return sorted;
	}
}
