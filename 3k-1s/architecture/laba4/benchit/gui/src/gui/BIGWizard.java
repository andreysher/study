package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.Vector;

import javax.swing.*;

import system.BIGInterface;

/**
 * <p>
 * Überschrift: BenchIT
 * </p>
 * <p>
 * Beschreibung:
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
public class BIGWizard extends JFrame {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private final Vector<JPanel> panels = new Vector<JPanel>();
	private final Vector<Action> actions = new Vector<Action>();

	// used for an icon or what...
	JPanel leftPanel = new JPanel();
	CardLayout rightPaneLayout = new CardLayout();
	JPanel rightPanel = new JPanel(rightPaneLayout);
	JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
	int actualPanel = 0;
	private Action finishAction = new DefaultCloseAction();
	boolean backActivated = true;
	JButton next, back, finish;

	public BIGWizard() {
		next = new JButton("Next >>");
		back = new JButton("<< Back");
		finish = new JButton("Exit");
		next.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				next(evt);
			}
		});
		back.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				back(evt);
			}
		});
		back.setEnabled(false);
		finish.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				// System.err.println("finish");
				// System.err.println(finishAction);
				finishAction.actionPerformed(evt);
			}
		});

		buttonPanel.add(back);
		buttonPanel.add(next);
		buttonPanel.add(finish);

		JLabel lab = new JLabel(new ImageIcon(system.BIGInterface.getInstance().getImgPath()
				+ File.separator + "splash.jpg"));
		lab.setBorder(BorderFactory.createEtchedBorder());
		leftPanel.add(lab, BorderLayout.CENTER);
		// putting all together
		JPanel contentPane = new JPanel();
		contentPane.setLayout(new BorderLayout());
		contentPane.add(leftPanel, BorderLayout.WEST);
		contentPane.add(rightPanel, BorderLayout.CENTER);
		contentPane.add(buttonPanel, BorderLayout.SOUTH);
		setContentPane(contentPane);
		setIconImage(new ImageIcon(BIGInterface.getInstance().getImgPath() + File.separator
				+ "clock.png").getImage());
	}

	public BIGWizard(String title) {
		this();
		setTitle(title);
	}

	public BIGWizard(String title, boolean backActivated) {
		this(title);
		this.backActivated = backActivated;
	}

	public JPanel[] getPanels() {
		JPanel[] pan = new JPanel[panels.size()];
		for (int i = 0; i < pan.length; i++) {
			pan[i] = panels.get(i);
		}
		return pan;
	}

	public JPanel getPanel(int position) {
		return panels.get(position);
	}

	public void next(ActionEvent evt) {
		rightPaneLayout.next(rightPanel);
		actualPanel++;
		setFinished(actualPanel == panels.size() - 1);
		rightPanel.revalidate();
		buttonPanel.revalidate();
		actions.get(actualPanel).actionPerformed(evt);
	}

	public void setFinished(boolean finished) {
		if (finished) {
			finish.setText("Finish");
		} else {
			finish.setText("Exit");
		}
		next.setEnabled(!finished && actualPanel < panels.size());
		back.setEnabled(!finished && backActivated && actualPanel >= 1);
	}

	public void back(ActionEvent evt) {
		rightPaneLayout.previous(rightPanel);
		actualPanel--;
		setFinished(false);
		rightPanel.revalidate();
		buttonPanel.revalidate();
		actions.get(actualPanel).actionPerformed(evt);
	}

	public void addPanel(JPanel pan) {
		panels.add(pan);
		actions.add(new DefaultAction());
		rightPanel.add(pan.toString(), pan);
	}

	public void addPanel(JPanel pan, String title) {
		pan.setBorder(BorderFactory.createTitledBorder(title));
		this.addPanel(pan);
	}

	public void addPanel(JPanel pan, int position) {
		panels.add(position, pan);
		actions.add(new DefaultAction());
		for (int i = position + 1; i < panels.size(); i++) {
			rightPanel.remove(panels.get(i));
			rightPanel.add(panels.get(i));
		}
	}

	public void addPanel(JPanel pan, Action act) {
		panels.add(pan);
		actions.add(act);
		rightPanel.add(pan.toString(), pan);
	}

	public void addPanel(JPanel pan, String title, Action act) {
		pan.setBorder(BorderFactory.createTitledBorder(title));
		this.addPanel(pan, act);
	}

	public void addPanel(JPanel pan, int position, Action act) {
		panels.add(position, pan);
		actions.add(position, act);
		for (int i = position + 1; i < panels.size(); i++) {
			rightPanel.remove(panels.get(i));
			rightPanel.add(panels.get(i));
		}
	}

	public void removePanel(JPanel pan) {
		int index = panels.indexOf(pan);
		panels.remove(index);
		actions.remove(index);
		rightPanel.remove(pan);
	}

	public void setFinishAction(Action action) {
		finishAction = action;
	}

	public void start() {
		if (panels.size() == 0)
			return;
		if (panels.size() == 1) {
			next.setEnabled(false);
		}
		pack();
		rightPanel.revalidate();
		int w = getGraphicsConfiguration().getDevice().getDisplayMode().getWidth();
		int h = getGraphicsConfiguration().getDevice().getDisplayMode().getHeight();
		this.setBounds((w - getWidth()) / 2, (h - getHeight()) / 2, getWidth(), getHeight());

		setVisible(true);
	}

	class DefaultAction extends AbstractAction {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;

		public void actionPerformed(ActionEvent evt) {

			// System.err.println("default");
		}
	};

	class DefaultCloseAction extends AbstractAction {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;

		public void actionPerformed(ActionEvent evt) {
			// System.err.println("defaultClose");
			setVisible(false);
		}
	};

}
