/*********************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGGUI.java Author: SWTP
 * Nagel 1 Last change by: $Author: tschuet $ $Revision: 1.47 $ $Date: 2009/01/07 12:06:48 $
 *********************************************************************/
package gui;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.awt.print.*;
import java.io.*;
import java.util.*;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.tree.*;

import plot.data.*;
import plot.gui.BIGPlot;
import system.*;
import admin.BIGAdmin;
import conn.MyJFrame;

/**
 * The BIGGUI handles all possible entries by the given host in the BIGInterface. You can change the value of each entry
 * to specify your local def file.<br>
 * The BIGGUI is also synchronized with the BIGEditor, so changing values here changed also the value in the editor.<br>
 * <br>
 * 
 * @see BIGEditor
 * @author Carsten Luxig <a href="mailto:c.luxig@lmcsoft.com">c.luxig@lmcsoft.com</a> some changes by Robert Schoene <a
 *         href="mailto:roberts@zhr.tu-dresden.de">roberts@zhr.tu-dresden.de</a> interface redesign by Robert Wloch <a
 *         href="mailto:wloch@zhr.tu-dresden.de">wloch@zhr.tu-dresden.de</a>
 */

public class BIGGUI extends JFrame {
	private static final long serialVersionUID = 1L;
	private Process executedProcess = null;
	// Variables declaration - do not modify
	private final BIGInterface bigInterface = BIGInterface.getInstance();
	final BIGConsole console = bigInterface.getConsole();
	// private BIGHelp help = new BIGHelp();
	private BIGAboutWindow about = null;
	// private BIGExecuteWindow executeWindow;
	private static BIGQuickViewWindow quickViewWindow;
	private static BIGKernelScriptWindow kernelScriptWindow;
	private static String VIEWPANEL = "Graphical View";
	private static String TEXTPANEL = "Plain Text View";
	private static String KERNELS_TAB = "Kernels";
	private static String RESULTS_TAB = "Results";
	private static boolean initDone;
	private int dimx;
	private BIGResultMixer[] resultMixer;

	// represents the current state of the GUI
	private GUIState state;

	// private JComponent adminPanel;
	private MyJFrame connectionFrame;
	private JComponent normalPanel;
	private JComponent[] resultViewPanels;

	private Action adminItemAction;
	private Action databaseItemAction;
	private Action localdefsItemAction;
	// private Action resultsItemAction;
	// private Action graficalViewItemAction;
	// private Action textualViewItemAction;
	private Action toggleViewItemAction;
	private Action executeSelectedItemAction;
	private Action showResultItemAction;

	// Menu variables
	private JMenu fileMenu;
	private JMenu setupMenu;
	private JMenu measureMenu;
	private JMenu evaluateMenu;
	private JMenu databaseMenu;
	private JMenu helpMenu;
	// remote menu is not present itself, but used to get it's menu items
	private BIGRemoteMenu remoteMenu;
	// menu items
	private JMenuItem adminMenuItem;
	private JMenuItem dbMenuItem;
	private JMenuItem localdefMenuItem;
	// private JMenuItem resultsMenuItem;
	private JMenuItem preferencesMenuItem;
	private JMenuItem quitMenuItem;
	// private JMenuItem graphicViewMenuItem;
	// private JMenuItem textViewMenuItem;
	private JMenuItem toggleViewMenuItem;
	private JMenuItem loadMenuItem;
	private JMenuItem saveMenuItem;
	private JMenuItem saveAsMenuItem;
	private JMenuItem executeSelectedMenuItem;
	private JMenuItem updateKernelTreeMenuItem;
	private JMenuItem showResultMenuItem;
	private JMenuItem loadUpMenuItem;
	private JMenuItem updateResultTreeMenuItem;
	private JMenuItem helpMenuItem;
	private JMenuItem tipMenuItem;
	private JMenuItem aboutMenuItem;
	private JMenuItem printMenuItem;

	private BIGAdmin adminTool;
	private JToolBar toolbar;
	private JToolBar statusBar;
	private JPanel mainViewPanel;
	private JSplitPane mainSplitPane;
	private JSplitPane rightSplitPane;
	BIGPlot plotWindow;

	private JTabbedPane listTabs;
	private JPanel cardsButtonPanel;
	private JPanel viewPanel;
	private JPanel textPanel;

	BIGGUIObserverProgress statusProgress;
	JLabel statusLabel;

	// private JButton viewButton;
	// private JButton textButton;
	private JComboBox<?> localdefFilesComboBox;
	private JComboBox<?> viewLevelComboBox;
	// private JCheckBox showCheckBox;
	private String localDefFilesComboBoxEntries[];
	private String viewLevelComboBoxEntries[];
	private final String[] choiceTypes = {"compile only", "run only", "compile and run"};
	private final int defaultSelected = 2;
	private JList<String> choiceList;

	private final String kernelRootName = "all kernels";
	private BIGKernelTree kernelTree;
	private JScrollPane kernelTreeScrollPane;
	private final String resultRootName = "all results";
	BIGResultTree resultTree;

	/*
	 * private JTextPane editor; private JPopupMenu priorityPopup; private JButton priorityButton, quickviewButton;
	 * private JButton startButton, closeButton;
	 */
	private Font labelFont;
	// private JProgressBar progressBar;
	private DetailLevel detailLevel = DetailLevel.Low;
	private final HashMap<String, BIGEditor> editorFileRelation = new HashMap<String, BIGEditor>();
	private boolean textViewMode;
	private int rightSplit, mainSplit;
	// debug
	private final int debug = bigInterface.getDebug("BIGGUI");
	// added by rschoene for quickLoad
	// private boolean fullLoaded=false;
	private String staticTitle;
	private JPanel consoleScrollPane;

	// End of variables declaration
	/**
	 * Creates new form BIGGUI.
	 * 
	 * @param startProgress progress in loading/creating will be shown here
	 * @param progressLabel information about the progress will be shown here
	 */
	public BIGGUI(final JProgressBar startProgress, final JLabel progressLabel) {
		// This is an undefined state, but it is neccessary to set it
		// at the beginning of the initialization.
		// The last statement of this constructor must set a defined
		// and allowed state.
		state = GUIState.LOADING;
		kernelScriptWindow = null;
		// adminPanel = null;
		connectionFrame = null;
		normalPanel = null;
		resultViewPanels = null;
		initDone = false;
		rightSplit = -1;
		mainSplit = -1;
		// this.fullLoaded=fullLoad;

		setIconImage(new ImageIcon(bigInterface.getImgPath() + File.separator + "clock.png").getImage());

		progressLabel.setText("Initializing GUI: GUI components");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}
		initComponents(startProgress, progressLabel);
		updateTitle();
		pack();
		// enabling window-close
		enableEvents(AWTEvent.WINDOW_EVENT_MASK);
		// setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		// just for fun,
		// setting the windows look and feel by cmd arg: win
		if (bigInterface.getLookAndFeel().equals("win")) {
			try {
				UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
			} catch (Exception ex) {
				System.out.println(ex.toString());
			}
			SwingUtilities.updateComponentTreeUI(getContentPane());
		}
		// just to prevent any unwanted start up state
		if (bigInterface.isRestart()) {
			setGUIState(GUIState.RESULTS);
		} else {
			setGUIState(GUIState.LOCALDEFS);
		}
		bigInterface.getHostObservable().addObserver(new Observer() {
			public void update(Observable arg0, Object arg1) {
				updateTitle();
				if (bigInterface.getHost().equals(bigInterface.getOriginalHost()))
					console.postMessage("WARNING: Please be aware that the"
							+ " current loaded LOCALDEFS belong to a different host"
							+ " than the one this BenchIT GUI is running\non. If you"
							+ " intend to to a remote execute this might be just"
							+ " wanted. However, if you're going to run kernels on"
							+ " the\npresent host, make sure you didn't alter the"
							+ " wrong LOCALDEFS by accident.", BIGConsole.WARNING);
			}
		});

		// now start a thread to load the input_[architecture|display]
		// localdef files
		Thread loadThread = new Thread() {
			@Override
			public void run() {
				// check, if daily tip should be shown
				boolean showDailyTip = false;
				try {
					BIGConfigFileParser parser = bigInterface.getBIGConfigFileParser();
					showDailyTip = parser.boolCheckOut(BIGDailyTip.configFileKeyShow);
				} catch (Exception e) {
					showDailyTip = false;
				}
				if (showDailyTip && !bigInterface.isRestart()) {
					BIGDailyTip dailyTip = new BIGDailyTip(BIGGUI.this);
					dailyTip.requestFocus();
				}
				startProgress.setValue(100);
				progressLabel.setText("done.");
			}
		};
		loadThread.start();
	}

	public BIGRemoteMenu getRemoteMenu() {
		return remoteMenu;
	}

	/**
	 * this method is used for saving the settings to the config file
	 * 
	 * @param e only handled, when window is closed/BIG is closed
	 **/
	@Override
	protected void processWindowEvent(WindowEvent e) {

		super.processWindowEvent(e);
		if (e.getID() == WindowEvent.WINDOW_CLOSING) {
			saveAndExit();
		}

	}

	/**
	 * This method is called from within the constructor to initialize the GUI and it's layout.
	 * 
	 * @param startProgress progress in loading will be visible there
	 * @param progressLabel information about progress
	 */
	private void initComponents(JProgressBar startProgress, JLabel progressLabel) {
		dimx = 1024;
		// First of all get the current screen resolution
		GraphicsConfiguration gc = getGraphicsConfiguration();
		if (gc != null) {
			GraphicsDevice gd = gc.getDevice();
			if (gd != null) {
				dimx = gd.getDisplayMode().getWidth();
			}
		}
		progressLabel.setText("Initializing GUI: Creating menu items");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}

		// create menu items
		adminItemAction = new AdminItemAction("Administrate LOCALDEF variables", null, true);
		adminMenuItem = new JMenuItem(adminItemAction);
		adminMenuItem.setText("Localdef Specification");

		databaseItemAction = new DatabaseItemAction("View results stored in the BenchIT Web Database",
				KeyStroke.getKeyStroke(KeyEvent.VK_D, ActionEvent.CTRL_MASK), true);
		dbMenuItem = new JMenuItem(databaseItemAction);
		dbMenuItem.setText("BenchIT Database");

		localdefsItemAction = new LocaldefsItemAction(
				"Set the LOCALDEF parameters for the next kernel execution", KeyStroke.getKeyStroke(
						KeyEvent.VK_L, ActionEvent.CTRL_MASK), true);
		localdefMenuItem = new JMenuItem(localdefsItemAction);
		localdefMenuItem.setText("Localdefs");

		// part of old menu structure
		/*
		 * kernelsItemAction = new KernelsItemAction( "Select and execute the kernels for benchmarking", null, true);
		 * kernelsMenuItem = new JMenuItem( kernelsItemAction ); kernelsMenuItem.setText( "Kernels" ); // part of old menu
		 * structure resultsItemAction = new ResultsItemAction( "Select and display or up load a result file", null, true);
		 * resultsMenuItem = new JMenuItem( resultsItemAction ); resultsMenuItem.setText( "Results" );
		 */

		preferencesMenuItem = new JMenuItem("Preferences");
		preferencesMenuItem.setToolTipText("Customize BenchIT GUI");
		preferencesMenuItem
				.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_O, ActionEvent.CTRL_MASK));
		preferencesMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				bigInterface.getBIGConfigFileParser().showConfigDialog(BIGGUI.this, kernelTree);
			}
		});

		quitMenuItem = new JMenuItem("Quit", KeyEvent.VK_Q);
		quitMenuItem.setToolTipText("This will terminate the program");
		quitMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_Q, ActionEvent.CTRL_MASK));
		quitMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				saveAndExit();
			}
		});

		// part of old menu structure
		/*
		 * graficalViewItemAction = new GraficalViewItemAction( "Displays the selected information in a grafical way", null,
		 * true ); graphicViewMenuItem = new JMenuItem( graficalViewItemAction ); graphicViewMenuItem.setText(
		 * "Grafical View" );
		 */

		// part of old menu structure
		/*
		 * textualViewItemAction = new TextualViewItemAction( "Displays the selected information as plain text", null, true
		 * ); textViewMenuItem = new JMenuItem( textualViewItemAction ); textViewMenuItem.setText( "Textual View" );
		 */

		toggleViewItemAction = new ToggleViewItemAction(
				"Toggles display of selected information as plain text or " + "graphical components",
				KeyStroke.getKeyStroke(KeyEvent.VK_T, ActionEvent.CTRL_MASK), true);
		toggleViewMenuItem = new JMenuItem(toggleViewItemAction);
		toggleViewMenuItem.setText("Toggle View - Text/Graphic");

		loadMenuItem = new JMenuItem("Load");
		loadMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_L, ActionEvent.CTRL_MASK));
		loadMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				loadRequest();
			}
		});

		saveMenuItem = new JMenuItem("Save");
		saveMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_S, ActionEvent.CTRL_MASK));
		saveMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				saveRequest();
			}
		});

		saveAsMenuItem = new JMenuItem("Save As...");
		saveAsMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				saveAsRequest();
			}
		});

		printMenuItem = new JMenuItem("Print");
		printMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				print();
			}
		});
		printMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_P, ActionEvent.CTRL_MASK));

		JMenuItem screenshot = new JMenuItem("Screenshot");
		screenshot.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				BIGFileChooser fc = new BIGFileChooser();
				String[] exts = {"jpg", "png"};
				for (String ext : exts) {
					if (ImageIO.getImageWritersBySuffix(ext).hasNext())
						fc.addFileFilter(ext);
					else
						System.out.println("no " + ext);
				}

				if (fc.showSaveDialog(null) != JFileChooser.APPROVE_OPTION)
					return;
				String name = fc.getSelectedFile().getAbsolutePath();
				String extension = null;
				if (name.toLowerCase().endsWith(".png")) {
					extension = "png";
				}
				if (name.toLowerCase().endsWith(".jpg")) {
					extension = "jpg";
				}
				if (extension == null) {
					if (fc.getFileFilter().getDescription().indexOf("PNG") > -1) {
						name = name + ".png";
						extension = "png";
					} else {
						name = name + ".jpg";
						extension = "jpg";
					}
				}
				BufferedImage bi = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_RGB);
				Graphics2D g2d = bi.createGraphics();
				printAll(g2d);
				g2d.dispose();
				try {
					ImageIO.write(bi, extension, new File(name));
				} catch (IOException ex) {
					System.out.println("Writing screenshot failed");
				}
			}
		});

		executeSelectedItemAction = new ExecuteSelectedItemAction(
				"Executes the selected kernels on the local machine", KeyStroke.getKeyStroke(KeyEvent.VK_E,
						ActionEvent.CTRL_MASK), true);
		executeSelectedMenuItem = new JMenuItem(executeSelectedItemAction);
		executeSelectedMenuItem.setText("Execute Selected");

		updateKernelTreeMenuItem = new JMenuItem("Reload Kernel Tree");
		updateKernelTreeMenuItem.setToolTipText("Refresh the list of kernels");
		updateKernelTreeMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F4, 0));
		updateKernelTreeMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				kernelTree.updateKernelTree();
			}
		});

		showResultItemAction = new ShowResultItemAction(
				"Displays the selected kernel in the result tree", null, true);
		showResultMenuItem = new JMenuItem(showResultItemAction);
		showResultMenuItem.setText("Show Result");

		// to be implemented in the future
		loadUpMenuItem = new JMenuItem("Load Up");
		loadUpMenuItem.setToolTipText("Load up the result file into the BenchIT Web Database");
		loadUpMenuItem.setEnabled(false);
		loadUpMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// loadUp();
			}
		});

		updateResultTreeMenuItem = new JMenuItem("Reload Result Tree");
		updateResultTreeMenuItem.setToolTipText("Refresh the tree of the result files");
		updateResultTreeMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F5, 0));

		updateResultTreeMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				resultTree.updateResultTree(null);
				System.out.println("Result tree reloaded!");
			}
		});

		helpMenuItem = new JMenuItem("Help");
		helpMenuItem.setToolTipText("Get help about the program");
		helpMenuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_F1, 0));
		helpMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				BIGHelp help = new BIGHelp();
				help.setVisible(true);
			}
		});

		tipMenuItem = new JMenuItem("Tip of the day...");
		tipMenuItem.setToolTipText("Get tip of the day about the program");
		tipMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				new BIGDailyTip(BIGGUI.this);
			}
		});

		aboutMenuItem = new JMenuItem("About");
		aboutMenuItem.setToolTipText("Get general information about the program");
		aboutMenuItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (about == null) {
					about = new BIGAboutWindow(BIGGUI.this);
				} else {
					about.setVisible(!about.isVisible());
				}
			}
		});

		// instanciate an object of MyJFrame with creates and
		// contains all the GUI components needed to connect with
		// the BenchIT database.
		String server = null;
		try {
			server = bigInterface.getBIGConfigFileParser().stringCheckOut("serverIP");
		} catch (Exception ex) {
			System.err.println("could not read the serverIP from BGUI.cfg");
			databaseMenu = new JMenu("Could not find serverIP in BGUI.cfg");
		}
		if (server != null) {
			connectionFrame = new MyJFrame(server, this);
			databaseMenu = connectionFrame.getMenu();
			databaseMenu.setMnemonic(KeyEvent.VK_D);
		}

		progressLabel.setText("Initializing GUI: Creating menu bar");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}

		// now set up the menu bar
		JMenuBar menubar = new JMenuBar();
		// ///////////////////////////////////////////////
		// this is the new menu structure
		// ///////////////////////////////////////////////
		fileMenu = new JMenu("File");
		fileMenu.setMnemonic(KeyEvent.VK_F);
		setupMenu = new JMenu("Setup");
		setupMenu.setMnemonic(KeyEvent.VK_S);
		measureMenu = new JMenu("Measure");
		measureMenu.setMnemonic(KeyEvent.VK_M);
		evaluateMenu = new JMenu("Evaluate");
		evaluateMenu.setMnemonic(KeyEvent.VK_E);
		helpMenu = new JMenu("Help");
		helpMenu.setMnemonic(KeyEvent.VK_H);
		/*
		 * JMenuItem setSession= new JMenuItem(new AbstractAction() { public void actionPerformed(ActionEvent evt) {
		 * setSessionDialog(); } }); setSession.setText("Set Session"); fileMenu.add(setSession);
		 */
		fileMenu.add(loadMenuItem);
		fileMenu.add(saveMenuItem);
		fileMenu.add(saveAsMenuItem);
		fileMenu.addSeparator();
		fileMenu.add(printMenuItem);
		fileMenu.addSeparator();
		fileMenu.add(screenshot);
		fileMenu.addSeparator();
		fileMenu.add(quitMenuItem);
		menubar.add(fileMenu);

		// get the menu items for remote stuff
		remoteMenu = new BIGRemoteMenu(this);
		Map<String, List<JMenuItem>> remoteLists = remoteMenu.getRemoteMenus();
		// get the menu items for DB stuff
		Map<String, List<JMenuItem>> dbLists = connectionFrame.getDBMenus();

		setupMenu.add(adminMenuItem);
		setupMenu.add(localdefMenuItem);
		// add remote setup menu items to setup menu
		addListItems(remoteLists, setupMenu, "setup");
		// add DB setup menu items to setup menu
		addListItems(dbLists, setupMenu, "setup");
		setupMenu.addSeparator();
		setupMenu.add(preferencesMenuItem);
		JMenuItem item = new JMenuItem("Set Default Plot-Colors");
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				final JFrame f = new JFrame("Choose the standard colors for plotting");
				Container pane = f.getContentPane();
				final Paint[] p = BIGPlot.getDefaultPaintSequence();
				final Paint[] newP = new Paint[p.length];
				for (int i = 0; i < p.length; i++) {
					newP[i] = p[i];
				}
				final JComboBox<String> jcomb = new JComboBox<String>();
				for (int i = 0; i < p.length; i++) {
					jcomb.addItem("" + (i + 1));
				}
				jcomb.setSelectedIndex(0);
				final JColorChooser jcc = new JColorChooser((Color) p[0]);
				jcomb.addActionListener(new ActionListener() {
					int oldSelected = 0;

					public void actionPerformed(ActionEvent ae) {
						newP[oldSelected] = jcc.getColor();
						jcc.setColor((Color) newP[jcomb.getSelectedIndex()]);
						jcc.revalidate();
						oldSelected = jcomb.getSelectedIndex();
					}
				});
				Button ok = new Button("Okay");
				ok.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent ae) {
						newP[jcomb.getSelectedIndex()] = jcc.getColor();
						BIGPlot.setDefaultPaintSequence(newP);
						f.setVisible(false);
					}
				});

				Button cancel = new Button("Cancel");
				cancel.addActionListener(new ActionListener() {
					public void actionPerformed(ActionEvent ae) {
						f.setVisible(false);
					}
				});

				pane.setLayout(new BorderLayout());
				JPanel buttonPan = new JPanel(new FlowLayout());
				buttonPan.add(ok);
				buttonPan.add(cancel);
				JPanel comboPan = new JPanel(new FlowLayout());
				comboPan.add(new JLabel("Actual function"));
				comboPan.add(jcomb);

				pane.add(comboPan, BorderLayout.NORTH);
				pane.add(jcc, BorderLayout.CENTER);

				pane.add(buttonPan, BorderLayout.SOUTH);
				f.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
				f.pack();
				f.setVisible(true);
			}
		});
		setupMenu.add(item);
		item = new JMenuItem("Reset all Plot-Information");
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				for (int i = 0; i < resultMixer.length; i++) {
					resultMixer[i].cleanUp();
				}
				resultTree.removePlotInformations();
				if (actualFile != null) {
					// reloadOutput();
					loadResult(actualFile, false);
				}
			}
		});
		setupMenu.add(item);
		item = new JMenuItem("Set Plot Fonts");
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				// final JFrame f=new JFrame("Choose the Fonts for Axis");
				// final JComboBox jcomb=new JComboBox(BIGPlotable.fontNames);
				final Font fonts[] = new Font[BIGPlotable.fontNames.length];
				// do not use BIGUtility.getFont for using the real defaults
				fonts[0] = getFont("XAxis");
				if (fonts[0] == null) {
					System.out.println("Axis Font settings not found in BGUI.cfg");
					System.out.println("Inserting default");
					Font lFont = (new Font("SansSerif", Font.BOLD, 14));
					Font tFont = (new Font("SansSerif", Font.BOLD, 16));
					bigInterface.getBIGConfigFileParser().save();
					File bguicfg = bigInterface.getBIGConfigFileParser().getFile();
					String content = BIGFileHelper.getFileContent(bguicfg);
					StringBuffer sb = new StringBuffer(content);
					sb.append("\n# start auto filled axis font settings\n");
					for (int i = 0; i < BIGPlotable.fontNames.length; i++) {
						sb.append(BIGPlotable.fontNames[i] + "Font = " + lFont.getName() + "\n");
						sb.append(BIGPlotable.fontNames[i] + "FontStyle = " + lFont.getStyle() + "\n");
						if (i == 1) {
							sb.append(BIGPlotable.fontNames[i] + "FontSize = " + tFont.getSize() + "\n");
						} else {
							sb.append(BIGPlotable.fontNames[i] + "FontSize = " + lFont.getSize() + "\n");
						}
						sb.append(BIGPlotable.fontNames[i] + "Font = " + tFont.getName() + "\n");
					}
					sb.append("# end auto filled axis font settings");
					BIGFileHelper.saveToFile(sb.toString(), bguicfg);
					bigInterface.setBIGConfigFileParser(new BIGConfigFileParser(bguicfg.getAbsolutePath()));
					fonts[0] = lFont;
				}
				for (int i = 0; i < BIGPlotable.fontNames.length; i++) {
					fonts[i] = BIGUtility.getFont(BIGPlotable.fontNames[i].toString());
				}
				BIGFontDialog fontDlg = new BIGFontDialog(fonts);
				fontDlg.setVisible(true);
			}

			public Font getFont(String s) {
				try {
					String name = bigInterface.getBIGConfigFileParser().stringCheckOut(s + "Font");
					int style = bigInterface.getBIGConfigFileParser().intCheckOut(s + "FontStyle");
					int size = bigInterface.getBIGConfigFileParser().intCheckOut(s + "FontSize");
					return new Font(name, style, size);
				} catch (Exception ex) {
					return null;
				}

			}
		});
		setupMenu.add(item);
		item = new JMenuItem("Set Standard Plot Comment");
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				String comment = null;
				try {
					comment = bigInterface.getBIGConfigFileParser().stringCheckOut("plotComment");
				} catch (Exception ex) {

					bigInterface.getBIGConfigFileParser().save();
					File bguicfg = bigInterface.getBIGConfigFileParser().getFile();
					String content = BIGFileHelper.getFileContent(bguicfg);
					StringBuffer sb = new StringBuffer(content);
					sb.append("\n# start auto filled settings\n");
					sb.append("plotComment = \n");
					sb.append("plotCommentPercentX = 80\n");
					sb.append("plotCommentPercentY = 90\n");

					sb.append("# end auto filled settings");
					BIGFileHelper.saveToFile(sb.toString(), bguicfg);
					bigInterface.setBIGConfigFileParser(new BIGConfigFileParser(bguicfg.getAbsolutePath()));
					comment = "";
				}
				String newComment = JOptionPane
						.showInputDialog(
								"Insert the new Standard Plot Comment, use <> to find settings in output-file (eg. use <date>)",
								comment);
				if (newComment == null)
					return;
				bigInterface.getBIGConfigFileParser().set("plotComment", newComment);
				int i = -1;
				while (i == -1) {
					try {
						i = Integer.parseInt(JOptionPane.showInputDialog(
								"Start Position of x-coordinate for comment (Default is 80).", ""
										+ bigInterface.getBIGConfigFileParser().intCheckOut("plotCommentPercentX")));
					} catch (Exception ex1) {}
				}
				bigInterface.getBIGConfigFileParser().set("plotCommentPercentX", "" + i);
				i = -1;
				while (i == -1) {
					try {
						i = Integer.parseInt(JOptionPane.showInputDialog(
								"Start Position of y-coordinate for comment (Default is 90).", ""
										+ bigInterface.getBIGConfigFileParser().intCheckOut("plotCommentPercentY")));
					} catch (Exception ex1) {}
				}
				bigInterface.getBIGConfigFileParser().set("plotCommentPercentY", "" + i);

			}
		});
		setupMenu.add(item);
		item = new JMenuItem("Set Standard Title");
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent evt) {
				String comment = null;
				try {
					comment = bigInterface.getBIGConfigFileParser().stringCheckOut("standardTitle");
				} catch (Exception ex) {

					bigInterface.getBIGConfigFileParser().save();
					File bguicfg = bigInterface.getBIGConfigFileParser().getFile();
					String content = BIGFileHelper.getFileContent(bguicfg);
					StringBuffer sb = new StringBuffer(content);
					sb.append("\n# start auto filled settings\n");
					sb.append("standardTitle = \n");

					sb.append("# end auto filled settings");
					BIGFileHelper.saveToFile(sb.toString(), bguicfg);
					bigInterface.setBIGConfigFileParser(new BIGConfigFileParser(bguicfg.getAbsolutePath()));
					comment = "";
				}
				String newComment = JOptionPane
						.showInputDialog(
								"Insert the new Standard Title, use <> to find settings in output-file (eg. use <date>)",
								comment);
				if (newComment == null)
					return;
				bigInterface.getBIGConfigFileParser().set("standardTitle", newComment);

			}
		});
		setupMenu.add(item);
		item = new JMenuItem("Set Default Plot-Insets");
		item.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				Insets i = null;
				try {

					i = new Insets(bigInterface.getBIGConfigFileParser().intCheckOut("plotInsetsTop"),
							bigInterface.getBIGConfigFileParser().intCheckOut("plotInsetsLeft"), bigInterface
									.getBIGConfigFileParser().intCheckOut("plotInsetsBottom"), bigInterface
									.getBIGConfigFileParser().intCheckOut("plotInsetsRight"));
					// use standard if exception
				} catch (Exception ex) {
					bigInterface.getBIGConfigFileParser().set("plotInsetsTop", "4");
					bigInterface.getBIGConfigFileParser().set("plotInsetsLeft", "8");
					bigInterface.getBIGConfigFileParser().set("plotInsetsBottom", "4");
					bigInterface.getBIGConfigFileParser().set("plotInsetsRight", "32");
					i = new Insets(4, 8, 4, 32);
					bigInterface.getBIGConfigFileParser().save();
				}

				BIGInsetsPanel inpan = new BIGInsetsPanel(i);
				int ret = JOptionPane.showConfirmDialog(getContentPane(), inpan,
						"Select the new standard insets", JOptionPane.OK_CANCEL_OPTION);
				if (ret == JOptionPane.CANCEL_OPTION)
					return;
				else {
					bigInterface.getBIGConfigFileParser().set("plotInsetsTop", "" + inpan.getValues().top);
					bigInterface.getBIGConfigFileParser().set("plotInsetsLeft",
							"" + inpan.getValues().left);
					bigInterface.getBIGConfigFileParser().set("plotInsetsBottom",
							"" + inpan.getValues().bottom);
					bigInterface.getBIGConfigFileParser().set("plotInsetsRight",
							"" + inpan.getValues().right);
				}
			}
		});
		setupMenu.add(item);

		JMenuItem environmentEditorItem = new JMenuItem("Edit Environments");
		environmentEditorItem.setToolTipText("Edit Environments");
		environmentEditorItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				JFrame frame = new JFrame("Environments");
				frame.setIconImage(new ImageIcon(bigInterface.getImgPath() + File.separator + "clock.png")
						.getImage());
				frame.setGlassPane(new JPanel());
				frame.setContentPane(new BIGEnvironmentEditor());
				frame.pack();
				frame.setVisible(true);
			}
		});
		setupMenu.addSeparator();
		setupMenu.add(environmentEditorItem);

		menubar.add(setupMenu);
		if (BIGInterface.getSystem() == BIGInterface.UNIX_SYSTEM) {
			measureMenu.add(executeSelectedMenuItem);
		}

		JMenuItem stopItem = new JMenuItem("Stop Measurement");
		stopItem.setToolTipText("Stops local running Measurement");
		stopItem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (executedProcess != null) {
					executedProcess.destroy();
					executedProcess = null;
				} else {
					System.out.println("There is no running process.");
				}
			}
		});
		if (BIGInterface.getSystem() == BIGInterface.UNIX_SYSTEM) {
			measureMenu.add(stopItem);
		}

		measureMenu.addSeparator();
		// this menu is not need any longer as we have exec in remote folder now
		// measureMenu.add(executeSelectedRemoteMenuItem);
		// add remote measure menu items to measure menu
		addListItems(remoteLists, measureMenu, "measure", false);
		// add DB measure menu items to measure menu
		addListItems(dbLists, measureMenu, "measure");
		measureMenu.addSeparator();
		measureMenu.add(updateKernelTreeMenuItem);
		menubar.add(measureMenu);

		evaluateMenu.add(showResultMenuItem);
		// add remote evaluate menu items to evaluate menu
		addListItems(remoteLists, evaluateMenu, "evaluate");
		evaluateMenu.addSeparator();
		// this is not implemented up to now
		// evaluateMenu.add(loadUpMenuItem);
		evaluateMenu.add(dbMenuItem);
		// add DB evaluate menu items to evaluate menu
		addListItems(dbLists, evaluateMenu, "evaluate", false);
		evaluateMenu.addSeparator();
		evaluateMenu.add(updateResultTreeMenuItem);
		menubar.add(evaluateMenu);

		helpMenu.add(helpMenuItem);
		helpMenu.addSeparator();
		helpMenu.add(tipMenuItem);
		helpMenu.addSeparator();
		helpMenu.add(aboutMenuItem);
		menubar.add(helpMenu);

		// set the constructed menu bar
		setJMenuBar(menubar);

		progressLabel.setText("Initializing GUI: Creating toolbar");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}

		// now create the tool bar
		toolbar = new JToolBar();

		// we need the label font later for the item list of the
		// localdef file view
		labelFont = (new JButton(localdefsItemAction)).getFont();
		labelFont = labelFont.deriveFont(Font.PLAIN);

		JButton button = new JButton(localdefsItemAction);
		button.setMaximumSize(new Dimension(40, 40));
		ImageIcon icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "localdef.png");
		button.setIcon(icon);
		toolbar.add(button);

		button = new JButton(databaseItemAction);
		button.setMaximumSize(new Dimension(40, 40));
		icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "webdb.png");
		button.setIcon(icon);// button.setText("Web Database");
		toolbar.add(button);

		toolbar.addSeparator();

		choiceList = new JList<String>(choiceTypes);
		choiceList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		choiceList.setCellRenderer(new ListCellRenderer<String>() {
			JRadioButton[] radioButtons = new JRadioButton[3];

			public Component getListCellRendererComponent(JList<? extends String> list, String value,
					int index, boolean isSelected, boolean cellHasFocus) {
				JRadioButton label = new JRadioButton();
				label.setSelected(isSelected);
				label.setSize(25, 9);
				label.setMaximumSize(new Dimension(25, 9));
				radioButtons[index] = label;
				Font f = label.getFont();
				label.setText(value.toString());
				if (isSelected) {
					Font newF = new Font(f.getFontName(), f.getStyle(), 9);
					label.setForeground(Color.BLACK);
					label.setBackground(Color.WHITE);
					label.setFont(newF);
					label.setMaximumSize(label.getPreferredSize());
				} else {
					Font newF = new Font(f.getFontName(), f.getStyle(), 9);
					label.setFont(newF);
					label.setMaximumSize(label.getPreferredSize());
				}
				return label;
			}
		});
		choiceList.setFixedCellWidth(120);
		choiceList.setFixedCellHeight(12);
		choiceList.setBackground(toolbar.getBackground());
		choiceList.setBorder(BorderFactory.createEtchedBorder()); // Color.BLACK));
		choiceList.setSelectionForeground(Color.BLACK);
		choiceList.setSelectionBackground(Color.WHITE);
		choiceList.setBackground(Color.LIGHT_GRAY);

		choiceList.setSelectedIndex(defaultSelected);
		choiceList.setToolTipText("Choose the steps carried out on execution");

		toolbar.addSeparator();
		toolbar.add(choiceList);

		button = new JButton(executeSelectedItemAction);
		button.setMaximumSize(new Dimension(40, 40));
		icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "exec.png");
		button.setIcon(icon);
		if (BIGInterface.getSystem() == BIGInterface.UNIX_SYSTEM)
			toolbar.add(button);

		button = new JButton(showResultItemAction);
		button.setMaximumSize(new Dimension(40, 40));
		icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "show.png");
		button.setIcon(icon);
		toolbar.add(button);

		toolbar.addSeparator();

		button = new JButton(toggleViewItemAction);
		button.setMaximumSize(new Dimension(40, 40));
		icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "toggle.png");
		button.setIcon(icon);
		toolbar.add(button);

		if (debug > 0) {
			button = new JButton("DebugDB");
			button.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent ae) {
					bigInterface.debugDB();
				}
			});
			toolbar.add(button);
			toolbar.addSeparator();
		}

		boolean enableUpdate = false;
		try {
			enableUpdate = bigInterface.getBIGConfigFileParser().boolCheckOut("updateEnabled");
		} catch (Exception ex2) {
			System.err.println("option \"updateEnabled\" not found in BGUI.cfg");
			System.err.println("Setting it to true. (updateEnabled=1)");
			bigInterface.getBIGConfigFileParser().set("updateEnabled", "1");
		}
		if (enableUpdate) {
			progressLabel.setText("Looking for update");

			icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "update.png");

			boolean test = false;

			String upd = null;
			try {
				upd = bigInterface.getBIGConfigFileParser().stringCheckOut("updateServer");
			} catch (Exception ex2) {
				System.err.println("option \"updateServer\" not found in BGUI.cfg");
				bigInterface.getBIGConfigFileParser().set("updateServer", BIGUpdate.DEFAULTUPDATEURL);
				try {
					BIGUpdate.update(bigInterface.getBIGConfigFileParser().stringCheckOut("updateServer"),
							false);
				} catch (Exception ex1) {
					JOptionPane.showMessageDialog(this, "Update failed: " + ex1.getLocalizedMessage(),
							"Error", JOptionPane.ERROR_MESSAGE);
					return;
				}

			}
			toolbar.add(new JPanel());

			if (test) {
				button = new JButton(new AbstractAction() {
					/**
						 * 
						 */
					private static final long serialVersionUID = 1L;

					public void actionPerformed(ActionEvent ae) {
						(new Thread() {
							@Override
							public void run() {

								int ret = JOptionPane
										.showConfirmDialog(
												BIGGUI.this,
												"An update for BenchIT-GUI is available. Do you want to download it? (This may take a while)",
												"Update available", JOptionPane.YES_NO_OPTION);
								if (ret == JOptionPane.NO_OPTION)
									return;
								try {
									BIGUpdate.update(
											bigInterface.getBIGConfigFileParser().stringCheckOut("updateServer"), false);
								} catch (Exception ex) {
									ex.printStackTrace();
									System.err.println("option \"updateServer\" not found in BGUI.cfg");
									bigInterface.getBIGConfigFileParser().set("updateServer",
											BIGUpdate.DEFAULTUPDATEURL);
									try {
										BIGUpdate
												.update(bigInterface.getBIGConfigFileParser()
														.stringCheckOut("updateServer"), false);

									} catch (Exception ex1) {
										JOptionPane.showMessageDialog(BIGGUI.this,
												"Update failed: " + ex1.getLocalizedMessage(), "Error",
												JOptionPane.ERROR_MESSAGE);
										return;
									}
								}
								JOptionPane.showMessageDialog(BIGGUI.this,
										"Update was downloaded succesfully. Please restart the GUI now!");
							}
						}).start();

					}
				});

				if (icon != null) {
					button.setIcon(icon);
				}
				button.setEnabled(system.BIGUpdate.isUpdateAvailable(upd));
				button.setToolTipText("Update GUI");
				toolbar.add(button);

				// --------------------------------------------------------------------

				JButton txtUpdBtn = new JButton(new AbstractAction() {
					private static final long serialVersionUID = 1L;

					public void actionPerformed(ActionEvent evt) {
						(new Thread() {
							@Override
							public void run() {
								int ret = JOptionPane
										.showConfirmDialog(
												BIGGUI.this,
												"An txt-update for BenchIT-GUI is available. Do you want to download it? (This may take a while)",
												"Update available", JOptionPane.YES_NO_OPTION);

								if (ret == JOptionPane.NO_OPTION)
									return;

								try {
									BIGTextfileUpdate.update(
											bigInterface.getBIGConfigFileParser().stringCheckOut("updateServer"), false);
								} catch (Exception e1) {
									JOptionPane.showMessageDialog(BIGGUI.this,
											"Update failed: " + e1.getLocalizedMessage(), "Error",
											JOptionPane.ERROR_MESSAGE);
									return;
								}
								JOptionPane.showMessageDialog(BIGGUI.this,
										"Update was downloaded succesfully. Please restart the GUI now!");
							}
						}).start();
					}
				});
				if (icon != null) {
					txtUpdBtn.setIcon(icon);
				}
				txtUpdBtn.setToolTipText("Textfile-Update");
				txtUpdBtn.setEnabled(BIGTextfileUpdate.isTextfileUpdateAvailable(upd));
				toolbar.add(txtUpdBtn);
			}

			// stores information about available updates
			// 0 -> no update available
			// everything else -> update available
			// general update , update for text files
			final Integer updAvail[] = {new Integer(0), new Integer(0)};
			JButton updateBtn = new JButton(new AbstractAction() {
				private static final long serialVersionUID = 1L;

				public void actionPerformed(ActionEvent evt) {
					(new Thread() {
						@Override
						public void run() {
							int ret;
							boolean success = false;

							if (updAvail[1].intValue() > 0) {
								// textfile update vailable
								ret = JOptionPane
										.showConfirmDialog(
												BIGGUI.this,
												"An txt-update for BenchIT-GUI is available. Do you want to download it? (This may take a while)",
												"Update available", JOptionPane.YES_NO_OPTION);

								if (ret == JOptionPane.YES_OPTION) {
									try {
										BIGTextfileUpdate
												.update(bigInterface.getBIGConfigFileParser()
														.stringCheckOut("updateServer"), false);
										success = true;
									} catch (Exception e1) {
										e1.printStackTrace();
										System.err.println("option \"updateServer\" not found in BGUI.cfg");
										bigInterface.getBIGConfigFileParser().set("updateServer",
												BIGUpdate.DEFAULTUPDATEURL);
										try {
											BIGUpdate.update(
													bigInterface.getBIGConfigFileParser().stringCheckOut("updateServer"),
													false);
											success = true;
										} catch (Exception ex1) {
											JOptionPane.showMessageDialog(BIGGUI.this,
													"Update failed: " + ex1.getLocalizedMessage(), "Error",
													JOptionPane.ERROR_MESSAGE);
											return;
										}
									}
								}
							}
							if (updAvail[0].intValue() > 0) {
								// general update available
								ret = JOptionPane
										.showConfirmDialog(
												BIGGUI.this,
												"An update for BenchIT-GUI is available. Do you want to download it? (This may take a while)",
												"Update available", JOptionPane.YES_NO_OPTION);
								if (ret == JOptionPane.YES_OPTION) {
									try {
										BIGUpdate
												.update(bigInterface.getBIGConfigFileParser()
														.stringCheckOut("updateServer"), false);
										success = true;
									} catch (Exception ex) {
										JOptionPane.showMessageDialog(BIGGUI.this,
												"Error during update: " + ex.getMessage());
										System.err.println("Error during update: " + ex);
									}
								}
							}
							if (success) {
								JOptionPane.showMessageDialog(BIGGUI.this,
										"Update was downloaded succesfully. Please restart the GUI now!");
							}
						}
					}).start();
				}
			});
			updateBtn.setEnabled(false);
			if (BIGTextfileUpdate.isTextfileUpdateAvailable(upd)) {
				updAvail[1] = new Integer(1);
				updateBtn.setEnabled(true);
			}
			if (BIGUpdate.isUpdateAvailable(upd)) {
				updAvail[0] = new Integer(1);
				updateBtn.setEnabled(true);
			}
			icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "update.png");
			if (icon != null) {
				updateBtn.setIcon(icon);
			}
			updateBtn.setToolTipText("Update");
			toolbar.add(updateBtn);

			// --------------------------------------------------------------------
		}

		// localdefCombo
		localDefFilesComboBoxEntries = new String[3];
		localDefFilesComboBoxEntries[0] = "host info";
		localDefFilesComboBoxEntries[1] = "architecture info";
		localDefFilesComboBoxEntries[2] = "display info";
		localdefFilesComboBox = new JComboBox<Object>(localDefFilesComboBoxEntries);
		localdefFilesComboBox.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				reloadCards();
				viewPanel.revalidate();
				textPanel.revalidate();
				mainViewPanel.revalidate();
			}
		});
		localdefFilesComboBox.setToolTipText("Choose the LOCALDEF info you want to work with");
		// toolbar.add( localdefFilesComboBox );

		// viewlevelCombo
		DetailLevel[] levels = DetailLevel.values();
		viewLevelComboBoxEntries = new String[levels.length];
		for (int i = 0; i < levels.length; i++) {
			String levelString = new String("Detail level " + levels[i]);
			viewLevelComboBoxEntries[i] = levelString;
		}
		viewLevelComboBox = new JComboBox<Object>(viewLevelComboBoxEntries);
		viewLevelComboBox.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				JComboBox<?> cb = (JComboBox<?>) e.getSource();
				detailLevel = DetailLevel.values()[cb.getSelectedIndex()];
				// redraws the list of displayed items
				repaintTabs();
				// declares the components to be valid again
				viewPanel.revalidate();
				textPanel.revalidate();
				mainViewPanel.revalidate();
			}
		});
		viewLevelComboBox.setToolTipText("Choose the detail level of the shown information");
		// toolbar.add( viewLevelComboBox );
		// toolbar.addSeparator();

		// execlevelCombo
		// toolbar.add( choiceJCB );

		progressLabel.setText("Initializing GUI: Creating status bar");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}

		// create the status bar
		statusBar = new JToolBar();
		statusBar.setLayout(new BorderLayout());
		statusProgress = new BIGGUIObserverProgress(0, 100);
		statusLabel = new JLabel();
		statusLabel.setFont(new Font("SansSerif", Font.PLAIN, 10));
		statusLabel.setText("BenchIT starting");
		statusProgress.setPreferredSize(new Dimension(statusProgress.getMaximum(), 15));
		statusProgress.setMinimumSize(new Dimension(statusProgress.getMaximum(), 15));
		statusProgress.setMaximumSize(new Dimension(statusProgress.getMaximum(), 15));
		statusProgress.setFont(new Font("SansSerif", Font.PLAIN, 10));
		statusBar.add(statusProgress, BorderLayout.WEST);
		statusBar.addSeparator();
		statusBar.add(statusLabel, BorderLayout.CENTER);
		// setting a label, which shows free ressources
		final JLabel free = new JLabel();
		free.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent evt) {
				if (evt.getClickCount() > 1) {
					System.gc();
				}
			}
		});
		free.setFont(new Font("SansSerif", Font.PLAIN, 10));
		final TimerTask task = new TimerTask() {
			@Override
			public void run() {
				Thread t = new Thread() {
					@Override
					public void run() {
						free.setText((int) ((Runtime.getRuntime().totalMemory() - Runtime.getRuntime()
								.freeMemory()) / 1048576)
								+ " MB /"
								+ (int) (Runtime.getRuntime().totalMemory() / 1048576) + " MB");
					}
				};
				SwingUtilities.invokeLater(t);

			}
		};
		java.util.Timer t = new java.util.Timer();
		t.scheduleAtFixedRate(task, 0, 10000);
		statusBar.add(free, BorderLayout.EAST);

		bigInterface.setStatusProgress(statusProgress);
		bigInterface.setStatusLabel(statusLabel);

		// false: admin panel is integrated in another GUI
		adminTool = new BIGAdmin();
		statusProgress.setString("");
		statusProgress.setValue(statusProgress.getMinimum());
		statusLabel.setText("done.");
		// Now we create the panels for the actual view area.
		// The viewPanel and textPanel are the contents of a
		// CardLayout and are displayed according to the user's
		// choice to "Grafical View" or "Textual View".
		cardsButtonPanel = new JPanel(new GridBagLayout());
		viewPanel = new JPanel(new GridBagLayout());
		textPanel = new JPanel(new GridBagLayout());
		mainViewPanel = new JPanel(new CardLayout());
		mainViewPanel.add(viewPanel, VIEWPANEL);
		mainViewPanel.add(textPanel, TEXTPANEL);
		JPanel mainPanel = new JPanel(new BorderLayout());
		mainPanel.add(cardsButtonPanel, BorderLayout.NORTH);
		mainPanel.add(mainViewPanel, BorderLayout.CENTER);
		int numberOfMixers = bigInterface.getBIGConfigFileParser().intCheckOut("numberOfMixers", 1);
		resultMixer = new BIGResultMixer[numberOfMixers];
		for (int i = 0; i < numberOfMixers; i++) {
			resultMixer[i] = new BIGResultMixer(viewPanel, "title" + (i + 1));
			resultMixer[i].init(new File(bigInterface.getConfigPath() + File.separator + "mixer_" + i));
			resultMixer[i].setResultTree(resultTree);
		}
		// listTabs is the area to the left, containing list and trees
		listTabs = new JTabbedPane(SwingConstants.TOP);
		// initialize the kernel and result tree
		// only the result tree may be updated during runtime of this
		// program: resultTreeUpdate()
		progressLabel.setText("Initializing GUI: Initializing lists and trees");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}
		initListTabbs();
		// the trees are placed in a JScrollPane and those scroll panes
		// are added to the listTabbs
		kernelTreeScrollPane = new JScrollPane(kernelTree);
		icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "kernel16.png");
		listTabs.addTab(KERNELS_TAB, icon, kernelTreeScrollPane,
				"Select and execute the kernels for benchmarking");
		JScrollPane resultTreeScrollPane = new JScrollPane(resultTree);
		icon = new ImageIcon(bigInterface.getImgPath() + File.separator + "result16.png");
		listTabs.addTab(RESULTS_TAB, icon, resultTreeScrollPane,
				"Select and display or up load a result file");
		listTabs.setPreferredSize(new Dimension(140, 430));

		progressLabel.setText("Initializing GUI: Initializing view panels");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}
		initCards();

		consoleScrollPane = console.getDisplayPanel();
		consoleScrollPane.setPreferredSize(new Dimension(dimx - 200, 100));
		Border border = BorderFactory.createEmptyBorder();
		TitledBorder titledBorder = BorderFactory.createTitledBorder(border,
				"Output console for: Messages | Warnings | Errors");
		titledBorder.setTitleJustification(TitledBorder.CENTER);
		titledBorder.setTitleFont(new Font("SansSerif", Font.PLAIN, 10));
		consoleScrollPane.setBorder(titledBorder);

		rightSplitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, false, mainPanel, consoleScrollPane);
		// why doesn't the splitter move to it's new location??
		// anyone got an idea?
		if (rightSplit < 0) {
			rightSplit = 550;
		}
		rightSplitPane.setDividerLocation(rightSplit);
		rightSplitPane.setOneTouchExpandable(true);

		mainSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, false, listTabs, rightSplitPane);

		if (mainSplit < 0) {
			mainSplit = 280;
		}
		mainSplitPane.setDividerLocation(mainSplit);

		int w = dimx - mainSplitPane.getDividerLocation() - 35;
		int h = rightSplitPane.getDividerLocation() - 38;
		mainViewPanel.setPreferredSize(new Dimension(w, h));

		JPanel panel = new JPanel(new BorderLayout());
		panel.add(toolbar, BorderLayout.PAGE_START);
		toolbar.setFloatable(false);
		panel.add(statusBar, BorderLayout.PAGE_END);
		statusBar.setFloatable(false);
		panel.add(mainSplitPane, BorderLayout.CENTER);

		// We need this when switching states as we have to add
		// the toolbar to the different panels. Look at setGUIState()
		// for understanding the next statement
		normalPanel = mainSplitPane;

		setContentPane(panel);

		// as all components are initialized, add the listeners that can
		// conflict with uninitialized components earlier
		progressLabel.setText("Initializing GUI: Initializing listeners");
		startProgress.setValue(startProgress.getValue() + 1);
		try {
			Thread.sleep(0, 1);
		} catch (InterruptedException ie) {}

		resultTree.setListeners();
		kernelTree.setListeners();

		listTabs.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				if (state != GUIState.KERNELS && state != GUIState.RESULTS) {
					forceListTabbView();
				}
			}
		});
		listTabs.addFocusListener(new FocusListener() {
			public void focusLost(FocusEvent e) {}

			public void focusGained(FocusEvent e) {
				if (initDone == false) {
					initDone = true;
					return;
				}
			}
		});
		listTabs.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				forceListTabbView();
			}
		});
		/* Deactivate the external console frame as it's now within the GUI. */
		console.setVisible(false);

		// this.setSession(null);
	}

	/**
	 * Adds JMenutItems from a list of a certain category to a JMenu. The first item will be a separator. There are only 3
	 * categories: 0 = setup, 1 = measure, 2 = evaluate. The array lists must be of size 3.
	 * 
	 * @param remoteLists The array of Lists.
	 * @param menu The JMenu to add the menu items to.
	 * @param category The category to choose the items from.
	 **/
	private void addListItems(Map<String, List<JMenuItem>> remoteLists, JMenu menu, String category) {
		addListItems(remoteLists, menu, category, true);
	}

	/**
	 * Adds JMenutItems from a list of a certain category to a JMenu. The first item will be a separator. There are only 3
	 * categories: 0 = setup, 1 = measure, 2 = evaluate. The array lists must be of size 3.
	 * 
	 * @param remoteLists The array of Lists.
	 * @param menu The JMenu to add the menu items to.
	 * @param category The category to choose the items from.
	 * @param separator If true, the first item is a separator.
	 **/
	private void addListItems(Map<String, List<JMenuItem>> remoteLists, JMenu menu, String category,
			boolean separator) {
		if (remoteLists.get(category).size() > 0) {
			if (separator) {
				menu.addSeparator();
			}
			for (int i = 0; i < remoteLists.get(category).size(); i++) {
				Object obj = remoteLists.get(category).get(i);
				if (obj instanceof JMenuItem) {
					menu.add((JMenuItem) obj);
				} else if ((obj instanceof String) && ((String) obj).compareTo("Separator") == 0) {
					menu.addSeparator();
				}
			}
		}
	}

	/**
	 * This method is called when the user chose "load" from the menu or used the key combination CTRL+L. Depending on the
	 * current view state of the GUI a proper reaction is choosen.
	 **/
	public void loadRequest() {
		if (getGUIState() == GUIState.LOCALDEFS) {
			editorFileRelation.clear();
			Object[] possibleChoices = getAvailableLOCALDEFS();
			Object selectedValue = JOptionPane.showInputDialog(this,
					"Choose the hostname of which you want to load the LOCALDEFS.",
					"Load present LOCALDEF files", JOptionPane.INFORMATION_MESSAGE, null, possibleChoices,
					possibleChoices[0]);
			if ((selectedValue != null) && !selectedValue.toString().equals("")) {
				bigInterface.setHost(selectedValue.toString());
				loadNewHost();
				updateTitle();
			}
		}
	}

	/**
	 * This method is called when the user chose "save" from the menu or used the key combination CTRL+S. Depending on the
	 * current view state of the GUI a proper reaction is choosen.
	 **/
	public void saveRequest() {
		if (getGUIState() == GUIState.LOCALDEFS) {
			save();
		} else if ((getGUIState() == GUIState.KERNELS) && (kernelScriptWindow != null)) {
			BIGGUI.kernelScriptWindow.saveRequest(false);
		} else if (getGUIState() == GUIState.ADMIN) {
			adminTool.save();
		}
	}

	/**
	 * This method is called when the user chose "save as" from the menu. Depending on the current view state of the GUI a
	 * proper reaction is choosen.
	 **/
	public void saveAsRequest() {
		if (getGUIState() == GUIState.LOCALDEFS) {
			String host = JOptionPane.showInputDialog("Type in the host you want to save as.\nThe files "
					+ "for this host will be created or replaced in the " + "LOCALDEFS directory.");
			if ((host != null) && !host.equals("")) {
				saveAs(host);
			}
		} else if ((getGUIState() == GUIState.RESULTS) && (plotWindow != null)) {
			plotWindow.SaveAs();
		}
	}

	/**
	 * Sends the result file to the benchit database of results.
	 * 
	 * @param resultFile the absolute path name of the result file.
	 **/
	public void loadUpResultFile(String resultFile) {
		//
		//
		// needs to be implemented
		// then uncomment code in result tree popop to activate popup entry
		//
		//
	}

	private BIGOutputFile actualFile = null;

	/** Creates a BIGQuickViewWindow and displays the result graph. */
	public void loadResult(final BIGOutputFile file, boolean save) {
		if ((actualFile != null) && (save)) {
			final BIGOutputFile tempFile = actualFile;
			(new Thread() {
				@Override
				public void run() {
					tempFile.save();
				}
			}).start();
		}
		setGUIState(GUIState.RESULTS);
		quickViewWindow = new BIGQuickViewWindow(BIGGUI.this, file);
		actualFile = file;
		reloadCards();
		// declare the changed panels to be valid again
		cardsButtonPanel.revalidate();
		viewPanel.revalidate();
		textPanel.revalidate();
		mainViewPanel.revalidate();
		repaint();
		viewPanel.revalidate();
		textPanel.revalidate();
		mainViewPanel.revalidate();
		repaint();
		// for results it should be possible to switch to the text view
		toggleViewItemAction.setEnabled(true);
	}

	public Object[] getAvailableLOCALDEFS() {
		Object[] retval = null;
		String dirName = bigInterface.getBenchItPath() + File.separator + "LOCALDEFS" + File.separator;
		File file = null;
		try {
			file = new File(dirName);
		} catch (NullPointerException npe) {
			return retval;
		}
		if (file != null) {
			if (file.exists() && file.isDirectory()) {
				File[] obj = file.listFiles(new FileFilter() {
					public boolean accept(File pathname) {
						boolean ret = false;
						if (pathname != null) {
							if (pathname.exists() && pathname.isFile()) {
								String name = pathname.getName();
								if (!name.endsWith("_input_architecture") && !name.endsWith("_input_display")) {
									ret = true;
								}
							}
						}
						return ret;
					}
				});
				if (obj != null) {
					Arrays.sort(obj);
					retval = new String[obj.length];
					for (int c = 0; c < obj.length; c++) {
						retval[c] = (obj[c].getName());
					}
				}
			}
		}
		return retval;
	}

	/**
	 * Returns the state the GUI is in at the moment of invokation.
	 * 
	 * @return BIGGUI.[ADMIN|DATABASE|LOCALDEFS|KERNELS|RESULTS]
	 **/
	public GUIState getGUIState() {
		return state;
	}

	public JPanel getViewPanel() {
		return viewPanel;
	}

	public JPanel getTextPanel() {
		return textPanel;
	}

	public void setOnline(boolean on) {
		setOnline(on, consoleScrollPane);
	}

	public void setOnline(boolean on, JComponent console) {
		this.setOnline(on);
		if (!on) {
			rightSplitPane.setBottomComponent(console);
		}
		validate();
	}

	public void updateTitle() {
		staticTitle = "BenchIT - GUI (@" + bigInterface.getHost() + ")";
		switch (getGUIState()) {
			case ADMIN :
				setTitle(staticTitle + ": AdminTool");
				break;
			case DATABASE :
				setTitle("BenchIT - Web Database");
				break;
			case LOCALDEFS :
				setTitle(staticTitle + ": LocalDef settings");
				break;
			case KERNELS :
				setTitle(staticTitle + ": Execute kernels");
				break;
			case RESULTS :
				setTitle(staticTitle + ": View results");
				break;
			default :
				setTitle(staticTitle + ": Loading...");
				break;
		}
	}

	/**
	 * This Method sets the state of the GUI. The only recognized states are:<br>
	 * ADMIN, DATABASE, LOCALDEFS, KERNELS, and RESULTS.<br>
	 * If another state is requested, this method won't have an effect.
	 * 
	 * @param inState either one of ADMIN, DATABASE, LOCALDEFS, KERNELS, and RESULTS.
	 */
	public void setGUIState(GUIState inState) {
		if (state == inState)
			return;
		rightSplit = rightSplitPane.getDividerLocation();
		JPanel panel = null;
		// force save entries if old state was LOCALDEFS
		if (state == GUIState.LOCALDEFS)
			saveEntries();
		// set the new one as the current one
		state = inState;
		updateTitle();
		// enable all state switches
		adminItemAction.setEnabled(true);
		databaseItemAction.setEnabled(true);
		localdefsItemAction.setEnabled(true);
		// Disable Buttons
		choiceList.setEnabled(false);
		executeSelectedItemAction.setEnabled(false);
		showResultItemAction.setEnabled(false);
		toggleViewItemAction.setEnabled(true);
		loadMenuItem.setEnabled(false);
		saveMenuItem.setEnabled(false);
		saveAsMenuItem.setEnabled(false);
		// preset the state of the BIGInterface to normal mode
		// bigInterface.setLoadAdminTool( false );
		// preset the view mode to graphical view
		setTextViewMode(false);
		switch (state) {
			case ADMIN :
				adminItemAction.setEnabled(false);
				toggleViewItemAction.setEnabled(false);
				saveMenuItem.setEnabled(true);
				saveMenuItem.setToolTipText("Save LOCALDEF variable configuration to config.xml");
				break;
			case DATABASE :
				// if we are connected
				// if (this.connectionFrame.getMenu().isEnabled()) { this.connectionFrame.setConsole(this.thisBIGGUI); }
				databaseItemAction.setEnabled(false);
				toggleViewItemAction.setEnabled(false);
				break;
			case LOCALDEFS :
				localdefsItemAction.setEnabled(false);
				viewLevelComboBox.setEnabled(!textViewMode);
				loadMenuItem.setEnabled(true);
				saveMenuItem.setEnabled(true);
				saveAsMenuItem.setEnabled(true);
				loadMenuItem.setToolTipText("Load the LOCALDEFS of a different host");
				saveMenuItem.setToolTipText("Save the changes on the LOCALDEFS");
				saveAsMenuItem.setToolTipText("Save a copy of the current LOCALDEFS for a new host");
				break;
			case KERNELS :
				setTextViewMode(true);
				choiceList.setEnabled(true);
				executeSelectedItemAction.setEnabled(true);
				showResultItemAction.setEnabled(true);
				toggleViewItemAction.setEnabled(false);
				saveMenuItem.setEnabled(true);
				saveMenuItem.setToolTipText("Save the changes of kernel files");
				// bring kernel tabb in front
				listTabs.setSelectedIndex(listTabs.indexOfTab(KERNELS_TAB));
				break;
			case RESULTS :
				saveAsMenuItem.setEnabled(true);
				saveAsMenuItem.setToolTipText("Save plot to a new .bit file");
				// bring result tabb in front
				listTabs.setSelectedIndex(listTabs.indexOfTab(RESULTS_TAB));
				break;
			default :
		}
		panel = new JPanel(new BorderLayout());
		panel.add(toolbar, BorderLayout.PAGE_START);
		panel.add(statusBar, BorderLayout.PAGE_END);
		if (state == GUIState.DATABASE) {
			panel.add(connectionFrame.getDisplayPanel(), BorderLayout.CENTER);
			setContentPane(panel);
		} else {
			// got to be set because of the databaseconnection
			// this.rightSplitPane.setBottomComponent(this.consoleScrollPane);
			rightSplitPane.setDividerLocation(rightSplit);
			panel.add(normalPanel, BorderLayout.CENTER);
			setContentPane(panel);
			if (rightSplitPane.getBottomComponent() instanceof conn.MainEditPanel) {
				((conn.MainEditPanel) rightSplitPane.getBottomComponent()).setSelectedIndex(1);
			}
		}
		validate();
		reloadCards();
	}

	private void forceListTabbView() {
		String selectedTabb = listTabs.getTitleAt(listTabs.getSelectedIndex());
		if (selectedTabb.equals(KERNELS_TAB)) {
			setGUIState(GUIState.KERNELS);
		} else if (selectedTabb.equals(RESULTS_TAB)) {
			setGUIState(GUIState.RESULTS);
		}
	}

	/**
	 * Load the corresponding LOCALDEF-file for the host defined by entry in the bigInterface. This methode create a new
	 * thread for the load operation, so it doesn't block actual execution.
	 */
	public void loadNewHost() {
		Thread loadThread = new Thread() {
			@Override
			public void run() {
				try {
					bigInterface.load();
					reloadCards();
					repaint();
					mainViewPanel.revalidate();
				} catch (BIGParserException bpe) {
					System.err.println("Cannot load file\n" + bpe);
				}
			}
		};
		loadThread.start();
	}

	/**
	 * Load the corresponding LOCALDEF-file for the host defined by entry in the bigInterface. This methode doesn't create
	 * a new thread for the load operation, so it blocks actual execution.
	 */
	public void loadNewHostBlocked() {
		try {
			bigInterface.load();
			reloadCards();
			repaint();
			mainViewPanel.revalidate();
		} catch (BIGParserException bpe) {
			System.err.println("Cannot load file\n" + bpe);
		}
	}

	/**
	 * Initializes the kernel and result tree filling both with appropriate information.
	 **/
	private void initListTabbs() {
		DefaultMutableTreeNode rootNode = new DefaultMutableTreeNode(kernelRootName);
		DefaultTreeModel model = new DefaultTreeModel(rootNode);
		kernelTree = new BIGKernelTree(model, this);
		kernelTree.setupTree();

		rootNode = new DefaultMutableTreeNode(resultRootName);
		model = new DefaultTreeModel(rootNode);
		resultTree = new BIGResultTree(model, this);
		resultTree.setupTree();
	}

	/**
	 * Initalizes the main view panel according to the current state the GUI is in. This method is called after a new
	 * state is set with setGUIState().
	 */
	private void initCards() {
		switch (state) {
			case LOCALDEFS :
				// add combo boxes
				GridBagConstraints gc = new GridBagConstraints();
				gc.fill = GridBagConstraints.BOTH;
				gc.weightx = 1.0;
				cardsButtonPanel.add(localdefFilesComboBox, gc);
				cardsButtonPanel.add(viewLevelComboBox, gc);

				// the CardLayout has to display LOCALDEF information
				String curFile = null;
				String selection = (String) (localdefFilesComboBox.getSelectedItem());
				Object[] allFiles = bigInterface.getAllFilenames().toArray();
				for (int i = 0; i < allFiles.length; i++) {
					if (selection.equals(localDefFilesComboBoxEntries[i])) {
						curFile = (String) allFiles[i];
						break;
					}
				}
				if (curFile == null) {
					console.postMessage("BIGGUI: No match found for LOCALDEF ComboBox selection.",
							BIGConsole.ERROR);
					return;
				}
				// must be optimized in one pattern
				String viewFile = curFile.replaceAll("<hostname>", bigInterface.getHost());
				viewFile = viewFile.replaceAll("#lt;hostname#gt;", bigInterface.getHost());
				// <--
				ArrayList<BIGEntry> entries = bigInterface.getAllEntries(curFile);
				GridBagConstraints c = new GridBagConstraints();
				c.fill = GridBagConstraints.BOTH;
				c.weightx = 1.0;
				c.gridx = 0;
				c.gridy = 0;
				JPanel panel = new JPanel();
				GridBagLayout gbLayout = new GridBagLayout();
				panel.setLayout(gbLayout);

				int pos = 0;
				c.gridx = pos;
				pos++;
				for (int j = 0; j < entries.size(); j++) {
					BIGEntry entry = entries.get(j);
					if (!entry.getActiveStatus())
						continue;
					addEntry(entry, pos, gbLayout, c, panel);
					pos++;
				}
				createGlue(pos, gbLayout, c, panel);
				JScrollPane jsp = new JScrollPane(panel);
				jsp.setWheelScrollingEnabled(true);
				jsp.getVerticalScrollBar().setUnitIncrement(15);
				// add scroll pane with graphical representation of the file content
				c.fill = GridBagConstraints.BOTH;
				c.weighty = 1.0;
				viewPanel.add(jsp, c);

				// Inserts Text Editors for the host files
				BIGEditor editor = null;
				for (int i = 0; i < allFiles.length; i++) {
					String tmpFile = (String) allFiles[i];
					// Check, if an editor for this file already exists.
					// If so, reuse it. If not, create a new one.
					BIGEditor tmpEditor = editorFileRelation.get(tmpFile);
					editorFileRelation.remove(tmpFile);
					if (tmpEditor == null) {
						tmpEditor = new BIGEditor(bigInterface.getDefFileText(tmpFile));
						editorFileRelation.put(tmpFile, tmpEditor);
					} else {
						BIGStrings contains = tmpEditor.getAllLines();
						tmpEditor = new BIGEditor(contains);
						editorFileRelation.put(tmpFile, tmpEditor);
					}
					tmpEditor.setTokenMarker(new org.syntax.jedit.tokenmarker.ShellScriptTokenMarker());
					// extract the Editor for the selected file
					if (tmpFile.compareTo(curFile) == 0) {
						editor = tmpEditor;
					}
				}
				// display the selected Editor's content
				if (editor == null) {
					console.postMessage("BIGGUI: Unable to set Editor " + "contents.", BIGConsole.ERROR);
					return;
				}
				// JScrollPane editorJSP = new JScrollPane( editor );
				// add scroll pane with text content of the file
				c.fill = GridBagConstraints.BOTH;
				c.weighty = 1.0;
				textPanel.add(editor, c);
				break;
			case ADMIN :// the CardLayout has to display admin information
				if (adminTool == null)
					return;
				GridBagConstraints cAdmin = new GridBagConstraints();
				cAdmin.fill = GridBagConstraints.BOTH;
				cAdmin.weightx = 1.0;
				cAdmin.weighty = 1.0;
				cAdmin.gridx = 0;
				cAdmin.gridy = 0;
				JComponent comp = adminTool.getDisplayPanel();
				viewPanel.add(comp, cAdmin);
				break;
			case KERNELS : // the CardLayout has to display kernel information
				if (kernelScriptWindow == null)
					return;
				GridBagConstraints cKernels = new GridBagConstraints();
				cKernels.fill = GridBagConstraints.BOTH;
				cKernels.weightx = 1.0;
				cKernels.weighty = 1.0;
				cKernels.gridx = 0;
				cKernels.gridy = 0;
				// Inserts the text panel for the script files
				JComponent[] comps = kernelScriptWindow.getDisplayPanels();
				GridBagConstraints cKernels2 = new GridBagConstraints();
				cKernels2.fill = GridBagConstraints.BOTH;
				cKernels2.weightx = 1.0;
				if (comps.length > 1) {
					for (int i = 1; i < comps.length; i++) {
						cardsButtonPanel.add(comps[i], cKernels2);
					}
				}
				textPanel.add(comps[0], cKernels);
				break;
			case RESULTS : // the CardLayout has to display result information
				// first: get the selected result file
				if (resultTree.isSelectionEmpty())
					return;
				TreePath selectedPath = resultTree.getSelectionPath();
				DefaultMutableTreeNode node = (DefaultMutableTreeNode) selectedPath.getLastPathComponent();
				if (!(node.getUserObject() instanceof BIGOutputFile)) {
					return;
				}
				// check, if a file is selected, but the quickViewWindow is
				// not loaded yet then do it now
				if (quickViewWindow == null) {
					quickViewWindow = new BIGQuickViewWindow(BIGGUI.this,
							(BIGOutputFile) node.getUserObject());
				}
				// second: add the display panel for the selected
				// result file to the viewPanel
				resultViewPanels = quickViewWindow.getDisplayPanels();
				if (resultViewPanels.length != 2) {
					console.postMessage("QuickViewWindow didn't return 2 panels!", BIGConsole.WARNING);
					return;
				}
				GridBagConstraints cResults = new GridBagConstraints();
				cResults.fill = GridBagConstraints.BOTH;
				cResults.anchor = GridBagConstraints.NORTHWEST;
				cResults.weightx = 1.0;
				cResults.weighty = 1.0;
				cResults.gridx = 0;
				cResults.gridy = 0;
				viewPanel.add(resultViewPanels[0], cResults);
				textPanel.add(resultViewPanels[1], cResults);
				// without the repaint java doesn't paint the panel properly
				// however, with the repaint, you can still see the wrong painting for some millis
				// any suggestions why? you're welcome to mail me: wloch@zhr.tu-dresden.de
				repaint();
			default :
				break;
		}
	}

	private void removeCards() {
		cardsButtonPanel.removeAll();
		viewPanel.removeAll();
		textPanel.removeAll();
	}

	private void reloadCards() {
		removeCards();
		initCards();
		// declare the changed panels to be valid again
		viewPanel.revalidate();
		textPanel.revalidate();
		mainViewPanel.revalidate();
		repaint();
	}

	private List<BIGRunEntry> getRunEntries(BIGStrings kernelNames) {
		List<BIGRunEntry> result = new ArrayList<BIGRunEntry>(kernelNames.size());
		BIGRunType type = BIGRunType.values()[choiceList.getSelectedIndex()];
		for (int i = 0; i < kernelNames.size(); i++) {
			result.add(new BIGRunEntry(kernelNames.get(i), type));
		}

		return result;
	}

	private boolean getShutdownForExec() throws Exception {
		boolean shutdown = bigInterface.getBIGConfigFileParser().boolCheckOut("shutDownForExec", true);
		boolean askForShutdown = bigInterface.getBIGConfigFileParser().boolCheckOut(
				"askForShutDownForExec", true);
		if (!askForShutdown)
			return shutdown;

		JComponent message[] = new JComponent[4];
		String[] options = {"Okay", "Cancel"};

		message[0] = new JLabel("");
		if (!bigInterface.getHost().equals(bigInterface.getOriginalHost())) {
			String msg = "!!! Please note, that not your selected hostname\"" + bigInterface.getHost()
					+ "\" is used, but the original hostname\"" + bigInterface.getOriginalHost() + "\"";
			console.postMessage(msg, BIGConsole.WARNING);
			message[0] = new JLabel(msg);
		}
		message[1] = new JLabel(
				"Do you want to shut down the GUI, so the measurement isn't influenced?");
		message[2] = new JCheckBox("shut down BIG");
		((JCheckBox) message[2]).setSelected(shutdown);
		message[3] = new JCheckBox("ask again");
		((JCheckBox) message[3]).setSelected(true);
		int result = JOptionPane.showOptionDialog(this, message, "Shut down BenchIT GUI?",
				JOptionPane.DEFAULT_OPTION, JOptionPane.INFORMATION_MESSAGE, null, options, options[0]);
		// Cancel was pressed
		if (result == 1)
			throw new Exception("Canceled");
		shutdown = ((JCheckBox) message[2]).isSelected();
		askForShutdown = ((JCheckBox) message[3]).isSelected();
		bigInterface.getBIGConfigFileParser().set("shutDownForExec", shutdown);
		bigInterface.getBIGConfigFileParser().set("askForShutDownForExec", askForShutdown);
		return shutdown;
	}

	/**
	 * This method finds out all selected nodes in the kernel tree and execute the selected action ("compile only",
	 * "run only" or "compile and run") for every selected kernel.
	 */
	public void executeSelected() {
		final BIGStrings kernelNames = kernelTree.getSelected();
		if (kernelNames.size() == 0) {
			console.postMessage("No kernel selected for execution!", BIGConsole.WARNING);
			return;
		}
		BIGGUI.kernelScriptWindow.saveRequest(true);
		final List<BIGRunEntry> runEntries = getRunEntries(kernelNames);

		final boolean shutdown;
		try {
			// Don't even ask for shutdown on compile
			shutdown = choiceList.getSelectedIndex() != 0 && getShutdownForExec();
		} catch (Exception e) {
			return;
		}

		final Thread saveThread = save();

		final BIGGUI bgui = this;

		if (shutdown) { // if we shut down big
			Thread execThread = new Thread() {
				@Override
				public void run() {
					try {
						saveThread.join();
					} catch (InterruptedException ie) {}
					BIGExecute exec = BIGExecute.getInstance();
					exec.generatePostProcScript(runEntries, "postproc.sh", kernelTree, true);
					bigInterface.setConfigFileContents(bgui);
					saveAndExit();
				}
			};
			execThread.start();
		} else { // GUI stays alive during measurement
			Thread execThread = new Thread() {
				@Override
				public void run() {
					try {
						saveThread.join();
					} catch (InterruptedException ie) {}
					BIGExecute exec = BIGExecute.getInstance();
					String executeFile = exec.generatePostProcScript(runEntries, null, kernelTree, false);
					if (executeFile == null) {
						System.err.println("Could not create execution shell-script");
						return;
					}
					bigInterface.setConfigFileContents(bgui);
					try {
						Process p = Runtime.getRuntime().exec("sh " + executeFile);
						bigInterface.getConsole().addStream(p.getErrorStream(), BIGConsole.ERROR);
						bigInterface.getConsole().addStream(p.getInputStream(), BIGConsole.DEBUG);
						executedProcess = p;
						p.waitFor();
						executedProcess = null;
					} catch (Exception e) {}
					// Reload this in GUI thread!
					SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							kernelTree.reloadKernelsExecutables();
							resultTree.updateResultTree(null);
							showResultForSelectedKernel();
						}
					});
				}
			};
			execThread.start();
		}
	}

	private List<BIGRunEntry> getSelectedKernels(String reason) {
		BIGStrings kernelNames = kernelTree.getSelected();
		if (kernelNames.size() == 0) {
			console.postMessage("No kernel selected for " + reason, BIGConsole.WARNING);
			return null;
		}
		BIGGUI.kernelScriptWindow.saveRequest(true);
		try {
			save().join();
		} catch (InterruptedException e) {
			return null;
		}
		return getRunEntries(kernelNames);
	}

	public void executeSelectedOnOtherSystem() {
		final List<BIGRunEntry> runEntries = getSelectedKernels("execution!");
		(new Thread() {
			@Override
			public void run() {
				remoteMenu.startWorkOnAnotherMachine(runEntries);
			}
		}).start();
	}

	public void copySelectedToOtherSystem() {
		final List<BIGRunEntry> runEntries = getSelectedKernels("copying!");
		(new Thread() {
			@Override
			public void run() {
				remoteMenu.copyFilesToAnotherMachine(runEntries);
			}
		}).start();
	}

	/**
	 * Checks all entries wether they should be displayed or not (visible state). Sets an entry visible if the priority
	 * level <= current view level, otherwise unvisible.
	 */
	private void repaintTabs() {
		Object[] allFiles = bigInterface.getAllFilenames().toArray();

		// this is the upper limit of the for-loop
		// mostly when you change your files, you have fullLoaded
		// (quickLoad disabled or loaded after starting the program)
		int forUpperLimit = allFiles.length;
		// but if theres just <hostname> loaded, we save just this
		/*
		 * if (!fullLoaded) { forUpperLimit=1; }
		 */

		for (int i = 0; i < forUpperLimit; i++) {
			ArrayList<BIGEntry> entries = bigInterface.getAllEntries(allFiles[i].toString());

			for (int j = 0; j < entries.size(); j++) {
				BIGEntry entry = entries.get(j);
				JComponent comp = (JComponent) entry.getComponent();
				JComponent label = (JComponent) entry.getLabel();
				// checks if entry is part of the current selected view level
				if (comp != null) {
					boolean visible = entry.getDetailLevel().ordinal() <= detailLevel.ordinal();
					comp.setVisible(visible);
					label.setVisible(visible);
				}
				if (debug > 0) {
					System.out.println("BIGGUI: repaintTabs: " + entry.getName() + " visible (or null?): "
							+ (comp != null ? "" + comp.isVisible() : "null"));
				}
			}
		}
	}

	/**
	 * Saves the edit values from the components to the values in the BIGEntries
	 */
	@SuppressWarnings("unchecked")
	private void saveEntries() {
		Object[] allFiles = bigInterface.getAllFilenames().toArray();
		// this is the upper limit of the for-loop
		// mostly when you change your files, you have fullLoaded
		// (quickLoad disabled or loaded after starting the program)
		int forUpperLimit = allFiles.length;
		// but if theres just <hostname> loaded, we save just this
		/*
		 * if (!fullLoaded) { forUpperLimit=1; }
		 */
		statusLabel.setText("Saving entries...");
		for (int i = 0; i < forUpperLimit; i++) {
			ArrayList<BIGEntry> entries = bigInterface.getAllEntries(allFiles[i].toString());

			for (int j = 0; j < entries.size(); j++) {
				BIGEntry entry = entries.get(j);
				JComponent comp = (JComponent) entry.getComponent();
				if (comp != null) {
					switch (entry.getType()) {
						case None :
							break;
						case String :
							entry.setValue(((BIGTextField) comp).getText());
							break;
						case Integer :
							entry.setValue(((BIGSpinner) comp).getTextField().getIntegerValue());
							break;
						case Float :
							entry.setValue(((BIGSpinner) comp).getTextField().getValue());
							break;
						case Multiple :
							entry.setValue(((JComboBox<String>) comp).getSelectedItem().toString());
							break;
						case List :
							entry.setValue(new BIGStrings(((JList<String>) comp).getSelectedValuesList()));
							break;
						case Boolean :
							entry.setValue(new Boolean(((JCheckBox) comp).isSelected()));
							break;
						case FreeText :
						case RestrictedText :
							entry.setValue(((JTextArea) comp).getText());
							break;
						default :
							entry.setValue(null);
							break;
					}
				}
			}
		}
		statusLabel.setText("done");
	}

	/*
	 * Adds the entry to the panel with specified constraints.
	 */
	private void addEntry(BIGEntry entry, int pos, GridBagLayout gbLay, GridBagConstraints gbConst,
			JPanel panel) {
		BIGType type = entry.getType();
		String viewname = entry.getViewName().equals("") ? entry.getName() : entry.getViewName();
		Object value = entry.getValue();
		// wheter to use the default value or not
		if (value == null && type != BIGType.None) {
			// INTEGER null means not set
			if (entry.getType() != BIGType.Integer) {
				value = entry.getDefaultValue();
				viewname += " (!)";
				entry.setToolTipText(entry.getToolTipText() + " DEFAULT");
			}
		}

		gbConst.gridx = 0;
		gbConst.gridy = pos;
		gbConst.gridheight = 1;
		gbConst.gridwidth = 1;
		gbConst.fill = GridBagConstraints.NONE;
		gbConst.anchor = GridBagConstraints.NORTHWEST;
		gbConst.insets = new Insets(1, 3, 1, 3);
		// creates and add a new Label for the entry
		JLabel entryLabel = new JLabel(viewname);
		entryLabel.setFont(labelFont);/*
																	 * if (entry.getNecessity()){ entryLabel.setForeground(Color.RED); }
																	 */
		entryLabel.setToolTipText(entry.getToolTipText());
		gbLay.setConstraints(entryLabel, gbConst);

		gbConst.gridx = 1;
		// resize the entry component if it hasn't a BIGSpinner
		if (type != BIGType.Integer && type != BIGType.Float) {
			gbConst.weightx = 90;
			gbConst.fill = GridBagConstraints.HORIZONTAL;
		}
		gbConst.anchor = GridBagConstraints.NORTHWEST;

		// create and add the component for the entry
		JComponent comp = null;
		switch (type) {
			case String :
				comp = new BIGTextField(value, 20, BIGTextField.TEXT, entry.getNecessity());
				break;
			case Integer :
				BIGTextField field = new BIGTextField(value, 5, BIGTextField.INTEGER, entry.getNecessity());

				comp = new BIGSpinner(field, 1f);
				Dimension dim = field.getPreferredSize();
				((BIGSpinner) comp).setPreferredSize(new Dimension(dim.width + 15, dim.height));
				break;
			case Float :
				field = new BIGTextField(value, 5, BIGTextField.FLOAT, entry.getNecessity());
				comp = new BIGSpinner(field, 1f);
				field.setDecimalCount(3);
				dim = field.getPreferredSize();
				((BIGSpinner) comp).setPreferredSize(new Dimension(dim.width + 15, dim.height));
				break;
			case Multiple :
				JComboBox<String> comp2 = new JComboBox<String>(entry.getMultipleChoice().toArray());
				comp2.setSelectedItem(value);
				comp = comp2;
				break;
			case List :
				final BIGStrings values = entry.getMultipleChoice();
				final JList<String> list = new JList<String>(values.toArray());
				final int multiple = entry.getMultipleChoiceCount();

				// viewname must final for out printing
				final String thisname = viewname;
				// System.out.println(multiple);
				list.setSelectionMode(multiple > 1
						? ListSelectionModel.MULTIPLE_INTERVAL_SELECTION
						: ListSelectionModel.SINGLE_SELECTION);
				list.addListSelectionListener(new ListSelectionListener() {
					public void valueChanged(ListSelectionEvent e) {
						if (list.getSelectedIndices().length > multiple) {
							System.err.println("You have more than \"" + multiple + "\" values selected in \""
									+ thisname + "\".");
						}
					}
				});
				final BIGStrings defaultValue = (BIGStrings) value;
				for (int i = 0; i < defaultValue.size(); i++) {
					list.addSelectionInterval(values.find(defaultValue.get(i)),
							values.find(defaultValue.get(i)));
				}
				list.setVisibleRowCount(4);
				JScrollPane js = new JScrollPane(list);
				comp = js;
				break;
			case Boolean :
				comp = new JCheckBox();
				((JCheckBox) comp).setSelected(((Boolean) value).booleanValue());
				break;
			case None :
				comp = new JLabel();
				break;
			case FreeText :
				if (value.toString().endsWith("\n")) {
					comp = new JTextArea(value.toString());
				} else {
					comp = new JTextArea(value.toString() + "\n");
				}
				((JTextArea) comp).setBorder(BorderFactory.createLineBorder(Color.BLACK));
				break;
			case RestrictedText :
				comp = new BIGTextField(value, 20, BIGTextField.RESTRICTEDTEXT, entry.getNecessity());
				break;
			default :
				comp = null;
				break;
		}

		if (comp != null) {
			comp.setToolTipText(entry.getToolTipText());
			gbLay.setConstraints(comp, gbConst);
			// adding label
			panel.add(entryLabel);
			// adding compontent
			panel.add(comp);
			entry.setComponent(comp);
			entry.setLabel(entryLabel);
			// checks if entry is part of the current selected view level
			if (entry.getDetailLevel().ordinal() > detailLevel.ordinal()) {
				comp.setVisible(false);
				entryLabel.setVisible(false);
			}
		} else {
			System.err.println("BIGGUI: No type is matching for this entry: " + viewname + ".");
		}

		if (debug > 0) {
			System.out.println("BIGGUI: Entry \"" + entry.getName() + "\" added to pos: "
					+ entry.getPos());
		}
	}

	/**
	 * Adds a glue to the panel, so that the elements are always on top.
	 */
	private void createGlue(int pos, GridBagLayout gbLay, GridBagConstraints gbConst, JPanel panel) {
		gbConst.gridx = 0;
		gbConst.gridy = pos;
		gbConst.gridheight = 1;
		// align over to cells
		gbConst.gridwidth = 2;
		// resize to max
		gbConst.weightx = 100;
		gbConst.weighty = 100;
		// in both directions
		gbConst.fill = GridBagConstraints.BOTH;
		// always add on NORTH
		gbConst.anchor = GridBagConstraints.NORTH;
		gbConst.insets = new Insets(0, 0, 0, 0);
		// create and add empty panel
		JPanel glue = new JPanel();
		gbLay.setConstraints(glue, gbConst);
		panel.add(glue);
	}

	/**
	 * Sets the value of the component by the given type of entry
	 * 
	 * @param entry the entry
	 * @param comp the Component which were set to the value of the entry
	 */
	/*
	 * private void updateValue(BIGEntry entry, Component comp) { Object value = entry.getValue() == null ?
	 * entry.getDefaultValue() : entry.getValue(); final String name = entry.getName(); if (comp == null) return; try {
	 * switch (entry.getType()) { case STRING : ((BIGTextField) comp).setValue(value.toString()); break; case
	 * BIGInterface.INTEGER : case FLOAT : ((BIGSpinner) comp).setValue(value.toString()); break; case
	 * BIGInterface.MULTIPLE : ((JComboBox) comp).setSelectedItem(value); break; case LIST : final BIGStrings values =
	 * entry.getMultipleChoice(); final JList list = new JList(values.toArray()); final int multiple =
	 * entry.getMultipleChoiceCount(); System.out.println(multiple); list.setSelectionMode( multiple > 1 ?
	 * ListSelectionModel.MULTIPLE_INTERVAL_SELECTION : ListSelectionModel.SINGLE_SELECTION);
	 * list.addListSelectionListener(new ListSelectionListener() { public void valueChanged(ListSelectionEvent e) { if
	 * (list.getSelectedIndices().length > multiple) System.err.println( "You have more than \"" + multiple +
	 * "\" values selected in \"" + name + "\"."); } }); final BIGStrings defaultValue = (BIGStrings) value; for (int i =
	 * 0; i < defaultValue.size(); i++) list.addSelectionInterval( values.find(defaultValue.get(i)),
	 * values.find(defaultValue.get(i))); list.setVisibleRowCount(4); JScrollPane js = new JScrollPane(list); comp = js;
	 * break; case BOOLEAN : ((JCheckBox) comp).setSelected( ((Boolean) value).booleanValue()); break; /* // not
	 * implemented yet, because of less information about this both case DATETIME: break; case BIGInterface.VECTOR :
	 * break;
	 */
	/*
	 * } // comp.repaint(); } catch (ClassCastException cce) { console.postMessage( "unable to cast " + name + " with " +
	 * entry.getType(), BIGConsole.ERROR); } }
	 */

	/**
	 * Saves all files to db
	 * 
	 * @returns the Thread that carries out the action so you can wait for it
	 */
	private Thread save() {
		return save(!textViewMode);
	}

	private Thread save(boolean fromGraphicalView) {
		return save(bigInterface.getHost(), fromGraphicalView);
	}

	private Thread save(String host, boolean fromGraphicalView) {
		final String fHost = host;
		final boolean fFromGraphicalView = fromGraphicalView;
		Thread thread = new Thread() {
			@Override
			public void run() {
				if (fHost == null || fHost.equals(""))
					return;
				BIGDefFileParser defParser = new BIGDefFileParser();
				if (statusProgress != null) {
					defParser.getObservable().addObserver(statusProgress);
				}
				// CardLayout shows VIEWPANEL, save all entries in db first
				if (fFromGraphicalView) {
					saveEntries();
				}

				Object[] allFiles = bigInterface.getAllFilenames().toArray();
				// save every rendered file
				// this is the upper limit of the for-loop
				// mostly when you change your files, you have fullLoaded
				// (quickLoad disabled or loaded after starting the program)
				int forUpperLimit = allFiles.length;
				// but if there's just <hostname> loaded, we save just this
				/*
				 * if ( !fullLoaded ) { forUpperLimit = 1; }
				 */
				try {
					for (int i = 0; i < forUpperLimit; i++) {
						String curFile = allFiles[i].toString();
						if (statusLabel != null) {
							statusLabel.setText("Saving " + curFile.replace("<hostname>", fHost));
						}
						BIGEditor edit = editorFileRelation.get(curFile);
						if (edit != null) {
							// if clickIt is shown, render the file first
							if (fFromGraphicalView) {
								ArrayList<BIGEntry> entries = bigInterface.getAllEntries(curFile);

								BIGStrings tmp = new BIGStrings(entries.size());

								for (int c = 0; c < entries.size(); c++) {
									tmp.add(entries.get(c).getName());
								}

								// render the text
								try {
									BIGStrings newText = defParser.render(edit.getAllLines(), curFile, tmp);
									edit.setText(newText.toString());
								} catch (Exception e) {}
							}
							bigInterface.setDefFileText(edit.getAllLines(), curFile);
						}
						if (statusLabel != null)
							statusLabel.setText("done");
					}
					// sets the new host
					if (bigInterface.getHost() != fHost) {
						bigInterface.setHost(fHost);
						// init the new host and fill the tabs
						bigInterface.load();
						reloadCards();
					}
				} catch (Exception e) {
					e.printStackTrace();
					console.postMessage("Error during saving as \"" + fHost + "\" in the files. " + e,
							BIGConsole.ERROR, true);
				}
			} // end of run()
		};
		thread.start();
		return thread;
	}

	/**
	 * Saves the edited parameter in the files specified by the given host.
	 * 
	 * @param host the new host
	 */
	private Thread saveAs(String host) {
		return save(host, !textViewMode);
	}

	public void setKernelScriptWindow(BIGKernelScriptWindow ksw, boolean block) {
		final BIGKernelScriptWindow fksw = ksw;
		Thread t = new Thread() {
			@Override
			public void run() {
				kernelScriptWindow = fksw;
				reloadCards();
			}
		};
		if (block) {
			t.start();
			try {
				t.join();
			} catch (InterruptedException e) {}
		} else
			SwingUtilities.invokeLater(t);
	}

	public BIGKernelScriptWindow getKernelScriptWindow() {
		return kernelScriptWindow;
	}

	public void saveAndExit() {
		boolean save = bigInterface.getBIGConfigFileParser().boolCheckOut("loadAndSaveSettings", false);
		if ((actualFile != null) && save) {
			final BIGOutputFile tempFile = actualFile;
			(new Thread() {
				@Override
				public void run() {
					tempFile.save();
				}
			}).start();
		}
		if (resultMixer != null) {
			for (int i = 0; i < resultMixer.length; i++) {
				resultMixer[i].save();
			}
		}
		// first we set this windows changes to the configfileParser
		bigInterface.setConfigFileContents(this);
		// then we clean the memory
		System.gc();
		// and then we exit
		System.exit(0);
	}

	public void setValues(int inx, int iny, int inxSize, int inySize, int rs, int ms,
			DetailLevel detailLevel) {
		rightSplit = rs;
		rightSplitPane.setDividerLocation(rightSplit);
		mainSplit = ms;
		mainSplitPane.setDividerLocation(mainSplit);
		setBounds(inx, iny, inxSize, inySize);
		if (detailLevel != this.detailLevel) {
			this.detailLevel = detailLevel;
			// redraws the list of displayed items
			repaintTabs();
			// declares the components to be valid again
			viewPanel.revalidate();
			textPanel.revalidate();
			mainViewPanel.revalidate();
			viewLevelComboBox.setSelectedIndex(detailLevel.ordinal());
		}
	}

	public int getRightSplit() {
		return rightSplitPane.getDividerLocation();
	}

	public int getMainSplit() {
		return mainSplitPane.getDividerLocation();
	}

	public DetailLevel getDetailLevel() {
		return detailLevel;
	}

	public JLabel getStatusLabel() {
		return statusLabel;
	}

	public BIGResultMixer[] getResultMixer() {
		return resultMixer;
	}

	public BIGGUIObserverProgress getStatusProgress() {
		return statusProgress;
	}

	public BIGResultTree getResultTree() {
		return resultTree;
	}

	public BIGKernelTree getKernelTree() {
		return kernelTree;
	}

	public JPanel getConsoleScrollPane() {
		return consoleScrollPane;
	}

	public JSplitPane getRightSplitPane() {
		return rightSplitPane;
	}

	public void setTextViewMode(boolean textViewMode) {
		if (this.textViewMode == textViewMode)
			return;
		this.textViewMode = textViewMode;
		CardLayout cl = (CardLayout) (mainViewPanel.getLayout());
		if (textViewMode)
			cl.show(mainViewPanel, TEXTPANEL);
		else
			cl.show(mainViewPanel, VIEWPANEL);
	}

	/**
	 * Sets the result view to the result mixer.
	 **/
	public void setResultMixer(int whichResultMixer) {
		this.setResultMixer(resultMixer[whichResultMixer]);
	}

	/**
	 * Sets the result view to the result mixer.
	 **/
	public void setResultMixer(BIGResultMixer brm) {
		viewPanel.removeAll();
		viewPanel.revalidate();
		viewPanel.setLayout(new GridLayout(1, 1));
		viewPanel.add(brm.getPlot());
		viewPanel.revalidate();
		viewPanel.repaint();
		// switch to the graphic view
		setTextViewMode(false);
		// for ResultMixers it shouldn't be possible to switch to the text view
		toggleViewItemAction.setEnabled(false);
	}

	public void print() {
		switch (getGUIState()) {
			case LOCALDEFS :
				System.out.println("Not yet implemented for LOCALDEFS");
				break;
			case KERNELS :
				BIGGUI.kernelScriptWindow.getActualEditor().print();
				break;
			case RESULTS :
				if (textViewMode) {
					for (int i = 0; i < textPanel.getComponentCount(); i++) {
						if (textPanel.getComponent(i) instanceof BIGEditor) {
							((BIGEditor) textPanel.getComponent(i)).print();
						}
					}
				} else {
					PrinterJob printJob = PrinterJob.getPrinterJob();
					printJob.setPrintable(plotWindow.getChartPanel());
					if (printJob.printDialog()) {
						try {
							printJob.print();
						} catch (PrinterException ex2) {
							ex2.printStackTrace();
						}
					}
				}
				break;
			default :
				break;
		}
	}

	public void showResultForSelectedKernel() {
		TreePath path = kernelTree.getSelectionPath();
		if (path == null)
			return;
		Object obj = ((DefaultMutableTreeNode) path.getLastPathComponent()).getUserObject();
		if (!(obj instanceof BIGKernel))
			return;
		resultTree.selectKernelResultFile((BIGKernel) obj);
		/*
		 * BIGOutputFile file = resultTree.getKernelResultFile((BIGKernel) obj); String selected = ((BIGKernel)
		 * obj).getNameAfterSorting(resultTree.getSorting()); quickViewWindow = new BIGQuickViewWindow(BIGGUI.this, null);
		 * if (statusLabel != null) { statusLabel.setText("Repainting plot panel..."); } reloadCards(); if (statusLabel !=
		 * null) { statusLabel.setText("done"); }
		 */
	}
	abstract class ItemAction extends AbstractAction {
		private static final long serialVersionUID = 1L;

		public ItemAction(String text, String desc, Integer mnemonic) {
			super(text);
			putValue(SHORT_DESCRIPTION, desc);
			putValue(MNEMONIC_KEY, mnemonic);
		}

		public ItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super();
			putValue(SHORT_DESCRIPTION, desc);
			putValue(ACCELERATOR_KEY, accelerator);
			setEnabled(enabled);
		}

		public ItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, icon);
			putValue(SHORT_DESCRIPTION, desc);
			putValue(ACCELERATOR_KEY, accelerator);
			putValue(SMALL_ICON, icon);
			setEnabled(enabled);
		}
	}

	class AdminItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public AdminItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public AdminItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public AdminItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			setGUIState(GUIState.ADMIN);
		}
	}

	class DatabaseItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public DatabaseItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public DatabaseItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public DatabaseItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			setGUIState(GUIState.DATABASE);
		}
	}

	class LocaldefsItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public LocaldefsItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public LocaldefsItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public LocaldefsItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			setGUIState(GUIState.LOCALDEFS);
		}
	}

	class KernelsItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public KernelsItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public KernelsItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public KernelsItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			setGUIState(GUIState.KERNELS);
		}
	}

	class ResultsItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public ResultsItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public ResultsItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public ResultsItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			setGUIState(GUIState.RESULTS);
		}
	}

	class ToggleViewItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public ToggleViewItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public ToggleViewItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public ToggleViewItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			if (getGUIState() == GUIState.LOCALDEFS) {
				if (textViewMode) {
					// the gui switches to graphical view of the LOCALDEFS
					loadNewHostBlocked();
				}
				viewLevelComboBox.setEnabled(textViewMode);
				cardsButtonPanel.revalidate();
			}
			setTextViewMode(!textViewMode);
		}
	}

	class ExecuteSelectedItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public ExecuteSelectedItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public ExecuteSelectedItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public ExecuteSelectedItemAction(String text, String desc, ImageIcon icon,
				KeyStroke accelerator, boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			executeSelected();
		}
	}

	class ShowResultItemAction extends ItemAction {
		private static final long serialVersionUID = 1L;

		public ShowResultItemAction(String text, String desc, Integer mnemonic) {
			super(desc, desc, mnemonic);
		}

		public ShowResultItemAction(String desc, KeyStroke accelerator, boolean enabled) {
			super(desc, accelerator, enabled);
		}

		public ShowResultItemAction(String text, String desc, ImageIcon icon, KeyStroke accelerator,
				boolean enabled) {
			super(text, desc, icon, accelerator, enabled);
		}

		public void actionPerformed(ActionEvent e) {
			Thread thread = new Thread() {
				@Override
				public void run() {
					showResultForSelectedKernel();
				}
			};
			thread.start();
		}
	}
}

/******************************************************************************
 * Log-History $Log: BIGGUI.java,v $ Revision 1.47 2009/01/07 12:06:48 tschuet just reviewed Revision 1.46 2008/06/02
 * 10:14:20 tschuet different bug fixes (synch of localdef views, format check of BIGTextFields...) Revision 1.45
 * 2007/10/22 08:54:36 tschuet fixed bug of disappearring nodes in the result-tree after measurements (in case of
 * ceating a new main node like "numerical") Revision 1.44 2007/09/14 08:51:52 tschuet update use standardized
 * update-url (BIGUpdate.DEFAULTUPDATEURL) attempt to solve the memory consumption problem of plots Revision 1.43
 * 2007/07/10 10:41:30 tschuet actualize the gui-icon Revision 1.42 2007/07/03 11:28:26 tschuet insert a new benchit
 * icon into the frames Revision 1.41 2007/04/17 10:17:52 tschuet there was a spelling mistake: string 'preferrences' in
 * the setup menu has been corrected to 'preferences' Revision 1.40 2007/03/27 18:45:03 tschuet deactivate the
 * "toggle dispay" button while using a result mixer (it is no longer possible the switch to text view in a result
 * mixer) Revision 1.39 2007/02/27 12:37:49 tschuet different bugfixes (mixer, remote folder -> see meeting at february,
 * 27th) Revision 1.38 2007/01/15 06:47:20 rschoene empty Integer variables for LOCALDEFS are possible, compile/run
 * selection changed Revision 1.37 2006/12/18 08:09:34 rschoene new FontDialog Revision 1.36 2006/12/14 08:02:32
 * rschoene added BIGInterface.FREE_TEXT Revision 1.35 2006/12/06 06:54:19 rschoene added output for init-progressbar
 * "Looking for update" Revision 1.34 2006/11/06 13:18:44 rschoene added Screenshot Revision 1.33 2006/10/04 11:46:03
 * rschoene changed setOnline Revision 1.32 2006/08/16 13:21:38 rschoene global Insets Revision 1.31 2006/07/26 13:02:21
 * rschoene saves LOCALDEFS when running a kernel remote Revision 1.30 2006/07/05 09:28:22 rschoene debugged reset all
 * plot-information Revision 1.29 2006/07/03 14:19:48 rschoene new update-function Revision 1.28 2006/06/28 10:37:53
 * rschoene some changes for online-mode Revision 1.27 2006/06/26 10:47:34 rschoene integrated online- in result-mode
 * Revision 1.26 2006/06/25 12:38:54 rschoene added update support Revision 1.25 2006/06/23 10:06:43 rschoene set
 * resulttree for mixers Revision 1.24 2006/06/02 16:10:02 rschoene I hope it runs better now Revision 1.23 2006/05/27
 * 10:00:27 rschoene changed behaviour of Remote-measurement, removed bugs and debug printing Revision 1.22 2006/05/26
 * 12:06:18 rschoene removed a bug (font) Revision 1.21 2006/05/12 09:59:25 rschoene set title and legend fonts Revision
 * 1.20 2006/05/12 08:47:19 rschoene comment in plot Revision 1.19 2006/05/11 18:26:58 rschoene font settings Revision
 * 1.18 2006/05/11 10:30:45 rschoene changes for reset plot-info, save height/width for save/change default colors
 * Revision 1.17 2006/04/04 10:10:55 rschoene remoteexec now as thred Revision 1.16 2006/03/09 14:04:24 rschoene added
 * printing Revision 1.15 2006/03/07 12:49:20 rschoene handle ask for shutdown in BGUI.cfg Revision 1.14 2006/02/07
 * 16:36:55 rschoene all new plotting Revision 1.13 2006/01/31 21:12:20 rschoene prepared reset all plots, removed an
 * ineccessary call Revision 1.12 2006/01/10 10:52:37 rschoene moved a debug-message Revision 1.11 2006/01/10 10:43:30
 * rschoene plotting colors where �hm ... you now, but fixed, also removed debugMessage Revision 1.10 2006/01/09
 * 16:33:52 rschoene most functionality of BIGQuickViewWindow moved to BIGPlot Revision 1.9 2005/12/15 11:11:32 rschoene
 * now saves changes before run Revision 1.8 2005/12/14 16:00:50 rschoene added BIGEnvironmentEditor Revision 1.7
 * 2005/12/07 10:25:24 rschoene fun with parameter -host (hostname can be null and thats really s**t) Revision 1.6
 * 2005/11/23 14:59:56 rschoene F6 didnt work Revision 1.5 2005/11/17 09:38:43 rschoene rsome session-experiences
 * Revision 1.4 2005/11/11 13:44:20 rschoene now handles (compile/run) in a list Revision 1.3 2005/11/02 14:44:27
 * rschoene removed debug output Revision 1.2 2005/11/02 14:15:14 rschoene removed bugs and added multiple mixer support
 * Revision 1.1 2005/10/20 13:13:56 rschoene src3 add Revision 1.38 2005/06/16 10:44:01 rschoene execute only shown on
 * unix-systems Revision 1.37 2005/06/16 09:24:51 rschoene changed creation of conn.MyJFrame Revision 1.35 2005/05/26
 * 09:04:08 rschoene dividerLocation bug again >:( Revision 1.34 2005/05/26 08:42:03 rschoene save commit for editors
 * Revision 1.31 2005/05/11 10:01:54 rschoene deleted old structures (less memory using...) Revision 1.30 2005/05/11
 * 09:32:50 rschoene removed jumping-right-divider-bug 2 (ADMIN-Mode) Revision 1.29 2005/05/11 09:27:45 rschoene right
 * divider doesnt jump when clicking first on results Revision 1.28 2005/04/26 12:13:28 rschoene removed bug that didnt
 * load the editors, when loading a new host Revision 1.27 2005/04/22 12:41:09 rschoene some changes, like less
 * mem-using for changed files and removing of not needed stuff Revision 1.26 2005/04/14 16:11:57 wloch implemented
 * sorting of the kernel tree Revision 1.25 2005/04/04 10:34:28 rschoene removed GC-Problem with BIGPlots on more then 2
 * yaxis Revision 1.24 2005/03/22 14:16:23 rschoene removed debug message Revision 1.23 2005/03/15 10:41:42 wloch added
 * detail level restroration after restart Revision 1.22 2005/02/23 13:14:22 wloch assigned new accelerator keys to the
 * menu Revision 1.21 2005/02/22 17:01:53 wloch added subnode to BRM and instant loading Revision 1.20 2005/02/22
 * 11:41:00 wloch corrected remote execute menu entry in menu and popup Revision 1.19 2005/02/21 19:13:16 wloch major
 * changes to menu structure and popup menus Revision 1.18 2005/02/18 09:45:19 wloch implemented a better popup menu API
 ******************************************************************************/
