/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGEditor.java Author: SWTP
 * Nagel 1 Last change by: $Author: domke $ $Revision: 1.18 $ $Date: 2008/05/29 12:08:07 $
 ******************************************************************************/
// now bases on
/*
 * JEditTextArea.java - jEdit's text component Copyright (C) 1999 Slava Pestov You may use and
 * modify this package for any purpose. Redistribution is permitted, in both source and binary form,
 * provided that this notice remains intact in all source distributions of this package.
 */

package gui;

import java.awt.*;
import java.awt.datatransfer.*;
import java.awt.event.*;
import java.awt.geom.AffineTransform;
import java.awt.print.*;
import java.io.File;
import java.util.*;

import javax.swing.*;
import javax.swing.Timer;
import javax.swing.event.*;
import javax.swing.text.*;
import javax.swing.undo.*;

import org.syntax.jedit.*;
import org.syntax.jedit.tokenmarker.*;

import system.BIGStrings;

/**
 * The BIGEditor provides you, to edit the local def files manually. It is synchronized with the BIGGUI, so changing a
 * BIGEntry in the BIGEditor, the BIGGUI were also changed.<br>
 * The syntax highlighting is also possible to get, what kind of type an entry is.<br>
 * <br>
 */
public class BIGEditor extends JComponent implements Printable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Adding components with this name to the text area will place them left of the horizontal scroll bar. In jEdit, the
	 * status bar is added this way.
	 */
	public static String LEFT_OF_SCROLLBAR = "los";

	private File editingFile = null;

	// creates a new ArrayList that contains all marked lines, initial capacity of 10
	private final ArrayList<Integer> markedLines = new ArrayList<Integer>();

	public BIGEditor(File f) {
		this();
		String text = system.BIGFileHelper.getFileContent(f);
		text = text.replaceAll("\r\n", "\n");
		system.BIGFileHelper.saveToFile(text, f);
		editingFile = f;
		setDocument(new SyntaxDocument());
		String extension = f.getName()
				.substring(f.getName().lastIndexOf(".") + 1, f.getName().length());
		if (extension.equals("c") || extension.equals("h")) {
			setTokenMarker(new CTokenMarker());
		} else if (extension.equals("java")) {
			setTokenMarker(new JavaTokenMarker());
		} else if (extension.equals("cpp")) {
			setTokenMarker(new CCTokenMarker());
		} else if (extension.equals("cu")) {
			setTokenMarker(new CudaTokenMarker());
		} else if (extension.equals("sh") || extension.equals("SH") || (f.getName().indexOf(".") == -1)) {
			setTokenMarker(new ShellScriptTokenMarker());
		} else if (extension.equals("php") || extension.equals("php3") || extension.equals("phps")) {
			setTokenMarker(new PHPTokenMarker());
		} else if (extension.equals("perl") || extension.equals("pl") || extension.equals("pm")) {
			setTokenMarker(new PerlTokenMarker());
		} else if (extension.equals("py")) {
			setTokenMarker(new PythonTokenMarker());
		} else if (extension.toLowerCase().equals("f") || extension.toLowerCase().equals("f77")
				|| extension.toLowerCase().equals("f90") || extension.toLowerCase().equals("f95")
				|| extension.toLowerCase().equals("f03")) {
			setTokenMarker(new FortranTokenMarker());
		}

		/*
		 * else if (extension.equals("f")||extension.equals("f77")||extension.equals(".f90")) { this.setTokenMarker(new
		 * FortranTokenMarker()); }
		 */
		// here you could add with little changes the FortranTokenMarker from
		// http://www.devdaily.com/java/jwarehouse/jext-src-5.0.shtml
		// also some other languages are available under GPL
		setText(system.BIGFileHelper.getFileContent(f));
		document.tokenizeLines();
		// this.initText=this.getText();
		textChanged = false;
		initStuff();
	}

	/**
	 * Creates a new BIGEditor with the default settings.
	 */
	private BIGEditor() {
		this(TextAreaDefaults.getDefaults());
	}

	/**
	 * Creates a new BIGEditor with the specified settings.
	 * 
	 * @param defaults The default settings
	 */
	public BIGEditor(TextAreaDefaults defaults) {
		// Enable the necessary events
		enableEvents(AWTEvent.KEY_EVENT_MASK);

		// Initialize some misc. stuff
		// show row line
		// paint a line which tells the user when 80 characters are over
		painter = new TextAreaPainter(this, defaults) {
			private static final long serialVersionUID = 1L;

			@Override
			public void paint(Graphics gfx) {
				Color oldColor;
				Graphics2D g2d = (Graphics2D) gfx;
				AffineTransform at = new AffineTransform();
				at.setToTranslation(this.getFontMetrics().charWidth('W') * 5, 0);
				g2d.transform(at);

				super.paint(gfx);
				at.setToTranslation(this.getFontMetrics().charWidth('W') * -5, 0);
				g2d.transform(at);
				g2d.setPaint(new Color(230, 230, 230));
				// g2d.setPaint(Color.LIGHT_GRAY);
				g2d.fillRect(0, 0, this.getFontMetrics().charWidth('W') * 5, getFontMetrics().getHeight()
						* getLineCount());
				g2d.setPaint(Color.BLACK);
				g2d.setFont(getFont());
				// line number
				for (int i = firstLine; i < getLineCount() + 1; i++) {
					String s = "" + i;
					if (i < 10) {
						s = " " + s;
					}
					if (i < 100) {
						s = " " + s;
					}
					if (i < 1000) {
						s = " " + s;
					}
					if (markedLines.contains(new Integer(i - 1))) {
						oldColor = gfx.getColor();
						// gfx.setXORMode( Color.WHITE );
						gfx.setColor(new Color(0, 0, 255, 30));
						gfx.fillRect(0, (i - firstLine) * getFontMetrics().getHeight() + 2, getWidth(),
								getFontMetrics().getHeight() - 3);
						gfx.setColor(oldColor);
					}
					g2d.drawString(s, 0, (i - firstLine) * getFontMetrics().getHeight() - 3);
				}
				// line after 80 chars
				int linePos = this.getFontMetrics().charWidth('W') * 85 + getHorizontalOffset();
				int height = getHeight();
				Color temp = gfx.getColor();
				gfx.setColor(Color.LIGHT_GRAY);
				gfx.drawLine(linePos, 0, linePos, height);
				gfx.setColor(temp);
			}

		};
		documentHandler = new DocumentHandler();
		listenerList = new EventListenerList();
		caretEvent = new MutableCaretEvent();
		lineSegment = new Segment();
		bracketLine = bracketPosition = -1;
		blink = true;

		// Initialize the GUI
		setLayout(new ScrollLayout());
		add(CENTER, painter);
		add(RIGHT, vertical = new JScrollBar(Adjustable.VERTICAL));
		add(BOTTOM, horizontal = new JScrollBar(Adjustable.HORIZONTAL));
		painter.addMouseWheelListener(new java.awt.event.MouseWheelListener() {
			public void mouseWheelMoved(java.awt.event.MouseWheelEvent evt) {
				if (evt.isControlDown()) {
					horizontal.setValue(horizontal.getValue() + evt.getWheelRotation() * 50);
				} else {
					vertical.setValue(vertical.getValue() + evt.getWheelRotation()
							* vertical.getUnitIncrement());
				}
			}
		});
		// Add some event listeners
		vertical.addAdjustmentListener(new AdjustHandler());
		horizontal.addAdjustmentListener(new AdjustHandler());
		painter.addComponentListener(new ComponentHandler());
		painter.addMouseListener(new MouseHandler());
		painter.addMouseMotionListener(new DragHandler());
		addFocusListener(new FocusHandler());

		// Load the defaults
		setInputHandler(defaults.inputHandler);
		setDocument(defaults.document);
		editable = defaults.editable;
		caretVisible = defaults.caretVisible;
		caretBlinks = defaults.caretBlinks;
		electricScroll = defaults.electricScroll;

		popup = defaults.popup;

		// We don't seem to get the initial focus event?
		focusedComponent = this;
		setFocusTraversalKeys(KeyboardFocusManager.FORWARD_TRAVERSAL_KEYS, null);
	}

	final UndoManager undo = new UndoManager();

	private void initStuff() {
		UndoableEditListener uelis = new UndoableEditListener() {
			public void undoableEditHappened(UndoableEditEvent evt) {
				undo.addEdit(evt.getEdit());
			}

		};
		document.addUndoableEditListener(uelis);

		Font f = new Font("Monospaced", Font.PLAIN, system.BIGInterface.getInstance()
				.getBIGConfigFileParser().intCheckOut("editorTextSize", 12));
		painter.setFont(f);
		setFont(f);
		java.util.Dictionary<Object, Object> dic = document.getDocumentProperties();
		dic.remove(PlainDocument.tabSizeAttribute);
		dic.put(PlainDocument.tabSizeAttribute, new Integer(2));
		getPainter().addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {

				if ((SwingUtilities.isMiddleMouseButton(e)) && (e.getClickCount() == 1)) {
					pasteMidMouse(xyToOffset(e.getX(), e.getY()));

				}
			}
		});
		org.syntax.jedit.DefaultInputHandler han = new org.syntax.jedit.DefaultInputHandler() {
			@Override
			public void keyPressed(KeyEvent evt) {
				super.keyPressed(evt);
				if ((evt.getKeyCode() == 10)) {

					int iline = getCaretLine() - 1;
					String lastLine = getLineText(iline);
					String start = "";

					while ((lastLine.length() > 0) && (lastLine.charAt(0) < 33)) {
						char c = lastLine.charAt(0);
						lastLine = lastLine.substring(1);
						start = start + c;
					}
					int off = getCaretPosition();
					int firstLine = getFirstLine();
					try {
						document.insertString(off, start, null);
					} catch (BadLocationException ignored) {}
					setCaretPosition(off + start.length());
					setFirstLine(firstLine);
					document.tokenizeLines();
				}
				JPanel console = system.BIGInterface.getInstance().getConsole().getDisplayPanel();
				BIGGUI gui = (BIGGUI) console.getRootPane().getParent();
				// modifiers 2 means ctrl
				if ((evt.getModifiers() == 2)) {
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("B")) {
						// mark actual line
						Integer actLine = new Integer(getCaretLine() - 1);
						if (markedLines.contains(actLine)) {
							// actual line was already marked before -> line should not be marked any longer
							markedLines.remove(actLine);
						} else {
							// actual line is not marked yet -> so do it now
							markedLines.add(actLine);
						}
						painter.repaint();
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("P")) {
						// print
						print();
						return;
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("Z")) {
						// undo
						try {
							if (undo.canUndo()) {
								undo.undo();
								document.tokenizeLines();
							}
						} catch (CannotUndoException e) {}
						return;
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("Y")) {
						// redo
						try {
							if (undo.canRedo()) {
								undo.redo();
								document.tokenizeLines();
							}
						} catch (CannotUndoException e) {}
						return;
					}

					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("L")) {
						// swich to LOCALDEF
						gui.setGUIState(GUIState.LOCALDEFS);
						return;
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("E")) {
						// execute
						gui.executeSelected();
						return;
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("O")) {
						system.BIGInterface.getInstance().getBIGConfigFileParser()
								.showConfigDialog(gui, gui.getKernelTree());
						return;
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("Q")) {
						// quit
						gui.saveAndExit();
						return;
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("F")) {
						// find
						int caretPosition = 0;
						String find = "";
						while (true) {
							find = JOptionPane.showInputDialog(BIGEditor.this,
									"Please insert the String you want to find", find);
							if ((find != null) && (!find.equals(""))) {
								try {
									String text = getDocument().getText(0, getDocument().getLength() - 1);
									int position = text.toUpperCase().indexOf(find.toUpperCase(), caretPosition);
									if (position > -1) {
										setCaretPosition(position);
										setSelectionStart(position);
										setSelectionEnd(position + find.length());
										caretPosition = position + 1;
									} else {
										JOptionPane.showMessageDialog(BIGEditor.this, "\"" + find + "\" wasn't found.");
										return;
									}
								} catch (BadLocationException ex) {
									ex.printStackTrace();
									return;
								}
							} else
								return;
						}
					}
					if (KeyEvent.getKeyText(evt.getKeyCode()).equals("G")) {
						// goto to line
						String line = JOptionPane.showInputDialog(BIGEditor.this, "Goto which line?");
						try {
							int newLine = (new Integer(line)).intValue() - 1;
							setCaretPosition(getLineStartOffset(newLine));
							scrollTo(newLine, 0);
							System.out.println("There are only " + getLineCount() + " lines in this file.");
						} catch (NumberFormatException ex1) {
							System.err.println("Please insert a number.");
						} catch (IllegalArgumentException ex2) {
							System.err.println("This file has only " + getLineCount() + " lines.");
						}
					}
				}
				if (KeyEvent.getKeyText(evt.getKeyCode()).equals("F5")) {
					gui.getKernelTree().updateKernelTree();
				}
				if (KeyEvent.getKeyText(evt.getKeyCode()).equals("F6")) {
					gui.getResultTree().updateResultTree(null);
				}
				if (KeyEvent.getKeyText(evt.getKeyCode()).equals("F1")) {
					BIGHelp help = new BIGHelp();
					help.setVisible(true);
				}

			}
		};
		han.addDefaultKeyBindings();
		setInputHandler(han);
		setCaretPosition(0);
		java.util.Set<KeyStroke> forwardTraversalKeys = new java.util.HashSet<KeyStroke>();
		/*
		 * forwardTraversalKeys.add(KeyStroke.getKeyStroke( KeyEvent.VK_TAB,0));
		 */
		forwardTraversalKeys.add(KeyStroke.getKeyStroke(KeyEvent.VK_TAB, InputEvent.CTRL_MASK));
		setFocusTraversalKeys(KeyboardFocusManager.FORWARD_TRAVERSAL_KEYS, forwardTraversalKeys);

		java.util.Set<KeyStroke> backwardTraversalKeys = new java.util.HashSet<KeyStroke>();
		/*
		 * backwardTraversalKeys.add(KeyStroke.getKeyStroke( KeyEvent.VK_TAB, InputEvent.SHIFT_MASK));
		 */
		backwardTraversalKeys.add(KeyStroke.getKeyStroke(KeyEvent.VK_TAB, InputEvent.SHIFT_MASK
				| InputEvent.CTRL_MASK));
		setFocusTraversalKeys(KeyboardFocusManager.BACKWARD_TRAVERSAL_KEYS, backwardTraversalKeys);

	}

	/**
	 * Returns if this component can be traversed by pressing the Tab key. This returns false.
	 */
	/*
	 * public final boolean isManagingFocus() { return true; }
	 */

	/**
	 * Returns the object responsible for painting this text area.
	 */
	public final TextAreaPainter getPainter() {
		return painter;
	}

	/**
	 * Returns the input handler.
	 */
	public final InputHandler getInputHandler() {
		return inputHandler;
	}

	/**
	 * Sets the input handler.
	 * 
	 * @param inputHandler The new input handler
	 */
	public void setInputHandler(InputHandler inputHandler) {
		this.inputHandler = inputHandler;
	}

	/**
	 * Returns true if the caret is blinking, false otherwise.
	 */
	public final boolean isCaretBlinkEnabled() {
		return caretBlinks;
	}

	/**
	 * Toggles caret blinking.
	 * 
	 * @param caretBlinks True if the caret should blink, false otherwise
	 */
	public void setCaretBlinkEnabled(boolean caretBlinks) {
		this.caretBlinks = caretBlinks;
		if (!caretBlinks) {
			blink = false;
		}

		painter.invalidateSelectedLines();
	}

	/**
	 * Returns true if the caret is visible, false otherwise.
	 */
	public final boolean isCaretVisible() {
		return (!caretBlinks || blink) && caretVisible;
	}

	/**
	 * Sets if the caret should be visible.
	 * 
	 * @param caretVisible True if the caret should be visible, false otherwise
	 */
	public void setCaretVisible(boolean caretVisible) {
		this.caretVisible = caretVisible;
		blink = true;

		painter.invalidateSelectedLines();
	}

	/**
	 * Blinks the caret.
	 */
	public final void blinkCaret() {
		if (caretBlinks) {
			blink = !blink;
			painter.invalidateSelectedLines();
		} else {
			blink = true;
		}
	}

	/**
	 * Returns the number of lines from the top and button of the text area that are always visible.
	 */
	public final int getElectricScroll() {
		return electricScroll;
	}

	/**
	 * Sets the number of lines from the top and bottom of the text area that are always visible
	 * 
	 * @param electricScroll The number of lines always visible from the top or bottom
	 */
	public final void setElectricScroll(int electricScroll) {
		this.electricScroll = electricScroll;
	}

	/**
	 * Updates the state of the scroll bars. This should be called if the number of lines in the document changes, or when
	 * the size of the text are changes.
	 */
	public void updateScrollBars() {
		if (vertical != null && visibleLines != 0) {
			vertical.setValues(firstLine, visibleLines, 0, getLineCount());
			vertical.setUnitIncrement(2);
			vertical.setBlockIncrement(visibleLines);
		}

		int width = painter.getWidth();
		if (horizontal != null && width != 0) {
			horizontal.setValues(-horizontalOffset, width, 0, width * 5);
			horizontal.setUnitIncrement(painter.getFontMetrics().charWidth('w'));
			horizontal.setBlockIncrement(width / 2);
		}
	}

	/**
	 * Returns the line displayed at the text area's origin.
	 */
	public final int getFirstLine() {
		return firstLine;
	}

	/**
	 * Sets the line displayed at the text area's origin without updating the scroll bars.
	 */
	public void setFirstLine(int firstLine) {
		if (firstLine == this.firstLine)
			return;
		this.firstLine = firstLine;
		if (firstLine != vertical.getValue()) {
			updateScrollBars();
		}
		painter.repaint();
	}

	/**
	 * Returns the number of lines visible in this text area.
	 */
	public final int getVisibleLines() {
		return visibleLines;
	}

	/**
	 * Recalculates the number of visible lines. This should not be called directly.
	 */
	public final void recalculateVisibleLines() {
		if (painter == null)
			return;
		int height = painter.getHeight();
		int lineHeight = painter.getFontMetrics().getHeight();
		visibleLines = height / lineHeight;
		updateScrollBars();
	}

	/**
	 * Returns the horizontal offset of drawn lines.
	 */
	public final int getHorizontalOffset() {
		return horizontalOffset;
	}

	/**
	 * Sets the horizontal offset of drawn lines. This can be used to implement horizontal scrolling.
	 * 
	 * @param horizontalOffset offset The new horizontal offset
	 */
	public void setHorizontalOffset(int horizontalOffset) {
		if (horizontalOffset == this.horizontalOffset)
			return;
		this.horizontalOffset = horizontalOffset;
		if (horizontalOffset != horizontal.getValue()) {
			updateScrollBars();
		}
		painter.repaint();
	}

	/**
	 * A fast way of changing both the first line and horizontal offset.
	 * 
	 * @param firstLine The new first line
	 * @param horizontalOffset The new horizontal offset
	 * @return True if any of the values were changed, false otherwise
	 */
	public boolean setOrigin(int firstLine, int horizontalOffset) {
		boolean changed = false;

		if (horizontalOffset != this.horizontalOffset) {
			this.horizontalOffset = horizontalOffset;
			changed = true;
		}

		if (firstLine != this.firstLine) {
			this.firstLine = firstLine;
			changed = true;
		}

		if (changed) {
			updateScrollBars();
			painter.repaint();
		}

		return changed;
	}

	/**
	 * Ensures that the caret is visible by scrolling the text area if necessary.
	 * 
	 * @return True if scrolling was actually performed, false if the caret was already visible
	 */
	public boolean scrollToCaret() {
		int line = getCaretLine();
		int lineStart = getLineStartOffset(line);
		int offset = Math.max(0, Math.min(getLineLength(line) - 1, getCaretPosition() - lineStart));

		return scrollTo(line, offset);
	}

	/**
	 * Ensures that the specified line and offset is visible by scrolling the text area if necessary.
	 * 
	 * @param line The line to scroll to
	 * @param offset The offset in the line to scroll to
	 * @return True if scrolling was actually performed, false if the line and offset was already visible
	 */
	public boolean scrollTo(int line, int offset) {
		// visibleLines == 0 before the component is realized
		// we can't do any proper scrolling then, so we have
		// this hack...
		if (visibleLines == 0) {
			setFirstLine(Math.max(0, line - electricScroll));
			return true;
		}

		int newFirstLine = firstLine;
		int newHorizontalOffset = horizontalOffset;

		if (line < firstLine + electricScroll) {
			newFirstLine = Math.max(0, line - electricScroll);
		} else if (line + electricScroll >= firstLine + visibleLines) {
			newFirstLine = (line - visibleLines) + electricScroll + 1;
			if (newFirstLine + visibleLines >= getLineCount()) {
				newFirstLine = getLineCount() - visibleLines;
			}
			if (newFirstLine < 0) {
				newFirstLine = 0;
			}
		}

		int x = _offsetToX(line, offset);
		int width = painter.getFontMetrics().charWidth('w');

		if (x < 0) {
			newHorizontalOffset = Math.min(0, horizontalOffset - x + width + 5);
		} else if (x + width >= painter.getWidth()) {
			newHorizontalOffset = horizontalOffset + (painter.getWidth() - x) - width - 5;
		}

		return setOrigin(newFirstLine, newHorizontalOffset);
	}

	/**
	 * Converts a line index to a y co-ordinate.
	 * 
	 * @param line The line
	 */
	public int lineToY(int line) {
		FontMetrics fm = painter.getFontMetrics();
		return (line - firstLine) * fm.getHeight() - (fm.getLeading() + fm.getMaxDescent());
	}

	/**
	 * Converts a y co-ordinate to a line index.
	 * 
	 * @param y The y co-ordinate
	 */
	public int yToLine(int y) {
		FontMetrics fm = painter.getFontMetrics();
		int height = fm.getHeight();
		return Math.max(0, Math.min(getLineCount() - 1, y / height + firstLine));
	}

	/**
	 * Converts an offset in a line into an x co-ordinate. This is a slow version that can be used any time.
	 * 
	 * @param line The line
	 * @param offset The offset, from the start of the line
	 */
	public final int offsetToX(int line, int offset) {
		// don't use cached tokens
		painter.currentLineTokens = null;
		return _offsetToX(line, offset);
	}

	/**
	 * Converts an offset in a line into an x co-ordinate. This is a fast version that should only be used if no changes
	 * were made to the text since the last repaint.
	 * 
	 * @param line The line
	 * @param offset The offset, from the start of the line
	 */
	public int _offsetToX(int line, int offset) {
		TokenMarker tokenMarker = getTokenMarker();

		/* Use painter's cached info for speed */
		FontMetrics fm = painter.getFontMetrics();

		getLineText(line, lineSegment);

		int segmentOffset = lineSegment.offset;
		int x = horizontalOffset;

		/* If syntax coloring is disabled, do simple translation */
		if (tokenMarker == null) {
			lineSegment.count = offset;
			return x + Utilities.getTabbedTextWidth(lineSegment, fm, x, painter, 0);
		}
		/*
		 * If syntax coloring is enabled, we have to do this because tokens can vary in width
		 */
		else {
			Token tokens;
			if (painter.currentLineIndex == line && painter.currentLineTokens != null) {
				tokens = painter.currentLineTokens;
			} else {
				painter.currentLineIndex = line;
				tokens = painter.currentLineTokens = tokenMarker.markTokens(lineSegment, line);
			}

			Font defaultFont = painter.getFont();
			SyntaxStyle[] styles = painter.getStyles();

			for (;;) {
				byte id = tokens.id;
				if (id == Token.END)
					return x;

				if (id == Token.NULL) {
					fm = painter.getFontMetrics();
				} else {
					fm = styles[id].getFontMetrics(defaultFont);
				}

				int length = tokens.length;

				if (offset + segmentOffset < lineSegment.offset + length) {
					lineSegment.count = offset - (lineSegment.offset - segmentOffset);
					return x + Utilities.getTabbedTextWidth(lineSegment, fm, x, painter, 0);
				} else {
					lineSegment.count = length;
					x += Utilities.getTabbedTextWidth(lineSegment, fm, x, painter, 0);
					lineSegment.offset += length;
				}
				tokens = tokens.next;
			}
		}
	}

	/**
	 * Converts an x co-ordinate to an offset within a line.
	 * 
	 * @param line The line
	 * @param x The x co-ordinate
	 */
	public int xToOffset(int line, int x) {
		TokenMarker tokenMarker = getTokenMarker();

		/* Use painter's cached info for speed */
		FontMetrics fm = painter.getFontMetrics();
		x = x - painter.getFontMetrics().charWidth('W') * 5;
		getLineText(line, lineSegment);

		char[] segmentArray = lineSegment.array;
		int segmentOffset = lineSegment.offset;
		int segmentCount = lineSegment.count;

		int width = horizontalOffset;

		if (tokenMarker == null) {
			for (int i = 0; i < segmentCount; i++) {
				char c = segmentArray[i + segmentOffset];
				int charWidth;
				if (c == '\t') {
					charWidth = (int) painter.nextTabStop(width, i) - width;
				} else {
					charWidth = fm.charWidth(c);
				}

				if (painter.isBlockCaretEnabled()) {
					if (x - charWidth <= width)
						return i;
				} else {
					if (x - charWidth / 2 <= width)
						return i;
				}

				width += charWidth;
			}

			return segmentCount;
		} else {
			Token tokens;
			if (painter.currentLineIndex == line && painter.currentLineTokens != null) {
				tokens = painter.currentLineTokens;
			} else {
				painter.currentLineIndex = line;
				tokens = painter.currentLineTokens = tokenMarker.markTokens(lineSegment, line);
			}

			int offset = 0;
			Font defaultFont = painter.getFont();
			SyntaxStyle[] styles = painter.getStyles();

			for (;;) {
				byte id = tokens.id;
				if (id == Token.END)
					return offset;

				if (id == Token.NULL) {
					fm = painter.getFontMetrics();
				} else {
					fm = styles[id].getFontMetrics(defaultFont);
				}

				int length = tokens.length;

				for (int i = 0; i < length; i++) {
					char c = segmentArray[segmentOffset + offset + i];
					int charWidth;
					if (c == '\t') {
						charWidth = (int) painter.nextTabStop(width, offset + i) - width;
					} else {
						charWidth = fm.charWidth(c);
					}

					if (painter.isBlockCaretEnabled()) {
						if (x - charWidth <= width)
							return offset + i;
					} else {
						if (x - charWidth / 2 <= width)
							return offset + i;
					}

					width += charWidth;
				}

				offset += length;
				tokens = tokens.next;
			}
		}
	}

	/**
	 * Converts a point to an offset, from the start of the text.
	 * 
	 * @param x The x co-ordinate of the point
	 * @param y The y co-ordinate of the point
	 */
	public int xyToOffset(int x, int y) {
		int line = yToLine(y);
		int start = getLineStartOffset(line);
		return start + xToOffset(line, x);
	}

	/**
	 * Returns the document this text area is editing.
	 */
	public final SyntaxDocument getDocument() {
		return document;
	}

	/**
	 * Sets the document this text area is editing.
	 * 
	 * @param document The document
	 */
	public void setDocument(SyntaxDocument document) {
		if (this.document == document)
			return;
		if (this.document != null) {
			this.document.removeDocumentListener(documentHandler);
		}
		this.document = document;

		document.addDocumentListener(documentHandler);

		select(0, 0);
		updateScrollBars();
		painter.repaint();
	}

	/**
	 * Returns the document's token marker. Equivalent to calling <code>getDocument().getTokenMarker()</code>.
	 */
	public final TokenMarker getTokenMarker() {
		return document.getTokenMarker();
	}

	/**
	 * Sets the document's token marker. Equivalent to caling <code>getDocument().setTokenMarker()</code>.
	 * 
	 * @param tokenMarker The token marker
	 */
	public final void setTokenMarker(TokenMarker tokenMarker) {
		document.setTokenMarker(tokenMarker);
	}

	/**
	 * Returns the length of the document. Equivalent to calling <code>getDocument().getLength()</code>.
	 */
	public final int getDocumentLength() {
		return document.getLength();
	}

	/**
	 * Returns the number of lines in the document.
	 */
	public final int getLineCount() {
		return document.getDefaultRootElement().getElementCount();
	}

	/**
	 * Returns the line containing the specified offset.
	 * 
	 * @param offset The offset
	 */
	public final int getLineOfOffset(int offset) {
		return document.getDefaultRootElement().getElementIndex(offset);
	}

	/**
	 * Returns the start offset of the specified line.
	 * 
	 * @param line The line
	 * @return The start offset of the specified line, or -1 if the line is invalid
	 */
	public int getLineStartOffset(int line) {
		Element lineElement = document.getDefaultRootElement().getElement(line);
		if (lineElement == null)
			return -1;
		else
			return lineElement.getStartOffset();
	}

	/**
	 * Returns the end offset of the specified line.
	 * 
	 * @param line The line
	 * @return The end offset of the specified line, or -1 if the line is invalid.
	 */
	public int getLineEndOffset(int line) {
		Element lineElement = document.getDefaultRootElement().getElement(line);
		if (lineElement == null)
			return -1;
		else
			return lineElement.getEndOffset();
	}

	/**
	 * Returns the length of the specified line.
	 * 
	 * @param line The line
	 */
	public int getLineLength(int line) {
		Element lineElement = document.getDefaultRootElement().getElement(line);
		if (lineElement == null)
			return -1;
		else
			return lineElement.getEndOffset() - lineElement.getStartOffset() - 1;
	}

	/**
	 * Returns the entire text of this text area.
	 */
	public String getText() {
		try {
			String text = document.getText(0, document.getLength());
			if (text.indexOf("\r\n") > -1) {
				System.err.println("Win");
				text = (text.replaceAll("\r\n", "\n"));
				textChanged = true;
			}
			return text;
		} catch (BadLocationException bl) {
			bl.printStackTrace();
			return null;
		}
	}

	/**
	 * Sets the entire text of this text area.
	 */
	public void setText(String text) {
		if (text.indexOf("\r\n") > -1) {
			System.err.println("Win");
			text = (text.replaceAll("\r\n", "\n"));
			textChanged = true;
		}
		if (text.indexOf("\n\r") > -1) {
			System.err.println("Win2");
			text = (text.replaceAll("\n\r", "\n"));
			textChanged = true;
		}

		// System.err.println("Setting text on area@"+this.hashCode()+" to text@"+text.hashCode());
		try {
			document.beginCompoundEdit();
			document.remove(0, document.getLength());
			document.insertString(0, text, null);
			document.endCompoundEdit();
		} catch (BadLocationException bl) {
			bl.printStackTrace();
		} finally {
			document.endCompoundEdit();
		}
	}

	/**
	 * Returns the specified substring of the document.
	 * 
	 * @param start The start offset
	 * @param len The length of the substring
	 * @return The substring, or null if the offsets are invalid
	 */
	public final String getText(int start, int len) {
		try {
			return document.getText(start, len);
		} catch (BadLocationException bl) {
			bl.printStackTrace();
			return null;
		}
	}

	/**
	 * Copies the specified substring of the document into a segment. If the offsets are invalid, the segment will contain
	 * a null string.
	 * 
	 * @param start The start offset
	 * @param len The length of the substring
	 * @param segment The segment
	 */
	public final void getText(int start, int len, Segment segment) {
		try {
			document.getText(start, len, segment);
		} catch (BadLocationException bl) {
			bl.printStackTrace();
			segment.offset = segment.count = 0;
		}
	}

	/**
	 * Returns the text on the specified line.
	 * 
	 * @param lineIndex The line
	 * @return The text, or null if the line is invalid
	 */
	public final String getLineText(int lineIndex) {
		int start = getLineStartOffset(lineIndex);
		return getText(start, getLineEndOffset(lineIndex) - start - 1);
	}

	/**
	 * Copies the text on the specified line into a segment. If the line is invalid, the segment will contain a null
	 * string.
	 * 
	 * @param lineIndex The line
	 */
	public final void getLineText(int lineIndex, Segment segment) {
		int start = getLineStartOffset(lineIndex);
		getText(start, getLineEndOffset(lineIndex) - start - 1, segment);
	}

	/**
	 * Returns the selection start offset.
	 */
	public final int getSelectionStart() {
		return selectionStart;
	}

	/**
	 * Returns the offset where the selection starts on the specified line.
	 */
	public int getSelectionStart(int line) {
		if (line == selectionStartLine)
			return selectionStart;
		else if (rectSelect) {
			Element map = document.getDefaultRootElement();
			int start = selectionStart - map.getElement(selectionStartLine).getStartOffset();

			Element lineElement = map.getElement(line);
			int lineStart = lineElement.getStartOffset();
			int lineEnd = lineElement.getEndOffset() - 1;
			return Math.min(lineEnd, lineStart + start);
		} else
			return getLineStartOffset(line);
	}

	/**
	 * Returns the selection start line.
	 */
	public final int getSelectionStartLine() {
		return selectionStartLine;
	}

	/**
	 * Sets the selection start. The new selection will be the new selection start and the old selection end.
	 * 
	 * @param selectionStart The selection start
	 * @see #select(int,int)
	 */
	public final void setSelectionStart(int selectionStart) {
		select(selectionStart, selectionEnd);
	}

	/**
	 * Returns the selection end offset.
	 */
	public final int getSelectionEnd() {
		return selectionEnd;
	}

	/**
	 * Returns the offset where the selection ends on the specified line.
	 */
	public int getSelectionEnd(int line) {
		if (line == selectionEndLine)
			return selectionEnd;
		else if (rectSelect) {
			Element map = document.getDefaultRootElement();
			int end = selectionEnd - map.getElement(selectionEndLine).getStartOffset();

			Element lineElement = map.getElement(line);
			int lineStart = lineElement.getStartOffset();
			int lineEnd = lineElement.getEndOffset() - 1;
			return Math.min(lineEnd, lineStart + end);
		} else
			return getLineEndOffset(line) - 1;
	}

	/**
	 * Returns the selection end line.
	 */
	public final int getSelectionEndLine() {
		return selectionEndLine;
	}

	/**
	 * Sets the selection end. The new selection will be the old selection start and the bew selection end.
	 * 
	 * @param selectionEnd The selection end
	 * @see #select(int,int)
	 */
	public final void setSelectionEnd(int selectionEnd) {
		select(selectionStart, selectionEnd);
	}

	/**
	 * Returns the caret position. This will either be the selection start or the selection end, depending on which
	 * direction the selection was made in.
	 */
	public final int getCaretPosition() {
		return (biasLeft ? selectionStart : selectionEnd);
	}

	/**
	 * Returns the caret line.
	 */
	public final int getCaretLine() {
		return (biasLeft ? selectionStartLine : selectionEndLine);
	}

	/**
	 * Returns the mark position. This will be the opposite selection bound to the caret position.
	 * 
	 * @see #getCaretPosition()
	 */
	public final int getMarkPosition() {
		return (biasLeft ? selectionEnd : selectionStart);
	}

	/**
	 * Returns the mark line.
	 */
	public final int getMarkLine() {
		return (biasLeft ? selectionEndLine : selectionStartLine);
	}

	/**
	 * Sets the caret position. The new selection will consist of the caret position only (hence no text will be selected)
	 * 
	 * @param caret The caret position
	 * @see #select(int,int)
	 */
	public final void setCaretPosition(int caret) {
		select(caret, caret);
	}

	/**
	 * Selects all text in the document.
	 */
	public final void selectAll() {
		select(0, getDocumentLength());
	}

	/**
	 * Moves the mark to the caret position.
	 */
	public final void selectNone() {
		select(getCaretPosition(), getCaretPosition());
	}

	/**
	 * Selects from the start offset to the end offset. This is the general selection method used by all other selecting
	 * methods. The caret position will be start if start &lt; end, and end if end &gt; start.
	 * 
	 * @param start The start offset
	 * @param end The end offset
	 */
	public void select(int start, int end) {
		int newStart, newEnd;
		boolean newBias;
		if (start <= end) {
			newStart = start;
			newEnd = end;
			newBias = false;
		} else {
			newStart = end;
			newEnd = start;
			newBias = true;
		}

		if (newStart < 0 || newEnd > getDocumentLength())
			throw new IllegalArgumentException("Bounds out of" + " range: " + newStart + "," + newEnd);

		// If the new position is the same as the old, we don't
		// do all this crap, however we still do the stuff at
		// the end (clearing magic position, scrolling)
		if (newStart != selectionStart || newEnd != selectionEnd || newBias != biasLeft) {
			int newStartLine = getLineOfOffset(newStart);
			int newEndLine = getLineOfOffset(newEnd);

			if (painter.isBracketHighlightEnabled()) {
				if (bracketLine != -1) {
					painter.invalidateLine(bracketLine);
				}
				updateBracketHighlight(end);
				if (bracketLine != -1) {
					painter.invalidateLine(bracketLine);
				}
			}

			painter.invalidateLineRange(selectionStartLine, selectionEndLine);
			painter.invalidateLineRange(newStartLine, newEndLine);

			document.addUndoableEdit(new CaretUndo(selectionStart, selectionEnd));

			selectionStart = newStart;
			selectionEnd = newEnd;
			selectionStartLine = newStartLine;
			selectionEndLine = newEndLine;
			biasLeft = newBias;

			fireCaretEvent();
		}

		// When the user is typing, etc, we don't want the caret
		// to blink
		blink = true;
		caretTimer.restart();

		// Disable rectangle select if selection start = selection end
		if (selectionStart == selectionEnd) {
			rectSelect = false;
		}

		// Clear the `magic' caret position used by up/down
		magicCaret = -1;

		scrollToCaret();
	}

	/**
	 * Returns the selected text, or null if no selection is active.
	 */
	public final String getSelectedText() {
		if (selectionStart == selectionEnd)
			return null;

		if (rectSelect) {
			// Return each row of the selection on a new line

			Element map = document.getDefaultRootElement();

			int start = selectionStart - map.getElement(selectionStartLine).getStartOffset();
			int end = selectionEnd - map.getElement(selectionEndLine).getStartOffset();

			// Certain rectangles satisfy this condition...
			if (end < start) {
				int tmp = end;
				end = start;
				start = tmp;
			}

			StringBuffer buf = new StringBuffer();
			Segment seg = new Segment();

			for (int i = selectionStartLine; i <= selectionEndLine; i++) {
				Element lineElement = map.getElement(i);
				int lineStart = lineElement.getStartOffset();
				int lineEnd = lineElement.getEndOffset() - 1;
				int lineLen = lineEnd - lineStart;

				lineStart = Math.min(lineStart + start, lineEnd);
				lineLen = Math.min(end - start, lineEnd - lineStart);

				getText(lineStart, lineLen, seg);
				buf.append(seg.array, seg.offset, seg.count);

				if (i != selectionEndLine) {
					buf.append('\n');
				}
			}

			return buf.toString();
		} else
			return getText(selectionStart, selectionEnd - selectionStart);
	}

	/**
	 * Replaces the selection with the specified text.
	 * 
	 * @param selectedText The replacement text for the selection
	 */
	public void setSelectedText(String selectedText) {
		if (!editable)
			throw new InternalError("Text component" + " read only");

		document.beginCompoundEdit();

		try {
			if (rectSelect) {
				Element map = document.getDefaultRootElement();

				int start = selectionStart - map.getElement(selectionStartLine).getStartOffset();
				int end = selectionEnd - map.getElement(selectionEndLine).getStartOffset();

				// Certain rectangles satisfy this condition...
				if (end < start) {
					int tmp = end;
					end = start;
					start = tmp;
				}

				int lastNewline = 0;
				int currNewline = 0;

				for (int i = selectionStartLine; i <= selectionEndLine; i++) {
					Element lineElement = map.getElement(i);
					int lineStart = lineElement.getStartOffset();
					int lineEnd = lineElement.getEndOffset() - 1;
					int rectStart = Math.min(lineEnd, lineStart + start);

					document.remove(rectStart, Math.min(lineEnd - rectStart, end - start));

					if (selectedText == null) {
						continue;
					}

					currNewline = selectedText.indexOf('\n', lastNewline);
					if (currNewline == -1) {
						currNewline = selectedText.length();
					}

					document.insertString(rectStart, selectedText.substring(lastNewline, currNewline), null);

					lastNewline = Math.min(selectedText.length(), currNewline + 1);
				}

				if (selectedText != null && currNewline != selectedText.length()) {
					int offset = map.getElement(selectionEndLine).getEndOffset() - 1;
					document.insertString(offset, "\n", null);
					document.insertString(offset + 1, selectedText.substring(currNewline + 1), null);
				}
			} else {
				document.remove(selectionStart, selectionEnd - selectionStart);
				if (selectedText != null) {
					document.insertString(selectionStart, selectedText, null);
				}
			}
		} catch (BadLocationException bl) {
			bl.printStackTrace();
			throw new InternalError("Cannot replace" + " selection");
		}
		// No matter what happends... stops us from leaving document
		// in a bad state
		finally {
			document.endCompoundEdit();
		}

		setCaretPosition(selectionEnd);
	}

	/**
	 * Returns true if this text area is editable, false otherwise.
	 */
	public final boolean isEditable() {
		return editable;
	}

	/**
	 * Sets if this component is editable.
	 * 
	 * @param editable True if this text area should be editable, false otherwise
	 */
	public final void setEditable(boolean editable) {
		this.editable = editable;
	}

	/**
	 * Returns the right click popup menu.
	 */
	public final JPopupMenu getRightClickPopup() {
		return popup;
	}

	/**
	 * Sets the right click popup menu.
	 * 
	 * @param popup The popup
	 */
	public final void setRightClickPopup(JPopupMenu popup) {
		this.popup = popup;
	}

	/**
	 * Returns the `magic' caret position. This can be used to preserve the column position when moving up and down lines.
	 */
	public final int getMagicCaretPosition() {
		if (magicCaret == -1) {
			magicCaret = (getCaretPosition() - getLineStartOffset(getCaretLine()) + 5)
					* painter.getFontMetrics().charWidth('W');
		}
		return magicCaret;
	}

	/**
	 * Sets the `magic' caret position. This can be used to preserve the column position when moving up and down lines.
	 * 
	 * @param magicCaret The magic caret position
	 */
	public final void setMagicCaretPosition(int magicCaret) {
		this.magicCaret = magicCaret;
	}

	/**
	 * Similar to <code>setSelectedText()</code>, but overstrikes the appropriate number of characters if overwrite mode
	 * is enabled.
	 * 
	 * @param str The string
	 * @see #setSelectedText(String)
	 * @see #isOverwriteEnabled()
	 */
	public void overwriteSetSelectedText(String str) {
		// Don't overstrike if there is a selection
		if (!overwrite || selectionStart != selectionEnd) {
			setSelectedText(str);
			return;
		}

		// Don't overstrike if we're on the end of
		// the line
		int caret = getCaretPosition();
		int caretLineEnd = getLineEndOffset(getCaretLine());
		if (caretLineEnd - caret <= str.length()) {
			setSelectedText(str);
			return;
		}

		document.beginCompoundEdit();

		try {
			document.remove(caret, str.length());
			document.insertString(caret, str, null);
		} catch (BadLocationException bl) {
			bl.printStackTrace();
		} finally {
			document.endCompoundEdit();
		}
	}

	/**
	 * Returns true if overwrite mode is enabled, false otherwise.
	 */
	public final boolean isOverwriteEnabled() {
		return overwrite;
	}

	/**
	 * Sets if overwrite mode should be enabled.
	 * 
	 * @param overwrite True if overwrite mode should be enabled, false otherwise.
	 */
	public final void setOverwriteEnabled(boolean overwrite) {
		this.overwrite = overwrite;
		painter.invalidateSelectedLines();
	}

	/**
	 * Returns true if the selection is rectangular, false otherwise.
	 */
	public final boolean isSelectionRectangular() {
		return rectSelect;
	}

	/**
	 * Sets if the selection should be rectangular.
	 * 
	 * @param overwrite True if the selection should be rectangular, false otherwise.
	 */
	public final void setSelectionRectangular(boolean rectSelect) {
		this.rectSelect = rectSelect;
		painter.invalidateSelectedLines();
	}

	/**
	 * Returns the position of the highlighted bracket (the bracket matching the one before the caret)
	 */
	public final int getBracketPosition() {
		return bracketPosition;
	}

	/**
	 * Returns the line of the highlighted bracket (the bracket matching the one before the caret)
	 */
	public final int getBracketLine() {
		return bracketLine;
	}

	/**
	 * Adds a caret change listener to this text area.
	 * 
	 * @param listener The listener
	 */
	public final void addCaretListener(CaretListener listener) {

		listenerList.add(CaretListener.class, listener);
	}

	/**
	 * Removes a caret change listener from this text area.
	 * 
	 * @param listener The listener
	 */
	public final void removeCaretListener(CaretListener listener) {
		listenerList.remove(CaretListener.class, listener);
	}

	/**
	 * Deletes the selected text from the text area and places it into the clipboard.
	 */
	public void cut() {
		if (editable) {
			copy();
			setSelectedText("");
		}
	}

	/**
	 * Places the selected text into the clipboard.
	 */
	public void copy() {
		if (selectionStart != selectionEnd) {
			Clipboard clipboard = getToolkit().getSystemClipboard();

			String selection = getSelectedText();

			int repeatCount = inputHandler.getRepeatCount();
			StringBuffer buf = new StringBuffer();
			for (int i = 0; i < repeatCount; i++) {
				buf.append(selection);
			}

			clipboard.setContents(new StringSelection(buf.toString()), null);
		}
	}

	/**
	 * Inserts the clipboard contents into the text.
	 */
	public void paste() {
		if (editable) {
			Clipboard clipboard = getToolkit().getSystemClipboard();
			try {
				// The MacOS MRJ doesn't convert \r to \n,
				// so do it here
				String selection = ((String) clipboard.getContents(this).getTransferData(
						DataFlavor.stringFlavor)).replace('\r', '\n');

				int repeatCount = inputHandler.getRepeatCount();
				StringBuffer buf = new StringBuffer();
				for (int i = 0; i < repeatCount; i++) {
					buf.append(selection);
				}
				selection = buf.toString();
				setSelectedText(selection);
			} catch (Exception e) {
				getToolkit().beep();
				System.err.println("Clipboard does not" + " contain a string");
			}
		}
	}

	/**
	 * Inserts the clipboard contents into the text.
	 */
	public void pasteMidMouse(int offset) {
		if (editable) {
			// System.err.println("Before\n"+this.getText());
			try {
				Toolkit tk = Toolkit.getDefaultToolkit();

				Clipboard systemClipboard = tk.getSystemSelection();

				Transferable transferData = systemClipboard.getContents(null);

				DataFlavor[] dataFlavor = transferData.getTransferDataFlavors();
				DataFlavor flavor = dataFlavor[0];

				Object content = transferData.getTransferData(flavor);
				int firstline = getFirstLine();
				int caret = getCaretPosition();
				document.beginCompoundEdit();
				document.insertString(offset, content.toString(), null);
				document.endCompoundEdit();
				setCaretPosition(caret);
				setFirstLine(firstline);
				document.tokenizeLines();
				document.endCompoundEdit();
				// System.err.println("After\n"+this.getText());
			} catch (Exception e) {
				getToolkit().beep();
				System.err.println("Clipboard does not" + " contain a string");
			}
		}
	}

	/**
	 * Called by the AWT when this component is removed from it's parent. This stops clears the currently focused
	 * component.
	 */
	@Override
	public void removeNotify() {
		super.removeNotify();
		if (focusedComponent == this) {
			focusedComponent = null;
		}
	}

	/**
	 * Forwards key events directly to the input handler. This is slightly faster than using a KeyListener because some
	 * Swing overhead is avoided.
	 */
	@Override
	public void processKeyEvent(KeyEvent evt) {
		if (inputHandler == null)
			return;
		switch (evt.getID()) {
			case KeyEvent.KEY_TYPED :
				inputHandler.keyTyped(evt);
				break;
			case KeyEvent.KEY_PRESSED :
				inputHandler.keyPressed(evt);
				break;
			case KeyEvent.KEY_RELEASED :
				inputHandler.keyReleased(evt);
				break;
		}
	}

	// protected members
	protected static String CENTER = "center";
	protected static String RIGHT = "right";
	protected static String BOTTOM = "bottom";

	protected static BIGEditor focusedComponent;
	protected static Timer caretTimer;

	protected TextAreaPainter painter;

	protected JPopupMenu popup;

	// protected EventListenerList listenerList;
	protected MutableCaretEvent caretEvent;

	protected boolean caretBlinks;
	protected boolean caretVisible;
	protected boolean blink;

	protected boolean editable;

	protected int firstLine;
	protected int visibleLines;
	protected int electricScroll;

	protected int horizontalOffset;

	protected JScrollBar vertical;
	protected JScrollBar horizontal;
	protected boolean scrollBarsInitialized;

	protected InputHandler inputHandler;
	private SyntaxDocument document;
	protected DocumentHandler documentHandler;

	protected Segment lineSegment;

	protected int selectionStart;
	protected int selectionStartLine;
	protected int selectionEnd;
	protected int selectionEndLine;
	protected boolean biasLeft;

	protected int bracketPosition;
	protected int bracketLine;

	protected int magicCaret;
	protected boolean overwrite;
	protected boolean rectSelect;

	protected void fireCaretEvent() {
		Object[] listeners = listenerList.getListenerList();
		for (int i = listeners.length - 2; i >= 0; i--) {
			if (listeners[i] == CaretListener.class) {
				((CaretListener) listeners[i + 1]).caretUpdate(caretEvent);
			}
		}
	}

	protected void updateBracketHighlight(int newCaretPosition) {
		if (newCaretPosition == 0) {
			bracketPosition = bracketLine = -1;
			return;
		}

		try {
			int offset = TextUtilities.findMatchingBracket(document, newCaretPosition - 1);
			if (offset != -1) {
				bracketLine = getLineOfOffset(offset);
				bracketPosition = offset - getLineStartOffset(bracketLine);
				return;
			}
		} catch (BadLocationException bl) {
			bl.printStackTrace();
		}

		bracketLine = bracketPosition = -1;
	}

	protected void documentChanged(DocumentEvent evt) {
		DocumentEvent.ElementChange ch = evt.getChange(document.getDefaultRootElement());

		int count;
		if (ch == null) {
			count = 0;
		} else {
			count = ch.getChildrenAdded().length - ch.getChildrenRemoved().length;
		}

		int line = getLineOfOffset(evt.getOffset());
		if (count == 0) {
			painter.invalidateLine(line);
		}
		// do magic stuff
		else if (line < firstLine) {
			setFirstLine(firstLine + count);
		}
		// end of magic stuff
		else {
			painter.invalidateLineRange(line, firstLine + visibleLines);
			updateScrollBars();
		}
		textChanged = true;
	}

	class ScrollLayout implements LayoutManager {
		public void addLayoutComponent(String name, Component comp) {
			if (name.equals(CENTER)) {
				center = comp;
			} else if (name.equals(RIGHT)) {
				right = comp;
			} else if (name.equals(BOTTOM)) {
				bottom = comp;
			} else if (name.equals(LEFT_OF_SCROLLBAR)) {
				leftOfScrollBar.addElement(comp);
			}
		}

		public void removeLayoutComponent(Component comp) {
			if (center == comp) {
				center = null;
			}
			if (right == comp) {
				right = null;
			}
			if (bottom == comp) {
				bottom = null;
			} else {
				leftOfScrollBar.removeElement(comp);
			}
		}

		public Dimension preferredLayoutSize(Container parent) {
			Dimension dim = new Dimension();
			Insets insets = getInsets();
			dim.width = insets.left + insets.right;
			dim.height = insets.top + insets.bottom;

			Dimension centerPref = center.getPreferredSize();
			dim.width += centerPref.width;
			dim.height += centerPref.height;
			Dimension rightPref = right.getPreferredSize();
			dim.width += rightPref.width;
			Dimension bottomPref = bottom.getPreferredSize();
			dim.height += bottomPref.height;

			return dim;
		}

		public Dimension minimumLayoutSize(Container parent) {
			Dimension dim = new Dimension();
			Insets insets = getInsets();
			dim.width = insets.left + insets.right;
			dim.height = insets.top + insets.bottom;

			Dimension centerPref = center.getMinimumSize();
			dim.width += centerPref.width;
			dim.height += centerPref.height;
			Dimension rightPref = right.getMinimumSize();
			dim.width += rightPref.width;
			Dimension bottomPref = bottom.getMinimumSize();
			dim.height += bottomPref.height;

			return dim;
		}

		public void layoutContainer(Container parent) {
			Dimension size = parent.getSize();
			Insets insets = parent.getInsets();
			int itop = insets.top;
			int ileft = insets.left;
			int ibottom = insets.bottom;
			int iright = insets.right;

			int rightWidth = right.getPreferredSize().width;
			int bottomHeight = bottom.getPreferredSize().height;
			int centerWidth = size.width - rightWidth - ileft - iright;
			int centerHeight = size.height - bottomHeight - itop - ibottom;

			center.setBounds(ileft, itop, centerWidth, centerHeight);

			right.setBounds(ileft + centerWidth, itop, rightWidth, centerHeight);

			// Lay out all status components, in order
			Enumeration<Component> status = leftOfScrollBar.elements();
			while (status.hasMoreElements()) {
				Component comp = status.nextElement();
				Dimension dim = comp.getPreferredSize();
				comp.setBounds(ileft, itop + centerHeight, dim.width, bottomHeight);
				ileft += dim.width;
			}

			bottom.setBounds(ileft, itop + centerHeight, size.width - rightWidth - ileft - iright,
					bottomHeight);
		}

		// private members
		private Component center;
		private Component right;
		private Component bottom;
		private final Vector<Component> leftOfScrollBar = new Vector<Component>();
	}

	static class CaretBlinker implements ActionListener {
		public void actionPerformed(ActionEvent evt) {
			if (focusedComponent != null && focusedComponent.hasFocus()) {
				focusedComponent.blinkCaret();
			}
		}
	}

	class MutableCaretEvent extends CaretEvent {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;

		MutableCaretEvent() {
			super(BIGEditor.this);
		}

		@Override
		public int getDot() {
			return getCaretPosition();
		}

		@Override
		public int getMark() {
			return getMarkPosition();
		}
	}

	class AdjustHandler implements AdjustmentListener {
		public void adjustmentValueChanged(final AdjustmentEvent evt) {
			if (!scrollBarsInitialized)
				return;

			// If this is not done, mousePressed events accumilate
			// and the result is that scrolling doesn't stop after
			// the mouse is released
			SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					if (evt.getAdjustable() == vertical) {
						setFirstLine(vertical.getValue());
					} else {
						setHorizontalOffset(-horizontal.getValue());
					}
				}
			});
		}
	}

	class ComponentHandler extends ComponentAdapter {
		@Override
		public void componentResized(ComponentEvent evt) {
			recalculateVisibleLines();
			scrollBarsInitialized = true;
		}
	}

	class DocumentHandler implements DocumentListener {
		public void insertUpdate(DocumentEvent evt) {
			documentChanged(evt);

			int offset = evt.getOffset();
			int length = evt.getLength();

			int newStart;
			int newEnd;

			if (selectionStart > offset || (selectionStart == selectionEnd && selectionStart == offset)) {
				newStart = selectionStart + length;
			} else {
				newStart = selectionStart;
			}

			if (selectionEnd >= offset) {
				newEnd = selectionEnd + length;
			} else {
				newEnd = selectionEnd;
			}

			select(newStart, newEnd);
		}

		public void removeUpdate(DocumentEvent evt) {
			documentChanged(evt);

			int offset = evt.getOffset();
			int length = evt.getLength();

			int newStart;
			int newEnd;

			if (selectionStart > offset) {
				if (selectionStart > offset + length) {
					newStart = selectionStart - length;
				} else {
					newStart = offset;
				}
			} else {
				newStart = selectionStart;
			}

			if (selectionEnd > offset) {
				if (selectionEnd > offset + length) {
					newEnd = selectionEnd - length;
				} else {
					newEnd = offset;
				}
			} else {
				newEnd = selectionEnd;
			}

			select(newStart, newEnd);
		}

		public void changedUpdate(DocumentEvent evt) {}
	}

	class DragHandler implements MouseMotionListener {
		public void mouseDragged(MouseEvent evt) {
			if (popup != null && popup.isVisible())
				return;

			setSelectionRectangular((evt.getModifiers() & InputEvent.CTRL_MASK) != 0);
			select(getMarkPosition(), xyToOffset(evt.getX(), evt.getY()));
		}

		public void mouseMoved(MouseEvent evt) {}
	}

	class FocusHandler implements FocusListener {
		public void focusGained(FocusEvent evt) {
			setCaretVisible(true);
			focusedComponent = BIGEditor.this;
		}

		public void focusLost(FocusEvent evt) {
			setCaretVisible(false);
			focusedComponent = null;
		}
	}

	class MouseHandler extends MouseAdapter {
		@Override
		public void mousePressed(MouseEvent evt) {
			requestFocus();

			// Focus events not fired sometimes?
			setCaretVisible(true);
			focusedComponent = BIGEditor.this;

			if ((evt.getModifiers() & InputEvent.BUTTON3_MASK) != 0 && popup != null) {
				popup.show(painter, evt.getX(), evt.getY());
				return;
			}

			int line = yToLine(evt.getY());
			int offset = xToOffset(line, evt.getX());
			int dot = getLineStartOffset(line) + offset;

			switch (evt.getClickCount()) {
				case 1 :
					doSingleClick(evt, line, offset, dot);
					break;
				case 2 :

					// It uses the bracket matching stuff, so
					// it can throw a BLE
					try {
						doDoubleClick(evt, line, offset, dot);
					} catch (BadLocationException bl) {
						bl.printStackTrace();
					}
					break;
				case 3 :
					doTripleClick(evt, line, offset, dot);
					break;
			}
		}

		private void doSingleClick(MouseEvent evt, int line, int offset, int dot) {
			if ((evt.getModifiers() & InputEvent.SHIFT_MASK) != 0) {
				rectSelect = (evt.getModifiers() & InputEvent.CTRL_MASK) != 0;
				select(getMarkPosition(), dot);
			} else {
				setCaretPosition(dot);
			}
		}

		private void doDoubleClick(MouseEvent evt, int line, int offset, int dot)
				throws BadLocationException {
			// Ignore empty lines
			if (getLineLength(line) == 0)
				return;

			try {
				int bracket = TextUtilities.findMatchingBracket(document, Math.max(0, dot - 1));
				if (bracket != -1) {
					int mark = getMarkPosition();
					// Hack
					if (bracket > mark) {
						bracket++;
						mark--;
					}
					select(mark, bracket);
					return;
				}
			} catch (BadLocationException bl) {
				bl.printStackTrace();
			}

			// Ok, it's not a bracket... select the word
			String lineText = getLineText(line);
			char ch = lineText.charAt(Math.max(0, offset - 1));

			String noWordSep = (String) document.getProperty("noWordSep");
			if (noWordSep == null) {
				noWordSep = "";
			}

			// If the user clicked on a non-letter char,
			// we select the surrounding non-letters
			boolean selectNoLetter = (!Character.isLetterOrDigit(ch) && noWordSep.indexOf(ch) == -1);

			int wordStart = 0;

			for (int i = offset - 1; i >= 0; i--) {
				ch = lineText.charAt(i);
				if (selectNoLetter ^ (!Character.isLetterOrDigit(ch) && noWordSep.indexOf(ch) == -1)) {
					wordStart = i + 1;
					break;
				}
			}

			int wordEnd = lineText.length();
			for (int i = offset; i < lineText.length(); i++) {
				ch = lineText.charAt(i);
				if (selectNoLetter ^ (!Character.isLetterOrDigit(ch) && noWordSep.indexOf(ch) == -1)) {
					wordEnd = i;
					break;
				}
			}

			int lineStart = getLineStartOffset(line);
			select(lineStart + wordStart, lineStart + wordEnd);

			/*
			 * String lineText = getLineText(line); String noWordSep = (String)document.getProperty("noWordSep"); int
			 * wordStart = TextUtilities.findWordStart(lineText,offset,noWordSep); int wordEnd =
			 * TextUtilities.findWordEnd(lineText,offset,noWordSep); int lineStart = getLineStartOffset(line);
			 * select(lineStart + wordStart,lineStart + wordEnd);
			 */
		}

		private void doTripleClick(MouseEvent evt, int line, int offset, int dot) {
			select(getLineStartOffset(line), getLineEndOffset(line) - 1);
		}
	}

	class CaretUndo extends AbstractUndoableEdit {
		/**
	 * 
	 */
		private static final long serialVersionUID = 1L;
		private int start;
		private int end;

		CaretUndo(int start, int end) {
			this.start = start;
			this.end = end;
		}

		@Override
		public boolean isSignificant() {
			return false;
		}

		@Override
		public String getPresentationName() {
			return "caret move";
		}

		@Override
		public void undo() throws CannotUndoException {
			super.undo();

			select(start, end);
		}

		@Override
		public void redo() throws CannotRedoException {
			super.redo();

			select(start, end);
		}

		@Override
		public boolean addEdit(UndoableEdit edit) {
			if (edit instanceof CaretUndo) {
				CaretUndo cedit = (CaretUndo) edit;
				start = cedit.start;
				end = cedit.end;
				cedit.die();

				return true;
			} else
				return false;
		}
	}

	static {
		caretTimer = new Timer(500, new CaretBlinker());
		caretTimer.setInitialDelay(500);
		caretTimer.start();
	}

	/**
	 * Constructor creates an instance of JTextPane.
	 * 
	 * @param text the text diplayed in the editor.
	 */
	public BIGEditor(BIGStrings text) {
		this(text.toString());
	}

	/**
	 * Constructor creates an instance of JTextPane.
	 * 
	 * @param text the text diplayed in the editor.
	 */
	public BIGEditor(String text) {
		this();
		setDocument(new SyntaxDocument());
		setText(text);
		document.tokenizeLines();
		setCaretPosition(0);
		textChanged = false;
		initStuff();
	}

	/**
	 * Returns the text in the Editor as a <code>BIGStrings</code>-Object, splitted by every line.
	 * 
	 * @return a <code>BIGStrings</code>-Object split by lines
	 */
	public BIGStrings getAllLines() {
		String text = getText();
		String[] lines = text.split("\n");
		BIGStrings ret = new BIGStrings(lines.length);
		for (int i = 0; i < lines.length; i++) {
			ret.add(lines[i]);
		}
		return ret;
	}

	/**
	 * This method tells if the text of the editor changed since it's creation or the last call to
	 * <code>textSaved()</code>.
	 * 
	 * @return true, if text changed, false else.
	 */
	public boolean textChanged() {
		return textChanged;
	}

	/**
	 * Resets the textChanged field of this class and sets this class' text property to the current text of the
	 * JEditorPane.
	 */
	public void textSaved() {
		textChanged = false;
	}

	private boolean textChanged;

	/*
	 * protected void finalize() { System.err.println("remove editor@"+this.hashCode()); }
	 */
	public File getEditingFile() {
		return editingFile;
	}

	public String getFileName() {
		return editingFile.getName();
	}

	public int print(Graphics g, PageFormat pf, int pageNr) {
		double widthForOneLine = 1.0 * painter.getFontMetrics().charWidth('W') * 82.0;
		double realSpaceForOneLine = pf.getImageableWidth();
		double scaleFactor = Math.min(realSpaceForOneLine / widthForOneLine, 1.0);

		Graphics2D g2d = (Graphics2D) g;
		int offSetX = (int) (pf.getImageableX() * 1.0 / scaleFactor + 1);
		int offSetY = (int) (pf.getImageableY() * 1.0 / scaleFactor + 1);
		int pixelsPerLine = (int) (painter.getFontMetrics().getHeight() * scaleFactor);
		int linesPerPage = (int) ((pf.getImageableHeight() / scaleFactor) / pixelsPerLine);
		int numberOfPagesVertical = ((getLineCount() - 1) / linesPerPage) + 1;
		System.err.println("PagesVert:" + numberOfPagesVertical);
		int maxSizeHor = 0;
		for (int i = 0; i < getLineCount(); i++)
			if (maxSizeHor < getLineLength(i)) {
				maxSizeHor = getLineLength(i);
			}
		StringBuffer sb = new StringBuffer(maxSizeHor);
		for (int i = 0; i < maxSizeHor; i++) {
			sb.append('W');
		}
		maxSizeHor = (painter.getFontMetrics().stringWidth(sb.toString()));
		int numberOfPagesHorizontal = ((maxSizeHor - 1) / (int) widthForOneLine) + 1;
		if (pageNr >= (numberOfPagesVertical * numberOfPagesHorizontal))
			return NO_SUCH_PAGE;
		// horizontal move
		for (int i = pageNr % numberOfPagesVertical; i > 0; i--) {
			offSetY = offSetY - linesPerPage * pixelsPerLine;
		}
		// vertical move
		for (int i = pageNr; i >= numberOfPagesVertical; i = i - numberOfPagesVertical) {
			offSetX = offSetX - (int) (widthForOneLine * scaleFactor);
		}
		/*
		 * if (startingLineForActualPage<this.getLineCount()) {
		 */
		// offSetY = offSetY-startingLineForActualPage*pixelsPerLine;//offSetY - startingLineForThisPage
		// * ( ( int ) pf.getHeight() ) ;
		g.translate(offSetX, offSetY);
		g2d.scale(scaleFactor, scaleFactor);
		painter.setLineHighlightEnabled(false);
		document.tokenizeLines();
		painter.paintComponent(g);
		painter.setLineHighlightEnabled(true);
		return PAGE_EXISTS;
		// }
	}

	public void print() {
		PrinterJob printJob = PrinterJob.getPrinterJob();
		printJob.setPrintable(BIGEditor.this);
		if (printJob.printDialog()) {
			try {
				printJob.print();
			} catch (PrinterException ex2) {
				ex2.printStackTrace();
			}
		}

	}
}

/*****************************************************************************
 * Log-History
 *****************************************************************************/
