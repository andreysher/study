/**
 * <p>
 * Überschrift: Legend for Plots
 * </p>
 * <p>
 * Beschreibung: 95% jfree StandardLegend + shapes...
 * </p>
 * <p>
 * Copyright: Copyright (c) 2006
 * </p>
 * <p>
 * Organisation: ZIH TU Dresden
 * </p>
 * 
 * @author Robert Schoene
 * @version 1.0
 */

package plot.gui;

import java.awt.*;
import java.awt.font.LineMetrics;
import java.awt.geom.*;
import java.io.*;

import org.jfree.chart.*;
import org.jfree.chart.event.LegendChangeEvent;
import org.jfree.io.SerialUtilities;
import org.jfree.util.ObjectUtils;

public class BIGPlotLegend extends StandardLegend implements Serializable {
	private static final long serialVersionUID = 1L;

	/** The default outer gap. */
	public static final Spacer DEFAULT_OUTER_GAP = new Spacer(Spacer.ABSOLUTE, 3, 3, 3, 3);

	/** The default inner gap. */
	public static final Spacer DEFAULT_INNER_GAP = new Spacer(Spacer.ABSOLUTE, 2, 2, 2, 2);

	/** The default outline stroke. */
	public static final Stroke DEFAULT_OUTLINE_STROKE = new BasicStroke();

	/** The default outline paint. */
	public static final Paint DEFAULT_OUTLINE_PAINT = Color.gray;

	/** The default background paint. */
	public static final Paint DEFAULT_BACKGROUND_PAINT = Color.white;

	/** The default title font. */
	public static final Font DEFAULT_TITLE_FONT = new Font("SansSerif", Font.BOLD, 11);

	/** The default item font. */
	public static final Font DEFAULT_ITEM_FONT = new Font("SansSerif", Font.BOLD, 14);

	/** The amount of blank space around the legend. */
	private Spacer outerGap;

	/** The pen/brush used to draw the outline of the legend. */
	private transient Stroke outlineStroke;

	/** The color used to draw the outline of the legend. */
	private transient Paint outlinePaint;

	/** The color used to draw the background of the legend. */
	private transient Paint backgroundPaint;

	/** The blank space inside the legend box. */
	private final Spacer innerGap;

	/** An optional title for the legend. */
	private String title;

	/** The font used to display the legend title. */
	private Font titleFont;

	/** The font used to display the legend item names. */
	private Font itemFont;

	/** The color used to display the legend item names. */
	private transient Paint itemPaint;

	/** A flag controlling whether or not outlines are drawn around shapes. */
	private boolean outlineShapes;

	/** The stroke used to outline item shapes. */
	private transient Stroke shapeOutlineStroke = new BasicStroke(0.5f);

	/** The paint used to outline item shapes. */
	private transient Paint shapeOutlinePaint = Color.lightGray;

	/** A flag that controls whether the legend displays the series shapes. */
	private boolean displaySeriesShapes;

	private plot.data.BIGPlotable plotable = null;

	/**
	 * Constructs a new legend with default settings.
	 * 
	 * @param chart the chart that the legend belongs to.
	 */
	public BIGPlotLegend(JFreeChart chart) {

		super(chart);
		outerGap = DEFAULT_OUTER_GAP;
		innerGap = DEFAULT_INNER_GAP;
		backgroundPaint = DEFAULT_BACKGROUND_PAINT;
		outlineStroke = DEFAULT_OUTLINE_STROKE;
		outlinePaint = DEFAULT_OUTLINE_PAINT;
		title = null;
		titleFont = DEFAULT_TITLE_FONT;
		itemFont = DEFAULT_ITEM_FONT;
		itemPaint = Color.black;
		displaySeriesShapes = false;

	}

	/**
	 * Returns the outer gap for the legend.
	 * <P>
	 * This is the amount of blank space around the outside of the legend.
	 * 
	 * @return the gap.
	 */
	@Override
	public Spacer getOuterGap() {
		return outerGap;
	}

	/**
	 * Sets the outer gap for the legend. A {@link LegendChangeEvent} is sent to all registered listeners.
	 * 
	 * @param outerGap the outer gap.
	 */
	@Override
	public void setOuterGap(Spacer outerGap) {
		this.outerGap = outerGap;
		notifyListeners(new LegendChangeEvent(this));
	}

	public plot.data.BIGPlotable getPlotable() {
		return plotable;
	}

	public void setPlotable(plot.data.BIGPlotable plotable) {
		this.plotable = plotable;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the background color for the legend.
	 * 
	 * @return the background color.
	 */
	@Override
	public Paint getBackgroundPaint() {
		return backgroundPaint;
	}

	/**
	 * Sets the background color of the legend.
	 * <P>
	 * Registered listeners are notified that the legend has changed.
	 * 
	 * @param paint the new background color.
	 */
	@Override
	public void setBackgroundPaint(Paint paint) {
		backgroundPaint = paint;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the outline pen/brush.
	 * 
	 * @return the outline pen/brush.
	 */
	@Override
	public Stroke getOutlineStroke() {
		return outlineStroke;
	}

	/**
	 * Sets the outline pen/brush.
	 * <P>
	 * Registered listeners are notified that the legend has changed.
	 * 
	 * @param stroke the new outline pen/brush.
	 */
	@Override
	public void setOutlineStroke(Stroke stroke) {
		outlineStroke = stroke;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the outline color.
	 * 
	 * @return the outline color.
	 */
	@Override
	public Paint getOutlinePaint() {
		return outlinePaint;
	}

	/**
	 * Sets the outline color.
	 * <P>
	 * Registered listeners are notified that the legend has changed.
	 * 
	 * @param paint the new outline color.
	 */
	@Override
	public void setOutlinePaint(Paint paint) {
		outlinePaint = paint;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Gets the title for the legend.
	 * 
	 * @return The title of the legend; which may be null.
	 */
	@Override
	public String getTitle() {
		return title;
	}

	/**
	 * Sets the title of the legend.
	 * 
	 * @param title The title to use. Pass <code>null</code> if you don't want a title.
	 */
	@Override
	public void setTitle(String title) {
		this.title = title;
	}

	/**
	 * Returns the title font.
	 * 
	 * @return The font.
	 */
	@Override
	public Font getTitleFont() {
		return titleFont;
	}

	/**
	 * Sets the title font.
	 * 
	 * @param font the new font.
	 */
	@Override
	public void setTitleFont(Font font) {
		titleFont = font;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the series label font.
	 * 
	 * @return the series label font.
	 */
	@Override
	public Font getItemFont() {
		return itemFont;
	}

	/**
	 * Sets the series label font.
	 * <P>
	 * Registered listeners are notified that the legend has changed.
	 * 
	 * @param font the new series label font.
	 */
	@Override
	public void setItemFont(Font font) {
		itemFont = font;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the series label color.
	 * 
	 * @return the series label color.
	 */
	@Override
	public Paint getItemPaint() {
		return itemPaint;
	}

	/**
	 * Sets the series label color.
	 * <P>
	 * Registered listeners are notified that the legend has changed.
	 * 
	 * @param paint the new series label color.
	 */
	@Override
	public void setItemPaint(Paint paint) {
		itemPaint = paint;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the flag that indicates whether or not outlines are drawn around shapes.
	 * 
	 * @return the flag.
	 */
	@Override
	public boolean getOutlineShapes() {
		return outlineShapes;
	}

	/**
	 * Sets the flag that controls whether or not outlines are drawn around shapes.
	 * 
	 * @param flag The flag.
	 */
	@Override
	public void setOutlineShapes(boolean flag) {
		outlineShapes = flag;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the stroke used to outline shapes.
	 * 
	 * @return the stroke.
	 */
	@Override
	public Stroke getShapeOutlineStroke() {
		return shapeOutlineStroke;
	}

	/**
	 * Sets the stroke used to outline shapes.
	 * <P>
	 * Registered listeners are notified of the change.
	 * 
	 * @param stroke the stroke.
	 */
	@Override
	public void setShapeOutlineStroke(Stroke stroke) {
		shapeOutlineStroke = stroke;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Returns the paint used to outline shapes.
	 * 
	 * @return the paint.
	 */
	@Override
	public Paint getShapeOutlinePaint() {
		return shapeOutlinePaint;
	}

	/**
	 * Sets the paint used to outline shapes.
	 * <P>
	 * Registered listeners are notified of the change.
	 * 
	 * @param paint the paint.
	 */
	@Override
	public void setShapeOutlinePaint(Paint paint) {
		shapeOutlinePaint = paint;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Sets a flag that controls whether or not the legend displays the series shapes.
	 * 
	 * @param flag the new value of the flag.
	 */
	@Override
	public void setDisplaySeriesShapes(boolean flag) {
		displaySeriesShapes = flag;
		notifyListeners(new LegendChangeEvent(this));
	}

	/**
	 * Draws the legend on a Java 2D graphics device (such as the screen or a printer).
	 * 
	 * @param g2 the graphics device.
	 * @param available the area within which the legend, and afterwards the plot, should be drawn.
	 * @return the area used by the legend.
	 */
	@Override
	public Rectangle2D draw(Graphics2D g2, Rectangle2D available) {

		return draw(g2, available, (getAnchor() & HORIZONTAL) != 0, (getAnchor() & INVERTED) != 0);

	}

	/**
	 * Draws the legend.
	 * 
	 * @param g2 the graphics device.
	 * @param available the area available for drawing the chart.
	 * @param horizontal a flag indicating whether the legend items are laid out horizontally.
	 * @param inverted ???
	 * @return the remaining available drawing area.
	 */
	@Override
	protected Rectangle2D draw(Graphics2D g2, Rectangle2D available, boolean horizontal,
			boolean inverted) {

		LegendItemCollection legendItems = getChart().getPlot().getLegendItems();

		if ((legendItems != null) && (legendItems.getItemCount() > 0)) {

			DrawableLegendItem legendTitle = null;

			Rectangle2D legendArea = new Rectangle2D.Double();
			double availableWidth = available.getWidth();
			double availableHeight = available.getHeight();

			// the translation point for the origin of the drawing system
			Point2D translation = new Point2D.Double();

			// Create buffer for individual rectangles within the legend
			DrawableLegendItem[] items = new DrawableLegendItem[legendItems.getItemCount()];

			// Compute individual rectangles in the legend, translation point as well
			// as the bounding box for the legend.
			if (horizontal) {
				double xstart = available.getX() + getOuterGap().getLeftSpace(availableWidth);
				double xlimit = available.getMaxX() + getOuterGap().getRightSpace(availableWidth) - 1;
				double maxRowWidth = 0;
				double xoffset = 0;
				double rowHeight = 0;
				double totalHeight = 0;
				boolean startingNewRow = true;

				if (title != null && !title.equals("")) {

					g2.setFont(getTitleFont());

					LegendItem titleItem = new LegendItem(title, title, null, Color.black,
							DEFAULT_OUTLINE_PAINT, DEFAULT_OUTLINE_STROKE);

					legendTitle = createDrawableLegendItem(g2, titleItem, xoffset, totalHeight);

					rowHeight = Math.max(rowHeight, legendTitle.getHeight());
					xoffset += legendTitle.getWidth();
				}

				g2.setFont(itemFont);
				for (int i = 0; i < legendItems.getItemCount(); i++) {
					items[i] = createDrawableLegendItem(g2, legendItems.get(i), xoffset, totalHeight);
					if ((!startingNewRow) && (items[i].getX() + items[i].getWidth() + xstart > xlimit)) {

						maxRowWidth = Math.max(maxRowWidth, xoffset);
						xoffset = 0;
						totalHeight += rowHeight;
						i--;
						startingNewRow = true;

					} else {
						rowHeight = Math.max(rowHeight, items[i].getHeight());
						xoffset += items[i].getWidth();
						startingNewRow = false;
					}
				}

				maxRowWidth = Math.max(maxRowWidth, xoffset);
				totalHeight += rowHeight;

				// Create the bounding box
				legendArea = new Rectangle2D.Double(0, 0, maxRowWidth, totalHeight);

				// The yloc point is the variable part of the translation point
				// for horizontal legends. xloc is constant.
				double yloc = (inverted) ? available.getMaxY() - totalHeight
						- getOuterGap().getBottomSpace(availableHeight) : available.getY()
						+ getOuterGap().getTopSpace(availableHeight);
				double xloc = available.getX() + available.getWidth() / 2 - maxRowWidth / 2;

				// Create the translation point
				translation = new Point2D.Double(xloc, yloc);
			} else { // vertical...
				double totalHeight = 0;
				double maxWidth = 0;

				if (title != null && !title.equals("")) {

					g2.setFont(getTitleFont());

					LegendItem titleItem = new LegendItem(title, title, null, Color.black,
							DEFAULT_OUTLINE_PAINT, DEFAULT_OUTLINE_STROKE);

					legendTitle = createDrawableLegendItem(g2, titleItem, 0, totalHeight);

					totalHeight += legendTitle.getHeight();
					maxWidth = Math.max(maxWidth, legendTitle.getWidth());
				}

				g2.setFont(itemFont);
				for (int i = 0; i < items.length; i++) {
					items[i] = createDrawableLegendItem(g2, legendItems.get(i), 0, totalHeight);
					totalHeight += items[i].getHeight();
					maxWidth = Math.max(maxWidth, items[i].getWidth());
				}

				// Create the bounding box
				legendArea = new Rectangle2D.Float(0, 0, (float) maxWidth, (float) totalHeight);

				// The xloc point is the variable part of the translation point
				// for vertical legends. yloc is constant.
				double xloc = (inverted) ? available.getMaxX() - maxWidth
						- getOuterGap().getRightSpace(availableWidth) : available.getX()
						+ getOuterGap().getLeftSpace(availableWidth);
				double yloc = available.getY() + (available.getHeight() / 2) - (totalHeight / 2);

				// Create the translation point
				translation = new Point2D.Double(xloc, yloc);
			}

			// Move the origin of the drawing to the appropriate location
			g2.translate(translation.getX(), translation.getY());

			// Draw the legend's bounding box
			g2.setPaint(backgroundPaint);
			g2.fill(legendArea);
			g2.setPaint(outlinePaint);
			g2.setStroke(outlineStroke);
			g2.draw(legendArea);
			/*
			 * double minx=legendArea.getMinX(); double miny=legendArea.getMinY(); double maxx=legendArea.getMaxX(); double
			 * maxy=legendArea.getMaxY(); //g2.drawLine((int)minx,(int)miny,(int)maxx,(int)miny);
			 * g2.drawLine((int)maxx,(int)miny,(int)maxx,(int)maxy); g2.drawLine((int)minx,(int)maxy,(int)maxx,(int)maxy);
			 * g2.drawLine((int)minx,(int)miny,(int)minx,(int)maxy); String s = "visit www.benchit.org" ; g2.setFont(new
			 * Font(this.itemFont.getName(),Font.PLAIN,12)); double top1=( legendArea.getWidth() / 2.0 -
			 * g2.getFontMetrics().stringWidth( s ) / 2.0 ); double top2=( legendArea.getWidth() / 2.0 +
			 * g2.getFontMetrics().stringWidth( s ) / 2.0 ); g2.drawLine((int)minx,(int)miny,(int)top1-1,(int)miny);
			 * g2.drawLine((int)top2+1,(int)miny,(int)maxx,(int)miny); g2.drawString(s, ( int )top1 , 4) ;
			 */

			// draw legend title
			if (legendTitle != null) {
				// XXX dsm - make title bold?
				g2.setPaint(legendTitle.getItem().getPaint());
				g2.setPaint(itemPaint);
				g2.setFont(getTitleFont());
				g2.drawString(legendTitle.getItem().getLabel(), (float) legendTitle.getLabelPosition()
						.getX(), (float) legendTitle.getLabelPosition().getY());
			}

			// Draw individual series elements
			for (int i = 0; i < items.length; i++) {
				g2.setPaint(items[i].getItem().getPaint());
				Shape keyBox = items[i].getMarker();
				if (plotable == null) {
					g2.fill(keyBox);
				} else {
					boolean fill = false;
					if (i < plotable.getNumberOfFunctionsForDataset(0)) {
						fill = plotable.fillShapes[0];
					} else {
						fill = plotable.fillShapes[1];
					}
					if (fill) {
						g2.fill(keyBox);
					}
				}
				// if (getOutlineShapes()) {
				// g2.setPaint(items[i].getItem().getPaint());
				g2.setStroke(shapeOutlineStroke);
				g2.draw(keyBox);
				// }
				g2.setPaint(itemPaint);
				g2.setFont(itemFont);
				g2.drawString(items[i].getItem().getLabel(), (float) items[i].getLabelPosition().getX(),
						(float) items[i].getLabelPosition().getY());
			}

			// translate the origin back to what it was prior to drawing the legend
			g2.translate(-translation.getX(), -translation.getY());

			if (horizontal) {
				// The remaining drawing area bounding box will have the same
				// x origin, width and height independent of the anchor's
				// location. The variable is the y coordinate. If the anchor is
				// SOUTH, the y coordinate is simply the original y coordinate
				// of the available area. If it is NORTH, we adjust original y
				// by the total height of the legend and the initial gap.
				double yy = available.getY();
				double yloc = (inverted) ? yy : yy + legendArea.getHeight()
						+ getOuterGap().getBottomSpace(availableHeight);

				// return the remaining available drawing area
				return new Rectangle2D.Double(available.getX(), yloc, availableWidth, availableHeight
						- legendArea.getHeight() - getOuterGap().getTopSpace(availableHeight)
						- getOuterGap().getBottomSpace(availableHeight));
			} else {
				// The remaining drawing area bounding box will have the same
				// y origin, width and height independent of the anchor's
				// location. The variable is the x coordinate. If the anchor is
				// EAST, the x coordinate is simply the original x coordinate
				// of the available area. If it is WEST, we adjust original x
				// by the total width of the legend and the initial gap.
				double xloc = (inverted) ? available.getX() : available.getX() + legendArea.getWidth()
						+ getOuterGap().getLeftSpace(availableWidth)
						+ getOuterGap().getRightSpace(availableWidth);

				// return the remaining available drawing area
				return new Rectangle2D.Double(xloc, available.getY(), availableWidth
						- legendArea.getWidth() - getOuterGap().getLeftSpace(availableWidth)
						- getOuterGap().getRightSpace(availableWidth), availableHeight);
			}
		} else
			return available;
	}

	/**
	 * Returns a rectangle surrounding a individual entry in the legend.
	 * <P>
	 * The marker box for each entry will be positioned next to the name of the specified series within the legend area.
	 * The marker box will be square and 70% of the height of current font.
	 * 
	 * @param graphics the graphics context (supplies font metrics etc.).
	 * @param legendItem the legend item.
	 * @param x the upper left x coordinate for the bounding box.
	 * @param y the upper left y coordinate for the bounding box.
	 * @return a DrawableLegendItem encapsulating all necessary info for drawing.
	 */
	private DrawableLegendItem createDrawableLegendItem(Graphics2D graphics, LegendItem legendItem,
			double x, double y) {

		int innerGap = 2;
		FontMetrics fm = graphics.getFontMetrics();
		LineMetrics lm = fm.getLineMetrics(legendItem.getLabel(), graphics);
		float textAscent = lm.getAscent();
		float lineHeight = textAscent + lm.getDescent() + lm.getLeading();

		DrawableLegendItem item = new DrawableLegendItem(legendItem);

		float xloc = (float) (x + innerGap + 1.15f * lineHeight);
		float yloc = (float) (y + innerGap + 0.15f * lineHeight + textAscent);

		item.setLabelPosition(new Point2D.Float(xloc, yloc));

		float boxDim = lineHeight * 0.70f;
		xloc = (float) (x + innerGap + 0.15f * lineHeight);
		yloc = (float) (y + innerGap + 0.15f * lineHeight);

		if (displaySeriesShapes) {
			Shape marker = legendItem.getShape();
			AffineTransform transformer = AffineTransform.getTranslateInstance(xloc + boxDim / 2, yloc
					+ boxDim / 2);
			marker = transformer.createTransformedShape(marker);
			item.setMarker(marker);
		} else {
			item.setMarker(new Rectangle2D.Float(xloc, yloc, boxDim, boxDim));
		}

		float width = (float) (item.getLabelPosition().getX() - x
				+ fm.getStringBounds(legendItem.getLabel(), graphics).getWidth() + 0.5 * textAscent);

		float height = 2 * innerGap + lineHeight;
		item.setBounds(x, y, width, height);
		return item;

	}

	/**
	 * Tests an object for equality with this legend.
	 * 
	 * @param obj the object.
	 * @return <code>true</code> or <code>false</code>.
	 */
	@Override
	public boolean equals(Object obj) {
		if (obj == null)
			return false;

		if (obj == this)
			return true;

		if (obj instanceof BIGPlotLegend) {
			BIGPlotLegend l = (BIGPlotLegend) obj;
			if (super.equals(obj)) {

				boolean b0 = ObjectUtils.equal(outerGap, l.outerGap);
				boolean b1 = ObjectUtils.equal(outlineStroke, l.outlineStroke);
				boolean b2 = ObjectUtils.equal(outlinePaint, l.outlinePaint);
				boolean b3 = ObjectUtils.equal(backgroundPaint, l.backgroundPaint);
				boolean b4 = ObjectUtils.equal(innerGap, l.innerGap);
				boolean b5 = ObjectUtils.equal(title, l.title);
				boolean b6 = ObjectUtils.equal(titleFont, l.titleFont);
				boolean b7 = ObjectUtils.equal(itemFont, l.itemFont);
				boolean b8 = ObjectUtils.equal(itemPaint, l.itemPaint);
				boolean b9 = (outlineShapes == l.outlineShapes);
				boolean b10 = ObjectUtils.equal(shapeOutlineStroke, l.shapeOutlineStroke);
				boolean b11 = ObjectUtils.equal(shapeOutlinePaint, l.shapeOutlinePaint);
				boolean b12 = (displaySeriesShapes == l.displaySeriesShapes);
				return b0 && b1 && b2 && b3 && b4 && b5 && b6 && b7 && b8 && b9 && b10 && b11 && b12;
			}
		}

		return false;
	}

	@Override
	public int hashCode() {
		return outerGap.hashCode() + outlineStroke.hashCode() + outlinePaint.hashCode()
				+ backgroundPaint.hashCode() + innerGap.hashCode() + title.hashCode()
				+ titleFont.hashCode() + itemFont.hashCode() + itemPaint.hashCode()
				+ new Boolean(outlineShapes).toString().hashCode() + shapeOutlineStroke.hashCode()
				+ shapeOutlinePaint.hashCode() + new Boolean(displaySeriesShapes).toString().hashCode();
	}

	/**
	 * Provides serialization support.
	 * 
	 * @param stream the output stream.
	 * @throws IOException if there is an I/O error.
	 */
	private void writeObject(ObjectOutputStream stream) throws IOException {
		stream.defaultWriteObject();
		SerialUtilities.writeStroke(outlineStroke, stream);
		SerialUtilities.writePaint(outlinePaint, stream);
		SerialUtilities.writePaint(backgroundPaint, stream);
		SerialUtilities.writePaint(itemPaint, stream);
		SerialUtilities.writeStroke(shapeOutlineStroke, stream);
		SerialUtilities.writePaint(shapeOutlinePaint, stream);
	}

	/**
	 * Provides serialization support.
	 * 
	 * @param stream the output stream.
	 * @throws IOException if there is an I/O error.
	 * @throws ClassNotFoundException if there is a classpath problem.
	 */
	private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
		stream.defaultReadObject();
		outlineStroke = SerialUtilities.readStroke(stream);
		outlinePaint = SerialUtilities.readPaint(stream);
		backgroundPaint = SerialUtilities.readPaint(stream);
		itemPaint = SerialUtilities.readPaint(stream);
		shapeOutlineStroke = SerialUtilities.readStroke(stream);
		shapeOutlinePaint = SerialUtilities.readPaint(stream);
	}

}
