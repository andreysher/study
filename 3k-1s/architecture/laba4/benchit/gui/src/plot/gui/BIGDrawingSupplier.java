package plot.gui;

import java.awt.*;

import org.jfree.chart.renderer.DefaultDrawingSupplier;

final class BIGDrawingSupplier extends DefaultDrawingSupplier {
	/**
	 * 
	 */
	private final BIGPlot bigPlot;
	/**
	 * @param bigPlot
	 */
	BIGDrawingSupplier(BIGPlot bigPlot) {
		this.bigPlot = bigPlot;
	}
	private static final long serialVersionUID = 1L;
	// the next shape. maybe one day set the shapes?
	private int nextShape = 0;
	// the next paint (can be set)
	private int nextPaint = 0;
	// used to get shapes
	private final org.jfree.chart.SeriesShapeFactory factory = new org.jfree.chart.SeriesShapeFactory();
	// the sequence of colors to use
	private final Paint[] PAINT_SEQUENCE = BIGPlot.getDefaultPaintSequence();
	// next paint used for filling shapes
	@Override
	public Paint getNextPaint() {
		return PAINT_SEQUENCE[(nextPaint++) % PAINT_SEQUENCE.length];
	}
	// next outline paint used for drawing shapes
	@Override
	public Paint getNextOutlinePaint() {
		return PAINT_SEQUENCE[(nextPaint++) % PAINT_SEQUENCE.length];
	}
	// next shape used as ... next shape (tadaa)
	@Override
	public Shape getNextShape() {
		return factory.getShape(nextShape++, 0, 0.0, 0.0, this.bigPlot.shapeSize);
	}
}