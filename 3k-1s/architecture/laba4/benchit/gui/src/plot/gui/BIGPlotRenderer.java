package plot.gui;

import java.awt.*;
import java.util.Vector;

import org.jfree.chart.renderer.*;

/**
 * <p>
 * Überschrift: BenchIT-Renderer for Shapes
 * </p>
 * <p>
 * Beschreibung: Uses setted Colors to draw the Shapes
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
public class BIGPlotRenderer extends StandardXYItemRenderer {
	private static final long serialVersionUID = 1L;

	// setted colors
	private Vector<Color> colors = null;

	// array of shapes that will be used
	public static final Shape[] BIG_SHAPES = DefaultDrawingSupplier.DEFAULT_SHAPE_SEQUENCE;

	// setted shapes
	private Vector<Shape> settedShapes = null;

	/**
	 * Constructor
	 * 
	 * @param colors1 Paint[] setted Color sequence
	 * @param shapes Shape[] setted Shape sequence
	 * @param plotShapes boolean whether to plot shapes (should be true)
	 * @param plotLines boolean whether to draw lines between shapes
	 * @param fillShapes boolean whether to fill the shapes
	 */
	public BIGPlotRenderer(Vector<Color> colors1, Vector<Shape> settedShapes, boolean plotShapes,
			boolean plotLines, boolean fillShapes) {
		super();
		// set colors
		this.colors = colors1;
		// set shapes
		this.settedShapes = settedShapes;
		// calls super.*();
		setPlotShapes(plotShapes);
		setPlotLines(plotLines);
		setDefaultShapeFilled(fillShapes);
	}

	// ------------------------------------------------------------------------------

	@Override
	public Paint getItemPaint(int i, int j, int k) {
		return getSeriesPaint(i, j);
	}

	@Override
	public Paint getSeriesPaint(int i, int j) {
		if (colors == null) {
			colors = new Vector<Color>();
		}
		if (colors.size() <= j)
			colors.setSize(j + 1);
		Color color = colors.get(j);
		if (color != null) {
			if (color.getRGB() != ((Color) super.getSeriesPaint(i, j)).getRGB()) {
				setSeriesPaint(j, new Color(color.getRGB()));
			}
		} else {
			color = (Color) super.getSeriesPaint(i, j);
			colors.set(j, color);
		}
		return color;
	}

	@Override
	public void setSeriesPaint(int i, int j, Paint color) {
		colors.set(j, (Color) color);
		super.setSeriesPaint(i, j, color);
	}

	@Override
	public void setSeriesPaint(int j, Paint color) {
		colors.set(j, (Color) color);
		super.setSeriesPaint(j, color);
	}

	// --------------------------------------------------------------------------------------------
	public void setSeriesShape(int j, int index) {
		Shape actualShape = BIG_SHAPES[index];
		setSeriesShape(j, actualShape);
	}

	/**
	 * BIG-Implementation for setting the shapes of a selected series
	 * 
	 * @param j int - the series to which the shape is setted
	 * @param shape Shape - the shape that is setted
	 */
	@Override
	public void setSeriesShape(int j, Shape shape) {
		settedShapes.set(j, shape);
		super.setSeriesShape(j, shape);
	}

	/**
	 * BIG-Implementation for setting the shapes of a selected series
	 * 
	 * @param i int - ???
	 * @param j int - ???
	 * @param shape Shape - the shape that will be setted
	 */
	@Override
	public void setSeriesShape(int i, int j, Shape shape) {
		settedShapes.set(j, shape);
		super.setSeriesShape(i, j, shape);
	}

	/**
	 * Returns the shape of the given series
	 * 
	 * @param i int - ???
	 * @param j int - ???
	 * @return the shape of the given series
	 */
	@Override
	public Shape getSeriesShape(int i, int j) {
		if (settedShapes == null) {
			settedShapes = new Vector<Shape>();
		}
		if (settedShapes.size() <= j)
			settedShapes.setSize(j + 1);

		Shape shape = settedShapes.get(j);
		if (shape != null) {
			if (!shape.equals(super.getSeriesShape(i, j))) {
				super.setSeriesShape(i, j, shape);
			}
		} else {
			shape = BIG_SHAPES[j % BIG_SHAPES.length];
			setSeriesShape(i, j, shape);
		}
		return shape;
	}

	/**
	 * @param i int
	 * @param j int
	 * @param k int
	 * @return the Shape at the selected position
	 */
	@Override
	public Shape getItemShape(int i, int j, int k) {
		return super.getItemShape(i, j, k);
	}
	// --------------------------------------------------------------------------------------------
}