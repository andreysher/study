package plot.gui;

import java.awt.Cursor;
import java.awt.event.*;
import java.awt.geom.Rectangle2D;

import javax.swing.*;

import org.jfree.chart.*;

import plot.data.BIGPlotable;

/**
 * <p>
 * Überschrift: BenchIT
 * </p>
 * <p>
 * Beschreibung:
 * </p>
 * Implements cool zoom (Mousewheel) and move (Mouse-drag) features when holding shift
 * <p>
 * Copyright: Copyright (c) 2007
 * </p>
 * <p>
 * Organisation: ZIH TU Dresden
 * </p>
 * 
 * @author Robert Schoene
 * @version 1.0
 */
public class PlotListen implements MouseListener, MouseMotionListener, MouseWheelListener {
	private final JFreeChart chart;
	private final ChartPanel chartPanel;
	private final BIGPlotable plotable;
	private boolean active = true;
	double x = -1;
	double y = -1;

	/**
	 * create a new Mouse(Motion/Wheel)Listener
	 * 
	 * @param chart JFreeChart the chart for the cool new zoom and move features
	 * @param pan ChartPanel the charts panel
	 */
	public PlotListen(JFreeChart chart, ChartPanel pan, BIGPlotable plot) {
		this.chart = chart;
		this.chartPanel = pan;
		plotable = plot;

		// for logarithmic axes we deactivate the zoom-function (because it doesn't work well)
		if (plotable.xData.Log >= 2) {
			active = false;
		}
		for (int i = 0; i < plotable.yAxis.size(); i++) {
			if (plotable.yAxis.get(i).Log >= 2) {
				active = false;
				break;
			}
		}
	}

	/**
	 * if drag and drop: do change min and max
	 * 
	 * @param evt MouseEvent
	 */
	public void mouseDragged(MouseEvent evt) {
		if (active) {
			if (evt.isShiftDown()) {
				actualizePlot(evt);
			}
			/*
			 * else { pan.setVerticalZoom( true ) ; pan.setHorizontalZoom( true ) ; }
			 */
			evt.consume();
		}
	}

	/**
	 * if started drag and drop: save position
	 * 
	 * @param evt MouseEvent
	 */
	public void mousePressed(MouseEvent evt) {
		if (active) {
			if (evt.isShiftDown()) {
				chartPanel.setVerticalZoom(false);
				chartPanel.setHorizontalZoom(false);
				Cursor handCursor = new Cursor(Cursor.HAND_CURSOR);
				chartPanel.setCursor(handCursor);
				if (x < 0 && y < 0) {
					x = chart.getXYPlot().getHorizontalValueAxis()
							.translateJava2DtoValue(evt.getX(), chartPanel.getChartRenderingInfo().getDataArea());

					y = chart.getXYPlot().getVerticalValueAxis()
							.translateJava2DtoValue(evt.getY(), chartPanel.getChartRenderingInfo().getDataArea());
				} else {
					// System.err.println("ELSE-BRANCH");
					// -----------------------------------------------
					x = chart.getXYPlot().getHorizontalValueAxis()
							.translateJava2DtoValue(evt.getX(), chartPanel.getChartRenderingInfo().getDataArea());

					y = chart.getXYPlot().getVerticalValueAxis()
							.translateJava2DtoValue(evt.getY(), chartPanel.getChartRenderingInfo().getDataArea());
					// -----------------------------------------------
				}
				evt.consume();

			} else {

				chartPanel.setVerticalZoom(true);
				chartPanel.setHorizontalZoom(true);
				evt.consume();
			}
		}
	}

	/**
	 * if ended drag and drop: do change min and max a last time
	 * 
	 * @param evt MouseEvent
	 */
	public void mouseReleased(MouseEvent evt) {
		Cursor normalCursor = new Cursor(Cursor.DEFAULT_CURSOR);
		chartPanel.setCursor(normalCursor);
		if (evt.isShiftDown() && (x >= 0) && (y >= 0) && active) {
			actualizePlot(evt);
		}
	}

	public void mouseMoved(MouseEvent e) {}

	public void mouseClicked(MouseEvent evt) {
		if (SwingUtilities.isMiddleMouseButton(evt)) {
			double finalY = chart.getXYPlot().getVerticalValueAxis()
					.translateJava2DtoValue(evt.getY(), chartPanel.getChartRenderingInfo().getDataArea());
			double finalX = chart.getXYPlot().getHorizontalValueAxis()
					.translateJava2DtoValue(evt.getX(), chartPanel.getChartRenderingInfo().getDataArea());
			if (!evt.isAltGraphDown()) {
				if (evt.isShiftDown()) {
					if (evt.isControlDown()) {
						try {
							finalY = Double.parseDouble(JOptionPane.showInputDialog(chartPanel,
									"Please specify the exact y position for the marker", new Double(finalY)));
						} catch (Exception ignored) {
							return;
						}
					} else {
						try {
							finalX = Double.parseDouble(JOptionPane.showInputDialog(chartPanel,
									"Please specify the exact x position for the marker", new Double(finalX)));
						} catch (Exception ignored) {
							return;
						}
					}
				}
				if (evt.isControlDown()) {
					chart.getXYPlot().addRangeMarker(new Marker(finalY));
				} else {
					chart.getXYPlot().addDomainMarker(new Marker(finalX));
				}
			}
			if (evt.isAltGraphDown()) {
				String text = JOptionPane.showInputDialog(chartPanel, "Please insert Text", "");
				if (text == null || text.equals("")) {
					return;
				}
				chart.getXYPlot().addAnnotation(
						new org.jfree.chart.annotations.XYTextAnnotation(text, 0, 0));
			}
		}
	}

	public void mouseEntered(MouseEvent e) {}

	public void mouseExited(MouseEvent e) {}

	/**
	 * if mousewheel: do zoom at this position
	 * 
	 * @param e MouseWheelEvent
	 */
	public void mouseWheelMoved(MouseWheelEvent e) {
		if (e.isShiftDown()) {
			double i = 1.0 * e.getScrollAmount() * e.getWheelRotation();
			double xsize = (chart.getXYPlot().getDomainAxis().getMaximumAxisValue() - chart.getXYPlot()
					.getDomainAxis().getMinimumAxisValue())
					* (1.0 + 0.05 * i);
			double ysize = (chart.getXYPlot().getRangeAxis().getMaximumAxisValue() - chart.getXYPlot()
					.getRangeAxis().getMinimumAxisValue())
					* (1.0 + 0.05 * i);
			double xMid = chart.getXYPlot().getHorizontalValueAxis()
					.translateJava2DtoValue(e.getX(), chartPanel.getChartRenderingInfo().getDataArea());
			double yMid = chart.getXYPlot().getVerticalValueAxis()
					.translateJava2DtoValue(e.getY(), chartPanel.getChartRenderingInfo().getDataArea());
			double toLeft = (xMid - chart.getXYPlot().getDomainAxis().getMinimumAxisValue())
					/ (chart.getXYPlot().getDomainAxis().getMaximumAxisValue() - chart.getXYPlot()
							.getDomainAxis().getMinimumAxisValue());

			double toDown = (yMid - chart.getXYPlot().getRangeAxis().getMinimumAxisValue())
					/ (chart.getXYPlot().getRangeAxis().getMaximumAxisValue() - chart.getXYPlot()
							.getRangeAxis().getMinimumAxisValue());

			double xmin = xMid - xsize * toLeft;
			if (xmin < 0) {
				xmin = 0;
			}
			double ymin = yMid - ysize * toDown;
			if (ymin < 0) {
				ymin = 0;
			}
			chart.getXYPlot().getDomainAxis().setMinimumAxisValue(0);
			chart.getXYPlot().getDomainAxis().setMaximumAxisValue(xmin + xsize);
			chart.getXYPlot().getDomainAxis().setMinimumAxisValue(xmin);

			chart.getXYPlot().getRangeAxis().setMinimumAxisValue(0);
			chart.getXYPlot().getRangeAxis().setMaximumAxisValue(ymin + ysize);
			chart.getXYPlot().getRangeAxis().setMinimumAxisValue(ymin);
		}
	}

	/**
	 * actualize the visualisation of the plot (e.g. x-/y-coordinates)
	 * 
	 * @param evt MouseEvent
	 */
	private void actualizePlot(MouseEvent evt) {
		chartPanel.setVerticalZoom(false);
		chartPanel.setHorizontalZoom(false);
		/*
		 * double yshift = chart.getXYPlot().getVerticalValueAxis(). translateJava2DtoValue( evt.getY() ,
		 * pan.getChartRenderingInfo(). getDataArea() ) - y ;
		 */
		double yshift = computeYShift(evt.getY(), chartPanel.getChartRenderingInfo().getDataArea(), y);
		// System.out.println("yshift: " + yshift );

		y = chart.getXYPlot().getVerticalValueAxis()
				.translateJava2DtoValue(evt.getY(), chartPanel.getChartRenderingInfo().getDataArea());

		/*
		 * double xshift = chart.getXYPlot().getHorizontalValueAxis(). translateJava2DtoValue( evt.getX() ,
		 * pan.getChartRenderingInfo(). getDataArea() ) - x ;
		 */
		double xshift = computeXShift(evt.getX(), chartPanel.getChartRenderingInfo().getDataArea(), x);
		// System.out.println("xshift: " + xshift );

		x = chart.getXYPlot().getHorizontalValueAxis()
				.translateJava2DtoValue(evt.getX(), chartPanel.getChartRenderingInfo().getDataArea());

		double min = chart.getXYPlot().getDomainAxis().getMinimumAxisValue();
		if (min + xshift < 0) {
			// System.out.println("min + xshift < 0");
			xshift = 0 - min;
		}
		chart.getXYPlot().getDomainAxis().setMinimumAxisValue(0);
		chart.getXYPlot().getDomainAxis()
				.setMaximumAxisValue(chart.getXYPlot().getDomainAxis().getMaximumAxisValue() + xshift);
		chart.getXYPlot().getDomainAxis().setMinimumAxisValue(min + xshift);

		// System.out.println("Xmin: " + chart.getXYPlot().getDomainAxis().getMinimumAxisValue() );
		// System.out.println("Xmax: " + chart.getXYPlot().getDomainAxis().getMaximumAxisValue() );

		min = chart.getXYPlot().getRangeAxis().getMinimumAxisValue();
		if (min + yshift < 0) {
			// System.out.println("min + yshift < 0");
			yshift = 0 - min;
		}
		chart.getXYPlot().getRangeAxis().setMinimumAxisValue(0);
		chart.getXYPlot().getRangeAxis()
				.setMaximumAxisValue(chart.getXYPlot().getRangeAxis().getMaximumAxisValue() + yshift);
		chart.getXYPlot().getRangeAxis().setMinimumAxisValue(min + yshift);
		// System.out.println("Ymin: " + chart.getXYPlot().getRangeAxis().getMinimumAxisValue() );
		// System.out.println("Ymax: " + chart.getXYPlot().getRangeAxis().getMaximumAxisValue() );
	}

	/**
	 * computes the difference between the last known x-position (xValue) and the new x-position
	 * 
	 * @param i x-position of event
	 * @param rect data area of the plot
	 * @param xValue - last known x-position
	 * @return difference between the old and new x-position
	 */
	private double computeXShift(int i, Rectangle2D rect, double xValue) {
		/*
		 * System.err.println("(XSHIFT): " + ( chart.getXYPlot().getHorizontalValueAxis(). translateJava2DtoValue( i , rect
		 * ) - xValue) );
		 */
		double shift = chart.getXYPlot().getHorizontalValueAxis().translateJava2DtoValue(i, rect)
				- xValue;
		/*
		 * double shift = xValue - chart.getXYPlot().getHorizontalValueAxis(). translateJava2DtoValue( i , rect );
		 */
		// System.err.println("XSHIFT: " + shift );
		/*
		 * return chart.getXYPlot().getHorizontalValueAxis(). translateJava2DtoValue( i , rect ) - xValue ;
		 */
		return shift;
	}

	/**
	 * computes the difference between the last known y-position (yValue) and the new y-position
	 * 
	 * @param i y-position of event
	 * @param rect data area of the plot
	 * @param yValue last known y-position
	 * @return difference between the old and new x-position
	 */
	private double computeYShift(int i, Rectangle2D rect, double yValue) {
		/*
		 * System.err.println("(YSHIFT): " + ( chart.getXYPlot().getVerticalValueAxis(). translateJava2DtoValue( i , rect )
		 * - yValue ) );
		 */
		double shift = chart.getXYPlot().getVerticalValueAxis().translateJava2DtoValue(i, rect)
				- yValue;
		/*
		 * double shift = yValue - chart.getXYPlot().getVerticalValueAxis(). translateJava2DtoValue( i , rect );
		 */
		// System.err.println("YSHIFT: " + shift );
		/*
		 * return chart.getXYPlot().getVerticalValueAxis(). translateJava2DtoValue( i , rect ) - yValue ;
		 */
		return shift;
	}
}
