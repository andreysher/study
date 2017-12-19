package plot.data;

import java.awt.*;
import java.util.Vector;

public class YAxisData extends AxisData {
	private boolean scaleLegends = false;
	/**
	 * colors for datasets
	 */
	public final Vector<Color> Colors = new Vector<Color>();
	/**
	 * shapes for datasets
	 */
	public final Vector<Shape> Shapes = new Vector<Shape>();

	/**
	 * data sets
	 */
	BIGDataSet Data;
	public BIGDataSet getData() {
		return Data;
	}
	public void setData(BIGDataSet data) {
		Data = data;
		if (Data == null)
			return;
		Data.setPre(getPre());
		Data.setPreNamesSelected(scaleLegends);
		Data.setMinimumRangeValue(getMin());
		Data.setMaximumRangeValue(getMax());
	}

	public YAxisData(String text, BIGDataSet data) {
		super(text);
		Data = data;
		if (Data == null)
			return;
		super.setPre(Data.getPre());
		scaleLegends = Data.getPreNamesSelected();
	}

	public void setMinMax() {
		if (Data == null)
			return;
		super.setMin(Data.getMinimumRangeValue());
		super.setMax(Data.getMaximumRangeValue());
		MinDefault = getMinAbs();
		MaxDefault = getMaxAbs();
	}

	public void setXPre(int xPre) {
		if (Data != null)
			Data.setXPre(xPre);
	}

	@Override
	public void setPre(int pre) {
		super.setPre(pre);
		if (Data != null)
			Data.setPre(pre);
	}

	@Override
	public void setMinAbs(double min) {
		super.setMinAbs(min);
		if (Data != null)
			Data.setMinimumRangeValue(min, true);
	}

	@Override
	public void setMaxAbs(double max) {
		super.setMaxAbs(max);
		if (Data != null)
			Data.setMaximumRangeValue(max, true);
	}

	public void setScaleLegends(boolean scaleLegends) {
		this.scaleLegends = scaleLegends;
		if (Data != null)
			Data.setPreNamesSelected(scaleLegends);
	}

	public boolean getScaleLegends() {
		return scaleLegends;
	}
}