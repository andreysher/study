package plot.data;

import system.BIGInterface;

public class AxisData {
	/**
	 * label-texts for yaxis
	 */
	public String Text;
	public String TextDefault;
	private Integer Pre = 3;
	/**
	 * Min as absolute (no prefix scale)
	 */
	private double Min = 0.0;
	/**
	 * Max as absolute (no prefix scale)
	 */
	private double Max = 1.0;
	public double MinDefault;
	public double MaxDefault;
	public String NumberFormat = "0.000";
	/**
	 * scale axis text
	 */
	public boolean scaleAxisText = false;

	/**
	 * logBase for yaxis. if <2 use numerical axis, not logarithmic
	 */
	public Integer Log = 0;
	public Integer LogDefault = 0;
	/**
	 * how many ticks for yaxis? -1 means set automatically
	 */
	public Integer Ticks = -1;
	public Integer TicksDefault = -1;

	public void reset() {
		Text = TextDefault;
		Log = LogDefault;
		Ticks = TicksDefault;
		setPre(3);
		setMinAbs(MinDefault);
		setMaxAbs(MaxDefault);
	}

	public double getMin() {
		return getMinAbs() * BIGDataSet.setModifiers[Pre];
	}

	/**
	 * @return Min absolute (no prefix scale)
	 */
	public double getMinAbs() {
		return Min;
	}

	/**
	 * @param min Min absolute (no prefix scale)
	 */
	public void setMinAbs(double min) {
		if (min < 0.0)
			Min = 0.0;
		else
			Min = min;
	}

	public void setMin(double min) {
		setMinAbs(min * BIGDataSet.setInverseModifiers[Pre]);
	}

	public double getMax() {
		return getMaxAbs() * BIGDataSet.setModifiers[Pre];
	}

	/**
	 * @return Max absolute (no prefix scale)
	 */
	public double getMaxAbs() {
		return Max;
	}

	/**
	 * @param max Max absolute (no prefix scale)
	 */
	public void setMaxAbs(double max) {
		if (max <= 0.0)
			Max = 1.0;
		else
			Max = max;
	}

	public void setMax(double max) {
		setMaxAbs(max * BIGDataSet.setInverseModifiers[Pre]);
	}

	public void setPre(int pre) {
		Pre = pre;
	}

	public int getPre() {
		return Pre;
	}

	public AxisData(String text) {
		Text = text;
		TextDefault = text;
		try {
			scaleAxisText = BIGInterface.getInstance().getBIGConfigFileParser()
					.boolCheckOut("AxisTextScaled");
		} catch (Exception ex) {
			BIGInterface.getInstance().getBIGConfigFileParser().set("AxisTextScaled", "0");
		}
	}

	public String getAxisText() {
		if (scaleAxisText) {
			// y axis text should be scaled
			return getPrefix() + Text;
		} else {
			// y1 axis text should not be scaled
			return Text;
		}
	}

	public String getPrefix() {
		return BIGDataSet.getPrefix(getPre());
	}
}