package plot.data;

public class XAxisData extends AxisData {
	private final BIGPlotable owner;

	public XAxisData(String text, BIGPlotable owner) {
		super(text);
		this.owner = owner;
	}

	@Override
	public void setPre(int pre) {
		super.setPre(pre);
		owner.setXPre(pre, false);
	}

	@Override
	public void setMinAbs(double min) {
		super.setMinAbs(min);
		owner.setXMinAbs(min, false);
	}

	@Override
	public void setMaxAbs(double max) {
		super.setMaxAbs(max);
		owner.setXMaxAbs(max, false);
	}
}