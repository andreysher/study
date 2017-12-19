/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: Measurements.java
 * Description: collection of measurements Copyright: Copyright (c) 2008 Company:ZIH (Center for
 * Information Services and High Performance Computing) Author: Anja Grundmann Last change by:
 * $Author$
 ******************************************************************************/

package reportgen;

import java.util.*;

public class Measurements {
	private ArrayList<Measurement> mms;

	public Measurements() {
		mms = new ArrayList<Measurement>();
	}

	public Measurements(Collection<Measurement> measurements) {
		mms = new ArrayList<Measurement>();
		mms.addAll(measurements);
	}

	public Measurements(Measurements measurements) {
		mms = new ArrayList<Measurement>();
		mms = measurements.mms;
	}

	public void addMeasurement(Measurement measurement) {
		mms.add(measurement);
	}

	public void addMeasurement(int index, Measurement measurement) {
		mms.add(index, measurement);
	}

	public void addMeasurements(Collection<Measurement> measurements) {
		mms.addAll(measurements);
	}

	public Measurement getMeasurement(int index) {
		return mms.get(index);
	}

	public ArrayList<Measurement> getMeasurements() {
		return mms;
	}

	public int getIndex(Measurement measurement) {
		return mms.indexOf(measurement);
	}

	public int getCount() {
		return mms.size();
	}

}