/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: BitFile.java
 * Description: collects all necessary information from the bit file Copyright: Copyright (c) 2008
 * Company:ZIH (Center for Information Services and High Performance Computing) Author: Anja
 * Grundmann Last change by: $Author$
 ******************************************************************************/

package reportgen;

import java.util.*;

public class BitFile {
	private int color;
	private String description;
	private ArrayList<String> vals;

	public BitFile() {
		vals = new ArrayList<String>();
		color = 0;
		description = new String();
	}

	public BitFile(BitFile bitfile) {
		vals = new ArrayList<String>();
		color = bitfile.color;
		description = bitfile.description;
		vals = bitfile.vals;
	}

	public BitFile(int color, String description, Collection<String> values) {
		vals = new ArrayList<String>();
		this.color = 0;
		this.description = new String(description);
		vals.addAll(values);
	}

	public void setGraphColorRGB(int color) {
		this.color = color;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public void addValue(String value) {
		vals.add(value);
	}

	public void addValue(int index, String value) {
		vals.add(index, value);
		// System.err.println("Value " + value + " was added at index " + index + " to list " +
		// this.toString() );
	}

	public void addValues(Collection<String> values) {
		vals.addAll(values);
	}

	public int getGraphColorRGB() {
		return color;
	}

	public String getDescription() {
		return description;
	}

	public String getValue(int index) {
		// System.err.println(" BITFILE RETURNED " + this.vals.get( index ) + " at index " + index);
		return vals.get(index);
	}

	public ArrayList<String> getValues() {
		return vals;
	}

	public int getIndex(String value) {
		return vals.indexOf(value);
	}

	public int getCount() {
		return vals.size();
	}
}