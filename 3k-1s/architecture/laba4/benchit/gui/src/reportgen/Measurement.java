/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: Measurement.java
 * Description: contains all information about a measurement Copyright: Copyright (c) 2008
 * Company:ZIH (Center for Information Services and High Performance Computing) Author: Anja
 * Grundmann Last change by: $Author$
 ******************************************************************************/

package reportgen;

import java.util.*;

import com.lowagie.text.Image;

public class Measurement {
	private Image graphic;
	private String title;
	private String origin;
	private ArchInfos archinfos;
	private ArrayList<BitFile> files;
	private int titlepagenumber;

	public Measurement() {
		files = new ArrayList<BitFile>();
		graphic = null;
		title = new String();
		origin = new String();
		archinfos = new ArchInfos();
		titlepagenumber = -1;
	}

	public Measurement(Measurement measurement) {
		files = new ArrayList<BitFile>();
		graphic = measurement.graphic;
		title = measurement.title;
		origin = measurement.origin;
		archinfos = measurement.archinfos;
		files = measurement.files;
		titlepagenumber = measurement.titlepagenumber;
	}

	public Measurement(Image graphic, String title, String origin, ArchInfos archinfos,
			Collection<BitFile> bitfiles, int titlepagenumber) {
		files = new ArrayList<BitFile>();
		this.graphic = graphic;
		this.title = title;
		this.origin = origin;
		this.archinfos = archinfos;
		files.addAll(bitfiles);
		this.titlepagenumber = titlepagenumber;
	}

	public void setGraphic(Image graphic) {
		this.graphic = graphic;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public void setOrigin(String origin) {
		this.origin = origin;
	}

	public void setArchInfos(ArchInfos archinfos) {
		this.archinfos = archinfos;
	}

	public void setTitlePageNumber(int titlepagenumber) {
		this.titlepagenumber = titlepagenumber;
	}

	public void addBitfile(BitFile bitfile) {
		files.add(bitfile);
	}

	public void addBitfile(int index, BitFile bitfile) {
		files.add(index, bitfile);
	}

	public void addBitfiles(Collection<BitFile> bitfiles) {
		files.addAll(bitfiles);
	}

	public Image getGraphic() {
		return graphic;
	}

	public String getTitle() {
		return title;
	}

	public String getOrigin() {
		return origin;
	}

	public ArchInfos getArchInfos() {
		return archinfos;
	}

	public int getTitlePageNumber() {
		return titlepagenumber;
	}

	public BitFile getBitFile(int index) {
		return files.get(index);
	}

	public ArrayList<BitFile> getBitFiles() {
		return files;
	}

	public int getIndex(BitFile bitfile) {
		return files.indexOf(bitfile);
	}

	public int getCountBitFiles() {
		return files.size();
	}

}
