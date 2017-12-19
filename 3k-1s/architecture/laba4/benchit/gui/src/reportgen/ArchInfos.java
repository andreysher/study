/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: ArchInfos.java
 * Description: contains all information about the architectural informations Copyright: Copyright
 * (c) 2008 Company:ZIH (Center for Information Services and High Performance Computing) Author:
 * Anja Grundmann Last change by: $Author$
 ******************************************************************************/

package reportgen;

import java.util.*;

public class ArchInfos {
	private final ArrayList<String> infos;
	private final ArrayList<String> showninfos;

	public ArchInfos() {
		infos = new ArrayList<String>();
		showninfos = new ArrayList<String>();
	}

	public ArchInfos(ArchInfos archinfos) {
		infos = archinfos.infos;
		showninfos = archinfos.showninfos;
	}

	public ArchInfos(Collection<String> archinfos, Collection<String> shownarchinfos) {
		infos = new ArrayList<String>();
		infos.addAll(archinfos);
		showninfos = new ArrayList<String>();
		showninfos.addAll(shownarchinfos);
		// System.err.println( shownarchinfos.size() + " ShownArchInfos were added to list " +
		// this.toString() );
	}

	public void addArchInfo(String archinfo) {
		infos.add(archinfo);
		// System.err.println("ArchInfo " + archinfo + " was added to list " + this.toString() );
	}

	public void addShownArchInfo(String archinfo) {
		showninfos.add(archinfo);
		// System.err.println("ShownArchInfo " + archinfo + " was added to list " + this.toString() );
	}

	public void addArchInfo(int index, String archinfo) {
		infos.add(index, archinfo);
	}

	public void addShownArchInfo(int index, String archinfo) {
		showninfos.add(index, archinfo);
		// System.err.println("ShownArchInfo " + archinfo + " was added to list " + this.toString() +
		// " at index " + index );
	}

	public void changeArchInfo(int index, String archinfo) {
		infos.set(index, archinfo);
	}

	public void changeShownArchInfo(int index, String archinfo) {
		showninfos.set(index, archinfo);
	}

	public void addArchInfos(Collection<String> archinfos) {
		infos.addAll(archinfos);
	}

	public void addShownArchInfos(Collection<String> archinfos) {
		showninfos.addAll(archinfos);
		// System.err.println( archinfos.size() + " ShownArchInfos were added to list " +
		// this.toString() );
	}

	public String getArchInfo(int index) {
		return infos.get(index);
	}

	public String getShownArchInfo(int index) {
		return showninfos.get(index);
	}

	public ArrayList<String> getArchInfos() {
		return infos;
	}

	public ArrayList<String> getShwonArchInfos() {
		return showninfos;
	}

	public int getIndex(String archinfo) {
		return infos.indexOf(archinfo);
	}

	public int getShownIndex(String archinfo) {
		return showninfos.indexOf(archinfo);
	}

	public int getCount() {
		return infos.size();
	}

	public int getShownCount() {
		return showninfos.size();
	}
}