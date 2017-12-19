package system;

import java.io.*;
import java.util.*;

import javax.swing.*;

public class BIGKernel implements Comparable<Object> {

	private static final boolean debug = false;
	// private String name;
	private final String relativePath;
	private final File absolutePath;
	/** the names of the executables */
	private String[] compiledNames = null;

	private int sorting = 0;
	private final String type;
	private String name;
	private String language;
	private String parallelLibraries;
	private String libraries;
	private final String dataType;

	public BIGKernel(String a) throws StringIndexOutOfBoundsException {
		this(new File(a));
	}

	public BIGKernel(File a) throws StringIndexOutOfBoundsException {
		super();
		if (debug) {
			System.err.println(a);
		}
		String n = a.getAbsolutePath();
		n = n.substring(BIGInterface.getInstance().getBenchItPath().length(), a.getAbsolutePath()
				.length());
		if (debug) {
			System.err.println("n=" + n);
		}
		if (n.startsWith(File.separator)) { // s is a separator ;)
			n = n.substring("skernels".length());
		} else {
			n = n.substring("kernels".length());
		}
		relativePath = n;
		if (debug) {
			System.err.println("relative path:" + relativePath);
		}
		type = n.substring(0, n.indexOf(File.separator));
		if (debug) {
			System.err.println("type " + type);
		}
		name = n.substring(n.indexOf(File.separator) + 1);
		language = name.substring(name.indexOf(File.separator) + 1);
		parallelLibraries = language.substring(language.indexOf(File.separator) + 1);
		libraries = parallelLibraries.substring(parallelLibraries.indexOf(File.separator) + 1);
		dataType = libraries.substring(libraries.indexOf(File.separator) + 1);
		if (debug) {
			System.err.println("name " + name);
		}
		if (debug) {
			System.err.println("language " + language);
		}
		if (debug) {
			System.err.println("parallelLibraries " + parallelLibraries);
		}
		if (debug) {
			System.err.println("libraries " + libraries);
		}
		if (debug) {
			System.err.println("dataType " + dataType);
		}
		name = name.substring(0, name.indexOf(File.separator));
		language = language.substring(0, language.indexOf(File.separator));
		parallelLibraries = parallelLibraries.substring(0, parallelLibraries.indexOf(File.separator));
		libraries = libraries.substring(0, libraries.indexOf(File.separator));
		if (debug) {
			System.err.println("name " + name);
		}
		if (debug) {
			System.err.println("language " + language);
		}
		if (debug) {
			System.err.println("parallelLibraries " + parallelLibraries);
		}
		if (debug) {
			System.err.println("libraries " + libraries);
		}
		if (debug) {
			System.err.println("dataType " + dataType);
		}
		// name = n;
		// relativePath = r;
		absolutePath = a;
		// subDirs = s;
		// scripts=BIGExecute.checkShellScriptExistance(getAbsolutePath());
		// String binName=relativePath.replaceAll(File.separator,".");

		/*
		 * if ((new File(this.absolutePath+File.separator+"RUN.SH")).exists()) { String[] newComNames=new
		 * String[compiledNames.length+1]; for ( int i = 0 ; i < compiledNames.length ; i++ ) { newComNames[ i ] =
		 * compiledNames[ i ] ; } newComNames[ compiledNames.length ] = ".." + File.separator + "kernel" + File.separator +
		 * relativePath+File.separator+"RUN.SH" ; compiledNames=newComNames; }
		 */
		reloadExecutables();
	}

	public int compareTo(Object o) {
		return name.compareTo(o.toString());
	}

	public int getNumberOfCompilations() {
		return compiledNames.length;
	}

	public void reloadExecutables() {
		final String finBinName = getNameAfterSorting(0);
		// checking for bin-path
		File binPath = new File(system.BIGInterface.getInstance().getBenchItPath() + File.separator
				+ "bin");
		if (!binPath.exists()) {
			compiledNames = new String[0];
		}
		// get the number of executables
		// within directory bin,
		// which start with binName
		compiledNames = binPath.list(new java.io.FilenameFilter() {
			public boolean accept(File path, String name) {
				if (name.startsWith(finBinName))
					return true;
				return false;
			}

		});
	}

	public String[] getCompiledNames() {
		return compiledNames;
	}

	public String[] askWhichKernel(java.awt.Component parent) {
		JList<?> selectionList = new JList<Object>(compiledNames);
		selectionList.setBorder(BorderFactory.createTitledBorder("available compilations"));
		Object[] o = new Object[4];
		o[0] = "More then one compiled version of the kernel\n" + name + "\n"
				+ "was found. Please select the version(s)\n" + "you want to start:";
		o[1] = selectionList;
		o[2] = "Please be aware that the last entry is\n" + "just a link to the last compiled version";
		JTextField jt = new JTextField();
		jt.setBorder(BorderFactory.createTitledBorder("regular expression (includes *'s and ?'s)"));
		o[3] = jt;

		int value = JOptionPane.showConfirmDialog(parent, o, "select compilations",
				JOptionPane.OK_CANCEL_OPTION, JOptionPane.QUESTION_MESSAGE);
		if (value == JOptionPane.CANCEL_OPTION)
			return null;
		else {
			String[] returnStrings = null;
			if ((jt.getText() != null) && (!jt.getText().equals(""))) {
				Vector<String> vOfEexecs = new Vector<String>();
				// getting matches in JList for regex
				for (int i = 0; i < compiledNames.length; i++) {
					if (gui.BIGRemoteMenu.isRegEx(jt.getText(), compiledNames[i])) {
						vOfEexecs.add(compiledNames[i]);
					}
				}
				// turn Objects into Strings
				o = vOfEexecs.toArray();
				returnStrings = new String[o.length];
				for (int i = 0; i < o.length; i++) {
					returnStrings[i] = (String) o[i];
				}
			} else {
				returnStrings = new String[selectionList.getSelectedValuesList().size()];
				for (int i = 0; i < returnStrings.length; i++) {
					returnStrings[i] = (String) (selectionList.getSelectedValuesList().get(i));
				}
				if (returnStrings.length == 0)
					return null;
			}
			return returnStrings;

		}
	}

	public String getType() {
		return type;
	}

	public String getLibraries() {
		return libraries;
	}

	public String getParallelLibraries() {
		return parallelLibraries;
	}

	public String getSourceLanguage() {
		return language;
	}

	public String getDataType() {
		return dataType;
	}

	public String[] getSortedNames() {
		String[] sortedNames = new String[6];
		// initial setting
		sortedNames[0] = type;
		sortedNames[1] = name;
		sortedNames[2] = language;
		sortedNames[3] = parallelLibraries;
		sortedNames[4] = libraries;
		sortedNames[5] = dataType;
		String tempString = sortedNames[sorting];
		for (int i = sorting; i > 0; i--) {
			sortedNames[i] = sortedNames[i - 1];
		}
		sortedNames[0] = tempString;
		return sortedNames;
	}

	public String getNameAfterSorting() {
		return getNameAfterSorting(sorting);
	}

	public String getNameAfterSorting(int sorting) {
		String[] names = getSortedNames();
		String name = new String();
		for (int i = 0; i < names.length; i++) {
			name = name + "." + names[i];
		}
		if (debug) {
			System.err.println(name.substring(1));
		}
		return name.substring(1);
	}

	public File[] getFiles() {
		File[] files = absolutePath.listFiles(new FileFilter() {
			public boolean accept(File f) {
				if (f.isDirectory())
					return false;
				return isFilenameWanted(f.getName());
			}
		});
		// for (int i=0;i<files.length;i++)
		// System.err.println(files[i]);
		return files;
	}

	/**
	 * checks whether to display the file or not
	 * 
	 * @param name String
	 * @return boolean
	 */
	public static boolean isFilenameWanted(String name) {
		String deprFiles = null;
		try {
			deprFiles = BIGInterface.getInstance().getBIGConfigFileParser()
					.stringCheckOut("DontShowFileList");
		} catch (Exception ex) {
			if (debug) {
				System.err.println("no list");
			}
			// no list
			return true;
		}
		StringTokenizer st = new StringTokenizer(deprFiles, ";");
		while (st.hasMoreTokens()) {
			String token = st.nextToken();
			if (gui.BIGRemoteMenu.isRegEx(token.trim(), name))
				return false;
		}
		return true;
	}

	public String getName() {
		return name;
	}

	public String getRelativePath() {
		return relativePath;
	}

	public String getAbsolutePath() {
		return absolutePath.getAbsolutePath();
	}

	@Override
	public String toString() {
		return getSortedNames()[5];
	}

	public void setSorting(int s) {
		sorting = s;
	}
}
