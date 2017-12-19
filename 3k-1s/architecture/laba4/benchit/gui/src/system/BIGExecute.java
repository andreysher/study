/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications BIGExecute.java Author: SWTP
 * Nagel 1 Last change by: $Author: rschoene $ $Revision: 1.7 $ $Date: 2006/12/06 06:48:19 $
 ******************************************************************************/
package system;

import gui.*;

import java.io.*;
import java.util.*;

import plot.data.BIGOutputParser;

/**
 * this class was written to execute external programs such as shell scripts and to return their output
 * 
 * @author <a href="mailto:pisi@pisi.de">Christoph Mueller</a>
 * @author <a href="mailto:fx@fx-world.de">Pascal Weyprecht</a>
 */
public class BIGExecute {
	private boolean debug = false;
	// my instance
	private static BIGExecute bigExecute;

	// the Interface instance
	private final BIGInterface db;

	// is there a programm running or not?!?
	private boolean running = false;

	/** The all indicator tag in the kernel list. */
	public static final String ALL_INDICATOR = "<all>";

	/** Indicator for map sorting according to the real directory structure. */
	public static final int KERNEL_DIRECTORY = 0;

	/**
	 * Indicator for map sorting according to the source language of kernels and then by the real directory structure.
	 */
	public static final int SOURCE_LANGUAGE = 1;

	/**
	 * The constructor.
	 **/
	private BIGExecute() {
		db = BIGInterface.getInstance();
		if (db.getDebug("BIGExecute") > 0) {
			debug = true;
		}
	}

	/**
	 * Returns the one and only instance of the execute. If there doesn't exist one, it creates one.
	 * 
	 * @return the instance pointer
	 **/
	public static BIGExecute getInstance() {
		if (bigExecute == null) {
			bigExecute = new BIGExecute();
		}
		return bigExecute;
	}

	/**
	 * Executes an external program given by <code>cmd</code> and waits until is has finished.
	 * 
	 * @param cmd the command to be executed.
	 * @return the return value of the external command.
	 **/
	public int execute(String cmd) {
		return this.execute(cmd, true);
	}

	/**
	 * Executes an external program given by <code>cmd</code> and waits until is has finished.
	 * 
	 * @param waitFor true if we want to wait until command has finished.
	 * @param cmd the command to be executed.
	 * @return the return value of the external command.
	 **/
	public int execute(String cmd, boolean waitFor) {
		if (waitFor)
			running = true;
		Process p = null;
		if (debug) {
			System.out.println("BIGExecute: executing:" + cmd);
		}
		// starting
		try {
			p = Runtime.getRuntime().exec(cmd);
		} catch (IOException e) {
			System.err.println("BIGExecute: failed to execute: " + cmd + "\n" + e.getMessage());
			// e.printStackTrace();
			return 1;
		}
		BIGInterface.getInstance().getConsole().addStream(p.getInputStream(), BIGConsole.DEBUG);
		BIGInterface.getInstance().getConsole().addStream(p.getErrorStream(), BIGConsole.WARNING);

		if (debug) {
			System.out.println("BIGExecute: command started - now waiting");
		}
		if (waitFor) {
			// waiting for finishing
			try {
				p.waitFor();
			} catch (InterruptedException e) {
				System.err.println("BIGExecute: waiting for started process has been interrupted");
				System.err.println(e.getMessage());
				// e.printStackTrace();
			}
			if (debug) {
				System.out.println("BIGExecute: command \"" + cmd + "\" finished");
			}
			running = false;
			return p.exitValue();
		} else
			return 0;
	}

	/**
	 * Returns the all indicator to replace it, if needed.
	 **/
	public String getAllIndicator() {
		return ALL_INDICATOR;
	}

	/**
	 * generate the file postproc.sh this is executed via "source postproc.sh" in BenchIT (shellscript) postproc contains
	 * commands for starting the kernels this is for benchmarking without java in the background.<br>
	 * Both parameter BIGStrings should have the same length, for every kernel one script to execute, kernels and scripts
	 * can exist twice in the BIGStrings.
	 * 
	 * @param kernels the kernels to be executed
	 * @param fileName use this as fileName "postProc.sh" when shutting down GUI
	 **/
	public String generatePostProcScript(List<BIGRunEntry> kernels, String fileName,
			BIGKernelTree kernelTree, boolean restartGui) {
		String standardComment;
		if (BIGInterface.getInstance().getOriginalHost().equals(BIGInterface.getInstance().getHost())) {
			standardComment = BIGInterface.getInstance()
					.getEntry("<hostname>", "BENCHIT_FILENAME_COMMENT").getValue().toString();
		} else {
			BIGOutputParser parser;
			try {
				parser = new BIGOutputParser(BIGInterface.getInstance().getBenchItPath() + File.separator
						+ "LOCALDEFS" + File.separator + BIGInterface.getInstance().getOriginalHost());
				standardComment = parser.getValue("BENCHIT_FILENAME_COMMENT");
			} catch (BIGParserException e) {
				standardComment = "";
			}
		}
		String target = "";
		String noParam = "";

		if (BIGInterface.getInstance().getBIGConfigFileParser()
				.intCheckOut("settedTargetActiveCompile", 0) > 0) {
			target = " --target="
					+ BIGInterface.getInstance().getBIGConfigFileParser()
							.stringCheckOut("settedTargetCompile", "");
		}
		if (BIGInterface.getInstance().getBIGConfigFileParser()
				.boolCheckOut("runWithoutParameter", false)) {
			noParam = " --no-parameter-file";
		}

		if (debug) {
			System.err.println("---------genBckPostProc(..)------------");
		}
		if (fileName == null) {
			fileName = "postproc" + System.currentTimeMillis() + ".sh";
		}
		String binPath = db.getGUIPath() + File.separator + "bin" + File.separator;
		fileName = binPath + fileName;

		String compileScript = BIGInterface.getInstance().getBenchItPath() + File.separator
				+ "COMPILE.SH";
		String runScript = BIGInterface.getInstance().getBenchItPath() + File.separator + "RUN.SH";

		BIGStrings script = new BIGStrings();

		// header of file
		script.add("#!/bin/sh");
		script.add("echo \"##### starting postproceeding / selected kernels\"");

		// kernel starts
		for (int pos = 0; pos < kernels.size(); pos++) {
			BIGRunEntry entry = kernels.get(pos);
			if (debug) {
				System.err.println("kernelName:" + entry.kernelName);
				System.err.println("script:" + entry.type);
			}
			BIGKernel k = kernelTree.findKernel(entry.kernelName);
			if (k == null) {
				System.err.println("BIGExecute: couldn't find kernel " + entry.kernelName);
				continue;
			}
			// maybe the comment is set in PARAMETERS
			String comment = standardComment;
			File files[] = k.getFiles();
			for (int l = 0; l < files.length; l++) {
				if (files[l].getName().equals("PARAMETERS")) {
					BIGOutputParser parser;
					try {
						parser = new BIGOutputParser(files[l].getName());
					} catch (BIGParserException e) {
						continue;
					}
					String paramComment = parser.getValue("BENCHIT_FILENAME_COMMENT");
					if (paramComment != null && !paramComment.isEmpty()) {
						comment = paramComment;
					}
				}
			}
			script.add("echo \"##### starting \\\"" + entry.kernelName + "\\\"\"");
			if (entry.type == BIGRunType.Both) {
				String compile = compileScript;
				compile += target;
				compile += " " + k.getNameAfterSorting(0);
				script.add(compile);

				String run = runScript;
				run += target;
				run += noParam;
				run += " " + k.getNameAfterSorting(0) + ".";
				if (comment == null || comment.isEmpty()) {
					run += "0";
				} else {
					run += comment;
				}
				script.add("if [ $? -eq 0 ]; then");
				script.add(run);
				script.add("fi");
			} else if (entry.type == BIGRunType.Compile) {
				if (debug) {
					System.err.println("Compile");
				}
				String compile = compileScript;
				compile += target;
				compile += " " + k.getNameAfterSorting(0);
				script.add(compile);
			} else {
				if (k.getNumberOfCompilations() == 0) {
					System.out.println("Please compile the kernel " + k.getName()
							+ "\nbefore trying to run it.");
					return null;
				} else if (k.getNumberOfCompilations() > 1) {
					String[] kernelExec = k.askWhichKernel(null);
					if (kernelExec == null)
						return null;
					try {
						if (BIGInterface.getInstance().getBIGConfigFileParser()
								.boolCheckOut("runWithoutParameter")) {
							noParam = " --no-parameter-file";
						}
					} catch (Exception ex1) {}
					for (int i = 0; i < kernelExec.length; i++) {
						String run = runScript;
						run += noParam;
						run += " " + kernelExec[i];

						script.add(run);

						script.add("echo \"##### finished \\\"" + entry.kernelName + "(" + kernelExec[i] + ")"
								+ "\\\"\"");
					}
				} else {
					String run = runScript;
					run += target;
					run += noParam;
					run += " " + k.getCompiledNames()[0];
					script.add(run);
				}
			}
			script.add("echo \"##### finished \\\"" + entry.kernelName + "(all Selected)" + "\\\"\"");
			script.add("echo \"##### removing shellscript \"\n");
			script.add("rm -f " + fileName);
		}
		if (restartGui) {
			script.add("./GUI.sh");
		}
		// write file
		try {
			script.saveToFile(fileName);
		} catch (Exception e) {
			System.err.println("BIGExecute: Error during writing of postproc.sh\n" + e);
		}

		return fileName;
	}

	/**
	 * get a BIGStrings of all available outputfiles
	 * 
	 * @return BIGStrings - the name of all outputfiles (*.bit)
	 * @see BIGStrings
	 **/
	public BIGStrings getResultList() {
		BIGStrings list = new BIGStrings();
		String name;
		File outputDir = new File(BIGInterface.getInstance().getOutputPath());
		File[] preList = outputDir.listFiles();
		if (preList == null)
			return new BIGStrings();
		if (debug) {
			System.out.println("BIGExecute: results found:");
		}
		for (int i = 0; i < preList.length; i++) {
			name = preList[i].getName();
			if (name.endsWith(".bit")) {
				list.add(name);
				if (debug) {
					System.out.println("BIGExecute: \t" + name);
				}
			}
		}
		if (debug) {
			System.out.println("BIGExecute: results found:");
		}
		list.sort();
		return list;
	}

	/**
	 * execute QUICKVIEW.SH with all outputfiles in list
	 * 
	 * @param list a list of valid outputfiles located in /output
	 * @deprecated
	 **/
	@Deprecated
	public void showResult(BIGStrings list) {
		String s = "";
		Iterator<String> it = list.iterator();
		while (it.hasNext()) {
			s += " ";
			s += it.next();
		}
		String cmd = db.getBenchItPath() + File.separator + "tools" + File.separator + "QUICKVIEW.SH "
				+ s;
		if (debug) {
			System.out.println("BIGExecute: executing:" + cmd);
		}
		this.execute(cmd, true);
	}

	public boolean isRunning() {
		return running;
	}
}

/*****************************************************************************
 * Log-History
 *****************************************************************************/
