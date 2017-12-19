package gui;

import java.io.File;

import javax.swing.*;
import javax.swing.filechooser.FileFilter;

import system.BIGInterface;

public final class BIGFileChooser extends JFileChooser {
	private static final long serialVersionUID = 1L;

	private boolean allowOverwrite = true;
	private boolean allowDirectory = false;
	private boolean allowNewFile = false;

	public BIGFileChooser() {
		this(BIGInterface.getInstance().getLastWorkPath());
	}

	public BIGFileChooser(File curPath) {
		super(curPath);
		setAcceptAllFileFilterUsed(false);
		setDialogTitle("Select the filename");
	}

	public BIGFileChooser(String curPath) {
		this(new File(curPath));
	}

	// when we selected a file
	@Override
	public void approveSelection() {
		BIGInterface.getInstance().setLastWorkPath(getCurrentDirectory());
		if (!allowDirectory && getSelectedFile().isDirectory()) {
			JOptionPane.showMessageDialog(this, "The selected file is a directory. Canceling.");
			cancelSelection();
			return;
		}
		if (this.getDialogType() == JFileChooser.SAVE_DIALOG) {
			// if there is the File with the same name
			if (getSelectedFile().exists()) {
				int options = JOptionPane.YES_NO_CANCEL_OPTION;
				String text = "Do you want to overwrite it?";
				if (!allowOverwrite) {
					options = JOptionPane.OK_CANCEL_OPTION;
					text = "Please select a new file!";
				}
				// we bring an optionDialog
				int result = JOptionPane.showConfirmDialog(this, "The selected file exists.\n" + text,
						"File exists", options, JOptionPane.QUESTION_MESSAGE);
				if (result == JOptionPane.CANCEL_OPTION) {
					cancelSelection();
					return;
				} else if (result == JOptionPane.NO_OPTION || result == JOptionPane.OK_OPTION)
					return;
			}
		} else if (this.getDialogType() == JFileChooser.OPEN_DIALOG) {
			if (!getSelectedFile().exists()) {
				int options = JOptionPane.YES_NO_CANCEL_OPTION;
				String text = "Do you want to create it?";
				if (!allowNewFile) {
					options = JOptionPane.OK_CANCEL_OPTION;
					text = "Please select a new file!";
				}
				// we bring an optionDialog
				int result = JOptionPane.showConfirmDialog(this, "Your selected file doesn't exist.\n"
						+ text, "File doesn't exists", options, JOptionPane.QUESTION_MESSAGE);
				if (result == JOptionPane.CANCEL_OPTION) {
					cancelSelection();
					return;
				} else if (result == JOptionPane.NO_OPTION || result == JOptionPane.OK_OPTION)
					return;
			}
		}
		super.approveSelection();
	}

	public void setAllowOverwrite(boolean allowOverwrite) {
		this.allowOverwrite = allowOverwrite;
	}

	public void setAllowDirectory(boolean allowDirectory) {
		this.allowDirectory = allowDirectory;
	}

	public void setAllowNewFile(boolean allowNewFile) {
		this.allowNewFile = allowNewFile;
	}

	@Override
	public File getSelectedFile() {
		File selectedFile = super.getSelectedFile();
		if (selectedFile != null && getFileFilter() != null) {
			// this will contain the extension of the file
			String extension = getFileFilter().getDescription();
			int dotPos = extension.lastIndexOf('.');
			if (dotPos >= 0) {
				extension = extension.substring(dotPos);
				if (!selectedFile.getName().endsWith(extension)) {
					selectedFile = new File(selectedFile.toString() + extension);
				}
			}
		}
		return selectedFile;
	}

	public void addFileFilter(final String extension) {
		addFileFilter(extension.toUpperCase() + " Files", extension);
	}

	public void addFileFilter(final String name, final String extension) {
		super.addChoosableFileFilter(new FileFilter() {
			@Override
			public boolean accept(File f) {
				if (f.isDirectory())
					return true;
				String actExtension = f.getName().substring(f.getName().lastIndexOf('.') + 1);
				if (actExtension.equals(extension))
					return true;
				return false;
			}

			// The description of this filter
			@Override
			public String getDescription() {
				return name + " *." + extension;
			}

		});
	}
}