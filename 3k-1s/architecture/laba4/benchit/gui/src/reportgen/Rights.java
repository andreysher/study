/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: Rights.java
 * Description: handles rights management settings Copyright: Copyright (c) 2008 Company:ZIH (Center
 * for Information Services and High Performance Computing) Author: Anja Grundmann Last change by:
 * $Author$
 ******************************************************************************/

package reportgen;

import java.util.ArrayList;

import com.lowagie.text.pdf.PdfWriter;

public class Rights {

	private String userpassword;
	private String ownerpassword;
	private final ArrayList<String> pdfrightstrings;
	private int pdfrights;
	private static final int ENCTYPE = PdfWriter.ENCRYPTION_AES_128;

	public Rights() {
		pdfrightstrings = new ArrayList<String>();
		pdfrightstrings.add("ALLOW_ASSEMBLY");
		pdfrightstrings.add("ALLOW_COPY");
		pdfrightstrings.add("ALLOW_DEGRADED_PRINTING");
		pdfrightstrings.add("ALLOW_MODIFY_ANNOTATIONS");
		pdfrightstrings.add("ALLOW_MODIFY_CONTENTS");
		pdfrightstrings.add("ALLOW_PRINTING");
		pdfrightstrings.add("ALLOW_SCREENREADERS");
		// this.pdfrights = PdfWriter.ALLOW_COPY | PdfWriter.ALLOW_PRINTING |
		// PdfWriter.ALLOW_SCREENREADERS;
	}

	public Rights(Rights rights) {
		pdfrightstrings = new ArrayList<String>();
		pdfrightstrings.add("ALLOW_ASSEMBLY");
		pdfrightstrings.add("ALLOW_COPY");
		pdfrightstrings.add("ALLOW_DEGRADED_PRINTING");
		pdfrightstrings.add("ALLOW_MODIFY_ANNOTATIONS");
		pdfrightstrings.add("ALLOW_MODIFY_CONTENTS");
		pdfrightstrings.add("ALLOW_PRINTING");
		pdfrightstrings.add("ALLOW_SCREENREADERS");
		userpassword = rights.userpassword;
		ownerpassword = rights.ownerpassword;
		pdfrights = rights.pdfrights;
	}

	public Rights(String userpassword, String ownerpassword, int pdfrights) {
		pdfrightstrings = new ArrayList<String>();
		pdfrightstrings.add("ALLOW_ASSEMBLY");
		pdfrightstrings.add("ALLOW_COPY");
		pdfrightstrings.add("ALLOW_DEGRADED_PRINTING");
		pdfrightstrings.add("ALLOW_MODIFY_ANNOTATIONS");
		pdfrightstrings.add("ALLOW_MODIFY_CONTENTS");
		pdfrightstrings.add("ALLOW_PRINTING");
		pdfrightstrings.add("ALLOW_SCREENREADERS");
		this.pdfrights = pdfrights;
		this.userpassword = userpassword;
		this.ownerpassword = ownerpassword;
	}

	public void setUserPassword(String userpassword) {
		this.userpassword = userpassword;
	}

	public void setOwnerPassword(String ownerpassword) {
		this.ownerpassword = ownerpassword;
	}

	public void setRight(int right, boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | right;
		} else {
			pdfrights = pdfrights & (~right);
		}
	}

	public void setRight(String right, boolean allowed) {
		right = right.toUpperCase();

		switch (pdfrightstrings.indexOf(right)) {
			case 0 :
				setAssembly(allowed);
				break;
			case 1 :
				setCopy(allowed);
				break;
			case 2 :
				setDegradedPrinting(allowed);
				break;
			case 3 :
				setModifyAnnotations(allowed);
				break;
			case 4 :
				setModifyContents(allowed);
				break;
			case 5 :
				setPrinting(allowed);
				break;
			case 6 :
				setScreenreaders(allowed);
				break;
			default :
				break;
		}
	}

	public void setAssembly(boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | PdfWriter.ALLOW_ASSEMBLY;
		} else {
			pdfrights = pdfrights & (~PdfWriter.ALLOW_ASSEMBLY);
		}
	}

	public void setCopy(boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | PdfWriter.ALLOW_COPY;
		} else {
			pdfrights = pdfrights & (~PdfWriter.ALLOW_COPY);
		}
	}

	public void setDegradedPrinting(boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | PdfWriter.ALLOW_DEGRADED_PRINTING;
		} else {
			pdfrights = pdfrights & (~PdfWriter.ALLOW_DEGRADED_PRINTING);
		}
	}

	public void setModifyAnnotations(boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | PdfWriter.ALLOW_MODIFY_ANNOTATIONS;
		} else {
			pdfrights = pdfrights & (~PdfWriter.ALLOW_MODIFY_ANNOTATIONS);
		}
	}

	public void setModifyContents(boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | PdfWriter.ALLOW_MODIFY_CONTENTS;
		} else {
			pdfrights = pdfrights & (~PdfWriter.ALLOW_MODIFY_CONTENTS);
		}
	}

	public void setPrinting(boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | PdfWriter.ALLOW_PRINTING;
		} else {
			pdfrights = pdfrights & (~PdfWriter.ALLOW_PRINTING);
		}
	}

	public void setScreenreaders(boolean allowed) {
		if (allowed == true) {
			pdfrights = pdfrights | PdfWriter.ALLOW_SCREENREADERS;
		} else {
			pdfrights = pdfrights & (~PdfWriter.ALLOW_SCREENREADERS);
		}
	}

	public String getUserPassword() {
		return userpassword;
	}

	public String getOwnerPassword() {
		return ownerpassword;
	}

	public int getRights() {
		return pdfrights;
	}

	public int getEncryptionType() {
		return Rights.ENCTYPE;
	}

	public boolean isAssembly() {
		return (pdfrights & PdfWriter.ALLOW_ASSEMBLY) == PdfWriter.ALLOW_ASSEMBLY;
	}

	public boolean isCopy() {
		return (pdfrights & PdfWriter.ALLOW_COPY) == PdfWriter.ALLOW_COPY;
	}

	public boolean isDegradedPrinting() {
		return (pdfrights & PdfWriter.ALLOW_DEGRADED_PRINTING) == PdfWriter.ALLOW_DEGRADED_PRINTING;
	}

	public boolean isModifyAnnotations() {
		return (pdfrights & PdfWriter.ALLOW_MODIFY_ANNOTATIONS) == PdfWriter.ALLOW_MODIFY_ANNOTATIONS;
	}

	public boolean isModifyContents() {
		return (pdfrights & PdfWriter.ALLOW_MODIFY_CONTENTS) == PdfWriter.ALLOW_MODIFY_CONTENTS;
	}

	public boolean isPrinting() {
		return (pdfrights & PdfWriter.ALLOW_PRINTING) == PdfWriter.ALLOW_PRINTING;
	}

	public boolean isScreenreaders() {
		return (pdfrights & PdfWriter.ALLOW_SCREENREADERS) == PdfWriter.ALLOW_SCREENREADERS;
	}
}
