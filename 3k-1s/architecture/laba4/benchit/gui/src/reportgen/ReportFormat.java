/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: ReportFormat.java
 * Description: manage supported format sizes Copyright: Copyright (c) 2008 Company:ZIH (Center for
 * Information Services and High Performance Computing) Author: Anja Grundmann Last change by:
 * $Author$
 ******************************************************************************/

package reportgen;

import java.util.ArrayList;

import com.lowagie.text.*;

public class ReportFormat {

	private Rectangle documentsize;
	private final ArrayList<String> documentsizes;
	private float leftmargin;
	private float rightmargin;
	private float topmargin;
	private float bottommargin;

	public float cmToPt(float value) {
		return value * 28.34646f;
	}

	public ReportFormat() {
		documentsizes = new ArrayList<String>();
		documentsizes.add("A3");
		documentsizes.add("A4");
		documentsizes.add("A5");
		documentsizes.add("LETTER");
		documentsizes.add("TABLOID");
		documentsize = new Rectangle(PageSize.A4);
		leftmargin = 60.0f;
		rightmargin = 60.0f;
		topmargin = 60.0f;
		bottommargin = 60.0f;
	}

	public ReportFormat(ReportFormat reportformat) {
		documentsizes = new ArrayList<String>();
		documentsizes.add("A3");
		documentsizes.add("A4");
		documentsizes.add("A5");
		documentsizes.add("LETTER");
		documentsizes.add("TABLOID");
		documentsize = reportformat.documentsize;
		leftmargin = reportformat.leftmargin;
		rightmargin = reportformat.rightmargin;
		topmargin = reportformat.topmargin;
		bottommargin = reportformat.bottommargin;
	}

	public ReportFormat(String documentsize) {
		documentsizes = new ArrayList<String>();
		documentsizes.add("A3");
		documentsizes.add("A4");
		documentsizes.add("A5");
		documentsizes.add("LETTER");
		documentsizes.add("TABLOID");
		this.documentsize = null;
		this.setDocumentSize(documentsize);
		leftmargin = 60.0f;
		rightmargin = 60.0f;
		topmargin = 60.0f;
		bottommargin = 60.0f;
	}

	public ReportFormat(String documentsize, float leftmargin, float rightmargin, float topmargin,
			float bottommargin, String munit) {
		documentsizes = new ArrayList<String>();
		documentsizes.add("A3");
		documentsizes.add("A4");
		documentsizes.add("A5");
		documentsizes.add("LETTER");
		documentsizes.add("TABLOID");
		this.documentsize = null;
		this.setDocumentSize(documentsize);

		if (munit.compareTo("cm") == 0) {
			this.leftmargin = cmToPt(leftmargin);
			this.rightmargin = cmToPt(rightmargin);
			this.topmargin = cmToPt(topmargin);
			this.bottommargin = cmToPt(bottommargin);
		} else {
			this.leftmargin = leftmargin;
			this.rightmargin = rightmargin;
			this.topmargin = topmargin;
			this.bottommargin = bottommargin;
		}
	}

	public void setDocumentSize(String documentsize) {
		documentsize = documentsize.toUpperCase();

		switch (documentsizes.indexOf(documentsize)) {
			case 0 :
				this.documentsize = PageSize.A3;
				break;
			case 1 :
				this.documentsize = PageSize.A4;
				break;
			case 2 :
				this.documentsize = PageSize.A5;
				break;
			case 3 :
				this.documentsize = PageSize.LETTER;
				break;
			case 4 :
				this.documentsize = PageSize.TABLOID;
				break;
			default :
				this.documentsize = PageSize.A4;
				break;
		}
	}

	public void setDocumentSize(Rectangle documentsize) {
		this.documentsize = documentsize;
	}

	public void setLeftMargin(float leftmargin, String munit) {
		if (munit.compareTo("cm") == 0) {
			this.leftmargin = cmToPt(leftmargin);
		} else {
			this.leftmargin = leftmargin;
		}
	}

	public void setRightMargin(float rightmargin, String munit) {
		if (munit.compareTo("cm") == 0) {
			this.rightmargin = cmToPt(rightmargin);
		} else {
			this.rightmargin = rightmargin;
		}
	}

	public void setTopMargin(float topmargin, String munit) {
		if (munit.compareTo("cm") == 0) {
			this.topmargin = cmToPt(topmargin);
		} else {
			this.topmargin = topmargin;
		}
	}

	public void setBottomMargin(float bottommargin, String munit) {
		if (munit.compareTo("cm") == 0) {
			this.bottommargin = cmToPt(bottommargin);
		} else {
			this.bottommargin = bottommargin;
		}
	}

	public Rectangle getDocumentSize() {
		return documentsize;
	}

	public float getPtLeftMargin() {
		return leftmargin;
	}

	public float getPtRightMargin() {
		return rightmargin;
	}

	public float getPtTopMargin() {
		return topmargin;
	}

	public float getPtBottomMargin() {
		return bottommargin;
	}

}