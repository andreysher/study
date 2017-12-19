/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: ReportGen.java
 * Description: handles the content of generated pdf document Copyright: Copyright (c) 2008
 * Company:ZIH (Center for Information Services and High Performance Computing) Author: Anja
 * Grundmann Last change by: $Author$
 ******************************************************************************/

package reportgen;

public class ReportGen {

	private PDFInfo pdfinfo;
	private Measurements measurements;
	private boolean istoc;

	public ReportGen() {
		pdfinfo = new PDFInfo();
		measurements = new Measurements();
		istoc = false;
	}

	public ReportGen(ReportGen reportgenerator) {
		pdfinfo = reportgenerator.pdfinfo;
		measurements = reportgenerator.measurements;
		istoc = false;
	}

	public ReportGen(PDFInfo pdfinfo, Measurements measurements) {
		this.pdfinfo = new PDFInfo(pdfinfo);
		this.measurements = new Measurements(measurements);
		istoc = false;
	}

	public void setPDFInfo(PDFInfo pdfinfo) {
		this.pdfinfo = pdfinfo;
	}

	public void setMeasurements(Measurements measurements) {
		this.measurements = measurements;
	}

	public void setIsToc(boolean istoc) {
		this.istoc = istoc;
	}

	public PDFInfo getPDFInfo() {
		return pdfinfo;
	}

	public Measurements getMeasurements() {
		return measurements;
	}

	public boolean getIsToc() {
		return istoc;
	}

}