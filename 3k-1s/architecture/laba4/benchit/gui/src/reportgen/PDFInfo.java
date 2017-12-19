/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: PDFInfo.java
 * Description: contains all pdf settings (like font family or size) Copyright: Copyright (c) 2008
 * Company:ZIH (Center for Information Services and High Performance Computing) Author: Anja
 * Grundmann Last change by: $Author$
 ******************************************************************************/

package reportgen;

import java.util.*;

import com.lowagie.text.*;

public class PDFInfo {
	private String author;
	private String title;
	private String date;
	private Rights rights;
	private ReportFormat reportformat;
	private Font textfont;
	private Font headingfont;

	public static final String CREATOR = "BenchIT Report Generator";
	public static final String SUBJECT = "BenchIT Measurement Report";

	private String getFormatedSystemDate() {
		String formatedsystemdate = "";
		Calendar cal = Calendar.getInstance();
		cal.setTime(new Date());
		formatedsystemdate += String.valueOf(cal.get(Calendar.DAY_OF_MONTH));
		formatedsystemdate += "." + String.valueOf(cal.get(Calendar.MONTH));
		formatedsystemdate += "." + String.valueOf(cal.get(Calendar.YEAR));

		return formatedsystemdate;
	}

	public PDFInfo() {
		author = new String("BenchIT User");
		title = new String("BenchIT Measurement Results");
		date = getFormatedSystemDate();
		rights = new Rights();
		reportformat = new ReportFormat();
		textfont = new Font(FontFactory.getFont(FontFactory.TIMES_ROMAN, 12.0f, Font.NORMAL));
		headingfont = new Font(FontFactory.getFont(FontFactory.TIMES_ROMAN, 16.0f, Font.BOLD));
	}

	public PDFInfo(PDFInfo pdfinfo) {
		author = pdfinfo.author;
		title = pdfinfo.title;
		date = pdfinfo.date;
		rights = pdfinfo.rights;
		reportformat = pdfinfo.reportformat;
		textfont = pdfinfo.textfont;
		headingfont = pdfinfo.headingfont;
	}

	public PDFInfo(String author, String title, String date) {
		this.author = new String(author);
		this.title = new String(title);
		this.date = new String(date);
		rights = new Rights();
		reportformat = new ReportFormat();
		textfont = new Font(FontFactory.getFont(FontFactory.TIMES_ROMAN, 12.0f, Font.NORMAL));
		headingfont = new Font(FontFactory.getFont(FontFactory.TIMES_ROMAN, 16.0f, Font.BOLD));

	}

	public PDFInfo(String author, String title, String date, Rights rights,
			ReportFormat reportformat, Font textfont, Font headingfont) {
		this.author = new String(author);
		this.title = new String(title);
		this.date = new String(date);
		this.rights = new Rights(rights);
		this.reportformat = new ReportFormat(reportformat);
		this.textfont = new Font(textfont);
		this.headingfont = new Font(headingfont);
	}

	public void setAuthor(String author) {
		this.author = author;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public void setDate(String date) {
		this.date = date;
	}

	public void setRights(Rights rights) {
		this.rights = rights;
	}

	public void setReportFormat(ReportFormat reportformat) {
		this.reportformat = reportformat;
	}

	public void setTextFont(Font textfont) {
		this.textfont = textfont;
	}

	public void setHeadingFont(Font headingfont) {
		this.headingfont = headingfont;
	}

	public String getAuthor() {
		return author;
	}

	public String getTitle() {
		return title;
	}

	public String getDate() {
		return date;
	}

	public Rights getRights() {
		return rights;
	}

	public ReportFormat getReportFormat() {
		return reportformat;
	}

	public Font getTextFont() {
		return textfont;
	}

	public Font getHeadingFont() {
		return headingfont;
	}

}