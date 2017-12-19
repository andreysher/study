/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: ReportPdfEvents.java
 * Description: handles the events during pdf generation Copyright: Copyright (c) 2008 Company:ZIH
 * (Center for Information Services and High Performance Computing) Author: Anja Grundmann Last
 * change by: $Author$
 ******************************************************************************/

package reportgen;

import java.io.File;

import system.BIGInterface;

import com.lowagie.text.*;
import com.lowagie.text.pdf.*;
// import com.lowagie.text.pdf.BaseFont;

public class ReportPdfEvents extends PdfPageEventHelper {

	private PdfPTable headtable;
	private PdfPTable headtable_firstpage;
	private PdfPTable foottable;
	private Phrase phrase;
	private Chunk head_foot;
	private String head_foot_font_family;
	private float head_foot_font_size;
	private float headheight;
	private float footheight;
	private float widths[];
	private static final float SPACINGTO_HEADFOOT = 10.0f;
	// private BaseFont bf = null;
	private Image benchit_logo;
	private Font head_foot_font;
	private final ReportGen reportgenerator;

	public ReportPdfEvents(ReportGen reportgenerator) {
		this.reportgenerator = reportgenerator;
	}

	@Override
	public void onOpenDocument(PdfWriter writer, Document benchitreport) {
		try {

			benchit_logo = Image.getInstance(BIGInterface.getInstance().getImgPath() + File.separator
					+ "benchIT_logo.png");

			benchit_logo.scaleAbsolute(120, 48);

			widths = new float[]{
					reportgenerator.getPDFInfo().getReportFormat().getDocumentSize().getWidth()
							- reportgenerator.getPDFInfo().getReportFormat().getPtLeftMargin()
							- reportgenerator.getPDFInfo().getReportFormat().getPtRightMargin() - 130.0f, 130.0f};

			headtable = new PdfPTable(2);

			headtable.getDefaultCell().setVerticalAlignment(Element.ALIGN_BOTTOM);
			headtable.getDefaultCell().setBorderWidthLeft(0.0f);
			headtable.getDefaultCell().setBorderWidthRight(0.0f);
			headtable.getDefaultCell().setBorderWidthTop(0.0f);

			phrase = new Phrase();

			head_foot_font = new Font(reportgenerator.getPDFInfo().getTextFont());

			head_foot_font_family = new String(head_foot_font.getFamilyname());

			head_foot_font_size = head_foot_font.getSize();

			Chunk head_foot = new Chunk(new String(reportgenerator.getPDFInfo().getTitle()), new Font(
					FontFactory.getFont(head_foot_font_family, head_foot_font_size, Font.NORMAL)));

			phrase.add(head_foot);

			headtable.addCell(phrase);

			headtable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_RIGHT);
			headtable.addCell(new Phrase(new Chunk(benchit_logo, 0, 0)));
			headtable.setTotalWidth(widths);
			headheight = headtable.getTotalHeight();

			foottable = new PdfPTable(1);

			foottable.getDefaultCell().setVerticalAlignment(Element.ALIGN_BOTTOM);
			foottable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_RIGHT);

			phrase = new Phrase();

			head_foot = new Chunk(new String(String.valueOf(writer.getPageNumber())), new Font(
					FontFactory.getFont(head_foot_font_family, head_foot_font_size, Font.NORMAL)));

			phrase.add(head_foot);

			foottable.addCell(phrase);

			foottable.setTotalWidth(reportgenerator.getPDFInfo().getReportFormat().getDocumentSize()
					.getWidth()
					- reportgenerator.getPDFInfo().getReportFormat().getPtLeftMargin()
					- reportgenerator.getPDFInfo().getReportFormat().getPtRightMargin());
			footheight = foottable.getTotalHeight();
			reportgenerator
					.getPDFInfo()
					.getReportFormat()
					.setTopMargin(
							reportgenerator.getPDFInfo().getReportFormat().getPtTopMargin() + headheight
									+ SPACINGTO_HEADFOOT, "pt");
			reportgenerator
					.getPDFInfo()
					.getReportFormat()
					.setBottomMargin(
							reportgenerator.getPDFInfo().getReportFormat().getPtBottomMargin() + footheight
									+ SPACINGTO_HEADFOOT, "pt");

		} catch (Exception e) {
			throw new ExceptionConverter(e);
		}
	}

	@Override
	public void onEndPage(PdfWriter writer, Document benchitreport) {
		if (writer.getPageNumber() != 1) {
			try {
				headtable.writeSelectedRows(0, -1, reportgenerator.getPDFInfo().getReportFormat()
						.getPtLeftMargin(), reportgenerator.getPDFInfo().getReportFormat().getDocumentSize()
						.getHeight()
						- reportgenerator.getPDFInfo().getReportFormat().getPtTopMargin()
						+ headheight
						+ SPACINGTO_HEADFOOT, writer.getDirectContent());

				if (reportgenerator.getIsToc() != true) {
					foottable = new PdfPTable(1);

					foottable.getDefaultCell().setVerticalAlignment(Element.ALIGN_BOTTOM);
					foottable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_RIGHT);
					foottable.getDefaultCell().setBorderWidthLeft(0.0f);
					foottable.getDefaultCell().setBorderWidthRight(0.0f);
					foottable.getDefaultCell().setBorderWidthBottom(0.0f);

					phrase = new Phrase();

					head_foot = new Chunk(new String(String.valueOf(writer.getPageNumber() - 1)), new Font(
							FontFactory.getFont(head_foot_font_family, head_foot_font_size, Font.NORMAL)));
					phrase.add(head_foot);

					foottable.addCell(phrase);

					foottable.setTotalWidth(reportgenerator.getPDFInfo().getReportFormat().getDocumentSize()
							.getWidth()
							- reportgenerator.getPDFInfo().getReportFormat().getPtLeftMargin()
							- reportgenerator.getPDFInfo().getReportFormat().getPtRightMargin());

					foottable.writeSelectedRows(0, -1, reportgenerator.getPDFInfo().getReportFormat()
							.getPtLeftMargin(), reportgenerator.getPDFInfo().getReportFormat()
							.getPtBottomMargin()
							- SPACINGTO_HEADFOOT, writer.getDirectContent());
				}

			} catch (Exception e) {
				throw new ExceptionConverter(e);
			}
		} else {
			try {
				headtable_firstpage = new PdfPTable(1);

				headtable_firstpage.getDefaultCell().setVerticalAlignment(Element.ALIGN_BOTTOM);
				headtable_firstpage.getDefaultCell().setHorizontalAlignment(Element.ALIGN_RIGHT);
				headtable_firstpage.getDefaultCell().setFixedHeight(headheight);
				headtable_firstpage.getDefaultCell().setBorderWidthLeft(0.0f);
				headtable_firstpage.getDefaultCell().setBorderWidthRight(0.0f);
				headtable_firstpage.getDefaultCell().setBorderWidthTop(0.0f);
				headtable_firstpage.getDefaultCell().setBorderWidthBottom(0.0f);
				headtable_firstpage.addCell(new Phrase(new Chunk(benchit_logo, 0, 0)));
				headtable_firstpage.setTotalWidth(reportgenerator.getPDFInfo().getReportFormat()
						.getDocumentSize().getWidth()
						- reportgenerator.getPDFInfo().getReportFormat().getPtLeftMargin()
						- reportgenerator.getPDFInfo().getReportFormat().getPtRightMargin());

				headtable_firstpage.writeSelectedRows(0, -1, reportgenerator.getPDFInfo().getReportFormat()
						.getPtLeftMargin(), reportgenerator.getPDFInfo().getReportFormat().getDocumentSize()
						.getHeight()
						- reportgenerator.getPDFInfo().getReportFormat().getPtTopMargin()
						+ headheight
						+ SPACINGTO_HEADFOOT, writer.getDirectContent());
			} catch (Exception e) {
				throw new ExceptionConverter(e);
			}
		}
	}

}
