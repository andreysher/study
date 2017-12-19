/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: ReportGenerator.java
 * Description: creates the pdf document Copyright: Copyright (c) 2008 Company:ZIH (Center for
 * Information Services and High Performance Computing) Author: Anja Grundmann Last change by:
 * $Author$
 ******************************************************************************/

package reportgen;

import java.awt.Color;
import java.io.*;
import java.util.*;

import javax.xml.parsers.*;

import com.lowagie.text.*;
import com.lowagie.text.pdf.*;

public class ReportGenerator {
	private ReportGen reportgenerator;

	public void addReportGenerator(ReportGen reportgenerator) {
		this.reportgenerator = new ReportGen(reportgenerator);
	}

	public static void main(String args[]) {

		/*
		 * ReportGenerator reportpdf = new ReportGenerator();
		 */

		File xmlfile = null;

		if (args.length > 0) {
			xmlfile = new File(args[0]);
		} else
			return;

		generateReport(xmlfile);

		/*
		 * String path = new String(xmlfile.getPath()); path = path.substring(0, path.lastIndexOf('/') + 1);
		 * FileOutputStream outputfile = null; Document benchitreport = new Document(); Image graphic; float freeforimage =
		 * 0.0f; /* SVGImage svgimage = new SVGImage(); Dimension2D dim; PdfPCell cell;
		 */

		/*
		 * Font reporttableheadfont; int i; int j; int k; int l; int temp, temp2; float widths[]; float numbercolumwidth;
		 * HashMap pagenumbers; int pageorder[]; int p; String pdfname; Calendar cal; int datepart = 0;
		 */
	}

	public static void generateReport(File xmlfile) {
		ReportGenerator reportpdf = new ReportGenerator();

		String path = xmlfile.getParent();

		FileOutputStream outputfile = null;
		Document benchitreport = new Document();

		Image graphic;
		float freeforimage = 0.0f;

		Font reporttableheadfont;

		int i;
		int j;
		int k;
		int l;
		int temp, temp2;

		float widths[];
		float numbercolumwidth;

		HashMap<String, Integer> pagenumbers;

		int pageorder[];
		int p;

		String pdfname;
		Calendar cal;
		int datepart = 0;

		if (xmlfile.length() > 0) {
			try {

				SAXParser parser = SAXParserFactory.newInstance().newSAXParser();
				parser.setProperty("http://apache.org/xml/properties/input-buffer-size", new Integer(
						(int) xmlfile.length()));

				parser.parse(xmlfile, new MyHandler(reportpdf, path));

				benchitreport.setPageSize(reportpdf.reportgenerator.getPDFInfo().getReportFormat()
						.getDocumentSize());

				cal = Calendar.getInstance();
				cal.setTimeInMillis(System.currentTimeMillis());

				pdfname = new String("BenchIT_Report_" + String.valueOf(cal.get(Calendar.YEAR)) + "_");

				datepart = cal.get(Calendar.MONTH) + 1;
				if (datepart < 10) {
					pdfname += "0";
				}
				pdfname += String.valueOf(datepart) + "_";

				datepart = cal.get(Calendar.DAY_OF_MONTH);
				if (datepart < 10) {
					pdfname += "0";
				}
				pdfname += String.valueOf(datepart) + "_";

				datepart = cal.get(Calendar.HOUR_OF_DAY);
				if (datepart < 10) {
					pdfname += "0";
				}
				pdfname += String.valueOf(datepart) + "_";

				datepart = cal.get(Calendar.MINUTE);
				if (datepart < 10) {
					pdfname += "0";
				}
				pdfname += String.valueOf(datepart) + "_";

				datepart = cal.get(Calendar.SECOND);
				if (datepart < 10) {
					pdfname += "0";
				}
				pdfname += String.valueOf(datepart) + "_";

				datepart = cal.get(Calendar.MILLISECOND);
				if (datepart < 100) {
					pdfname += "0";
				}
				if (datepart < 10) {
					pdfname += "0";
				}
				pdfname += String.valueOf(datepart) + ".pdf";

				outputfile = new FileOutputStream(path + File.separator + pdfname);

				PdfWriter writer = PdfWriter.getInstance(benchitreport, outputfile);

				writer.setLinearPageMode();
				writer.setPageEvent(new ReportPdfEvents(reportpdf.reportgenerator));

				benchitreport.addTitle(reportpdf.reportgenerator.getPDFInfo().getTitle());
				benchitreport.addAuthor(reportpdf.reportgenerator.getPDFInfo().getAuthor());
				benchitreport.addCreator(PDFInfo.CREATOR);
				benchitreport.addSubject(PDFInfo.SUBJECT);

				benchitreport.open();

				benchitreport.setMargins(reportpdf.reportgenerator.getPDFInfo().getReportFormat()
						.getPtLeftMargin(), reportpdf.reportgenerator.getPDFInfo().getReportFormat()
						.getPtRightMargin(), reportpdf.reportgenerator.getPDFInfo().getReportFormat()
						.getPtTopMargin(), reportpdf.reportgenerator.getPDFInfo().getReportFormat()
						.getPtBottomMargin());

				pagenumbers = new HashMap<String, Integer>();

				benchitreport.newPage();

				PdfPTable pagetable = new PdfPTable(1);

				Phrase phrase = new Phrase();
				Font reporttilefont = new Font(reportpdf.reportgenerator.getPDFInfo().getHeadingFont());

				String reporttilefontfamily = new String(reporttilefont.getFamilyname());

				Chunk celltext = new Chunk(new String(reportpdf.reportgenerator.getPDFInfo().getTitle()),
						new Font(FontFactory.getFont(reporttilefontfamily, 24.0f, Font.BOLD)));
				phrase.add(celltext);

				celltext = new Chunk(new String(reportpdf.reportgenerator.getPDFInfo().getAuthor()),
						new Font(FontFactory.getFont(reporttilefontfamily, 14.0f, Font.NORMAL)));

				phrase.add("\n\n\n");
				phrase.add(celltext);
				phrase.add("\n\n");

				celltext = new Chunk(new String(reportpdf.reportgenerator.getPDFInfo().getDate()),
						new Font(FontFactory.getFont(reporttilefontfamily, 14.0f, Font.NORMAL)));

				phrase.add(celltext);

				pagetable.getDefaultCell().setVerticalAlignment(Element.ALIGN_MIDDLE);
				pagetable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_CENTER);
				pagetable.getDefaultCell().setFixedHeight(
						benchitreport.getPageSize().getHeight() - benchitreport.topMargin()
								- benchitreport.bottomMargin());

				pagetable.getDefaultCell().setLeading(10.0f, 1.0f);

				pagetable.getDefaultCell().setBorderWidthLeft(0.0f);
				pagetable.getDefaultCell().setBorderWidthRight(0.0f);
				pagetable.getDefaultCell().setBorderWidthTop(0.0f);
				pagetable.getDefaultCell().setBorderWidthBottom(0.0f);

				pagetable.addCell(phrase);

				pagetable.setTotalWidth(benchitreport.getPageSize().getWidth() - benchitreport.leftMargin()
						- benchitreport.rightMargin());

				pagetable.writeSelectedRows(0, -1, benchitreport.leftMargin(), benchitreport.getPageSize()
						.getHeight() - benchitreport.topMargin(), writer.getDirectContent());

				reporttableheadfont = new Font(reportpdf.reportgenerator.getPDFInfo().getTextFont());
				reporttableheadfont.setStyle(Font.BOLD);

				Paragraph paragraph = new Paragraph();

				for (i = 0; i < reportpdf.reportgenerator.getMeasurements().getCount(); i++) {
					paragraph = new Paragraph();
					benchitreport.newPage();

					reportpdf.reportgenerator.getMeasurements().getMeasurement(i)
							.setTitlePageNumber(writer.getPageNumber() - 1);
					paragraph.add(new Chunk(new String((i + 1) + " "
							+ reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getTitle()),
							new Font(reportpdf.reportgenerator.getPDFInfo().getHeadingFont())));

					paragraph.setSpacingAfter(20.0f);

					paragraph.setLeading(reportpdf.reportgenerator.getPDFInfo().getHeadingFont().getSize()
							+ (reportpdf.reportgenerator.getPDFInfo().getHeadingFont().getSize() * 25) / 100);

					benchitreport.add(paragraph);

					freeforimage = benchitreport.getPageSize().getWidth() - benchitreport.leftMargin()
							- benchitreport.rightMargin() - 20.0f;
					graphic = reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getGraphic();
					if (graphic != null) {
						graphic.scaleAbsolute(freeforimage,
								(freeforimage * graphic.getHeight()) / graphic.getWidth());
						graphic.setAlignment(Element.ALIGN_CENTER);

						benchitreport.add(graphic);
					}

					pagetable = new PdfPTable(
							(reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getCountBitFiles()) + 1);

					pagetable.setSpacingBefore(20.0f);

					if (reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getOrigin()
							.compareTo("plot") == 0) {
						pagetable.setHeaderRows(1);

						phrase = new Phrase();
						phrase.add(new Chunk(new String("Architecture Information"), new Font(
								reporttableheadfont)));

						pagetable.addCell(phrase);

						phrase = new Phrase();
						phrase.add(new Chunk(new String("Value"), new Font(reporttableheadfont)));
						pagetable.addCell(phrase);
					} else {
						pagetable.setHeaderRows(2);

						phrase = new Phrase();

						phrase.add(new Chunk(new String("Architecture Information"), new Font(
								reporttableheadfont)));

						pagetable.addCell(phrase);

						for (j = 0; j < reportpdf.reportgenerator.getMeasurements().getMeasurement(i)
								.getCountBitFiles(); j++) {
							phrase = new Phrase();
							phrase
									.add(new Chunk(new String(reportpdf.reportgenerator.getMeasurements()
											.getMeasurement(i).getBitFile(j).getDescription()), new Font(
											reporttableheadfont)));
							pagetable.addCell(phrase);
						}

						phrase = new Phrase();

						phrase.add(new Chunk(new String("Graph Color"), new Font(reportpdf.reportgenerator
								.getPDFInfo().getTextFont())));

						pagetable.addCell(phrase);

						pagetable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_CENTER);

						for (j = 0; j < reportpdf.reportgenerator.getMeasurements().getMeasurement(i)
								.getCountBitFiles(); j++) {
							phrase = new Phrase();

							reporttableheadfont.setColor(new Color(reportpdf.reportgenerator.getMeasurements()
									.getMeasurement(i).getBitFile(j).getGraphColorRGB()));

							phrase.add(new Chunk(new String("*"), new Font(reporttableheadfont)));

							pagetable.addCell(phrase);
						}

						pagetable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_LEFT);

						reporttableheadfont.setColor(Color.BLACK);
					}

					for (k = 0; k < reportpdf.reportgenerator.getMeasurements().getMeasurement(i)
							.getArchInfos().getShownCount(); k++) {
						phrase = new Phrase();

						phrase.add(new Chunk(new String(reportpdf.reportgenerator.getMeasurements()
								.getMeasurement(i).getArchInfos().getShownArchInfo(k)), new Font(
								reportpdf.reportgenerator.getPDFInfo().getTextFont())));

						// System.err.println("ArchInfo for measurement i=" + i + " is " +
						// reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getArchInfos().getShownArchInfo(k)
						// );

						pagetable.addCell(phrase);

						// System.err.println("shownArchInfos for measurement i=" + i + " is " +
						// reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getArchInfos().getShownCount()
						// );

						/*
						 * for( int z=0; z<reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getArchInfos
						 * ().getShownCount(); z++) { System.err.println( z + ":\t" + reportpdf.reportgenerator.getMeasurements
						 * ().getMeasurement(i).getArchInfos().getShownArchInfo(z) ); }
						 */

						for (l = 0; l < reportpdf.reportgenerator.getMeasurements().getMeasurement(i)
								.getCountBitFiles(); l++) {
							phrase = new Phrase();

							// System.err.println("i = " + i + "\tk = " + k + "\tl = " + l);

							// --- ERROR --- //
							// System.err.println("VALUE IS " +
							// reportpdf.reportgenerator.getMeasurements().getMeasurement(i).getBitFile(l).getValue(k)
							// );

							phrase.add(new Chunk(new String(reportpdf.reportgenerator.getMeasurements()
									.getMeasurement(i).getBitFile(l).getValue(k)), new Font(reportpdf.reportgenerator
									.getPDFInfo().getTextFont())));

							pagetable.addCell(phrase);
						}
					}

					pagetable.setTotalWidth(benchitreport.getPageSize().getWidth()
							- benchitreport.leftMargin() - benchitreport.rightMargin() - 20.0f);
					pagetable.setLockedWidth(true);

					benchitreport.add(pagetable);

				}

				benchitreport.newPage();

				reportpdf.reportgenerator.setIsToc(true);

				pagenumbers.put("firsttocpage", new Integer(writer.getPageNumber()));

				paragraph = new Paragraph();

				paragraph.add(new Chunk(new String("Contents"), new Font(reportpdf.reportgenerator
						.getPDFInfo().getHeadingFont())));

				paragraph.setSpacingAfter(20.0f);

				paragraph.setLeading(reportpdf.reportgenerator.getPDFInfo().getHeadingFont().getSize()
						+ (reportpdf.reportgenerator.getPDFInfo().getHeadingFont().getSize() * 25) / 100);

				benchitreport.add(paragraph);

				pagetable = new PdfPTable(3);

				numbercolumwidth = (benchitreport.getPageSize().getWidth() - benchitreport.leftMargin() - benchitreport
						.rightMargin()) * 10 / 100;

				widths = new float[]{
						numbercolumwidth,
						(benchitreport.getPageSize().getWidth() - benchitreport.leftMargin()
								- benchitreport.rightMargin() - numbercolumwidth - numbercolumwidth),
						numbercolumwidth};

				pagetable.getDefaultCell().setBorderWidthLeft(0.0f);
				pagetable.getDefaultCell().setBorderWidthRight(0.0f);
				pagetable.getDefaultCell().setBorderWidthTop(0.0f);
				pagetable.getDefaultCell().setBorderWidthBottom(0.0f);
				pagetable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_RIGHT);

				phrase = new Phrase();
				phrase.add(new Chunk(new String("page"), new Font(reportpdf.reportgenerator.getPDFInfo()
						.getTextFont())));

				pagetable.addCell("");
				pagetable.addCell("");
				pagetable.addCell(phrase);
				pagetable.addCell("");
				pagetable.addCell("");
				pagetable.addCell("");

				pagetable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_LEFT);

				for (i = 0; i < reportpdf.reportgenerator.getMeasurements().getCount(); i++) {
					phrase = new Phrase();
					phrase.add(new Chunk(new String(String.valueOf(i + 1)), new Font(
							reportpdf.reportgenerator.getPDFInfo().getTextFont())));
					pagetable.addCell(phrase);

					phrase = new Phrase();
					phrase.add(new Chunk(new String(reportpdf.reportgenerator.getMeasurements()
							.getMeasurement(i).getTitle()), new Font(reportpdf.reportgenerator.getPDFInfo()
							.getTextFont())));
					pagetable.addCell(phrase);

					pagetable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_RIGHT);
					pagetable.getDefaultCell().setVerticalAlignment(Element.ALIGN_BOTTOM);

					phrase = new Phrase();
					phrase.add(new Chunk(new String(String.valueOf(reportpdf.reportgenerator
							.getMeasurements().getMeasurement(i).getTitlePageNumber())), new Font(
							reportpdf.reportgenerator.getPDFInfo().getTextFont())));
					pagetable.addCell(phrase);

					pagetable.getDefaultCell().setHorizontalAlignment(Element.ALIGN_LEFT);
					pagetable.getDefaultCell().setVerticalAlignment(Element.ALIGN_TOP);

					pagetable.addCell("");
					pagetable.addCell("");
					pagetable.addCell("");

				}

				pagetable.setTotalWidth(widths);
				pagetable.setLockedWidth(true);

				benchitreport.add(pagetable);

				benchitreport.newPage();

				pagenumbers.put("lasttocpage", new Integer(writer.reorderPages(null)));

				temp = new Integer(pagenumbers.get("firsttocpage").toString()).intValue() - 2;
				pagenumbers.put("countcontentpages", new Integer(temp));

				temp = new Integer(pagenumbers.get("lasttocpage").toString()).intValue()
						- new Integer(pagenumbers.get("firsttocpage").toString()).intValue() + 1;
				pagenumbers.put("counttocpages", new Integer(temp));

				pageorder = new int[new Integer(pagenumbers.get("lasttocpage").toString()).intValue()];

				pageorder[0] = 1;

				temp = new Integer(pagenumbers.get("counttocpages").toString()).intValue();
				for (p = 0; p < temp; p++) {
					temp2 = new Integer(pagenumbers.get("firsttocpage").toString()).intValue() + p;
					pageorder[p + 1] = temp2;
				}

				temp = new Integer(pagenumbers.get("countcontentpages").toString()).intValue();
				for (p = 0; p < temp; p++) {
					temp2 = new Integer(pagenumbers.get("counttocpages").toString()).intValue() + p + 1;
					pageorder[temp2] = p + 2;
				}

				writer.reorderPages(pageorder);

				benchitreport.close();
				System.out.println("PDF-File \"" + pdfname
						+ "\" successfully built into your output directory " + path);
			} catch (DocumentException de) {
				System.err.println(de.getMessage());
				de.printStackTrace();
				System.out.println("ERROR: Invalid XML-File!");
			} catch (Exception e) {
				e.printStackTrace();
				System.err.println(e.getMessage());
				System.out.println("ERROR: Invalid XML-File!");
			}
		} else {
			System.out.println("ERROR: Invalid XML-File!");
		}
	}

}
