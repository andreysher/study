/******************************************************************************
 * B e n c h I T - Performance Measurement for Scientific Applications Title: MyHandler.java
 * Description: process the xml-file and gather all relevant information Copyright: Copyright (c)
 * 2008 Company:ZIH (Center for Information Services and High Performance Computing) Author: Anja
 * Grundmann Last change by: $Author$
 ******************************************************************************/

package reportgen;

import java.io.File;
import java.util.ArrayList;

import org.xml.sax.*;
import org.xml.sax.helpers.DefaultHandler;

import com.lowagie.text.*;

public class MyHandler extends DefaultHandler {
	private final ReportGenerator reportpdf;

	private ReportGen reportgenerator;
	private PDFInfo pdfinfo;
	private ReportFormat reportformat;
	private Rights rights;
	private Measurements measurements;
	private Measurement measurement;
	private ArchInfos archinfos;
	private BitFile bitfile;
	private int xmltagtype;
	private boolean ispdfinfo;
	private boolean istext;
	// private Font font;
	private String fontfamily;
	private int fontstyle;
	private float fontsize;
	private final ArrayList<String> fontfamilies;
	private String dump;
	private final String path;
	private int bitfilecounter;
	// private String munit;
	// additon: we want to handle seperate units for every margin
	private String leftMarginUnit;
	private String rightMarginUnit;
	private String topMarginUnit;
	private String bottomMarginUnit;

	private static final String xmltags[] = {"benchit:ReportGenerator", "benchit:PDFInfo",
			"benchit:Author", "benchit:Title", "benchit:Date", "benchit:Rights", "benchit:Right",
			"benchit:Format", "benchit:Documentsize", "benchit:Margins", "benchit:LeftMargin",
			"benchit:RightMargin", "benchit:TopMargin", "benchit:BottomMargin", "benchit:Text",
			"benchit:FontType", "benchit:FontSize", "benchit:Heading", "benchit:Measurements",
			"benchit:Measurement", "benchit:Graphic", "benchit:ArchInfos", "benchit:Info",
			"benchit:BitFiles", "benchit:BitFile", "benchit:Color", "benchit:Description", "benchit:Dump"};

	private static final int TAG_REPORTGENERATOR = 0;
	private static final int TAG_PDFINFO = 1;
	private static final int TAG_AUTHOR = 2;
	private static final int TAG_TITLE = 3;
	private static final int TAG_DATE = 4;
	private static final int TAG_RIGHTS = 5;
	private static final int TAG_RIGHT = 6;
	private static final int TAG_FORMAT = 7;
	private static final int TAG_DOCUMENTSIZE = 8;
	// private static final int TAG_MARGINS = 9;
	private static final int TAG_LEFTMARGIN = 10;
	private static final int TAG_RIGHTMARGIN = 11;
	private static final int TAG_TOPMARGIN = 12;
	private static final int TAG_BOTTOMMARGIN = 13;
	private static final int TAG_TEXT = 14;
	private static final int TAG_FONTTYPE = 15;
	private static final int TAG_FONTSIZE = 16;
	private static final int TAG_HEADING = 17;
	private static final int TAG_MEASUREMENTS = 18;
	private static final int TAG_MEASUREMENT = 19;
	private static final int TAG_GRAPHIC = 20;
	private static final int TAG_ARCHINFOS = 21;
	private static final int TAG_INFO = 22;
	// private static final int TAG_BITFILES = 23;
	private static final int TAG_BITFILE = 24;
	private static final int TAG_COLOR = 25;
	private static final int TAG_DESCRIPTION = 26;
	private static final int TAG_DUMP = 27;

	public MyHandler(ReportGenerator reportpdf, String path) {
		fontfamilies = new ArrayList<String>();
		fontfamilies.add("COURIER");
		fontfamilies.add("HELVETICA");
		fontfamilies.add("TIMES_ROMAN");

		ispdfinfo = false;
		istext = false;
		fontfamily = new String();
		fontstyle = 0;
		fontsize = 0.0f;

		this.reportpdf = reportpdf;
		xmltagtype = -1;

		dump = new String();

		this.path = path;
		// System.err.println("Path is: " + this.path);
	}

	public void setFontStyle(Attributes a) {

		if (a.getValue("bold").compareTo("true") == 0) {
			fontstyle = fontstyle | Font.BOLD;
		}
		if (a.getValue("italic").compareTo("true") == 0) {
			fontstyle = fontstyle | Font.ITALIC;

		}
		if (a.getValue("underline").compareTo("true") == 0 && istext != true) {
			fontstyle = fontstyle | Font.UNDERLINE;
		}
	}

	/*
	 * public void mySeperateDumpData() { int archindex, boi_index, eoi_index; final String boi =
	 * "beginofidentifierstrings"; final String eoi = "endofidentifierstrings"; String allIdentifiers; boi_index =
	 * dump.indexOf( boi ); eoi_index = dump.indexOf( eoi ); allIdentifiers = dump.substring( boi_index, eoi_index ); for(
	 * archindex = 0; archindex < archinfos.getCount(); archindex++ ) { } }
	 */

	public void seperateDumpData() {
		int i;
		// int j;
		int foundindex;
		int foundbiindex;
		int foundequalto;
		int last;
		int next;
		String archinfo[];
		boolean addfoundvalue;
		boolean foundlast;
		boolean changearchinfo;

		foundbiindex = dump.indexOf("beginofidentifierstrings");

		// System.err.println("dump = " + dump);

		// pattern of all investigated entries: archinfos=value

		for (i = 0; i < archinfos.getCount(); i++) {
			foundindex = -1;
			addfoundvalue = false;
			changearchinfo = false;

			// System.err.println("-------------\nSuche nach " + archinfos.getArchInfo(i));

			do {
				foundindex = dump.indexOf(archinfos.getArchInfo(i).concat("="), foundindex + 1);
				// System.err.println("foundindex = " + foundindex);

				if (foundindex > -1) {
					foundequalto = -1;
					last = -1;
					next = -1;
					archinfo = new String[2];
					foundlast = false;

					foundequalto = dump.indexOf("=", foundindex);

					// System.err.println(dump.substring(foundindex, foundindex+100));
					// System.err.println("foundequalto = " + foundequalto);

					if (foundequalto > 0) {
						last = dump.lastIndexOf("\n", foundindex);
						// System.err.println("last = " + last);

						// if value consists of characters then the value can be
						// found between quotes
						// this means: archinfo="value"
						if (dump.charAt(foundequalto + 1) == '\"') {
							next = foundequalto + 1;
							// System.err.println("nach gleichheitszeichen kommt anfuehrungszeichen");
							// System.err.println("next = " + next);
							// System.err.println("suche nach zweiten gleichheitszeichen");
							do {
								// System.err.println("schleife");
								next = dump.indexOf("\"", next + 1);
								// System.err.println("next = " + next);
								if (dump.charAt(next - 1) != '\\') {
									foundlast = true;
									// System.err.println("letztes anfuehrungszeichen gefunden");
									// System.err.println("next = " + next);
								}
							} while (foundlast != true);
						} else {
							// but there are also numerical values without quotes
							// corresponding pattern: archinfo=value
							next = dump.indexOf("\n", foundindex);
							// System.err.println("nach gleichheitszeichen kommt KEIN anfuehrungszeichen");
							// System.err.println("next auf zeilenende setzen");
							// System.err.println("next = " + next);
						}

						if (last < 0) {
							last = 0;
						}
						if (next < 0) {
							next = dump.length() - 1;
						}

						// System.err.println("nach anpassung\n" + "next = " + next + "\nlast = " + last);

						archinfo = dump.substring(last, next).trim().split("=", 2);
						// System.err.println("archinfo0 = " + archinfo[0] + "   archinfo1 = " + archinfo[1]);

						if (archinfos.getArchInfo(i).compareToIgnoreCase(archinfo[0].trim()) == 0) {
							archinfo[1] = archinfo[1].trim();
							if (archinfo[1].length() > 0) {
								if (archinfo[1].charAt(0) == '\"') {
									archinfo[1] = archinfo[1].substring(1);
								}

								archinfo[1] = archinfo[1].replaceAll("\n", "");
								archinfo[1] = archinfo[1].replaceAll("\t", "");
								if (foundindex > foundbiindex && addfoundvalue != false && changearchinfo != true
										&& bitfilecounter == 1) {
									archinfos.addShownArchInfo(i, new String(archinfo[1]));
									changearchinfo = true;
								} else {
									if (addfoundvalue != true) {
										bitfile.addValue(i, new String(archinfo[1]));
										// System.err.println("bitfile.addValue( " + i + ", " + archinfo[1] + " )");
										addfoundvalue = true;
									}
									if (bitfilecounter != 1) {
										changearchinfo = true;
									}
								}
							}

							// special case: entries with numerical values (e.g. numberofprocessors) can have
							// an empty string at this point if the values is not set
							// this looks like: numberofprocessors=
							if (archinfo[1].length() == 0) {
								if (foundindex > foundbiindex && addfoundvalue != false && changearchinfo != true
										&& bitfilecounter == 1) {
									archinfos.addShownArchInfo(i, new String(archinfo[1]));
									changearchinfo = true;
								} else {
									if (addfoundvalue != true) {
										bitfile.addValue(i, new String(archinfo[1]));
										// System.err.println("bitfile.addValue( " + i + ", " + archinfo[1] + " )");
										addfoundvalue = true;
									}
									if (bitfilecounter != 1) {
										changearchinfo = true;
									}
								}
							}
						}
					}
				} else {
					// System.err.println("foundindex < 0");
					if (addfoundvalue != true) {
						bitfile.addValue(i, new String());
						// System.err.println("bitfile.addValue( " + i + ", newString() )");
						addfoundvalue = true;
					}
					if (changearchinfo != true) {
						archinfos.addShownArchInfo(i, new String(archinfos.getArchInfo(i)));
						changearchinfo = true;
					}
				}
			} while (foundindex >= 0 && (addfoundvalue != true || changearchinfo != true));
		}
	}

	@Override
	public void characters(char[] buf, int offset, int len) throws SAXException {
		String s = new String(buf, offset, len);

		if (s.length() > 0) {
			switch (xmltagtype) {
				case TAG_AUTHOR :
					pdfinfo.setAuthor(s);
					break;
				case TAG_TITLE :
					if (ispdfinfo == true) {
						pdfinfo.setTitle(s);
					} else {
						measurement.setTitle(s);
					}
					break;
				case TAG_DATE :
					pdfinfo.setDate(s);
					break;
				case TAG_RIGHT :
					rights.setRight(s, true);
					break;
				case TAG_DOCUMENTSIZE :
					reportformat.setDocumentSize(s);
					break;
				case TAG_LEFTMARGIN :
					// reportformat.setLeftMargin(Float.parseFloat(s), munit);
					reportformat.setLeftMargin(Float.parseFloat(s), leftMarginUnit);
					break;
				case TAG_RIGHTMARGIN :
					// reportformat.setRightMargin(Float.parseFloat(s), munit);
					reportformat.setRightMargin(Float.parseFloat(s), rightMarginUnit);
					break;
				case TAG_TOPMARGIN :
					// reportformat.setTopMargin(Float.parseFloat(s), munit);
					reportformat.setTopMargin(Float.parseFloat(s), topMarginUnit);
					break;
				case TAG_BOTTOMMARGIN :
					// reportformat.setBottomMargin(Float.parseFloat(s), munit);
					reportformat.setBottomMargin(Float.parseFloat(s), bottomMarginUnit);
					break;
				case TAG_FONTTYPE :
					fontfamily = s;
					break;
				case TAG_FONTSIZE :
					fontsize = Float.parseFloat(s);
					break;
				case TAG_GRAPHIC :
					try {
						if (path.length() > 0) {
							s = path + File.separator + s;
						}
						measurement.setGraphic(Image.getInstance(new File(s).toURI().toURL()));
					} catch (Exception e) {
						// e.printStackTrace();
						// TODO: Fix svg loading!
					}
					break;
				case TAG_INFO :
					archinfos.addArchInfo(s);
					break;
				case TAG_COLOR :
					bitfile.setGraphColorRGB(Integer.parseInt(s));
					break;
				case TAG_DESCRIPTION :
					bitfile.setDescription(s);
					break;
				case TAG_DUMP :
					dump = dump.concat(s);
					break;
				default :
					break;
			}
		}
	}

	private int findTag(String tag) {
		int i, retval = -1;

		for (i = 0; i < xmltags.length; i++) {
			if (tag.equals(xmltags[i]) == true) {
				retval = i;
				i = xmltags.length;
			}
		}
		return retval;
	}

	@Override
	public void startElement(String uri, String localName, String qName, Attributes attributes)
			throws SAXException {
		xmltagtype = findTag(qName);

		switch (xmltagtype) {
			case TAG_REPORTGENERATOR :
				reportgenerator = new ReportGen();
				break;
			case TAG_PDFINFO :
				pdfinfo = new PDFInfo();
				ispdfinfo = true;
				break;
			case TAG_RIGHTS :
				rights = new Rights();
				rights.setUserPassword(attributes.getValue("userpassword"));
				rights.setOwnerPassword(attributes.getValue("ownerpassword"));
				break;
			case TAG_FORMAT :
				reportformat = new ReportFormat();
				break;
			/*
			 * case TAG_MARGINS: munit = new String(attributes.getValue("unit")); break;
			 */
			case TAG_LEFTMARGIN :
				leftMarginUnit = new String(attributes.getValue("unit"));
				break;
			case TAG_RIGHTMARGIN :
				rightMarginUnit = new String(attributes.getValue("unit"));
				break;
			case TAG_TOPMARGIN :
				topMarginUnit = new String(attributes.getValue("unit"));
				break;
			case TAG_BOTTOMMARGIN :
				bottomMarginUnit = new String(attributes.getValue("unit"));
				break;
			case TAG_TEXT :
				istext = true;
				break;
			case TAG_FONTTYPE :
				setFontStyle(attributes);
				break;
			case TAG_MEASUREMENTS :
				measurements = new Measurements();
				break;
			case TAG_MEASUREMENT :
				measurement = new Measurement();
				measurement.setOrigin(attributes.getValue("origin"));
				bitfilecounter = 0;
				break;
			case TAG_ARCHINFOS :
				archinfos = new ArchInfos();
				break;
			case TAG_BITFILE :
				bitfile = new BitFile();
				bitfilecounter++;
				break;
			case TAG_DUMP :
				dump = new String();
				break;
			default :
				break;
		}
	}

	@Override
	public void endElement(String uri, String localName, String qName) throws SAXException {
		xmltagtype = findTag(qName);

		switch (xmltagtype) {
			case TAG_REPORTGENERATOR :
				reportpdf.addReportGenerator(reportgenerator);
				break;
			case TAG_PDFINFO :
				reportgenerator.setPDFInfo(pdfinfo);
				ispdfinfo = false;
				break;
			case TAG_RIGHTS :
				pdfinfo.setRights(rights);
				break;
			case TAG_FORMAT :
				pdfinfo.setReportFormat(reportformat);
				break;
			case TAG_TEXT :
				pdfinfo.setTextFont(new Font(FontFactory.getFont(fontfamily, fontsize, fontstyle)));
				fontfamily = new String();
				fontstyle = 0;
				fontsize = 0.0f;
				istext = false;
				break;
			case TAG_HEADING :
				pdfinfo.setHeadingFont(new Font(FontFactory.getFont(fontfamily, fontsize, fontstyle)));
				fontfamily = new String();
				fontstyle = 0;
				fontsize = 0.0f;
				break;
			case TAG_MEASUREMENTS :
				reportgenerator.setMeasurements(measurements);
				break;
			case TAG_MEASUREMENT :
				measurement.setArchInfos(archinfos);
				measurements.addMeasurement(measurement);
				break;
			case TAG_ARCHINFOS :
				break;
			case TAG_BITFILE :
				measurement.addBitfile(bitfile);
				break;
			case TAG_DUMP :
				seperateDumpData();
				break;
			default :
				break;
		}
		xmltagtype = -1;
	}
}
