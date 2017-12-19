package org.syntax.jedit.tokenmarker;

/*
 * FortranTokenMarker.java - FORTRAN token marker Copyright (C) 1998, 1999 Slava Pestov You may use
 * and modify this package for any purpose. Redistribution is permitted, in both source and binary
 * form, provided that this notice remains intact in all source distributions of this package.
 */

import javax.swing.text.Segment;

import org.syntax.jedit.KeywordMap;

/**
 * FORTRAN token marker.
 * 
 * @author Slava Pestov
 * @version $Id: FortranTokenMarker.java,v 1.1 2008/05/29 11:57:21 domke Exp $
 */
public class FortranTokenMarker extends TokenMarker {
	public FortranTokenMarker() {
		this(true, getKeywords());
	}

	public FortranTokenMarker(boolean cpp, KeywordMap keywords) {
		this.cpp = cpp;
		this.keywords = keywords;
	}

	@Override
	public byte markTokensImpl(byte token, Segment line, int lineIndex) {
		char[] array = line.array;
		int offset = line.offset;
		lastOffset = offset;
		lastKeyword = offset;
		int length = line.count + offset;
		boolean backslash = false;

		loop : for (int i = offset; i < length; i++) {
			int i1 = (i + 1);

			char c = array[i];
			if (c == '\\') {
				backslash = !backslash;
				continue;
			}

			switch (token) {
				case Token.NULL :
					switch (c) {
						case '#' :
							if (backslash) {
								backslash = false;
							} else if (cpp) {
								if (doKeyword(line, i, c)) {
									break;
								}
								addToken(i - lastOffset, token);
								addToken(length - i, Token.KEYWORD2);
								lastOffset = lastKeyword = length;
								break loop;
							}
							break;
						case '"' :
							doKeyword(line, i, c);
							if (backslash) {
								backslash = false;
							} else {
								addToken(i - lastOffset, token);
								token = Token.LITERAL1;
								lastOffset = lastKeyword = i;
							}
							break;
						case '\'' :
							doKeyword(line, i, c);
							if (backslash) {
								backslash = false;
							} else {
								addToken(i - lastOffset, token);
								token = Token.LITERAL2;
								lastOffset = lastKeyword = i;
							}
							break;
						case ':' :
							if (lastKeyword == offset) {
								if (doKeyword(line, i, c)) {
									break;
								}
								backslash = false;
								addToken(i1 - lastOffset, Token.LABEL);
								lastOffset = lastKeyword = i1;
							} else if (doKeyword(line, i, c)) {
								break;
							}
							break;
						case '!' :
							backslash = false;
							doKeyword(line, i, c);
							if (length - i > 1) {
								switch (array[i1]) {
									case ' ' :
										addToken(i - lastOffset, token);
										addToken(length - i, Token.COMMENT1);
										lastOffset = lastKeyword = length;
										break loop;
									default :
										addToken(i - lastOffset, token);
										addToken(length - i, Token.COMMENT1);
										lastOffset = lastKeyword = length;
										break loop;
								}
							}
							break;
						case 'C' :
							backslash = false;
							doKeyword(line, i, c);
							if (length - i > 1) {
								switch (array[i1]) {
									case ' ' :
										addToken(i - lastOffset, token);
										addToken(length - i, Token.COMMENT1);
										lastOffset = lastKeyword = length;
										break loop;
									case 'C' :
										addToken(i - lastOffset, token);
										addToken(length - i, Token.COMMENT1);
										lastOffset = lastKeyword = length;
										break loop;
								}
							}
							break;
						default :
							backslash = false;
							if (!Character.isLetterOrDigit(c) && c != '_') {
								doKeyword(line, i, c);
							}
							break;
					}
					break;
				case Token.COMMENT1 :
				case Token.COMMENT2 :
				case Token.LITERAL1 :
					if (backslash) {
						backslash = false;
					} else if (c == '"') {
						addToken(i1 - lastOffset, token);
						token = Token.NULL;
						lastOffset = lastKeyword = i1;
					}
					break;
				case Token.LITERAL2 :
					if (backslash) {
						backslash = false;
					} else if (c == '\'') {
						addToken(i1 - lastOffset, Token.LITERAL1);
						token = Token.NULL;
						lastOffset = lastKeyword = i1;
					}
					break;
				default :
					throw new InternalError("Invalid state: " + token);
			}
		}

		if (token == Token.NULL) {
			doKeyword(line, length, '\0');
		}

		switch (token) {
			case Token.LITERAL1 :
			case Token.LITERAL2 :
				addToken(length - lastOffset, Token.INVALID);
				token = Token.NULL;
				break;
			case Token.KEYWORD2 :
				addToken(length - lastOffset, token);
				if (!backslash) {
					token = Token.NULL;
				}
			default :
				addToken(length - lastOffset, token);
				break;
		}

		return token;
	}

	public static KeywordMap getKeywords() {
		if (cKeywords == null) {
			cKeywords = new KeywordMap(false);
			cKeywords.add("allocatable", Token.KEYWORD1);
			cKeywords.add("Allocatable", Token.KEYWORD1);
			cKeywords.add("ALLOCATABLE", Token.KEYWORD1);
			cKeywords.add("allocate", Token.KEYWORD1);
			cKeywords.add("Allocate", Token.KEYWORD1);
			cKeywords.add("ALLOCATE", Token.KEYWORD1);
			cKeywords.add("assign", Token.KEYWORD1);
			cKeywords.add("Assign", Token.KEYWORD1);
			cKeywords.add("ASSIGN", Token.KEYWORD1);
			cKeywords.add("assignment", Token.KEYWORD1);
			cKeywords.add("Assignment", Token.KEYWORD1);
			cKeywords.add("ASSIGNMENT", Token.KEYWORD1);
			cKeywords.add("block", Token.KEYWORD3);
			cKeywords.add("Block", Token.KEYWORD3);
			cKeywords.add("BLOCK", Token.KEYWORD3);
			cKeywords.add("data", Token.KEYWORD3);
			cKeywords.add("Data", Token.KEYWORD3);
			cKeywords.add("DATA", Token.KEYWORD3);
			cKeywords.add("call", Token.KEYWORD1);
			cKeywords.add("Call", Token.KEYWORD1);
			cKeywords.add("CALL", Token.KEYWORD1);
			cKeywords.add("case", Token.KEYWORD1);
			cKeywords.add("Case", Token.KEYWORD1);
			cKeywords.add("CASE", Token.KEYWORD1);
			cKeywords.add("character", Token.KEYWORD3);
			cKeywords.add("Character", Token.KEYWORD3);
			cKeywords.add("CHARACTER", Token.KEYWORD3);
			cKeywords.add("common", Token.KEYWORD3);
			cKeywords.add("Common", Token.KEYWORD3);
			cKeywords.add("COMMON", Token.KEYWORD3);
			cKeywords.add("complex", Token.KEYWORD3);
			cKeywords.add("Complex", Token.KEYWORD3);
			cKeywords.add("COMPLEX", Token.KEYWORD3);
			cKeywords.add("contains", Token.KEYWORD1);
			cKeywords.add("Contains", Token.KEYWORD1);
			cKeywords.add("CONTAINS", Token.KEYWORD1);
			cKeywords.add("continue", Token.KEYWORD1);
			cKeywords.add("Continue", Token.KEYWORD1);
			cKeywords.add("CONTINUE", Token.KEYWORD1);
			cKeywords.add("cycle", Token.KEYWORD1);
			cKeywords.add("Cycle", Token.KEYWORD1);
			cKeywords.add("CYCLE", Token.KEYWORD1);
			cKeywords.add("deallocate", Token.KEYWORD1);
			cKeywords.add("Deallocate", Token.KEYWORD1);
			cKeywords.add("DEALLOCATE", Token.KEYWORD1);
			cKeywords.add("default", Token.KEYWORD1);
			cKeywords.add("Default", Token.KEYWORD1);
			cKeywords.add("DEFAULT", Token.KEYWORD1);
			cKeywords.add("do", Token.KEYWORD1);
			cKeywords.add("Do", Token.KEYWORD1);
			cKeywords.add("DO", Token.KEYWORD1);
			cKeywords.add("double", Token.KEYWORD3);
			cKeywords.add("Double", Token.KEYWORD3);
			cKeywords.add("DOUBLE", Token.KEYWORD3);
			cKeywords.add("precision", Token.KEYWORD3);
			cKeywords.add("Precision", Token.KEYWORD3);
			cKeywords.add("PRECISION", Token.KEYWORD3);
			cKeywords.add("else", Token.KEYWORD1);
			cKeywords.add("Else", Token.KEYWORD1);
			cKeywords.add("ELSE", Token.KEYWORD1);
			cKeywords.add("elsewhere", Token.KEYWORD1);
			cKeywords.add("Elsewhere", Token.KEYWORD1);
			cKeywords.add("ELSEWHERE", Token.KEYWORD1);
			cKeywords.add("end", Token.KEYWORD1);
			cKeywords.add("End", Token.KEYWORD1);
			cKeywords.add("END", Token.KEYWORD1);
			cKeywords.add("entry", Token.KEYWORD1);
			cKeywords.add("Entry", Token.KEYWORD1);
			cKeywords.add("ENTRY", Token.KEYWORD1);
			cKeywords.add("equivalence", Token.KEYWORD1);
			cKeywords.add("Equivalence", Token.KEYWORD1);
			cKeywords.add("EQUIVALENCE", Token.KEYWORD1);
			cKeywords.add("exit", Token.KEYWORD1);
			cKeywords.add("Exit", Token.KEYWORD1);
			cKeywords.add("EXIT", Token.KEYWORD1);
			cKeywords.add("external", Token.KEYWORD1);
			cKeywords.add("External", Token.KEYWORD1);
			cKeywords.add("EXTERNAL", Token.KEYWORD1);
			cKeywords.add("function", Token.KEYWORD1);
			cKeywords.add("Function", Token.KEYWORD1);
			cKeywords.add("FUNCTION", Token.KEYWORD1);
			cKeywords.add("go", Token.KEYWORD1);
			cKeywords.add("Go", Token.KEYWORD1);
			cKeywords.add("GO", Token.KEYWORD1);
			cKeywords.add("to", Token.KEYWORD1);
			cKeywords.add("To", Token.KEYWORD1);
			cKeywords.add("TO", Token.KEYWORD1);
			cKeywords.add("if", Token.KEYWORD1);
			cKeywords.add("If", Token.KEYWORD1);
			cKeywords.add("IF", Token.KEYWORD1);
			cKeywords.add("implicit", Token.KEYWORD1);
			cKeywords.add("Implicit", Token.KEYWORD1);
			cKeywords.add("IMPLICIT", Token.KEYWORD1);
			cKeywords.add("in", Token.KEYWORD1);
			cKeywords.add("In", Token.KEYWORD1);
			cKeywords.add("IN", Token.KEYWORD1);
			cKeywords.add("inout", Token.KEYWORD1);
			cKeywords.add("Inout", Token.KEYWORD1);
			cKeywords.add("INOUT", Token.KEYWORD1);
			cKeywords.add("integer", Token.KEYWORD3);
			cKeywords.add("Integer", Token.KEYWORD3);
			cKeywords.add("INTEGER", Token.KEYWORD3);
			cKeywords.add("intent", Token.KEYWORD1);
			cKeywords.add("Intent", Token.KEYWORD1);
			cKeywords.add("INTENT", Token.KEYWORD1);
			cKeywords.add("interface", Token.KEYWORD3);
			cKeywords.add("Interface", Token.KEYWORD3);
			cKeywords.add("INTERFACE", Token.KEYWORD3);
			cKeywords.add("intrinsic", Token.KEYWORD1);
			cKeywords.add("Intrinsic", Token.KEYWORD1);
			cKeywords.add("INTRINSIC", Token.KEYWORD1);
			cKeywords.add("kind", Token.KEYWORD3);
			cKeywords.add("Kind", Token.KEYWORD3);
			cKeywords.add("KIND", Token.KEYWORD3);
			cKeywords.add("len", Token.KEYWORD3);
			cKeywords.add("Len", Token.KEYWORD3);
			cKeywords.add("LEN", Token.KEYWORD3);
			cKeywords.add("logical", Token.KEYWORD3);
			cKeywords.add("Logical", Token.KEYWORD3);
			cKeywords.add("LOGICAL", Token.KEYWORD3);
			cKeywords.add("module", Token.KEYWORD1);
			cKeywords.add("Module", Token.KEYWORD1);
			cKeywords.add("MODULE", Token.KEYWORD1);
			cKeywords.add("namelist", Token.KEYWORD1);
			cKeywords.add("Namelist", Token.KEYWORD1);
			cKeywords.add("NAMELIST", Token.KEYWORD1);
			cKeywords.add("nullify", Token.KEYWORD1);
			cKeywords.add("Nullify", Token.KEYWORD1);
			cKeywords.add("NULLIFY", Token.KEYWORD1);
			cKeywords.add("only", Token.KEYWORD1);
			cKeywords.add("Only", Token.KEYWORD1);
			cKeywords.add("ONLY", Token.KEYWORD1);
			cKeywords.add("operator", Token.KEYWORD1);
			cKeywords.add("Operator", Token.KEYWORD1);
			cKeywords.add("OPERATOR", Token.KEYWORD1);
			cKeywords.add("optional", Token.KEYWORD1);
			cKeywords.add("Optional", Token.KEYWORD1);
			cKeywords.add("OPTIONAL", Token.KEYWORD1);
			cKeywords.add("out", Token.KEYWORD1);
			cKeywords.add("Out", Token.KEYWORD1);
			cKeywords.add("OUT", Token.KEYWORD1);
			cKeywords.add("parameter", Token.KEYWORD1);
			cKeywords.add("Parameter", Token.KEYWORD1);
			cKeywords.add("PARAMETER", Token.KEYWORD1);
			cKeywords.add("pause", Token.KEYWORD1);
			cKeywords.add("Pause", Token.KEYWORD1);
			cKeywords.add("PAUSE", Token.KEYWORD1);
			cKeywords.add("pointer", Token.KEYWORD1);
			cKeywords.add("Pointer", Token.KEYWORD1);
			cKeywords.add("POINTER", Token.KEYWORD1);
			cKeywords.add("private", Token.KEYWORD1);
			cKeywords.add("Private", Token.KEYWORD1);
			cKeywords.add("PRIVATE", Token.KEYWORD1);
			cKeywords.add("program", Token.KEYWORD1);
			cKeywords.add("Program", Token.KEYWORD1);
			cKeywords.add("PROGRAM", Token.KEYWORD1);
			cKeywords.add("public", Token.KEYWORD1);
			cKeywords.add("Public", Token.KEYWORD1);
			cKeywords.add("PUBLIC", Token.KEYWORD1);
			cKeywords.add("real", Token.KEYWORD3);
			cKeywords.add("Real", Token.KEYWORD3);
			cKeywords.add("REAL", Token.KEYWORD3);
			cKeywords.add("recursive", Token.KEYWORD1);
			cKeywords.add("Recursive", Token.KEYWORD1);
			cKeywords.add("RECURSIVE", Token.KEYWORD1);
			cKeywords.add("result", Token.KEYWORD1);
			cKeywords.add("Result", Token.KEYWORD1);
			cKeywords.add("RESULT", Token.KEYWORD1);
			cKeywords.add("return", Token.KEYWORD1);
			cKeywords.add("Return", Token.KEYWORD1);
			cKeywords.add("RETURN", Token.KEYWORD1);
			cKeywords.add("save", Token.KEYWORD1);
			cKeywords.add("Save", Token.KEYWORD1);
			cKeywords.add("SAVE", Token.KEYWORD1);
			cKeywords.add("select", Token.KEYWORD1);
			cKeywords.add("Select", Token.KEYWORD1);
			cKeywords.add("SELECT", Token.KEYWORD1);
			cKeywords.add("stop", Token.KEYWORD1);
			cKeywords.add("Stop", Token.KEYWORD1);
			cKeywords.add("STOP", Token.KEYWORD1);
			cKeywords.add("subroutine", Token.KEYWORD1);
			cKeywords.add("Subroutine", Token.KEYWORD1);
			cKeywords.add("SUBROUTINE", Token.KEYWORD1);
			cKeywords.add("target", Token.KEYWORD1);
			cKeywords.add("Target", Token.KEYWORD1);
			cKeywords.add("TARGET", Token.KEYWORD1);
			cKeywords.add("then", Token.KEYWORD1);
			cKeywords.add("Then", Token.KEYWORD1);
			cKeywords.add("THEN", Token.KEYWORD1);
			cKeywords.add("type", Token.KEYWORD1);
			cKeywords.add("Type", Token.KEYWORD1);
			cKeywords.add("TYPE", Token.KEYWORD1);
			cKeywords.add("use", Token.KEYWORD1);
			cKeywords.add("Use", Token.KEYWORD1);
			cKeywords.add("USE", Token.KEYWORD1);
			cKeywords.add("where", Token.KEYWORD1);
			cKeywords.add("Where", Token.KEYWORD1);
			cKeywords.add("WHERE", Token.KEYWORD1);
			cKeywords.add("while", Token.KEYWORD1);
			cKeywords.add("While", Token.KEYWORD1);
			cKeywords.add("WHILE", Token.KEYWORD1);
			cKeywords.add("backspace", Token.KEYWORD1);
			cKeywords.add("Backspace", Token.KEYWORD1);
			cKeywords.add("BACKSPACE", Token.KEYWORD1);
			cKeywords.add("close", Token.KEYWORD1);
			cKeywords.add("Close", Token.KEYWORD1);
			cKeywords.add("CLOSE", Token.KEYWORD1);
			cKeywords.add("endfile", Token.KEYWORD1);
			cKeywords.add("Endfile", Token.KEYWORD1);
			cKeywords.add("ENDFILE", Token.KEYWORD1);
			cKeywords.add("format", Token.KEYWORD1);
			cKeywords.add("Format", Token.KEYWORD1);
			cKeywords.add("FORMAT", Token.KEYWORD1);
			cKeywords.add("inquire", Token.KEYWORD1);
			cKeywords.add("Inquire", Token.KEYWORD1);
			cKeywords.add("INQUIRE", Token.KEYWORD1);
			cKeywords.add("open", Token.KEYWORD1);
			cKeywords.add("Open", Token.KEYWORD1);
			cKeywords.add("OPEN", Token.KEYWORD1);
			cKeywords.add("print", Token.KEYWORD1);
			cKeywords.add("Print", Token.KEYWORD1);
			cKeywords.add("PRINT", Token.KEYWORD1);
			cKeywords.add("read", Token.KEYWORD1);
			cKeywords.add("Read", Token.KEYWORD1);
			cKeywords.add("READ", Token.KEYWORD1);
			cKeywords.add("rewind", Token.KEYWORD1);
			cKeywords.add("Rewind", Token.KEYWORD1);
			cKeywords.add("REWIND", Token.KEYWORD1);
			cKeywords.add("write", Token.KEYWORD1);
			cKeywords.add("Write", Token.KEYWORD1);
			cKeywords.add("WRITE", Token.KEYWORD1);
			cKeywords.add("not", Token.KEYWORD1);
			cKeywords.add("Not", Token.KEYWORD1);
			cKeywords.add("NOT", Token.KEYWORD1);
			cKeywords.add("and", Token.KEYWORD1);
			cKeywords.add("And", Token.KEYWORD1);
			cKeywords.add("AND", Token.KEYWORD1);
			cKeywords.add("or", Token.KEYWORD1);
			cKeywords.add("Or", Token.KEYWORD1);
			cKeywords.add("OR", Token.KEYWORD1);
			cKeywords.add("eqv", Token.KEYWORD1);
			cKeywords.add("Eqv", Token.KEYWORD1);
			cKeywords.add("EQV", Token.KEYWORD1);
			cKeywords.add("neqv", Token.KEYWORD1);
			cKeywords.add("Neqv", Token.KEYWORD1);
			cKeywords.add("NEQV", Token.KEYWORD1);
			cKeywords.add("eq", Token.KEYWORD1);
			cKeywords.add("Eq", Token.KEYWORD1);
			cKeywords.add("EQ", Token.KEYWORD1);
			cKeywords.add("ne", Token.KEYWORD1);
			cKeywords.add("Ne", Token.KEYWORD1);
			cKeywords.add("NE", Token.KEYWORD1);
			cKeywords.add("lt", Token.KEYWORD1);
			cKeywords.add("Lt", Token.KEYWORD1);
			cKeywords.add("LT", Token.KEYWORD1);
			cKeywords.add("gt", Token.KEYWORD1);
			cKeywords.add("Gt", Token.KEYWORD1);
			cKeywords.add("GT", Token.KEYWORD1);
			cKeywords.add("le", Token.KEYWORD1);
			cKeywords.add("Le", Token.KEYWORD1);
			cKeywords.add("LE", Token.KEYWORD1);
			cKeywords.add("ge", Token.KEYWORD1);
			cKeywords.add("Ge", Token.KEYWORD1);
			cKeywords.add("GE", Token.KEYWORD1);
			cKeywords.add("true", Token.LITERAL2);
			cKeywords.add("True", Token.LITERAL2);
			cKeywords.add("TRUE", Token.LITERAL2);
			cKeywords.add("false", Token.LITERAL2);
			cKeywords.add("False", Token.LITERAL2);
			cKeywords.add("FALSE", Token.LITERAL2);
			cKeywords.add("ichar", Token.KEYWORD1);
			cKeywords.add("Ichar", Token.KEYWORD1);
			cKeywords.add("ICHAR", Token.KEYWORD1);
			cKeywords.add("iachar", Token.KEYWORD1);
			cKeywords.add("Iachar", Token.KEYWORD1);
			cKeywords.add("IACHAR", Token.KEYWORD1);
			cKeywords.add("int", Token.KEYWORD1);
			cKeywords.add("Int", Token.KEYWORD1);
			cKeywords.add("INT", Token.KEYWORD1);
			cKeywords.add("dble", Token.KEYWORD1);
			cKeywords.add("Dble", Token.KEYWORD1);
			cKeywords.add("DBLE", Token.KEYWORD1);
			cKeywords.add("cmplx", Token.KEYWORD1);
			cKeywords.add("Cmplx", Token.KEYWORD1);
			cKeywords.add("CMPLX", Token.KEYWORD1);
			cKeywords.add("char", Token.KEYWORD1);
			cKeywords.add("Char", Token.KEYWORD1);
			cKeywords.add("CHAR", Token.KEYWORD1);
			cKeywords.add("achar", Token.KEYWORD1);
			cKeywords.add("Achar", Token.KEYWORD1);
			cKeywords.add("ACHAR", Token.KEYWORD1);
			cKeywords.add("selected_int_kind", Token.KEYWORD1);
			cKeywords.add("Selected_int_kind", Token.KEYWORD1);
			cKeywords.add("SELECTED_INT_KIND", Token.KEYWORD1);
			cKeywords.add("selected_real_kind", Token.KEYWORD1);
			cKeywords.add("Selected_real_kind", Token.KEYWORD1);
			cKeywords.add("SELECTED_REAL_KIND", Token.KEYWORD1);
			cKeywords.add("aint", Token.KEYWORD1);
			cKeywords.add("Aint", Token.KEYWORD1);
			cKeywords.add("AINT", Token.KEYWORD1);
			cKeywords.add("anint", Token.KEYWORD1);
			cKeywords.add("Anint", Token.KEYWORD1);
			cKeywords.add("ANINT", Token.KEYWORD1);
			cKeywords.add("nint", Token.KEYWORD1);
			cKeywords.add("Nint", Token.KEYWORD1);
			cKeywords.add("NINT", Token.KEYWORD1);
			cKeywords.add("ceiling", Token.KEYWORD1);
			cKeywords.add("Ceiling", Token.KEYWORD1);
			cKeywords.add("CEILING", Token.KEYWORD1);
			cKeywords.add("floor", Token.KEYWORD1);
			cKeywords.add("Floor", Token.KEYWORD1);
			cKeywords.add("FLOOR", Token.KEYWORD1);
			cKeywords.add("abs", Token.KEYWORD1);
			cKeywords.add("Abs", Token.KEYWORD1);
			cKeywords.add("ABS", Token.KEYWORD1);
			cKeywords.add("mod", Token.KEYWORD1);
			cKeywords.add("Mod", Token.KEYWORD1);
			cKeywords.add("MOD", Token.KEYWORD1);
			cKeywords.add("modulo", Token.KEYWORD1);
			cKeywords.add("Modulo", Token.KEYWORD1);
			cKeywords.add("MODULO", Token.KEYWORD1);
			cKeywords.add("sign", Token.KEYWORD1);
			cKeywords.add("Sign", Token.KEYWORD1);
			cKeywords.add("SIGN", Token.KEYWORD1);
			cKeywords.add("dim", Token.KEYWORD1);
			cKeywords.add("Dim", Token.KEYWORD1);
			cKeywords.add("DIM", Token.KEYWORD1);
			cKeywords.add("max", Token.KEYWORD1);
			cKeywords.add("Max", Token.KEYWORD1);
			cKeywords.add("MAX", Token.KEYWORD1);
			cKeywords.add("min", Token.KEYWORD1);
			cKeywords.add("Min", Token.KEYWORD1);
			cKeywords.add("MIN", Token.KEYWORD1);
			cKeywords.add("aimag", Token.KEYWORD1);
			cKeywords.add("Aimag", Token.KEYWORD1);
			cKeywords.add("AIMAG", Token.KEYWORD1);
			cKeywords.add("conjg", Token.KEYWORD1);
			cKeywords.add("Conjg", Token.KEYWORD1);
			cKeywords.add("CONJG", Token.KEYWORD1);
			cKeywords.add("sqrt", Token.KEYWORD1);
			cKeywords.add("Sqrt", Token.KEYWORD1);
			cKeywords.add("SQRT", Token.KEYWORD1);
			cKeywords.add("exp", Token.KEYWORD1);
			cKeywords.add("Exp", Token.KEYWORD1);
			cKeywords.add("EXP", Token.KEYWORD1);
			cKeywords.add("log", Token.KEYWORD1);
			cKeywords.add("Log", Token.KEYWORD1);
			cKeywords.add("LOG", Token.KEYWORD1);
			cKeywords.add("log10", Token.KEYWORD1);
			cKeywords.add("Log10", Token.KEYWORD1);
			cKeywords.add("LOG10", Token.KEYWORD1);
			cKeywords.add("sin", Token.KEYWORD1);
			cKeywords.add("Sin", Token.KEYWORD1);
			cKeywords.add("SIN", Token.KEYWORD1);
			cKeywords.add("cos", Token.KEYWORD1);
			cKeywords.add("Cos", Token.KEYWORD1);
			cKeywords.add("COS", Token.KEYWORD1);
			cKeywords.add("tan", Token.KEYWORD1);
			cKeywords.add("Tan", Token.KEYWORD1);
			cKeywords.add("TAN", Token.KEYWORD1);
			cKeywords.add("asin", Token.KEYWORD1);
			cKeywords.add("Asin", Token.KEYWORD1);
			cKeywords.add("ASIN", Token.KEYWORD1);
			cKeywords.add("acos", Token.KEYWORD1);
			cKeywords.add("Acos", Token.KEYWORD1);
			cKeywords.add("ACOS", Token.KEYWORD1);
			cKeywords.add("atan", Token.KEYWORD1);
			cKeywords.add("Atan", Token.KEYWORD1);
			cKeywords.add("ATAN", Token.KEYWORD1);
			cKeywords.add("atan2", Token.KEYWORD1);
			cKeywords.add("Atan2", Token.KEYWORD1);
			cKeywords.add("ATAN2", Token.KEYWORD1);
			cKeywords.add("sinh", Token.KEYWORD1);
			cKeywords.add("Sinh", Token.KEYWORD1);
			cKeywords.add("SINH", Token.KEYWORD1);
			cKeywords.add("cosh", Token.KEYWORD1);
			cKeywords.add("Cosh", Token.KEYWORD1);
			cKeywords.add("COSH", Token.KEYWORD1);
			cKeywords.add("tanh", Token.KEYWORD1);
			cKeywords.add("Tanh", Token.KEYWORD1);
			cKeywords.add("TANH", Token.KEYWORD1);
			cKeywords.add("lge", Token.KEYWORD1);
			cKeywords.add("Lge", Token.KEYWORD1);
			cKeywords.add("LGE", Token.KEYWORD1);
			cKeywords.add("lgt", Token.KEYWORD1);
			cKeywords.add("Lgt", Token.KEYWORD1);
			cKeywords.add("LGT", Token.KEYWORD1);
			cKeywords.add("lle", Token.KEYWORD1);
			cKeywords.add("Lle", Token.KEYWORD1);
			cKeywords.add("LLE", Token.KEYWORD1);
			cKeywords.add("llt", Token.KEYWORD1);
			cKeywords.add("Llt", Token.KEYWORD1);
			cKeywords.add("LLT", Token.KEYWORD1);
			cKeywords.add("len_trim", Token.KEYWORD1);
			cKeywords.add("Len_trim", Token.KEYWORD1);
			cKeywords.add("LEN_TRIM", Token.KEYWORD1);
			cKeywords.add("trim", Token.KEYWORD1);
			cKeywords.add("Trim", Token.KEYWORD1);
			cKeywords.add("TRIM", Token.KEYWORD1);
			cKeywords.add("adjustl", Token.KEYWORD1);
			cKeywords.add("Adjustl", Token.KEYWORD1);
			cKeywords.add("ADJUSTL", Token.KEYWORD1);
			cKeywords.add("adjustr", Token.KEYWORD1);
			cKeywords.add("Adjustr", Token.KEYWORD1);
			cKeywords.add("ADJUSTR", Token.KEYWORD1);
			cKeywords.add("repeat", Token.KEYWORD1);
			cKeywords.add("Repeat", Token.KEYWORD1);
			cKeywords.add("REPEAT", Token.KEYWORD1);
			cKeywords.add("index", Token.KEYWORD1);
			cKeywords.add("Index", Token.KEYWORD1);
			cKeywords.add("INDEX", Token.KEYWORD1);
			cKeywords.add("scan", Token.KEYWORD1);
			cKeywords.add("Scan", Token.KEYWORD1);
			cKeywords.add("SCAN", Token.KEYWORD1);
			cKeywords.add("verify", Token.KEYWORD1);
			cKeywords.add("Verify", Token.KEYWORD1);
			cKeywords.add("VERIFY", Token.KEYWORD1);
			cKeywords.add("reshape", Token.KEYWORD1);
			cKeywords.add("Reshape", Token.KEYWORD1);
			cKeywords.add("RESHAPE", Token.KEYWORD1);
			cKeywords.add("merge", Token.KEYWORD1);
			cKeywords.add("Merge", Token.KEYWORD1);
			cKeywords.add("MERGE", Token.KEYWORD1);
			cKeywords.add("pack", Token.KEYWORD1);
			cKeywords.add("Pack", Token.KEYWORD1);
			cKeywords.add("PACK", Token.KEYWORD1);
			cKeywords.add("unpack", Token.KEYWORD1);
			cKeywords.add("Unpack", Token.KEYWORD1);
			cKeywords.add("UNPACK", Token.KEYWORD1);
			cKeywords.add("spread", Token.KEYWORD1);
			cKeywords.add("Spread", Token.KEYWORD1);
			cKeywords.add("SPREAD", Token.KEYWORD1);
			cKeywords.add("allocated", Token.KEYWORD1);
			cKeywords.add("Allocated", Token.KEYWORD1);
			cKeywords.add("ALLOCATED", Token.KEYWORD1);
			cKeywords.add("lbound", Token.KEYWORD1);
			cKeywords.add("Lbound", Token.KEYWORD1);
			cKeywords.add("LBOUND", Token.KEYWORD1);
			cKeywords.add("ubound", Token.KEYWORD1);
			cKeywords.add("Ubound", Token.KEYWORD1);
			cKeywords.add("UBOUND", Token.KEYWORD1);
			cKeywords.add("all", Token.KEYWORD1);
			cKeywords.add("All", Token.KEYWORD1);
			cKeywords.add("ALL", Token.KEYWORD1);
			cKeywords.add("any", Token.KEYWORD1);
			cKeywords.add("Any", Token.KEYWORD1);
			cKeywords.add("ANY", Token.KEYWORD1);
			cKeywords.add("count", Token.KEYWORD1);
			cKeywords.add("Count", Token.KEYWORD1);
			cKeywords.add("COUNT", Token.KEYWORD1);
			cKeywords.add("size", Token.KEYWORD1);
			cKeywords.add("Size", Token.KEYWORD1);
			cKeywords.add("SIZE", Token.KEYWORD1);
			cKeywords.add("shape", Token.KEYWORD1);
			cKeywords.add("Shape", Token.KEYWORD1);
			cKeywords.add("SHAPE", Token.KEYWORD1);
			cKeywords.add("minval", Token.KEYWORD1);
			cKeywords.add("Minval", Token.KEYWORD1);
			cKeywords.add("MINVAL", Token.KEYWORD1);
			cKeywords.add("maxval", Token.KEYWORD1);
			cKeywords.add("Maxval", Token.KEYWORD1);
			cKeywords.add("MAXVAL", Token.KEYWORD1);
			cKeywords.add("minloc", Token.KEYWORD1);
			cKeywords.add("Minloc", Token.KEYWORD1);
			cKeywords.add("MINLOC", Token.KEYWORD1);
			cKeywords.add("maxloc", Token.KEYWORD1);
			cKeywords.add("Maxloc", Token.KEYWORD1);
			cKeywords.add("MAXLOC", Token.KEYWORD1);
			cKeywords.add("dot_product", Token.KEYWORD1);
			cKeywords.add("Dot_product", Token.KEYWORD1);
			cKeywords.add("DOT_PRODUCT", Token.KEYWORD1);
			cKeywords.add("matmul", Token.KEYWORD1);
			cKeywords.add("Matmul", Token.KEYWORD1);
			cKeywords.add("MATMUL", Token.KEYWORD1);
			cKeywords.add("transpose", Token.KEYWORD1);
			cKeywords.add("Transpose", Token.KEYWORD1);
			cKeywords.add("TRANSPOSE", Token.KEYWORD1);
			cKeywords.add("cshift", Token.KEYWORD1);
			cKeywords.add("Cshift", Token.KEYWORD1);
			cKeywords.add("CSHIFT", Token.KEYWORD1);
			cKeywords.add("eoshift", Token.KEYWORD1);
			cKeywords.add("Eoshift", Token.KEYWORD1);
			cKeywords.add("EOSHIFT", Token.KEYWORD1);
			cKeywords.add("sum", Token.KEYWORD1);
			cKeywords.add("Sum", Token.KEYWORD1);
			cKeywords.add("SUM", Token.KEYWORD1);
			cKeywords.add("produkt", Token.KEYWORD1);
			cKeywords.add("Produkt", Token.KEYWORD1);
			cKeywords.add("PRODUKT", Token.KEYWORD1);
			cKeywords.add("associated", Token.KEYWORD1);
			cKeywords.add("Associated", Token.KEYWORD1);
			cKeywords.add("ASSOCIATED", Token.KEYWORD1);
			cKeywords.add("null", Token.LITERAL2);
			cKeywords.add("Null", Token.LITERAL2);
			cKeywords.add("NULL", Token.LITERAL2);
			cKeywords.add("cpu_time", Token.KEYWORD1);
			cKeywords.add("Cpu_time", Token.KEYWORD1);
			cKeywords.add("CPU_TIME", Token.KEYWORD1);
			cKeywords.add("date_and_time", Token.KEYWORD1);
			cKeywords.add("Date_and_time", Token.KEYWORD1);
			cKeywords.add("DATE_AND_TIME", Token.KEYWORD1);
			cKeywords.add("mvbits", Token.KEYWORD1);
			cKeywords.add("Mvbits", Token.KEYWORD1);
			cKeywords.add("MVBITS", Token.KEYWORD1);
			cKeywords.add("random_number", Token.KEYWORD1);
			cKeywords.add("Random_number", Token.KEYWORD1);
			cKeywords.add("RANDOM_NUMBER", Token.KEYWORD1);
			cKeywords.add("random_seed", Token.KEYWORD1);
			cKeywords.add("Random_seed", Token.KEYWORD1);
			cKeywords.add("RANDOM_SEED", Token.KEYWORD1);
			cKeywords.add("system_clock", Token.KEYWORD1);
			cKeywords.add("System_clock", Token.KEYWORD1);
			cKeywords.add("SYSTEM_CLOCK", Token.KEYWORD1);
		}
		return cKeywords;
	}

	// private members
	private static KeywordMap cKeywords;

	private boolean cpp;
	private KeywordMap keywords;
	private int lastOffset;
	private int lastKeyword;

	private boolean doKeyword(Segment line, int i, char c) {
		int i1 = i + 1;

		int len = i - lastKeyword;
		byte id = keywords.lookup(line, lastKeyword, len);
		if (id != Token.NULL) {
			if (lastKeyword != lastOffset) {
				addToken(lastKeyword - lastOffset, Token.NULL);
			}
			addToken(len, id);
			lastOffset = i;
		}
		lastKeyword = i1;
		return false;
	}
}
