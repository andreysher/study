package org.syntax.jedit.tokenmarker;

import javax.swing.text.Segment;

import org.syntax.jedit.KeywordMap;

public class CudaTokenMarker extends CTokenMarker {

	public CudaTokenMarker() {
		super(true, getKeywords());
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
						case '<' :
						case '>' :
							if (length - i > 2) {
								if (array[i1] == c && array[i1 + 1] == c) {
									if (doKeyword(line, i, c))
										break;
									addToken(i - lastOffset, token);
									addToken(3, Token.LABEL);
									i += 2;
									lastOffset = lastKeyword = i + 1;
								}
							}
							break;
						case '#' :
							if (backslash)
								backslash = false;
							else if (cpp) {
								if (doKeyword(line, i, c))
									break;
								addToken(i - lastOffset, token);
								token = Token.KEYWORD2;
								lastOffset = lastKeyword = i;
								break;
							}
							break;
						case '"' :
							doKeyword(line, i, c);
							if (backslash)
								backslash = false;
							else {
								addToken(i - lastOffset, token);
								token = Token.LITERAL1;
								lastOffset = lastKeyword = i;
							}
							break;
						case '\'' :
							doKeyword(line, i, c);
							if (backslash)
								backslash = false;
							else {
								addToken(i - lastOffset, token);
								token = Token.LITERAL2;
								lastOffset = lastKeyword = i;
							}
							break;
						case ':' :
							if (lastKeyword == offset) {
								if (doKeyword(line, i, c))
									break;
								backslash = false;
								addToken(i1 - lastOffset, Token.LABEL);
								lastOffset = lastKeyword = i1;
							} else if (doKeyword(line, i, c))
								break;
							break;
						case '/' :
							backslash = false;
							doKeyword(line, i, c);
							if (length - i > 1) {
								switch (array[i1]) {
									case '*' :
										addToken(i - lastOffset, token);
										lastOffset = lastKeyword = i;
										if (length - i > 2 && array[i + 2] == '*')
											token = Token.COMMENT2;
										else
											token = Token.COMMENT1;
										break;
									case '/' :
										addToken(i - lastOffset, token);
										addToken(length - i, Token.COMMENT1);
										lastOffset = lastKeyword = length;
										break loop;
								}
							}
							break;
						default :
							backslash = false;
							if (!Character.isLetterOrDigit(c) && c != '_')
								doKeyword(line, i, c);
							break;
					}
					break;
				case Token.COMMENT1 :
				case Token.COMMENT2 :
					backslash = false;
					if (c == '*' && length - i > 1) {
						if (array[i1] == '/') {
							i++;
							addToken((i + 1) - lastOffset, token);
							token = Token.NULL;
							lastOffset = lastKeyword = i + 1;
						}
					}
					break;
				case Token.LITERAL1 :
					if (backslash)
						backslash = false;
					else if (c == '"') {
						addToken(i1 - lastOffset, token);
						token = Token.NULL;
						lastOffset = lastKeyword = i1;
					}
					break;
				case Token.LITERAL2 :
					if (backslash)
						backslash = false;
					else if (c == '\'') {
						addToken(i1 - lastOffset, Token.LITERAL1);
						token = Token.NULL;
						lastOffset = lastKeyword = i1;
					}
					break;
				case Token.KEYWORD2 :
					if (backslash)
						backslash = false;
					break;
				default :
					throw new InternalError("Invalid state: " + token);
			}
		}

		if (token == Token.NULL)
			doKeyword(line, length, '\0');

		switch (token) {
			case Token.LITERAL1 :
			case Token.LITERAL2 :
				if (!backslash) {
					addToken(length - lastOffset, Token.INVALID);
					token = Token.NULL;
				} else {
					addToken(length - lastOffset, token);
				}
				break;
			case Token.KEYWORD2 :
				addToken(length - lastOffset, token);
				if (!backslash)
					token = Token.NULL;
				break;
			default :
				addToken(length - lastOffset, token);
				break;
		}

		return token;
	}
	public static KeywordMap getKeywords() {
		if (ccKeywords == null) {
			ccKeywords = CCTokenMarker.getKeywords();
			ccKeywords.add("__global__", Token.KEYWORD1);
			ccKeywords.add("__host__", Token.KEYWORD1);
			ccKeywords.add("__device__", Token.KEYWORD1);
			ccKeywords.add("__constant__", Token.KEYWORD1);
			ccKeywords.add("__shared__", Token.KEYWORD1);
			ccKeywords.add("gridDim", Token.KEYWORD2);
			ccKeywords.add("blockIdx", Token.KEYWORD2);
			ccKeywords.add("blockDim", Token.KEYWORD2);
			ccKeywords.add("threadIdx", Token.KEYWORD2);
			ccKeywords.add("int1", Token.KEYWORD3);
			ccKeywords.add("uint1", Token.KEYWORD3);
			ccKeywords.add("int2", Token.KEYWORD3);
			ccKeywords.add("uint2", Token.KEYWORD3);
			ccKeywords.add("int3", Token.KEYWORD3);
			ccKeywords.add("uint3", Token.KEYWORD3);
			ccKeywords.add("int4", Token.KEYWORD3);
			ccKeywords.add("uint4", Token.KEYWORD3);
			ccKeywords.add("float1", Token.KEYWORD3);
			ccKeywords.add("float2", Token.KEYWORD3);
			ccKeywords.add("float3", Token.KEYWORD3);
			ccKeywords.add("float4", Token.KEYWORD3);
			ccKeywords.add("char1", Token.KEYWORD3);
			ccKeywords.add("char2", Token.KEYWORD3);
			ccKeywords.add("char3", Token.KEYWORD3);
			ccKeywords.add("char4", Token.KEYWORD3);
			ccKeywords.add("uchar1", Token.KEYWORD3);
			ccKeywords.add("uchar2", Token.KEYWORD3);
			ccKeywords.add("uchar3", Token.KEYWORD3);
			ccKeywords.add("uchar4", Token.KEYWORD3);
			ccKeywords.add("short1", Token.KEYWORD3);
			ccKeywords.add("short2", Token.KEYWORD3);
			ccKeywords.add("short3", Token.KEYWORD3);
			ccKeywords.add("short4", Token.KEYWORD3);
			ccKeywords.add("dim1", Token.KEYWORD3);
			ccKeywords.add("dim2", Token.KEYWORD3);
			ccKeywords.add("dim3", Token.KEYWORD3);
			ccKeywords.add("dim4", Token.KEYWORD3);
			ccKeywords.add("tex1D", Token.KEYWORD3);
			ccKeywords.add("tex1Dfetch", Token.KEYWORD3);
			ccKeywords.add("tex2D", Token.KEYWORD3);
			ccKeywords.add("__float_as_int(", Token.LABEL);
			ccKeywords.add("__int_as_float(", Token.LABEL);
			ccKeywords.add("__float2int_rn(", Token.LABEL);
			ccKeywords.add("__float2int_rz(", Token.LABEL);
			ccKeywords.add("__float2int_ru(", Token.LABEL);
			ccKeywords.add("__float2int_rd(", Token.LABEL);
			ccKeywords.add("__float2uint_rn(", Token.LABEL);
			ccKeywords.add("__float2uint_rz(", Token.LABEL);
			ccKeywords.add("__float2uint_ru(", Token.LABEL);
			ccKeywords.add("__float2uint_rd(", Token.LABEL);
			ccKeywords.add("__int2float_rn(", Token.LABEL);
			ccKeywords.add("__int2float_rz(", Token.LABEL);
			ccKeywords.add("__int2float_ru(", Token.LABEL);
			ccKeywords.add("__int2float_rd(", Token.LABEL);
			ccKeywords.add("__uint2float_rn(", Token.LABEL);
			ccKeywords.add("__uint2float_rz(", Token.LABEL);
			ccKeywords.add("__uint2float_ru(", Token.LABEL);
			ccKeywords.add("__uint2float_rd(", Token.LABEL);
			ccKeywords.add("__fadd_rz(", Token.LABEL);
			ccKeywords.add("__fmul_rz(", Token.LABEL);
			ccKeywords.add("__fdividef(", Token.LABEL);
			ccKeywords.add("__mul24(", Token.LABEL);
			ccKeywords.add("__umul24(", Token.LABEL);
			ccKeywords.add("__mulhi(", Token.LABEL);
			ccKeywords.add("__umulhi(", Token.LABEL);
			ccKeywords.add("__mul64hi(", Token.LABEL);
			ccKeywords.add("__umul64hi(", Token.LABEL);
			ccKeywords.add("min(", Token.KEYWORD2);
			ccKeywords.add("umin(", Token.KEYWORD2);
			ccKeywords.add("fminf(", Token.KEYWORD2);
			ccKeywords.add("fmin(", Token.KEYWORD2);
			ccKeywords.add("max(", Token.KEYWORD2);
			ccKeywords.add("umax(", Token.KEYWORD2);
			ccKeywords.add("fmaxf(", Token.KEYWORD2);
			ccKeywords.add("fmax(", Token.KEYWORD2);
			ccKeywords.add("abs(", Token.KEYWORD2);
			ccKeywords.add("fabsf(", Token.KEYWORD2);
			ccKeywords.add("fabs(", Token.KEYWORD2);
			ccKeywords.add("sqrtf(", Token.KEYWORD2);
			ccKeywords.add("sqrt(", Token.KEYWORD2);
			ccKeywords.add("sinf(", Token.KEYWORD2);
			ccKeywords.add("__sinf(", Token.LABEL);
			ccKeywords.add("sin(", Token.KEYWORD2);
			ccKeywords.add("cosf(", Token.KEYWORD2);
			ccKeywords.add("__cosf(", Token.LABEL);
			ccKeywords.add("cos(", Token.KEYWORD2);
			ccKeywords.add("sincosf(", Token.KEYWORD2);
			ccKeywords.add("__sincosf(", Token.LABEL);
			ccKeywords.add("expf(", Token.KEYWORD2);
			ccKeywords.add("__expf(", Token.LABEL);
			ccKeywords.add("exp(", Token.KEYWORD2);
			ccKeywords.add("logf(", Token.KEYWORD2);
			ccKeywords.add("__logf(", Token.LABEL);
			ccKeywords.add("log(", Token.KEYWORD2);
			ccKeywords.add("__syncthreads", Token.LABEL);
		}
		return ccKeywords;
	}

	// private members
	private static KeywordMap ccKeywords;
}
