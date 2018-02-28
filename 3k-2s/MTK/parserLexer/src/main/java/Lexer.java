import java.io.*;

class Lexer {
    private int currentChar;
    private Reader reader;
    Lexer(String fileName){
        try {
            reader = new BufferedReader(new FileReader(fileName));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            currentChar =  reader.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    Lexema getLexem() throws IOException, ParseFileException {
        Lexema lexem = new Lexema();
        while (Character.isSpaceChar(currentChar)){
                currentChar = reader.read();
        }
        if(Character.isDigit(currentChar)) {
            lexem.value += (char) currentChar;
            currentChar = reader.read();
            while (Character.isDigit(currentChar)) {
                lexem.value += (char) currentChar;
                currentChar = reader.read();
            }
            lexem.type = LexemTypes.NUMBER;
            return lexem;
        }
        switch (currentChar){
            case (int)'+':
                lexem.value = "+";
                lexem.type = LexemTypes.PLUS;
                currentChar = reader.read();
                return lexem;
            case (int)'-' :
                lexem.value = "-";
                lexem.type = LexemTypes.MINUS;
                currentChar = reader.read();
                return lexem;
            case (int)'*':
                lexem.value = "*";
                lexem.type = LexemTypes.MULTIPLY;
                currentChar = reader.read();
                return lexem;
            case (int)'(':
                lexem.value = "(";
                lexem.type = LexemTypes.BRACKET_OPEN;
                currentChar = reader.read();
                return lexem;
            case (int)')':
                lexem.value = ")";
                lexem.type = LexemTypes.BRACKET_CLOSE;
                currentChar = reader.read();
                return lexem;
            case (int)'^':
                lexem.value = "^";
                lexem.type = LexemTypes.POWER;
                currentChar = reader.read();
                return lexem;
            case (int)'/':
                lexem.value = "/";
                lexem.type = LexemTypes.DIVIZION;
                currentChar = reader.read();
                return lexem;
        }
        if(currentChar == -1){
            lexem.value = null;
            lexem.type = LexemTypes.EOF;
            return lexem;
        }
        throw new ParseFileException();
    }
}
