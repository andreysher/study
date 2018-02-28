import java.io.IOException;

public class Parser {
    private Lexer lexer;
    private Lexema currentLexem;
    private int bracketCounter = 0;

    public Parser(String fileName){
        lexer = new Lexer(fileName);
        try {
            currentLexem = lexer.getLexem();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseFileException e) {
            e.printStackTrace();
        }
    }

    private int parseAtom() throws IOException, ParseFileException {
        if(currentLexem.type == LexemTypes.BRACKET_OPEN){
            bracketCounter++;
            currentLexem = lexer.getLexem();
            return parseExpression();
        }
        if(currentLexem.type == LexemTypes.EOF){
            System.out.println("parseAtom(EOF)");
        }
        if(currentLexem.type == LexemTypes.BRACKET_CLOSE){
            bracketCounter--;
            currentLexem = lexer.getLexem();
        }
        if(currentLexem.type == LexemTypes.NUMBER){
            int num = Integer.parseInt(currentLexem.value);
            currentLexem = lexer.getLexem();
            return num;
        }
        return 0;
    }

    private int parsePower() throws IOException, ParseFileException {
        if(currentLexem.type == LexemTypes.MINUS){
            currentLexem = lexer.getLexem();
            return (-1)*parseAtom();
        }
        else{
            return parseAtom();
        }
    }

    private int parseFaction() throws IOException, ParseFileException {
        int temp = parsePower();
        while(currentLexem.type == LexemTypes.POWER){
            currentLexem = lexer.getLexem();
            temp = (int) Math.pow(temp, parseTerm());
        }
        return temp;
    }

    private int parseTerm() throws IOException, ParseFileException {
        int temp = parseFaction();
        while(currentLexem.type == LexemTypes.MULTIPLY || currentLexem.type == LexemTypes.DIVIZION){
            if(currentLexem.type == LexemTypes.MULTIPLY){
                currentLexem = lexer.getLexem();
                temp *= parseFaction();
            }
            if(currentLexem.type == LexemTypes.DIVIZION){
                currentLexem = lexer.getLexem();
                temp /= parseFaction();
            }
        }
        return temp;
    }

    private int parseExpression() throws IOException, ParseFileException {
        int temp = parseTerm();
        if(currentLexem.type == LexemTypes.BRACKET_CLOSE){
            parseAtom();
        }
        while (currentLexem.type == LexemTypes.PLUS || currentLexem.type == LexemTypes.MINUS){
            if(currentLexem.type == LexemTypes.PLUS){
                currentLexem = lexer.getLexem();
                temp += parseTerm();
            }
            else {
                currentLexem = lexer.getLexem();
                temp -= parseTerm();
            }
        }
        return temp;
    }

    private static int calculate(String fileName){
        Parser parser = new Parser(fileName);
        int temp = 0;
        try {
            temp = parser.parseExpression();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseFileException e) {
            e.printStackTrace();
        }
        System.out.println(temp);
        return temp;
    }

    public static void main(String[] args) {
        calculate(args[0]);
    }
}
