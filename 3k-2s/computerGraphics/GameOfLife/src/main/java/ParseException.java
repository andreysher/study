public class ParseException extends Exception{
    String reason;
    public ParseException(String reason){
        this.reason = reason;
    }
    public void printMessage(){
        System.out.println(reason);
    }
}
