package Model;

public class ParseException extends Exception{
    String reason;
    ParseException(String reason){
        this.reason = reason;
    }
    public void printMessage(){
        System.out.println(reason);
    }
}
