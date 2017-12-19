import java.nio.ByteBuffer;

class Message {
    private ByteBuffer message;
    private long lastSendTime = 0;

    Message(ByteBuffer msg) {  message = msg; }
    void resetTime() {
        lastSendTime = System.currentTimeMillis();
    }
    boolean isOld() {
        return System.currentTimeMillis() - lastSendTime > Constants.SEND_TIMEOUT;
    }
    ByteBuffer getMessage() {
        return message;
    }
}
