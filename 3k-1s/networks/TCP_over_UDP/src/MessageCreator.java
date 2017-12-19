import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

class MessageCreator {
    private MessageType msgType;
    private int messageId;
    private ByteBuffer payload;
    private int payloadId = 0;
    private byte[] hash;
    private int ACKED = 0;

    MessageCreator(byte[] data) {
        ByteBuffer msg = ByteBuffer.wrap(data);
        byte type = msg.get();

        switch (type) {
            case 1:
                parseSYNMsg(msg);
                break;
            case 2:
                parseAckMsg(msg);
                break;
            case 3:
                parseSYNAckMsg(msg);
                break;
            case 4:
                parsePayloadMsg(msg);
                break;
            case 5:
                parseDeadMsg(msg);
                break;
            case 6:
                parseDeadAckMsg(msg);
                break;
        }
    }

    MessageCreator(int messageId) {
        this.messageId = messageId;
    }

    MessageCreator(int messageId, ByteBuffer payload) {
        this.messageId = messageId;
        this.payload = payload;
    }

    int getPayloadId() {
        return payloadId;
    }
    int getACKED() {
        return ACKED;
    }
    int getMessageId() {
        return messageId;
    }

    boolean isValid(){
        try {
            return Arrays.equals(hash, MessageDigest.getInstance("SHA-1").digest(payload.array()));
        } catch (NoSuchAlgorithmException ex){
            return false;
        }
    }

    MessageType messageType() {
        return msgType;
    }
    ByteBuffer getPayload() {
        return payload;
    }

    ByteBuffer genSYNMsg() {
        ByteBuffer res = ByteBuffer.allocate(1 + 4);

        res.put((byte) 1);
        res.putInt(messageId);
        res.flip();
        return res;
    }

    ByteBuffer genAckMsg(int ACK) {
        ByteBuffer res = ByteBuffer.allocate(1 + 4 + 4);

        res.put((byte) 2);
        res.putInt(ACK);
        res.flip();
        return res;
    }

    ByteBuffer genSYNAckMsg(int ACK) {
        ByteBuffer res = ByteBuffer.allocate(1 + 4 + 4);

        res.put((byte) 3);
        res.putInt(messageId);
        res.putInt(ACK);//положили id ack
        res.flip();
        return res;
    }

    ByteBuffer genPayloadMsg(int payloadId) {
        ByteBuffer res = ByteBuffer.allocate(1 + 4 + 4 + 4 + 20 + payload.limit());

        res.put((byte) 4);
        res.putInt(messageId);
        res.putInt(payloadId);
        res.putInt(payload.limit());
        try {
            res.put(MessageDigest.getInstance("SHA-1").digest(payload.array()));//////////////////чтоб деджест был только на полезную часть а не на весь буфер
        } catch (NoSuchAlgorithmException ex){
            System.out.println("Error in hash");
            res.put(new byte[20]);
        }
        res.put(payload);
        res.flip();
        return res;
    }

    ByteBuffer genDeadMsg() {
        ByteBuffer res = ByteBuffer.allocate(1 + 4);

        res.put((byte) 5);
        res.putInt(messageId);
        res.flip();
        return res;
    }

    ByteBuffer genDeadAckMsg(int ACK) {
        ByteBuffer res = ByteBuffer.allocate(1 + 4 + 4);

        res.put((byte) 6);
        res.putInt(messageId);
        res.putInt(ACK);
        return res;
    }

    private void parseSYNMsg(ByteBuffer msg) {
        msgType = MessageType.SYN;
        messageId = msg.getInt();
    }

    private void parseAckMsg(ByteBuffer msg) {
        msgType = MessageType.ACK;
        ACKED = msg.getInt();
    }

    private void parseSYNAckMsg(ByteBuffer msg) {
        msgType = MessageType.SYNACK;
        messageId = msg.getInt();//id самого сообщ
        ACKED = msg.getInt();//id сообщ которого подтвердили
    }

    private void parsePayloadMsg(ByteBuffer msg) {
        hash = new byte[20];
        msgType = MessageType.PAYLOAD;
        messageId = msg.getInt();
        payloadId = msg.getInt();//id куска

        byte[] data = new byte[msg.getInt()];//достаем размер данных сообщ и создаем массив этого размера
        msg.get(hash);
        msg.get(data);
        payload = ByteBuffer.allocate(data.length);
        payload.put(data);
        payload.flip();
    }

    private void parseDeadMsg(ByteBuffer msg) {
        msgType = MessageType.DEAD;
        messageId = msg.getInt();
    }

    private void parseDeadAckMsg(ByteBuffer msg) {
        msgType = MessageType.DEADACK;
        messageId = msg.getInt();
        ACKED = msg.getInt();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (null == obj) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }

        MessageCreator msg = (MessageCreator) obj;
        return messageId == msg.messageId;
    }
}
