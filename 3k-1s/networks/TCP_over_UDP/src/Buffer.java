import java.nio.ByteBuffer;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

class Buffer {
    private static final Object monitorForSend = new Object();
    private static final Object monitorForRecv = new Object();

    private boolean isConnected = false;
    private Map<Integer, Message> sendBuf = new TreeMap<>();
    private SortedMap<Integer, ByteBuffer> receiveBuf = new TreeMap<>();

    private int sendMessageId = 0;
    private int sendPayloadID = 1;
    private int prevPayloadID = 1;

    private boolean isDead = false;
    private long deadTime = 0;

    int getSendMsgId() {
        synchronized (monitorForSend) {
            return sendMessageId;
        }
    }

    void putToSendMessage(ByteBuffer msg) {
        synchronized (monitorForSend) {
            sendBuf.put(sendMessageId++, new Message(msg));
        }
    }

    void removeFromSendMessage(int sendMessageId) { //удаляем из отправленных сообщ
        synchronized (monitorForSend) {
            sendBuf.remove(sendMessageId);
        }
    }

    void putToReceive(int payloadID, ByteBuffer msg) {
        synchronized (monitorForRecv) {
            if (receiveBuf.containsKey(payloadID)) {
                return;
            }

            receiveBuf.put(payloadID, msg);
            monitorForRecv.notify();//уведомление пользоват потока о том что пришли данные
        }
    }

    Map<Integer, Message> getMsgsToSend() {
        synchronized (monitorForSend) {
            return new TreeMap<>(sendBuf);
        }
    }

    boolean isDead() {
        return isDead && (Constants.DEAD_TIME < System.currentTimeMillis() - deadTime);
    }

    void dead() {
        isDead = true;
        System.out.println("umer");
        deadTime = System.currentTimeMillis();
    }

    boolean hasMessages() {
        synchronized (monitorForSend) {
            return !sendBuf.isEmpty();
        }
    }

    boolean isConnected() {
        synchronized (monitorForRecv) {
            return isConnected;
        }
    }

    void connected() {
        synchronized (monitorForSend) {
            isConnected = true;
        }
    }

    void cleanSendBuffer() {
        synchronized (monitorForSend) {
            sendBuf.clear();
        }
    }

    void putUserPayload(byte[] buffer, int len) {
        synchronized (monitorForSend) {
            int length = len;
            int off = 0;
            while (Constants.MAX_MSG_SIZE < length) {//ЛУЧШЕ ЕДИНЫЙ ТРЕД ЗАНИМ ОТПР{
                ByteBuffer data = ByteBuffer.allocate(Constants.MAX_MSG_SIZE);
                data.put(buffer, off, Constants.MAX_MSG_SIZE);

                data.flip();

                sendBuf.put(sendMessageId,
                new Message((new MessageCreator(sendMessageId, data)).genPayloadMsg(sendPayloadID++)));
                //положили в очередь на отпр
                sendMessageId++;

                length -= Constants.MAX_MSG_SIZE;
                off += Constants.MAX_MSG_SIZE;
            }
            if (0 != length) {//если еще остались данные на отправку
                ByteBuffer data = ByteBuffer.allocate(length);
                data.put(buffer, off, length);

                data.flip();

                sendBuf.put(sendMessageId, new Message((new MessageCreator(sendMessageId, data)).genPayloadMsg(sendPayloadID++)));
                sendMessageId++;
            }
        }
    }

    int getUserPayload(byte[] buffer, int offset) {
        synchronized (monitorForRecv) {
            return readData(buffer, offset);//записывает буфер со смещением.вернет колво счит эл-ов
        }
    }

    private int readData(byte[] buffer, int offset) {
        if (!receiveBuf.containsKey(prevPayloadID)) {//если мы этот мессендж не приняли - ждем

            try {
                while (!receiveBuf.containsKey(prevPayloadID)) {
                    monitorForRecv.wait();
                }
            } catch (InterruptedException ex) {
                System.out.println(ex.getMessage());
            }
        }
        int count = 0;

        while (receiveBuf.containsKey(prevPayloadID)) {//до тех пор пока есть куски данных считываем
            ByteBuffer buf = receiveBuf.get(prevPayloadID);

            if (buf.remaining() < buffer.length - offset - count) {//смотрим есть ли места чтоб записать еще данные в buffer
                int tmpCount = buf.remaining();
                buf.get(buffer, count + offset, buf.remaining());//
                count += tmpCount;//сколько было считано
            } else {
                buf.get(buffer, count + offset, buffer.length - offset - count);//считываем сколько можем считать
                count += buffer.length - offset - count;
            }

            if (!buf.hasRemaining()) {
                receiveBuf.remove(prevPayloadID);//удалили кусок от куда его переносили
                prevPayloadID++;
            } else {
                return count;
            }
        }

        return count;
    }
}