import java.net.DatagramPacket;
import java.nio.ByteBuffer;

public class MyPack {

    public static final byte SIN = 1;
    public static final byte ACK = 2;
    public static final byte RST = 4;
    public static final byte FIN = 8;

//для проверки флагов будем просто минусовать с больших до маленьких, если остается > 0,флаг стоит
    public static final int DATASIZE = 1015;
    public int packNumber;
    public int ackNumber;
    byte flags;
    public byte[] data;
//протестировано
    public static void intToByteArray(int val, byte[] buf, int offset){
        buf[offset] = (byte) (val >> 24);
        buf[offset+1] = (byte) (val >> 16);
        buf[offset+2] = (byte) (val >> 8);
        buf[offset+3] = (byte) val;
    }
//протестировано
    public static int byteArrayToInt(byte[] buf, int offset){
        return buf[offset] << 24 | (buf[offset + 1] & 0xFF) << 16
                | (buf[offset + 2] & 0xFF) << 8 | (buf[offset + 3] & 0xFF);
    }

    public MyPack(int packNum, int ackNum, byte fl, byte[] data){
        packNumber = packNum;
        ackNumber = ackNum;
        flags = fl;
        this.data = data;
    }

    public static int getPackNumber(byte[] buf){
        return byteArrayToInt(buf, 0);
    }

    public static int getAckNumber(byte[] buf){
        return byteArrayToInt(buf, 4);
    }

    public static byte getFlags(byte[] buf) {return buf[8];}

    public static byte[] getMassageData(byte[] buf){
        ByteBuffer b = ByteBuffer.wrap(buf);
        byte [] data = new byte[DATASIZE];
        b.get(data, 9, DATASIZE);
        return data;
    }

    public byte[] getData(){
        byte[] buf = new byte[1024];
        intToByteArray(this.packNumber, buf, 0);
        intToByteArray(this.ackNumber, buf, 4);
        buf[8] = this.flags;
        if(this.data != null) {
            for (int i = 9; i < data.length; i++) {
                buf[i] = data[i - 9];
            }
        }
        //иначе массив сам заполняется нулями при инициализации
        return buf;
    }

    public static MyPack myPackFromBuf(byte[] buffer){
        return new MyPack(getPackNumber(buffer), getAckNumber(buffer),
                getFlags(buffer), getMassageData(buffer));
    }
}
