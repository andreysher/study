import java.io.IOException;
import java.net.DatagramPacket;

public class ReceiverThread implements Runnable{
    /*
* Принимает пакеты, выстраивает их в правильном порядке в очередь(сортирует)
* */
    private MySock sock;

    public ReceiverThread(MySock socket){
        sock = socket;
    }

    @Override
    public void run() {
        /*принимаем пакеты, если ак имеет значение то меняем номер последнего полученого ака,
        * если пришел следующий требуемый нам пакет, то меняем значение отправляемого ака,
        * если пришел syn+ack(единственный пакет без данных), е отправляем его в буфер, а повышаем значения аков*/
        while (true) {
            byte[] buf = new byte[1024];
            DatagramPacket pack = new DatagramPacket(buf, buf.length);
            try {
                sock.udpSock.receive(pack);
                byte[] packData = pack.getData();
                if((MyPack.getFlags(packData) == 3)||(MyPack.getFlags(packData) == 2)
                        ||(MyPack.getFlags(packData) == 10)){
                    //то есть в нем есть ack - значит acknumber стоит смотреть
                    sock.lastAckNum = MyPack.getAckNumber(packData);
                }
                if(MyPack.getPackNumber(packData) == sock.acknowledge + 1){//следующий требуемый нам пакет
                    sock.acknowledge++;
                }
                if(MyPack.getFlags(packData) != 3) {//если не syn+ack
                    MyPack mPack = MyPack.myPackFromBuf(packData);
                    sock.input.put(mPack);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
