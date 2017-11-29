import java.net.InetSocketAddress;

class ControlledTask {
    int[] task;
    InetSocketAddress clientAddr;
    long time;

    ControlledTask(int[] task, InetSocketAddress clientAddr, long time){
        this.task = task;
        this.clientAddr = clientAddr;
        this.time = time;
    }
}
