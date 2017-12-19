class Constants {
    static final int MAX_BUFFER_SIZE = 1024;
    static final int MAX_MSG_SIZE = MAX_BUFFER_SIZE - 4 - 4 - 4 - 1 - 20; //-id,размер,idкуска,1-тип,20-хэш
    static final int REC_TIMEOUT = 50;//спит 50милисек
    static final int SEND_TIMEOUT = 100;
    static final int COUNT_EFF = 50;
    public static final int BUF_SZ = 4096;
    static final int DEAD_TIME = 500;
}
