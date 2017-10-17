package Model;

/**
 * Created by andrey on 05.05.17.
 */
public class ModelConfig {
    public static int COUNT_CELLS_X = 4;
    public static int COUNT_CELLS_Y = 4;

    /* В оригинальной игре есть небольшой шанс, что появится плитка со значением не 2, а 4
       Этот шанс (в процентах) определяется здесь */
    public static final int CHANCE_OF_LUCKY_SPAWN = 17; //%

    /* Состояния новосозданых клеток (при условии срабатывания CHANCE_OF_LUCKY_SPAWN и без него)*/
    public static final int LUCKY_INITIAL_CELL_STATE = 4;
    public static final int INITIAL_CELL_STATE = 2;

    /* Количество определённых к первому ходу пользователя ячеек */
    public static final int COUNT_INITITAL_CELLS = 2;
}
