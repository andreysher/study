package View;

import Model.ModelConfig;

/**
 * Created by andrey on 05.05.17.
 */
public class ViewConfig {
    public static final int CELL_SIZE = 64;
    public static final int SCREEN_WIDTH = ModelConfig.COUNT_CELLS_X * CELL_SIZE;
    public static final int SCREEN_HEIGHT = ModelConfig.COUNT_CELLS_Y * CELL_SIZE;
    public static final String SCREEN_NAME = "My 2048";
}
