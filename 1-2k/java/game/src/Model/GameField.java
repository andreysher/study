package Model;

import Control.ErrorCatcher;
import Model.ModelConfig;

import java.util.Random;

/**
 * Created by andrey on 05.05.17.
 */
public class GameField {
    /**
     * Состояние всех ячеек поля.
     */
    private int[][] theField;

    /**
     * Инициализирует поле и заполняет его нулями
     */
    public GameField(){
        theField = new int[ModelConfig.COUNT_CELLS_X][ModelConfig.COUNT_CELLS_Y];

        for(int i=0; i<theField.length;i++){
            for(int j=0; j<theField[i].length; j++){
                theField[i][j]=0;
            }
        }
    }

    /**
     * Возвращает состояние ячейки поля по координатам
     *
     * @param x Координата ячейки X
     * @param y Координата ячейки Y
     * @return Состояние выбранной ячейки
     */
    public int getState(int x, int y){
        return theField[x][y];
    }

    /**
     * Изменяет состояние ячейки поля по координатам
     *
     * @param x Координата ячейки X
     * @param y Координата ячейки Y
     * @param state Новое состояние для этой ячейки
     */
    public void setState(int x, int y, int state){
        //TODO check input maybe?

        theField[x][y] = state;
    }

    /**
     * Изменяет столбец под номером i
     *
     * @param i Номер изменяемого столбца
     * @param newColumn Массив новых состояний ячеек столбца
     */
    public void setColumn(int i, int[] newColumn) {
        theField[i] = newColumn;
    }

    /**
     * Возвращает массив состояний ячеек столбца под номером i
     *
     * @param i Номер запрашиваемого столбца
     * @return Массив состояний ячеек столбца
     */
    public int[] getColumn(int i) {
        return theField[i];
    }

    /**
     * Изменяет строку под номером i
     *
     * @param i Номер изменяемой строки
     * @param newLine Массив новых состояний ячеек строки
     */
    public void setLine(int i, int[] newLine) {
        for(int j = 0; j< ModelConfig.COUNT_CELLS_X; j++){
            theField[j][i] = newLine[j];
        }
    }

    /**
     * Возвращает массив состояний ячеек строки под номером i
     *
     * @param i Номер запрашиваемой строки
     * @return Массив состояний ячеек строки
     */
    public int[] getLine(int i) {
        int[] ret = new int[ModelConfig.COUNT_CELLS_X];

        for(int j = 0; j< ModelConfig.COUNT_CELLS_X; j++){
            ret[j] = theField[j][i];
        }

        return ret;
    }

    /**
     * Создаёт в случайной пустой клетке игрового поля плитку (с ненулевым состоянием).
     */
    public void generateNewCell() {
        int state = (new Random().nextInt(100) <= ModelConfig.CHANCE_OF_LUCKY_SPAWN)
                ? ModelConfig.LUCKY_INITIAL_CELL_STATE
                : ModelConfig.INITIAL_CELL_STATE;

        int randomX, randomY;

        randomX = new Random().nextInt(ModelConfig.COUNT_CELLS_X);
        int currentX = randomX;

        randomY = new Random().nextInt(ModelConfig.COUNT_CELLS_Y);
        int currentY = randomY;



        boolean placed = false;
        while(!placed){
            if(this.getState(currentX, currentY) == 0) {
                this.setState(currentX, currentY, state);
                placed = true;
            }else{
                if(currentX+1 < ModelConfig.COUNT_CELLS_X) {
                    currentX++;
                }else{
                    currentX = 0;
                    if(currentY+1 < ModelConfig.COUNT_CELLS_Y) {
                        currentY++;
                    }else{
                        currentY = 0;
                    }
                }

                if ((currentX == randomX) && (currentY==randomY) ) {  //No place -> Something went wrong
                    ErrorCatcher.cellCreationFailure();
                }
            }
        }

        GameState.addScore(state);
    }
}
