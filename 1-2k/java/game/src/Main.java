import Model.GameState;

/**
 * Created by andrey on 05.05.17.
 */
public class Main {

    /**
     * Точка входа. Содержит все необходимые действия для одного игрового цикла.
     */
    public static void main(String[] args) {
        GameState.startNewGame();
        while(!GameState.getFlag()){
            GameState.playGame();
        }

        GameState.endingOfGame();
    }
}
