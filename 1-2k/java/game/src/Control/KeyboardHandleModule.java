package Control;

import Model.Direction;

/**
 * Created by andrey on 05.05.17.
 */
public interface KeyboardHandleModule {
    /**
     * Считывание последних данных из стека событий, если можулю это необходимо
     */
    void update();

    /**
     * @return Возвращает направление последней нажатой "стрелочки",
     * либо AWAITING, если не было нажато ни одной
     */
    Direction lastDirectionKeyPressed();

    /**
     * @return Возвращает информацию о том, был ли нажат ESCAPE за последнюю итерацию
     */
    boolean wasEscPressed();
}
