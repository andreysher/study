package Control;

/**
 * Created by andrey on 05.05.17.
 */
public class ErrorCatcher {
    /**
     * Ошибка создания новой ячейки
     */
    public static void cellCreationFailure() {
        System.err.println("Main class failed to create new cell.");
        System.exit(-1);
    }

    /**
     * Передача неверного параметра Direction в метод сдвига
     */
    public static void shiftFailureWrongParam() {
        System.err.println("Main class failed to shift cells on field. Wrong parameter.");
        System.exit(-2);
    }

    /**
     * Внутренняя ошибка графического модуля
     *
     * @param e Выброшенное исключение
     */
    public static void graphicsFailure(Exception e) {
        System.err.println("GraphicsModule failed.");
        e.printStackTrace();
        System.exit(-3);
    }
}
