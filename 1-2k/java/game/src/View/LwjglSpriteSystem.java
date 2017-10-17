package View;

import java.util.HashMap;

/**
 * Created by andrey on 05.05.17.
 */
public class LwjglSpriteSystem {
    /**
     * Хранит в себе ссылки на все доступные в игре текстуры с ключом, равным изображённой на текстуре цифре
     */
    private HashMap<Integer, LwjglSprite> spriteByNumber = new HashMap<>();

    /**
     * Инициализирует HashMap spriteByNumber и записывает в него ссылки
     * на все доступные в игре текстуры с ключом, равным изображённой на текстуре цифре
     */
    LwjglSpriteSystem() {
//        LwjglSprite  vv = LwjglSprite.valueOf("CELL2");
        for (LwjglSprite sprite : LwjglSprite.values()) {
            if (sprite.getSpriteNumber() != null) {
                spriteByNumber.put(sprite.getSpriteNumber(), sprite);
            }
        }
    }

    /**
     * @param n Число, которое должна изображать текстура
     * @return Текстура, изображающее число, переданное в параметре. Если такого нет, возвращает EMPTY.
     */
    public LwjglSprite getSpriteByNumber(int n) {
        return spriteByNumber.getOrDefault(n, LwjglSprite.EMPTY);
    }
}
