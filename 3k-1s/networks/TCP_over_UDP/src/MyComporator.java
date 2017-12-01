import java.util.Comparator;

public class MyComporator implements Comparator<MyPack> {
    @Override
    public int compare(MyPack a, MyPack b) {
        return a.packNumber - b.packNumber;
    }
}