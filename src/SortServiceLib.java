import java.util.ArrayList;

public interface SortServiceLib {
    String getName();

    ArrayList<Integer> sort(ArrayList<Integer> data);

    int getChangeCnt();
}
