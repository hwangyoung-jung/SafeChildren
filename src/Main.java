import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        Profiler profiler = new Profiler();

        ArrayList<Integer> input = new ArrayList<>(
            Arrays.asList(5, 2, 4, 1, 3, 9, 1, 1, 7, 8, 2, 1)
        );

        List<SortServiceLib> libs = Arrays.asList(
            new BubbleSort(),
            new SelectionSort(),
            new HeapSort()
        );

        for (SortServiceLib lib : libs) {
            profiler.setData(input);
            profiler.setLib(lib);
            profiler.runLib();
            profiler.showResult();
            System.out.println("----------");
        }
    }
}
