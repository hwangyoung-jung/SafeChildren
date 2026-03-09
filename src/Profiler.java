import java.util.ArrayList;

public class Profiler {
    private ArrayList<Integer> inputData;
    private SortServiceLib sortServiceLib;
    private ArrayList<Integer> sortedResult;

    public void setData(ArrayList<Integer> input) {
        this.inputData = new ArrayList<>(input);
        this.sortedResult = null;
    }

    public void setLib(SortServiceLib lib) {
        this.sortServiceLib = lib;
        this.sortedResult = null;
    }

    public void runLib() {
        validateState();

        ArrayList<Integer> runTarget = new ArrayList<>(inputData);
        sortedResult = sortServiceLib.sort(runTarget);
    }

    public void showResult() {
        validateState();

        if (sortedResult == null) {
            throw new IllegalStateException("runLib() must be called before showResult().");
        }

        System.out.println(sortServiceLib.getName() + " 테스트");
        System.out.println("정렬결과: " + sortedResult);
        System.out.println("swap횟수: " + sortServiceLib.getChangeCnt() + "회");
    }

    private void validateState() {
        if (inputData == null) {
            throw new IllegalStateException("Input data is not set. Call setData() first.");
        }

        if (sortServiceLib == null) {
            throw new IllegalStateException("Sort library is not set. Call setLib() first.");
        }
    }
}
