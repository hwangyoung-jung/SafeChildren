import java.util.ArrayList;

public class BubbleSort implements SortServiceLib {
    private int changeCnt;

    @Override
    public String getName() {
        return "bubbleSort";
    }

    @Override
    public ArrayList<Integer> sort(ArrayList<Integer> data) {
        changeCnt = 0;

        int n = data.size();
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (data.get(j) > data.get(j + 1)) {
                    swap(data, j, j + 1);
                }
            }
        }
        return data;
    }

    @Override
    public int getChangeCnt() {
        return changeCnt;
    }

    private void swap(ArrayList<Integer> data, int i, int j) {
        int temp = data.get(i);
        data.set(i, data.get(j));
        data.set(j, temp);
        changeCnt++;
    }
}
