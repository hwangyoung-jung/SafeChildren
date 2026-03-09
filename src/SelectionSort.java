import java.util.ArrayList;

public class SelectionSort implements SortServiceLib {
    private int changeCnt;

    @Override
    public String getName() {
        return "selectionSort";
    }

    @Override
    public ArrayList<Integer> sort(ArrayList<Integer> data) {
        changeCnt = 0;

        int n = data.size();
        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;
            for (int j = i + 1; j < n; j++) {
                if (data.get(j) < data.get(minIndex)) {
                    minIndex = j;
                }
            }
            if (minIndex != i) {
                swap(data, i, minIndex);
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
