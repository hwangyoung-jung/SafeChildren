import java.util.ArrayList;

public class HeapSort implements SortServiceLib {
    private int changeCnt;

    @Override
    public String getName() {
        return "heapSort";
    }

    @Override
    public ArrayList<Integer> sort(ArrayList<Integer> data) {
        changeCnt = 0;

        int n = data.size();
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(data, n, i);
        }

        for (int i = n - 1; i > 0; i--) {
            swap(data, 0, i);
            heapify(data, i, 0);
        }

        return data;
    }

    @Override
    public int getChangeCnt() {
        return changeCnt;
    }

    private void heapify(ArrayList<Integer> data, int heapSize, int rootIndex) {
        int largest = rootIndex;
        int left = 2 * rootIndex + 1;
        int right = 2 * rootIndex + 2;

        if (left < heapSize && data.get(left) > data.get(largest)) {
            largest = left;
        }

        if (right < heapSize && data.get(right) > data.get(largest)) {
            largest = right;
        }

        if (largest != rootIndex) {
            swap(data, rootIndex, largest);
            heapify(data, heapSize, largest);
        }
    }

    private void swap(ArrayList<Integer> data, int i, int j) {
        int temp = data.get(i);
        data.set(i, data.get(j));
        data.set(j, temp);
        changeCnt++;
    }
}
