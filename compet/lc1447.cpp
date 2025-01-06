class Solution {
public:
    int minSumOfLengths(vector<int>& arr, int target) {
        long long sum = 0;
        int res = INT_MAX;

        unordered_map<int,int> m;
        m[0] = -1;

        map<int, int> min_lens;
        int min_len = INT_MAX;

        for (int end = 0;end < arr.size();end++) {
            sum += arr[end];
            m[sum] = end;

            if (m.find(sum - target) == m.end()) {
                continue;
            }
            int start = m[sum - target];
            
            auto first_sub = min_lens.upper_bound(start);
            if (first_sub != min_lens.begin()) {
                first_sub--;
                res = min(res, first_sub->second + end - start);
            }

            min_len = min(min_len, end - start);
            min_lens[end] = min_len;
        }
        return res == INT_MAX ? -1 : res;
    }
};
/*
3 2 2 4  3
3 5 7 11 14


*/
    template <typename K, typename V>
    void print(const map<K, V>& map) {
        for (const auto& pair : map) {
            cout << pair.first << ":" << pair.second << " ";
        }
        cout << endl;
    }

    template <typename K, typename V>
    void print(const unordered_map<K, V>& map) {
        for (const auto& pair : map) {
            cout << pair.first << ":" << pair.second << " ";
        }
        cout << endl;
    }

    template <typename T>
    void print(const vector<T>& vec) {
        for (const auto& element : vec) {
            cout << element << " ";
        }
        cout << endl;
    }
};
/*
3 2 2 4  3
3 5 7 11 14


*/