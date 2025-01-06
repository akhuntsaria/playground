class Solution {
public:
    int minimumEffort(vector<vector<int>>& tasks) {
        sort(tasks.begin(), tasks.end(), [](const auto& a, const auto& b) {
            return (b[1] - b[0]) < (a[1] - a[0]);
        });
        //print(tasks);
        int l = 0, r = INT_MAX, res = INT_MAX;
        while (l <= r) {
            int m = l + (r - l) / 2;
            //cout << l << " " << r << " " << m << endl;
            if (valid(tasks, m)) {
                res = min(res, m);
                //cout << res << endl;
                r = m - 1;
            } else {
                l = m + 1;
            }
        }
        return res;
    }

    bool valid(vector<vector<int>>& tasks, int e) {
        //cout << "valid: ";
        for (auto t : tasks) {
            if (t[1] > e) {
                //cout << endl;
                return false;
            }
            e -= t[0];
            //cout << e << " ";
        }
        //cout << endl;
        return true;
    }

    template <typename T>
    void print(const vector<T>& vec) {
        for (const auto& element : vec) {
            cout << element << " ";
        }
        cout << endl;
    }

    template <typename T>
    void print(const vector<vector<T>>& vec) {
        for (const auto& inner : vec) {
            for (const auto& element : inner) {
                cout << element << " ";
            }
            cout << endl;
        }
    }
};
/*
Binary search
*/