class Solution {
public:
    int minOperations(vector<int>& nums1, vector<int>& nums2) {
        auto sum1 = 0, sum2 = 0;
        for (int n : nums1) {
            sum1 += n;
        }
        for (int n : nums2) {
            sum2 += n;
        }

        auto ops = 0;
        if (sum1 == sum2) {
            return ops;
        }

        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());

        if (sum1 < sum2) {
            nums1.swap(nums2);
        }

        auto diff = sum1 - sum2;

        for (int end1 = nums1.size() - 1,start2 = 0;diff != 0;) {
            if (diff > 0) {
                if (diff <= nums1[end1] - 1) {
                    ops++;
                    diff = 0;
                    end1--;
                } else {
                    diff -= nums1[end1] - 1;
                    ops++;
                    end1--;
                }
            } else {
                if (diff <= nums1[end1] - 1) {
                    ops++;
                    diff = 0;
                    end1--;
                } else {
                    diff -= nums1[end1] - 1;
                    ops++;
                    end1--;
                }
            }

            if (end1 < 0 || start2)
        }
        return ops;
    }
};
/*
1 2 3 4 5 6
1 1 2 2 2 2

21
10

1 2 3 4 5 1
6 1 2 2 2 2
*/