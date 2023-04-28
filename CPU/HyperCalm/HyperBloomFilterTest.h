#ifndef HYPER_BLOOM_FILTER_REF_H
#define HYPER_BLOOM_FILTER_REF_H

#include <random>

namespace HyperBF {

// Test mode HBF mimics the behavior of the original HBF,
// but use a whole 64-bit to store a cell.
// It's used for testing the accuracy of the HBF,
// meanwhile also useful for understanding the original algorithm.
template <size_t CellBits,
          size_t _MaxReportSize = (1 << CellBits) - 1,
          size_t _CellPerBucket = sizeof(uint64_t) * 8 / CellBits>
class TestModeHyperBF {
    static constexpr size_t CellPerBucket = _CellPerBucket;
    static constexpr size_t TableNum = 8;
public:
    uint64_t* buckets;
    uint64_t* counters;
    double time_threshold;
    uint32_t bucket_num;
    uint32_t seeds[TableNum + 1];
    static constexpr size_t MaxReportSize = _MaxReportSize;

    TestModeHyperBF(uint32_t memory, double time_threshold, int seed = 123) :
        time_threshold(time_threshold), bucket_num(memory / sizeof(uint64_t)) {
        buckets = new (align_val_t { 64 }) uint64_t[bucket_num * CellPerBucket] {};
        counters = new (align_val_t { 64 }) uint64_t[bucket_num * CellPerBucket] {};
        mt19937 rng(seed);
        for (int i = 0; i <= TableNum; ++i) {
            seeds[i] = rng();
        }
        printf("d = %d\t (Number of mimic arrays in HyperBF)\n", bucket_num);
    }

    ~TestModeHyperBF() {
        delete[] buckets;
        delete[] counters;
    }

    int insert_cnt(int key, double time) {
        int first_bucket_pos = CalculatePos(key, TableNum) % bucket_num & ~(TableNum - 1);
        int min_cnt = MaxReportSize, max_cnt = 0;
        for (int i = 0; i < TableNum; ++i) {
            int cell_pos = CalculatePos(key, i) % CellPerBucket;
            int bucket_pos = (first_bucket_pos + i) * CellPerBucket;

            int now_tag = int(time / time_threshold + 1.0 * i / TableNum) % 3 + 1;
            int ban_tag = now_tag % 3 + 1;

            for (int j = 0; j < CellPerBucket; ++j) {
                if (buckets[bucket_pos + j] == ban_tag) {
                    buckets[bucket_pos + j] = 0;
                    counters[bucket_pos + j] = 0;
                }
            }
            buckets[bucket_pos + cell_pos] = now_tag;
            int cnt = counters[bucket_pos + cell_pos];
            max_cnt = max(max_cnt, cnt);
            min_cnt = min(min_cnt, cnt);
            if (cnt != MaxReportSize) {
                ++counters[bucket_pos + cell_pos];
            }
        }
        return min_cnt;
    }
    bool insert(int key, double time) {
        return insert_cnt(key, time) == 0;
    }

private:
    inline uint32_t CalculatePos(uint32_t key, int i) {
        return (key * seeds[i]) >> 15;
    }
};

}; // namespace HyperBF

#endif // HYPER_BLOOM_FILTER_REF_H
