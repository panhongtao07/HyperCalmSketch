#ifndef _HYPERCALM_H_
#define _HYPERCALM_H_

#include "CalmSpaceSaving.h"
#include "HyperBloomFilter.h"

class HyperCalm {
private:
    using HBF = HyperBloomFilter<>;
    CalmSpaceSaving css;
    HBF hbf;

    inline int suggestHBFMemory(int memory, double time_threshold) {
        int suggest_max;
        if (time_threshold > 0.001) {
            suggest_max = 50000;
        } else {
            suggest_max = 2000;
        }
        return min(memory / 2, suggest_max);
    }

public:
#define hbfmem suggestHBFMemory(memory, time_threshold)
#define sz max(1, memory / 1000)
    HyperCalm(double time_threshold, double unit_time, int memory, int seed)
        : css(time_threshold, unit_time, memory - hbfmem, 3, sz, sz),
          hbf(hbfmem, time_threshold, seed) {}
#undef sz
#undef hbfmem
    void insert(int key, double time) {
        bool b = hbf.insert(key, time);
        css.insert(key, time, b);
    }

    void insert_filter(int key, double time, size_t min_size) {
        int size = hbf.insert_cnt(key, time) + 1;
        if (size >= min_size)
            css.insert(key, time, size == min_size);
    }

    template <size_t min_size>
    void insert_filter(int key, double time) {
        static_assert(min_size <= HBF::MaxReportSize);
        int size = hbf.insert_cnt(key, time) + 1;
        if (size >= min_size)
            css.insert(key, time, size == min_size);
    }

    vector<pair<pair<int, int16_t>, int>> get_top_k(int k) const {
        return css.get_top_k(k);
    }
};

#endif  // _HYPERCALM_H_
