#ifndef _CLOCKSKETCH_H_
#define _CLOCKSKETCH_H_

#include <cstring>
#include <random>

template <bool use_counter = false>
class ClockSketch {
    static constexpr size_t TableNum = 4;
    static constexpr size_t CellBits = 4;
    static constexpr size_t CellsPerBucket = (sizeof(uint64_t) * 8 / CellBits);
public:
    uint64_t* buckets;
    int* counters;
    double time_threshold;
    int64_t la_time;
    uint32_t bucket_num, la_pos;
    uint32_t seeds[TableNum + 1];

    ClockSketch(uint32_t memory, double time_threshold, int seed = 233)
        : counters(nullptr), time_threshold(time_threshold), la_time(0), la_pos(0) {
        bucket_num = memory / sizeof(uint64_t);
        buckets = new uint64_t[bucket_num] {};
        std::mt19937 rng(seed);
        for (int i = 0; i <= TableNum; ++i) {
            seeds[i] = rng();
        }
        printf(" %d\t (Number of arrays in Clock-Sketch)\n",bucket_num);
    }

    ~ClockSketch() {
        delete[] buckets;
        delete[] counters;
    }

    bool insert(int key, double time) {
        int64_t nt = time * ((1 << CellBits) - 2) * (bucket_num * CellsPerBucket) / time_threshold;
        if (la_time) {
            int d = nt - la_time;
            if (d >= (int)bucket_num * CellsPerBucket * CellBits) {
                memset(buckets, 0, bucket_num * sizeof(*buckets));
                d = 0;
            }
            for (; d > 0 && la_pos % CellsPerBucket; --d) {
                uint64_t v = uint64_t(1) << (la_pos % CellsPerBucket * CellBits);
                if (buckets[la_pos / CellsPerBucket] & v * ((1 << CellBits) - 1))
                    buckets[la_pos / CellsPerBucket] -= v;
                (++la_pos) %= (bucket_num * CellsPerBucket);
            }
            constexpr uint64_t onePerCell = oneForEachCell(CellBits);
            for (; d >= CellsPerBucket; d -= CellsPerBucket) {
                uint64_t x = buckets[la_pos / CellsPerBucket];
                if constexpr (CellBits % 8 == 0) x |= x >> 4;
                if constexpr (CellBits % 4 == 0) x |= x >> 2;
                if constexpr (CellBits % 2 == 0) x |= x >> 1;
                buckets[la_pos / CellsPerBucket] -= x & onePerCell;
                (la_pos += CellsPerBucket) %= (bucket_num * CellsPerBucket);
            }
            for (; d > 0; --d) {
                uint64_t v = uint64_t(1) << (la_pos % CellsPerBucket * CellBits);
                if (buckets[la_pos / CellsPerBucket] & v * ((1 << CellBits) - 1))
                    buckets[la_pos / CellsPerBucket] -= v;
                (++la_pos) %= (bucket_num * CellsPerBucket);
            }
        }
        la_time = nt;
        bool ans = 0;
        for (int i = 0; i < TableNum; ++i) {
            int pos = CalculatePos(key, i) % (bucket_num * CellsPerBucket);
            uint64_t v = uint64_t((1 << CellBits) - 1) << (pos % CellsPerBucket * CellBits);
            if ((buckets[pos / CellsPerBucket] & v) == 0)
                ans = 1;
            buckets[pos / CellsPerBucket] |= v;
        }
        return ans;
    }

private:
    constexpr uint64_t oneForEachCell(size_t cellbit) {
        if (cellbit == 0 || cellbit > sizeof(uint64_t) * 8)
            throw std::invalid_argument("cellbit must be in [1, 64] and be a power of 2");
        if (cellbit == sizeof(uint64_t) * 8)
            return 0x1;
        return (oneForEachCell(cellbit * 2) << cellbit) | oneForEachCell(cellbit * 2);
    }

    inline uint32_t CalculatePos(uint32_t key, int i) {
        return (key * seeds[i]) >> 15;
    }
};

#endif //_CLOCKSKETCH_H_
