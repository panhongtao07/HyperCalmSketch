#include <cmath>
#include <cstdio>

#include "params.h"

using namespace std;

#include "../HyperCalm/HyperBloomFilter.h"
#include "../ComparedAlgorithms/ClockSketch.h"
#include "../ComparedAlgorithms/SWAMP.h"
#include "../ComparedAlgorithms/TOBF.h"
#include "../ComparedAlgorithms/groundtruth.h"

using groundtruth::Record;

template <typename Algorithm>
void size_test(
    const vector<Record>& input,
    const vector<int>& realtime_sizes,
    Algorithm&& algo
) {
    constexpr bool can_overflow = true;
    constexpr int cellbits = 2;
    constexpr int overflow_limit = (1 << cellbits) + can_overflow;
    double ARE = 0, AAE = 0;
    int acc_cnt = 0, overflow_cnt = 0, overflow_acc = 0;
    constexpr int small_thereshold = 1, large_thereshold = overflow_limit - can_overflow;
    int small = 0, middle = 0, large = 0;
    double sARE = 0, mARE = 0, lARE = 0;
    long long sTAE = 0, mTAE = 0, lTAE = 0;
    int large_acc = 0, report_large = 0;
    int small_acc = 0;
    for (int i = 0; i < input.size(); ++i) {
        auto &[key, time] = input[i];
        int raw_real_size = realtime_sizes[i];
        int size = min(algo.insert_cnt(key, time) + 1, overflow_limit);
        int real_size = min(raw_real_size, overflow_limit);
        int diff = abs(size - real_size);
        double relative_error = double(diff) / real_size;
        AAE += diff;
        ARE += relative_error;
        if (!diff) ++acc_cnt;
        if (raw_real_size > overflow_limit) {
            ++overflow_cnt;
            if (!diff) ++overflow_acc;
        }
        if (size >= large_thereshold) {
            ++report_large;
            if (!diff) ++large_acc;
        }
        if (real_size <= small_thereshold) {
            ++small;
            if (!diff) ++small_acc;
            sTAE += diff;
            sARE += relative_error;
        } else if (real_size >= large_thereshold) {
            ++large;
            lTAE += diff;
            lARE += relative_error;
        } else {
            ++middle;
            mTAE += diff;
            mARE += relative_error;
        }
    }
    AAE /= input.size();
    ARE /= input.size();
    sARE /= small;
    mARE /= middle;
    lARE /= large;
    printf("---------------------------------------------\n");
    printf("Realtime size test\n");
    printf("Accuracy: %.2lf%%, No overflow: %.2lf%%, Overflow: %.2lf%%\n",
           double(acc_cnt) / input.size() * 100,
           double(acc_cnt - overflow_acc) / (input.size() - overflow_cnt) * 100,
           double(overflow_acc) / overflow_cnt * 100);
    printf("ARE: %.5lf, AAE: %.3lf\n", ARE, AAE);
    printf("Thereshold: %d < middle < %d\n", small_thereshold, large_thereshold);
    printf("Small: %d, Middle: %d, Large: %d\n", small, middle, large);
    printf("Small recall: %.2lf%%\n", double(small_acc) / small * 100);
    double pre = double(large_acc) / report_large * 100, rec = double(large_acc) / large * 100;
    printf("Large precision: %.2lf%%, recall: %.2lf%%, F1: %.2lf%%\n",
           double(large_acc) / report_large * 100,
           double(large_acc) / large * 100,
           2 * pre * rec / (pre + rec));
    printf("sARE: %.5lf, sAAE: %.3lf, sTAE: %lld\n", sARE, double(sTAE) / small, sTAE);
    printf("mARE: %.5lf, mAAE: %.3lf, mTAE: %lld\n", mARE, double(mTAE) / middle, mTAE);
    printf("lARE: %.5lf, lAAE: %.3lf, lTAE: %lld\n", lARE, double(lTAE) / large, lTAE);
}

extern void ParseArgs(int argc, char** argv);
extern vector<Record> load_data(const string& fileName);

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    printf("---------------------------------------------\n");
    auto input = load_data(fileName);
    printf("---------------------------------------------\n");
    printName(sketchName);
    printf("---------------------------------------------\n");
    groundtruth::item_count(input);
    groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
    auto realtime_sizes = groundtruth::realtime_size(input, BATCH_TIME);
    int seed = 0;
    if (sketchName == 1)
        size_test(input, realtime_sizes, HyperBloomFilter(memory, BATCH_TIME, seed));
    else if (sketchName == 2)
        size_test(input, realtime_sizes, ClockSketch<true>(memory, BATCH_TIME, seed));
    else if (sketchName == 3)
        size_test(input, realtime_sizes, TOBF<true>(memory, BATCH_TIME, 4, seed));
    else if (sketchName == 4)
        size_test(input, realtime_sizes, SWAMP<int, float, true>(memory, BATCH_TIME));
    printf("---------------------------------------------\n");
}
