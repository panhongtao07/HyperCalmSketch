#include <cstdlib>
#include <cassert>
#include <ctime>
#include <vector>

#include "params.h"

using namespace std;

#include "../ComparedAlgorithms/ClockUSS.h"
#include "../HyperCalm/HyperCalm.h"
#include "../ComparedAlgorithms/groundtruth.h"

using PeriodicKey = pair<int, int16_t>;

template <typename Sketch>
tuple<int, long long, double> single_test(
    const Sketch& sketch,
    const vector<pair<uint32_t, float>>& input,
    const vector<pair<PeriodicKey, int>>& ans
) {
    int corret_count = 0;
    long long sae = 0;
    double sre = 0;
    for (auto &[tkey, ttime] : input) {
        sketch.insert(tkey, ttime);
    }
    vector<pair<PeriodicKey, int>> our = sketch.get_top_k(TOPK_THRESHOLD);
    sort(our.begin(), our.end());
    int j = 0;
    for (auto &[key, freq] : our) {
        while (j + 1 < ans.size() && ans[j].first < key)
            ++j;
        if (j < ans.size() && ans[j].first == key) {
            ++corret_count;
            auto diff = abs(ans[j].second - freq);
            sae += diff;
            sre += diff / double(ans[j].second);
        }
    }
    return {corret_count, sae, sre};
}

void periodic_test(const vector<pair<uint32_t, float>>& input) {
    groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
    groundtruth::item_count(input);
    auto batches = groundtruth::batch(input, BATCH_TIME, BATCH_SIZE_LIMIT).first;
    auto ans = groundtruth::topk(input, batches, UNIT_TIME, TOPK_THRESHOLD);
    printf("BATCH_TIME = %f\n", BATCH_TIME);
    printf("UNIT_TIME = %f\n", UNIT_TIME);
    printf("---------------------------------------------\n");
    if (sketchName == 1) {
        puts("Test HyperCalm");
    } else {
        puts("Test Clock+USS");
    }
    sort(ans.begin(), ans.end());
    int corret_count = 0;
    double sae = 0, sre = 0;
    timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int t = 0; t < repeat_time; ++t) {
        tuple<int, long long, double> res;
        if (sketchName == 1)
            res = single_test(HyperCalm(BATCH_TIME, UNIT_TIME, memory, t), input, ans);
        else
            res = single_test(ClockUSS(BATCH_TIME, UNIT_TIME, memory, t), input, ans);
        corret_count += get<0>(res);
        sae += get<1>(res);
        sre += get<2>(res);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    uint64_t time_ns =
        uint64_t(end_time.tv_sec - start_time.tv_sec) * 1000000000 +
        (end_time.tv_nsec - start_time.tv_nsec);
    printf("---------------------------------------------\n");
    printf("Results:\n");
    printf("Average Speed:\t %f M/s\n",
           1e3 * input.size() * repeat_time / time_ns);
    printf("Recall Rate:\t %f\n",
           1.0 * corret_count / ans.size() / repeat_time);
    printf("AAE:\t\t %f\n", sae / corret_count);
    printf("ARE:\t\t %f\n", sre / corret_count);
}
