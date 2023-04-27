#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cassert>

#include "params.h"

using namespace std;

#include "../HyperCalm/HyperCalm.h"
#include "../ComparedAlgorithms/groundtruth.h"

using PeriodicKey = pair<int, int16_t>;

constexpr size_t BatchSize = 3;

template <bool use_counter>
void periodic_size_test(
    const vector<pair<uint32_t, float>>& input,
    vector<pair<pair<int, int16_t>, int>>& ans
) {
    if (sketchName != 1) {
        printf("Test is not implemented");
    }
    sort(ans.begin(), ans.end());
    vector<pair<PeriodicKey, int>> our;
    int corret_count = 0;
    double sae = 0, sre = 0;
    timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int t = 0; t < repeat_time; ++t) {
        tuple<int, long long, double> res;
        HyperCalm sketch(BATCH_TIME, UNIT_TIME, memory, t);
        for (auto &[tkey, ttime] : input) {
            if constexpr (use_counter)
                sketch.insert_filter<BatchSize>(tkey, ttime);
            else
                sketch.insert(tkey, ttime);
        }
        our = sketch.get_top_k(TOPK_THRESHOLD);
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
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    uint64_t time_ns =
        uint64_t(end_time.tv_sec - start_time.tv_sec) * 1000000000 +
        (end_time.tv_nsec - start_time.tv_nsec);
    printf("---------------------------------------------\n");
    if constexpr (use_counter)
        printf("Results with counter:\n");
    else
        printf("Results without counter:\n");
    printf("Average Speed:\t %f M/s\n",
           1e3 * input.size() * repeat_time / time_ns);
    printf("Recall Rate:\t %f\n",
           1.0 * corret_count / ans.size() / repeat_time);
    printf("AAE:\t\t %f\n", sae / corret_count);
    printf("ARE:\t\t %f\n", sre / corret_count);
}

extern void ParseArgs(int argc, char** argv);
extern vector<pair<uint32_t, float>> load_data(const string& fileName);

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    printf("---------------------------------------------\n");
    auto input = load_data(fileName);
    printf("---------------------------------------------\n");
    groundtruth::item_count(input);
    groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
    auto batches = groundtruth::batch(input, BATCH_TIME, BatchSize).first;
    auto ans = groundtruth::topk(input, batches, UNIT_TIME, TOPK_THRESHOLD);
    printf("Answer batchsize = %ld\n", BatchSize);
    printf("BATCH_TIME = %f\n", BATCH_TIME);
    printf("UNIT_TIME = %f\n", UNIT_TIME);
    printf("---------------------------------------------\n");
    periodic_size_test<true>(input, ans);
    printf("---------------------------------------------\n");
    periodic_size_test<false>(input, ans);
    printf("---------------------------------------------\n");
}
