#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <iostream>

#include "params.h"

using namespace std;

#include "../HyperCalm/HyperCalm.h"
#include "../ComparedAlgorithms/ClockUSS.h"
#include "../ComparedAlgorithms/groundtruth.h"

using namespace groundtruth::type_info;

constexpr size_t cellbits = 2;
constexpr size_t BatchSize = (1 << cellbits);

template <bool use_counter>
void periodic_size_test(
    const vector<Record>& input,
    vector<pair<PeriodicKey, int>>& ans
) {
    printName(sketchName);
    sort(ans.begin(), ans.end());
    vector<pair<PeriodicKey, int>> our;
    int corret_count = 0;
    double sae = 0, sre = 0;
    uint64_t time_ns = 0;
    auto check = [&] (auto sketch) {
        timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (auto &[key, time] : input) {
            if constexpr (use_counter)
                sketch.template insert_filter<BatchSize>(key, time);
            else
                sketch.insert(key, time);
        }
        our = sketch.get_top_k(TOPK_THRESHOLD);
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_ns += (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
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
    };
    for (int t = 0; t < repeat_time; ++t) {
        if (sketchName == 1)
            check(HyperCalm(BATCH_TIME, UNIT_TIME, memory, t));
        else if (sketchName == 2)
            check(ClockUSS<use_counter>(BATCH_TIME, UNIT_TIME, memory, t));
    }
    cout << "---------------------------------------------" << endl;
    if constexpr (use_counter)
        cout << "Results with counter:" << endl;
    else
        cout << "Results without counter:" << endl;
    cout << "Average Speed:\t " << 1e3 * input.size() * repeat_time / time_ns << " M/s" << endl;
    cout << "Recall Rate:\t " << 1.0 * corret_count / ans.size() / repeat_time << endl;
    cout << "AAE:\t\t " << sae / corret_count << endl;
    cout << "ARE:\t\t " << sre / corret_count << endl;
}

extern void ParseArgs(int argc, char** argv);
extern vector<Record> load_data(const string& fileName);

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    cout << "---------------------------------------------" << '\n';
    auto input = load_data(fileName);
    cout << "---------------------------------------------" << '\n';
    groundtruth::item_count(input);
    groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
    auto batches = groundtruth::batch(input, BATCH_TIME, BatchSize).first;
    auto ans = groundtruth::topk(input, batches, UNIT_TIME, TOPK_THRESHOLD);
    printf("BATCH_TIME = %f, UNIT_TIME = %f\n", BATCH_TIME, UNIT_TIME);
    if (memory > 1000)
        cout << "Total Memory: " << memory / 1000. << " KB";
    else
        cout << "Total Memory: " << memory << " B";
    cout << ", Top K: " << TOPK_THRESHOLD << '\n';
    cout << "---------------------------------------------" << '\n';
    periodic_size_test<true>(input, ans);
    // cout << "---------------------------------------------" << endl;
    // periodic_size_test<false>(input, ans);
    cout << "---------------------------------------------" << endl;
}
