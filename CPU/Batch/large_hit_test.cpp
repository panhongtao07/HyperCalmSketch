#include <cstdio>
#include <cassert>
#include <ctime>

#include "params.h"

using namespace std;

#include "../HyperCalm/HyperBloomFilter.h"
#include "../ComparedAlgorithms/ClockSketch.h"
#include "../ComparedAlgorithms/SWAMP.h"
#include "../ComparedAlgorithms/TOBF.h"
#include "../ComparedAlgorithms/groundtruth.h"


using namespace groundtruth::type_info;

void printName(int sketchName) {
    if (sketchName == 1) {
        printf("Test Hyper Bloom filter\n");
    } else if (sketchName == 2) {
        printf("Test Clock-Sketch\n");
    } else if (sketchName == 3) {
        printf("Test Time-Out Bloom filter\n");
    } else if (sketchName == 4) {
        printf("Test SWAMP\n");
    } else assert(false);
}

template <typename Sketch>
tuple<int, int> single_test(
    Sketch&& sketch,
    const vector<Record>& input,
    const map<ItemKey, vector<BatchTimeRange>>& item_batches
) {
    int correct_count = 0, report_count = 0;
    map<ItemKey, int> last_batch;
    for (int i = 0; i < input.size(); ++i) {
        auto& [key, time] = input[i];
        if (sketch.insert_cnt(key, time) + 1 != BATCH_SIZE_LIMIT)
            continue;
        ++report_count;
        if (!item_batches.count(key))
            continue;
        auto& batches = item_batches.at(key);
        int batch_id = last_batch[key];
        while (batch_id < int(batches.size()) && batches[batch_id].second < i)
            ++batch_id;
        last_batch[key] = batch_id;
        if (batch_id < int(batches.size()) && batches[batch_id].first <= i) {
            ++correct_count;
            ++last_batch[key]; // avoid duplicate
        }
    }
    return make_tuple(correct_count, report_count);
}

void large_hit_test(const vector<Record>& input) {
    groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
    groundtruth::item_count(input);
    timespec start_time, end_time;
    uint64_t time_ns;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    auto item_batches = groundtruth::item_batches(input, BATCH_TIME, BATCH_SIZE_LIMIT);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    time_ns = (end_time.tv_sec - start_time.tv_sec) * 1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    printf("groundtruth time:\t %f s\n", time_ns / 1e9);
    int total_count = 0;
    for (auto& [key, batches] : item_batches)
        total_count += int(batches.size());
    printf("---------------------------------------------\n");
    printName(sketchName);
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    int correct_count = 0, report_count = 0;
    for (int t = 0; t < repeat_time; ++t) {
        tuple<int, int> res;
        if (sketchName == 1)
            res = single_test(HyperBloomFilter(memory, BATCH_TIME, t), input, item_batches);
        else if (sketchName == 3)
            res = single_test(TOBF<true>(memory, BATCH_TIME, 4, t), input, item_batches);
        else if (sketchName == 4)
            res = single_test(SWAMP<int, float, true>(memory, BATCH_TIME), input, item_batches);
        else assert(false);
        correct_count += get<0>(res);
        report_count += get<1>(res);
    }
    printf("---------------------------------------------\n");
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    printf("Results:\n");
    time_ns = (end_time.tv_sec - start_time.tv_sec) * 1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    printf("Algorithm time:\t %f s\n", time_ns / 1e9);
    printf("Average Speed:\t %f M/s\n", 1e3 * input.size() * repeat_time / time_ns);
    auto recall = 1.0 * correct_count / total_count / repeat_time;
    auto precision = 1.0 * correct_count / report_count / repeat_time;
    printf("Precision: %f, Recall: %f\n", precision, recall);
    if (verbose) {
        printf("F1 Score: %f\n", 2 * recall * precision / (recall + precision));
        printf("Detail: %d correct, %d wrong, %d missed\n",
                correct_count, report_count - correct_count,
                total_count - correct_count);
    }
}

extern void ParseArgs(int argc, char** argv);
extern vector<pair<uint32_t, float>> load_data(const string& fileName);

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    printf("---------------------------------------------\n");
    auto input = load_data(fileName);
    printf("---------------------------------------------\n");
    large_hit_test(input);
    printf("---------------------------------------------\n");
}
