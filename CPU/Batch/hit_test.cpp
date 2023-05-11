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

template <typename Sketch>
vector<Index> insert_result(Sketch&& sketch, const vector<Record>& input) {
    vector<int> res;
    for (int i = 0; i < input.size(); ++i) {
        auto& [tkey, ttime] = input[i];
        if (sketch.insert(tkey, ttime)) {
            res.push_back(i);
        }
    }
	return res;
}

tuple<int, int> single_hit_test(
    const vector<Index>& results,
    const vector<Index>& objects,
    const vector<Index>& batches
) {
    int object_count = 0, correct_count = 0;
    int j = 0, k = 0;
    for (int i : results) {
        while (j + 1 < int(batches.size()) && batches[j] < i)
            ++j;
        if (j < int(batches.size()) && batches[j] == i) {
            ++correct_count;
        }
        while (k + 1 < int(objects.size()) && objects[k] < i)
            ++k;
        if (k < int(objects.size()) && objects[k] == i) {
            ++object_count;
        }
    }
    return make_tuple(object_count, correct_count);
}

void hit_test(const vector<Record>& input) {
    constexpr bool use_counter = false;
    groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
    groundtruth::item_count(input);
    auto [objects, batches] = groundtruth::batch(input, BATCH_TIME, BATCH_SIZE_LIMIT);
    printf("---------------------------------------------\n");
    printName(sketchName);
    uint64_t time_ns = 0;
    int object_count = 0, correct_count = 0, tot_our_size = 0;
    for (int t = 0; t < repeat_time; ++t) {
        timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        vector<Index> res;
        if (sketchName == 1)
            res = insert_result(HyperBloomFilter(memory, BATCH_TIME, t), input);
        else if (sketchName == 2)
            res = insert_result(ClockSketch<use_counter>(memory, BATCH_TIME, t), input);
        else if (sketchName == 3)
            res = insert_result(TOBF<use_counter>(memory, BATCH_TIME, 4, t), input);
        else if (sketchName == 4)
            res = insert_result(SWAMP<int, float, use_counter>(memory, BATCH_TIME), input);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        time_ns += (end_time.tv_sec - start_time.tv_sec) * uint64_t(1e9);
        time_ns += (end_time.tv_nsec - start_time.tv_nsec);
        tuple<int, int> test_result = single_hit_test(res, objects, batches);
        object_count += get<0>(test_result);
        correct_count += get<1>(test_result);
        tot_our_size += res.size();
    }
    printf("---------------------------------------------\n");
    auto recall = 1.0 * object_count / objects.size() / repeat_time;
    auto precision = 1.0 * correct_count / tot_our_size / repeat_time;
    printf("Results:\n");
    printf("Average Speed:\t %f M/s\n", 1e3 * input.size() * repeat_time / time_ns);
    printf("Recall Rate:\t %f\n", recall);
    printf("Precision Rate:\t %f\n", precision);
    if (verbose) {
        auto f1 = recall || precision ? 2 * recall * precision / (recall + precision) : 0.;
        printf("F1 Score:\t %f\n", f1);
        printf("Detail:\t %d correct, %d wrong, %d missed\n",
                correct_count, tot_our_size - correct_count, int(objects.size()) - object_count);
    }
}
