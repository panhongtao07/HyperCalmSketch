#include <cmath>

#include "params.h"

using namespace std;

#include "../HyperCalm/HyperBloomFilter.h"
#include "../ComparedAlgorithms/groundtruth.h"

void size_test(const vector<pair<uint32_t, float>>& input) {
    groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
    auto realtime_sizes = groundtruth::realtime_size(input, BATCH_TIME);
    double ARE = 0, AAE = 0;
    if (sketchName != 1) {
        printf("Test is not implemented");
    }
    HyperBloomFilter hbf(memory, BATCH_TIME, 0);
    for (int i = 0; i < input.size(); ++i) {
        auto &[key, time] = input[i];
        int real_size = realtime_sizes[i];
        int size = hbf.insert_cnt(key, time) + 1;
        real_size = min(real_size, 4);
        int diff = abs(size - real_size);
        double relative_error = double(diff) / real_size;
        AAE += diff;
        ARE += relative_error;
    }
    AAE /= input.size();
    ARE /= input.size();
    printf("Realtime size test\n");
    printf("ARE: %.3lf, AAE: %.3lf\n", ARE, AAE);
}

extern void ParseArgs(int argc, char** argv);
extern vector<pair<uint32_t, float>> load_data(const string& fileName);

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    printf("---------------------------------------------\n");
    auto input = load_data(fileName);
    printf("---------------------------------------------\n");
    groundtruth::item_count(input);
    size_test(input);
    printf("---------------------------------------------\n");
}
