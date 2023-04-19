#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>
using namespace std;

#include <boost/program_options.hpp>

#include "../datasets/trace.h"
#include "../ComparedAlgorithms/clockSketch.h"
#include "../ComparedAlgorithms/SWAMP.h"
#include "../ComparedAlgorithms/TOBF.h"
#include "../ComparedAlgorithms/groundtruth.h"
#include "../HyperCalm/HyperBloomFilter.h"

using namespace boost::program_options;

string fileName;
int sketchName;
double BATCH_TIME, UNIT_TIME;
bool verbose = false;
int repeat_time = 1, BATCH_SIZE_LIMIT = 1, memory = 1e4;

void ParseArgs(int argc, char** argv) {
	options_description opts("Options");
	opts.add_options()
		("fileName,f", value<string>()->required(), "file name")
		("sketchName,s", value<int>()->required(), "sketch name")
		("time,t", value<int>()->required(), "repeat time")
		("topk,k", value<int>()->required(), "topk, unused")
		("batch_size,l", value<int>()->required(), "batch size threshold")
		("memory,m", value<int>()->required(), "memory")
		("batch_time,b", value<double>()->required(), "batch time")
		("unit_time,u", value<double>()->required(),"unit time")
		("verbose,V", "show verbose output");
	variables_map vm;

	store(parse_command_line(argc, argv, opts), vm);
	if (vm.count("fileName")) {
		fileName = vm["fileName"].as<string>();
	} else {
		printf("please use -f to specify the path of dataset.\n");
		exit(0);
	}
	if (vm.count("sketchName")) {
		sketchName = vm["sketchName"].as<int>();
		if (sketchName < 1 || sketchName > 4) {
			printf("sketchName < 1 || sketchName > 4\n");
			exit(0);
		}
	} else {
		printf("please use -s to specify the name of sketch.\n");
		exit(0);
	}
	if (vm.count("time"))
		repeat_time = vm["time"].as<int>();
	if (vm.count("batch_size"))
		BATCH_SIZE_LIMIT = vm["batch_size"].as<int>();
	if (vm.count("memory"))
		memory = vm["memory"].as<int>();
	if (vm.count("batch_time"))
		BATCH_TIME = vm["batch_time"].as<double>();
	if (vm.count("unit_time"))
		UNIT_TIME = vm["unit_time"].as<double>();
	if (vm.count("verbose"))
		verbose = true;
}

template <typename Sketch>
tuple<int, int, int> single_hit_test(
	Sketch&& sketch,
	const vector<pair<uint32_t, float>>& input,
	vector<int>& objects,
	vector<int>& batches
) {
	int object_count = 0, correct_count = 0, tot_our_size = 0;
	int j = 0, k = 0;
	for (int i = 0; i < input.size(); ++i) {
		auto& [tkey, ttime] = input[i];
		if (sketch.insert(tkey, ttime)) {
			++tot_our_size;
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
	}
	return make_tuple(object_count, correct_count, tot_our_size);
}

void hit_test(const vector<pair<uint32_t, float>>& input) {
	groundtruth::adjust_params(input, BATCH_TIME, UNIT_TIME);
	groundtruth::item_count(input);
	auto [objects, batches] = groundtruth::batch(input, BATCH_TIME, BATCH_SIZE_LIMIT);
	printf("---------------------------------------------\n");
	if (sketchName == 1) {
		puts("Test Hyper Bloom filter");
	} else if (sketchName == 2) {
		puts("Test Clock-Sketch");
	} else if (sketchName == 3) {
		puts("Test Time-Out Bloom filter");
	} else if (sketchName == 4) {
		puts("Test SWAMP");
	}
	int object_count = 0, correct_count = 0, tot_our_size = 0;
	timespec start_time, end_time;
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	for (int t = 0; t < repeat_time; ++t) {
		tuple<int, int, int> res;
		if (sketchName == 1)
			res = single_hit_test(HyperBloomFilter(memory, BATCH_TIME, t),input, objects, batches);
		else if (sketchName == 2)
			res = single_hit_test(clockSketch(memory, BATCH_TIME, t), input, objects, batches);
		else if (sketchName == 3)
			res = single_hit_test(TOBF(memory, BATCH_TIME, 4, t), input, objects, batches);
		else if (sketchName == 4)
			res = single_hit_test(SWAMP<int, float>(memory, BATCH_TIME), input, objects, batches);
		object_count += get<0>(res);
		correct_count += get<1>(res);
		tot_our_size += get<2>(res);
	}
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	uint64_t time_ns = uint64_t(end_time.tv_sec - start_time.tv_sec) * 1000000000 + (end_time.tv_nsec - start_time.tv_nsec);
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
	printf("---------------------------------------------\n");
}

int main(int argc, char** argv) {
	ParseArgs(argc, argv);
	printf("---------------------------------------------\n");
	vector<pair<uint32_t, float>> input;
	if (fileName.back() == 't')
		input = loadCAIDA(fileName.c_str());
	else
		input = loadCRITEO(fileName.c_str());
	printf("---------------------------------------------\n");
	hit_test(input);
}