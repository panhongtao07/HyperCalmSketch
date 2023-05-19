#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "../datasets/trace.h"

#include "params.h"

using namespace std;
using namespace boost::program_options;

void ParseArgs(int argc, char** argv) {
    options_description opts("Options");
    opts.add_options()
        ("fileName,f", value<string>()->required(), "file name")
        ("sketchName,s", value<int>()->required(), "sketch name")
        ("time,t", value<int>()->required(), "repeat time")
        ("topk,k", value<int>()->required(), "topk")
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
        if (sketchName < 1 || sketchName > 2) {
            printf("sketchName < 1 || sketchName > 2\n");
            exit(0);
        }
    } else {
        printf("please use -s to specify the name of sketch.\n");
        exit(0);
    }
    if (vm.count("time"))
        repeat_time = vm["time"].as<int>();
    if (vm.count("topk"))
        TOPK_THRESHOLD = vm["topk"].as<int>();
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

vector<pair<uint32_t, float>> load_data(const string& fileName) {
    vector<pair<uint32_t, float>> input;
    if (fileName.substr(0, 2) == "[L") {
        int total_num = 60;
        if (fileName.find(']') == string::npos) {
            throw runtime_error("invalid file name");
        }
        if (auto nums = fileName.substr(2, fileName.find(']') - 2); !nums.empty()) {
            total_num = stoi(nums);
        }
        string fname = fileName.substr(fileName.find(']') + 1);
        char buf[100];
        double last_time = 0;
        for (int i = 0; i < total_num; ++i) {
            snprintf(buf, 100, fname.c_str(), i);
            auto single_data = load_data(buf);
            for (auto& [key, time] : single_data) {
                input.emplace_back(key, time + last_time);
            }
            last_time += single_data.back().second;
        }
        return input;
    }
    if (fileName.back() == 't')
        input = loadCAIDA(fileName.c_str());
    else
        input = loadCRITEO(fileName.c_str());
    return input;
}
