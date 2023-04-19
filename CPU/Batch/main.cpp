#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>

#include "params.h"

using namespace std;

extern void ParseArgs(int argc, char** argv);
extern vector<pair<uint32_t, float>> load_data(const string& fileName);

extern void hit_test(const vector<pair<uint32_t, float>>& input);

int main(int argc, char** argv) {
	ParseArgs(argc, argv);
	printf("---------------------------------------------\n");
	auto input = load_data(fileName);
	printf("---------------------------------------------\n");
	hit_test(input);
}
