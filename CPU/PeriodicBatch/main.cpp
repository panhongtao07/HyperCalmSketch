#include <vector>
using namespace std;

#include "params.h"

extern void ParseArgs(int argc, char** argv);
extern vector<pair<uint32_t, float>> load_data(const string& fileName);

extern void periodic_test(const vector<pair<uint32_t, float>>& input);

int main(int argc, char** argv) {
	ParseArgs(argc, argv);
	printf("---------------------------------------------\n");
	auto input = load_data(fileName);
	printf("---------------------------------------------\n");
	periodic_test(input);
	printf("---------------------------------------------\n");
}