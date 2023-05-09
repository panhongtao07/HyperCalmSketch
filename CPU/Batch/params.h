#ifndef _BATCH_PARAMS_H_
#define _BATCH_PARAMS_H_

#include <string>

inline std::string fileName;
inline int sketchName;
inline double BATCH_TIME, UNIT_TIME;
inline bool verbose = false;
inline int repeat_time = 1, BATCH_SIZE_LIMIT = 1, memory = 1e4;

static void printName(int sketchName) {
    if (sketchName == 1) {
        printf("Test Hyper Bloom filter\n");
    } else if (sketchName == 2) {
        printf("Test Clock-Sketch\n");
    } else if (sketchName == 3) {
        printf("Test Time-Out Bloom filter\n");
    } else if (sketchName == 4) {
        printf("Test SWAMP\n");
    };
}

#endif
