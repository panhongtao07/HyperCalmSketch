#ifndef _BATCH_PARAMS_H_
#define _BATCH_PARAMS_H_

#include <string>

inline std::string fileName;
inline int sketchName;
inline double BATCH_TIME, UNIT_TIME;
inline bool verbose = false;
inline int repeat_time = 1, BATCH_SIZE_LIMIT = 1, memory = 1e4;

#endif
