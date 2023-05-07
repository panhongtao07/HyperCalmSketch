#ifndef _GROUNDTRUTH_H_
#define _GROUNDTRUTH_H_

#include <algorithm>
#include <map>
#include <vector>
#include <utility>

namespace groundtruth {

inline namespace type_info {

using Index = int;
using ItemKey = uint32_t;
using TimeStamp = float;
using Record = pair<ItemKey, TimeStamp>;
using BatchTimeRange = pair<Index, Index>;
using PeriodicKey = pair<int, int16_t>;

}  // namespace type_info

enum RecordPos : int {
	START,
	INTER,
};

#ifndef REC_POS
#define REC_POS RecordPos::START
#endif

void adjust_params(
	const vector<Record>& input,
	double& batch_time_threshold,
	double& unit_time
) {
	map<int, double> la;
	if (!batch_time_threshold) {
		double sum = 0;
		int cnt = 0;
		for (auto& [key, time] : input) {
			if (la.count(key)) {
				sum += time - la[key];
				++cnt;
			}
			la[key] = time;
		}
		batch_time_threshold = sum / cnt / 1 /*0*/;
	}
	if (!unit_time) {
		unit_time = batch_time_threshold * 10;
	}
}

map<ItemKey, int> item_count(const vector<Record>& input) {
	map<ItemKey, int> cnt;
	int max_cnt = 0;
	for (auto& [key, time] : input) {
		max_cnt = max(max_cnt, ++cnt[key]);
	}
	printf("Freq. of the hottest item = %d\n", max_cnt);
	printf("distinct items: %zu\n", cnt.size());
	printf("items: %zu\n", input.size());
	return cnt;
}

vector<int> realtime_size(
	const vector<Record>& input,
	double BATCH_TIME_THRESHOLD
) {
	map<ItemKey, double> last_time;
	map<ItemKey, int> last_cnt;
	vector<int> realtime_sizes;
	int max_size = 0;
	for (auto& [key, time] : input) {
		if (last_time.count(key) && time - last_time[key] <= BATCH_TIME_THRESHOLD) {
			max_size = max(max_size, ++last_cnt[key]);
		} else {
			last_cnt[key] = 1;
		}
		last_time[key] = time;
		realtime_sizes.push_back(last_cnt[key]);
	}
	printf("Largest batch: %d\n", max_size);
	return realtime_sizes;
}

template <RecordPos recordPos = RecordPos(REC_POS)>
pair<vector<Index>, vector<Index>> batch(
	const vector<Record>& input,
	double BATCH_TIME_THRESHOLD,
	int BATCH_SIZE_THRESHOLD
) {
	map<ItemKey, Index> la_start;
	vector<Index> objects;
	vector<Index> batches;
	auto realtime_sizes = realtime_size(input, BATCH_TIME_THRESHOLD);
	for (int i = 0; i < realtime_sizes.size(); ++i) {
		auto cnt = realtime_sizes[i];
		auto key = input[i].first;
		if (cnt == 1) {
			la_start[key] = i;
			batches.push_back(i);
		}
		if (cnt == BATCH_SIZE_THRESHOLD) {
			if constexpr (recordPos == RecordPos::START)
				objects.push_back(la_start[key]);
			else if constexpr (recordPos == RecordPos::INTER)
				objects.push_back(i);
		}
	}
	if (BATCH_SIZE_THRESHOLD > 1) {
		sort(objects.begin(), objects.end());
		printf("batch size threshold = %d\n", BATCH_SIZE_THRESHOLD);
		printf("filtered batches: %d\n", int(batches.size() - objects.size()));
	}
	printf("object batches: %zu\n", objects.size());
	return {objects, batches};
}

#undef REC_POS

map<ItemKey, vector<BatchTimeRange>> item_batches(
	const vector<Record>& input,
	double time_threshold,
	int size_threshold
) {
	map<ItemKey, Index> batch_start, batch_end;
	map<ItemKey, vector<BatchTimeRange>> item_batches;
	map<ItemKey, int> batch_size;
	auto realtime_sizes = realtime_size(input, time_threshold);
	for (int i = 0; i < realtime_sizes.size(); ++i) {
		auto cnt = realtime_sizes[i];
		auto key = input[i].first;
		if (cnt == 1) {
			if (batch_size[key] >= size_threshold) {
				item_batches[key].push_back({batch_start[key], batch_end[key]});
			}
			batch_start[key] = i;
		}
		batch_end[key] = i;
		batch_size[key] = cnt;
	}
	for (auto& [key, size] : batch_size) {
		if (size >= size_threshold) {
			item_batches[key].push_back({batch_start[key], batch_end[key]});
		}
	}
	return item_batches;
}

vector<pair<PeriodicKey, int>> topk(
	const vector<Record>& input,
	const vector<Index>& batches,
	double UNIT_TIME,
	int TOPK_THRESHOLD
) {
	map<PeriodicKey, int> cnt;
	map<int, double> las;
	for (auto i : batches) {
		auto [v, t] = input[i];
		if (las.count(v)) {
			++cnt[{v, (t - las[v]) / UNIT_TIME}];
		}
		las[v] = t;
	}

	vector<pair<int, PeriodicKey>> q;
	for (auto& [pr, c] : cnt)
		q.push_back({-c, pr});
	if (q.size() < TOPK_THRESHOLD) {
		printf("Not enough periodical batches: %zu < %d\n", q.size(), TOPK_THRESHOLD);
		printf("Please increase the value of BATCH_TIME to get more batches.\n");
		exit(0);
	}
	nth_element(q.begin(), q.begin() + TOPK_THRESHOLD, q.end());
	vector<pair<PeriodicKey, int>> q1;
	int mn_c = 1e9;
	for (int i = 0; i < TOPK_THRESHOLD; ++i) {
		q1.push_back({q[i].second, -q[i].first});
		mn_c = min(mn_c, -q[i].first);
	}
	return q1;
}

}  // namespace groundtruth

#endif  //_GROUNDTRUTH_H_