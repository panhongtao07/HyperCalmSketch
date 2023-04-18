#ifndef _GROUNDTRUTH_H_
#define _GROUNDTRUTH_H_
#include <algorithm>

namespace groundtruth {

enum RecordPos : int {
	START,
	INTER,
};

#ifndef REC_POS
#define REC_POS RecordPos::START
#endif

void adjust_params(
	const vector<pair<uint32_t, float>>& input,
	double& batch_time_threshold,
	double& unit_time
) {
	map<int, double> la;
	if (!batch_time_threshold) {
		double sum = 0;
		int cnt = 0;
		for (auto& [tkey, ttime] : input) {
			if (la.count(tkey)) {
				sum += ttime - la[tkey];
				++cnt;
			}
			la[tkey] = ttime;
		}
		batch_time_threshold = sum / cnt / 1 /*0*/;
	}
	if (!unit_time) {
		unit_time = batch_time_threshold * 10;
	}
}

template <RecordPos recordPos = RecordPos(REC_POS)>
pair<vector<int>, vector<int>> batch(
	const vector<pair<uint32_t, float>>& input,
	double& BATCH_TIME_THRESHOLD,
	double& UNIT_TIME,
	int BATCH_SIZE_THRESHOLD
) {
	adjust_params(input, BATCH_TIME_THRESHOLD, UNIT_TIME);
	map<int, double> la;
	map<int, int> cnt;
	int mx_cnt = 0;
	for (auto& [tkey, ttime] : input) {
		mx_cnt = max(mx_cnt, ++cnt[tkey]);
	}
	printf("# distinct items: %d\n", cnt.size());

	map<int, int> la_cnt;
	map<int, int> la_start;
	vector<int> objects;
	vector<int> batches;
	for (int i = 0; i < input.size(); ++i) {
		auto [tkey, ttime] = input[i];
		if (la.count(tkey) && ttime - la[tkey] <= BATCH_TIME_THRESHOLD) {
			++la_cnt[tkey];
		} else {
			la_cnt[tkey] = 1;
			la_start[tkey] = i;
			batches.push_back(i);
		}
		if (la_cnt[tkey] == BATCH_SIZE_THRESHOLD) {
			if constexpr (recordPos == RecordPos::START)
				objects.push_back(la_start[tkey]);
			else if constexpr (recordPos == RecordPos::INTER)
				objects.push_back(i);
		}
		la[tkey] = ttime;
	}
	if (BATCH_SIZE_THRESHOLD > 1) sort(objects.begin(), objects.end());

	printf("# items = %d\n", input.size());
	if (BATCH_SIZE_THRESHOLD > 1)
	printf("# batch size threshold = %d\n", BATCH_SIZE_THRESHOLD);
	printf("# object batches %d\n", objects.size());
	if (BATCH_SIZE_THRESHOLD > 1)
	printf("# filtered batches %d\n", int(batches.size() - objects.size()));
	printf("Freq. of the hottest item = %d\n", mx_cnt);

	return {objects, batches};
}

#undef REC_POS

vector<pair<pair<int, int16_t>, int>> topk(
	const vector<pair<uint32_t, float>>& input,
	const vector<int>& batches,
	double UNIT_TIME,
	int TOPK_THRESHOLD
) {
	map<pair<int, int16_t>, int> cnt;
	map<int, double> las;
	for (auto i : batches) {
		auto [v, t] = input[i];
		if (las.count(v)) {
			++cnt[{v, (t - las[v]) / UNIT_TIME}];
		}
		las[v] = t;
	}

	vector<pair<int, pair<int, int16_t>>> q;
	for (auto& [pr, c] : cnt)
		q.push_back({-c, pr});
	if (q.size() < TOPK_THRESHOLD) {
		printf("Not enough periodical batches: %d < %d\n", q.size(), TOPK_THRESHOLD);
		printf("Please increase the value of BATCH_TIME to get more batches.\n");
		exit(0);
	}
	nth_element(q.begin(), q.begin() + TOPK_THRESHOLD, q.end());
	vector<pair<pair<int, int16_t>, int>> q1;
	int mn_c = 1e9;
	for (int i = 0; i < TOPK_THRESHOLD; ++i) {
		q1.push_back({q[i].second, -q[i].first});
		mn_c = min(mn_c, -q[i].first);
	}
	return q1;
}

}  // namespace groundtruth

#endif  //_GROUNDTRUTH_H_