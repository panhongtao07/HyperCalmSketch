#ifndef _HYPERBLOOMFILTER_H_
#define _HYPERBLOOMFILTER_H_

#include <immintrin.h>
#include <random>

namespace HyperBF {

static constexpr uint64_t ODD_BIT_MASK = 0x5555555555555555;
enum Bits : uint64_t {
	_00 = 0x0000000000000000,
	_01 = 0x5555555555555555,
	_10 = 0xaaaaaaaaaaaaaaaa,
	_11 = 0xffffffffffffffff,
};
static constexpr uint64_t STATE_MASKS[] = { Bits::_01, Bits::_10, Bits::_11 };
static constexpr uint64_t MASK[] = { ~Bits::_01, ~Bits::_10, ~Bits::_11 };
static constexpr size_t kStateNum = 3;

enum CounterType {
	None,
	SyncWithBucket,
	FastSync,
};

// HyperBloomFilter is a time-sensitive variant of Bloom Filter.
template <size_t CellBits = 2, CounterType counterType = SyncWithBucket>
class HyperBloomFilter {
	static constexpr size_t counter_type = counterType;
	static constexpr size_t CellPerBucket = sizeof(uint64_t) * 8 / CellBits;
	static constexpr uint64_t CellMask = (1 << CellBits) - 1;
	static constexpr size_t TableNum = 8;
	static_assert(kStateNum + 1 <= (1 << CellBits));

	static constexpr size_t getMaxReportSize() {
		if constexpr(counterType == None)
			return 1;
		else if constexpr(counterType == FastSync)
			return CellMask;
		else
			return CellMask + 1;
	}
public:
	uint64_t* buckets;
	uint64_t* counters;
	double time_threshold;
	uint32_t bucket_num;
	uint32_t seeds[TableNum + 1];
#ifdef SIMD
	static constexpr bool use_simd = true;
#else
	static constexpr bool use_simd = false;
#endif
	static constexpr bool use_counter = counterType != None;
	static constexpr size_t MaxReportSize = getMaxReportSize();

	HyperBloomFilter(uint32_t memory, double time_threshold, int seed = 123);
	~HyperBloomFilter() {
		delete[] buckets;
		delete[] counters;
	}

	int insert_cnt(int key, double time);
	bool insert(int key, double time);

private:
	inline uint32_t CalculatePos(uint32_t key, int i) {
		return (key * seeds[i]) >> 15;
	}
};


template <size_t CellBits, CounterType counterType>
HyperBloomFilter<CellBits, counterType>::HyperBloomFilter(
	uint32_t memory, double time_threshold, int seed
) : counters(nullptr), time_threshold(time_threshold) {
	uint32_t memsize = memory / sizeof(uint64_t);
	bucket_num = memsize - memsize % TableNum;
	buckets = new (align_val_t { 64 }) uint64_t[bucket_num] {};
	if constexpr(use_counter) {
		// Use extra memory to store counter, just testing, record it later
		counters = new (align_val_t { 64 }) uint64_t[bucket_num] {};
	}
	mt19937 rng(seed);
	for (int i = 0; i <= TableNum; ++i) {
		seeds[i] = rng();
	}
	if (memory >= 1024)
		printf("Memory = %.1f KB\t (Memory used in HyperBF)\n", memory / 1000.0);
	else
		printf("Memory = %u B\t (Memory used in HyperBF)\n", memory);
	printf("d = %d\t (Number of arrays in HyperBF)\n", bucket_num);
}

template <>
int HyperBloomFilter<2>::insert_cnt(int key, double time) {
	constexpr size_t CellBits = 2;
	int first_bucket_pos = CalculatePos(key, TableNum) % bucket_num & ~(TableNum - 1);
	int min_cnt = MaxReportSize, max_cnt = 0;
	if constexpr(use_simd) {
		__m512i* x = (__m512i*)(buckets + first_bucket_pos);
		uint64_t b[8];
		for (int i = 0; i < 8; ++i) {
			int now_tag = int(time / time_threshold + 1.0 * i / TableNum) % 3 + 1;
			int ban_tag_m1 = now_tag % 3;
			b[7 - i] = STATE_MASKS[ban_tag_m1];
		}
		__m512i is_ban_bits = _mm512_set_epi64(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
		is_ban_bits = _mm512_xor_epi64(is_ban_bits, *x);
		is_ban_bits = _mm512_or_epi64(is_ban_bits, _mm512_srli_epi64(is_ban_bits, 1));
		is_ban_bits = _mm512_and_epi64(is_ban_bits, _mm512_set1_epi64(ODD_BIT_MASK));
		__m512i mask = _mm512_or_epi64(is_ban_bits, _mm512_slli_epi64(is_ban_bits, 1));
		*x = _mm512_and_epi64(*x, mask);
		// TODO: support counter
		for (int i = 0; i < TableNum; ++i) {
			int cell_pos = CalculatePos(key, i) % CellPerBucket;
			int bucket_pos = (first_bucket_pos + i);

			int now_tag = int(time / time_threshold + 1.0 * i / TableNum) % 3 + 1;

			int old_tag = (buckets[bucket_pos] >> (CellBits * cell_pos)) & CellMask;
			if (old_tag == 0)
				min_cnt = 0;
			buckets[bucket_pos] ^= uint64_t(now_tag ^ old_tag) << (CellBits * cell_pos);
		}
	} else {
		for (int i = 0; i < TableNum; ++i) {
			int cell_pos = CalculatePos(key, i) % CellPerBucket;
			int bucket_pos = (first_bucket_pos + i);

			int now_tag = int(time / time_threshold + 1.0 * i / TableNum) % 3 + 1;
			int ban_tag_m1 = now_tag % 3;

			uint64_t is_ban_bits = buckets[bucket_pos] ^ STATE_MASKS[ban_tag_m1];
			// 1 = any(not same), 0 = all(same)
			is_ban_bits |= is_ban_bits >> 1;
			is_ban_bits &= ODD_BIT_MASK;
			// if all(same), clear
			uint64_t mask = is_ban_bits | (is_ban_bits << 1);
			uint64_t bucket = buckets[bucket_pos];
			bucket &= mask;

			auto move_bits = CellBits * cell_pos;
			if constexpr(counter_type == None) {
				int old_tag = (bucket >> move_bits) & CellMask;
				if (old_tag == 0)
					min_cnt = 0;
				bucket ^= uint64_t(now_tag ^ old_tag) << move_bits;
				buckets[bucket_pos] = bucket;
			} else {
				bool with_header = false;
				if constexpr(counter_type == SyncWithBucket) {
					with_header = (bucket & (CellMask << move_bits));
				}
				bucket &= ~(CellMask << move_bits);
				bucket |= uint64_t(now_tag) << move_bits;
				buckets[bucket_pos] = bucket;

				uint64_t counter = counters[bucket_pos];
				counter &= mask;
				if constexpr(counter_type == SyncWithBucket) {
					if (!with_header) {
						min_cnt = 0;
						// leave the counter empty, state will record the header,
						// so that we can know whether it's a new batch.
						counters[bucket_pos] = counter;
						continue;
					}
				}
				int cnt = (counter >> move_bits) & CellMask;
				min_cnt = min(min_cnt, cnt + with_header); // add the header
				if (cnt != CellMask) {
					counter ^= uint64_t(cnt + 1 ^ cnt) << move_bits;
				}
				counters[bucket_pos] = counter;
			}
		}
	}
	return min_cnt;
}

template <size_t CellBits, CounterType counterType>
bool HyperBloomFilter<CellBits, counterType>::insert(int key, double time) {
	return insert_cnt(key, time) == 0;
}

} // namespace HyperBF

#include "HyperBloomFilterTest.h"

using HyperBF::HyperBloomFilter;

#endif // _HYPERBLOOMFILTER_H_
