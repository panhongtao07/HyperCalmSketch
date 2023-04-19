#ifndef _HYPERBLOOMFILTER_H_
#define _HYPERBLOOMFILTER_H_

#include <immintrin.h>
#include <random>

#include "../lib/param.h"


namespace {

static constexpr size_t kCellBits = 2;
static constexpr size_t kCellPerBucket = sizeof(uint64_t) * 8 / kCellBits;
static constexpr uint64_t kCellMask = (1 << kCellBits) - 1;
static constexpr size_t kTableNum = 8;

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
static_assert(kStateNum + 1 <= (1 << kCellBits));

}



class HyperBloomFilter {
public:
	uint64_t* buckets;
	uint64_t* counters;
	double time_threshold;
	uint32_t bucket_num;
	uint32_t seeds[kTableNum + 1];
#ifdef SIMD
	static constexpr bool use_simd = true;
#else
	static constexpr bool use_simd = false;
#endif
	static constexpr bool use_counter = !use_simd;

	HyperBloomFilter(uint32_t memory, double time_threshold_, int seed = 123)
		: counters(nullptr), time_threshold(time_threshold_) {
		uint32_t memsize = memory / sizeof(uint64_t);
		bucket_num = memsize;
		buckets = new (align_val_t { 64 }) uint64_t[bucket_num] {};
		if constexpr(use_counter) {
			// Use extra memory to store counter, just testing, record it later
			counters = new (align_val_t { 64 }) uint64_t[bucket_num] {};
		}
		mt19937 rng(seed);
		for (int i = 0; i <= kTableNum; ++i) {
			seeds[i] = rng();
		}
        printf("d = %d\t (Number of arrays in HyperBF)\n",bucket_num);
	}

	~HyperBloomFilter() {
		delete[] buckets;
	}

	template <size_t CellBits = kCellBits>
	int insert_cnt(int key, double time);
	template <size_t CellBits = kCellBits>
	bool insert(int key, double time);

private:
	inline uint32_t CalculatePos(uint32_t key, int i) {
		return CalculateBucketPos(key, seeds[i]);
	}
};


template <>
int HyperBloomFilter::insert_cnt<2>(int key, double time) {
	static_assert(kCellBits == 2, "Insert is specified for 2 bit cell!");
	int first_bucket_pos = CalculatePos(key, kTableNum) % bucket_num & ~(kTableNum - 1);
	int min_cnt = kCellMask, max_cnt = 0;
	if constexpr(use_simd) {
		__m512i* x = (__m512i*)(buckets + first_bucket_pos);
		uint64_t b[8];
		for (int i = 0; i < 8; ++i) {
			int now_tag = int(time / time_threshold + 1.0 * i / kTableNum) % 3 + 1;
			int ban_tag_m1 = now_tag % 3;
			b[7 - i] = STATE_MASKS[ban_tag_m1];
		}
		__m512i is_ban_bits = _mm512_set_epi64(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
		is_ban_bits = _mm512_xor_epi64(is_ban_bits, *x);
		is_ban_bits = _mm512_or_epi64(is_ban_bits, _mm512_srli_epi64(is_ban_bits, 1));
		is_ban_bits = _mm512_and_epi64(is_ban_bits, _mm512_set1_epi64(ODD_BIT_MASK));
		__m512i mask = _mm512_or_epi64(is_ban_bits, _mm512_slli_epi64(is_ban_bits, 1));
		*x = _mm512_and_epi64(*x, mask);
		for (int i = 0; i < kTableNum; ++i) {
			int cell_pos = CalculatePos(key, i) % kCellPerBucket;
			int bucket_pos = (first_bucket_pos + i);

			int now_tag = int(time / time_threshold + 1.0 * i / kTableNum) % 3 + 1;

			int old_tag = (buckets[bucket_pos] >> (kCellBits * cell_pos)) & kCellMask;
			if (old_tag == 0)
				min_cnt = 0;
			buckets[bucket_pos] ^= uint64_t(now_tag ^ old_tag) << (kCellBits * cell_pos);
		}
	} else {
		for (int i = 0; i < kTableNum; ++i) {
			int cell_pos = CalculatePos(key, i) % kCellPerBucket;
			int bucket_pos = (first_bucket_pos + i);

			int now_tag = int(time / time_threshold + 1.0 * i / kTableNum) % 3 + 1;
			int ban_tag_m1 = now_tag % 3;

			uint64_t is_ban_bits = buckets[bucket_pos] ^ STATE_MASKS[ban_tag_m1];
			// 1 = any(not same), 0 = all(same)
			is_ban_bits |= is_ban_bits >> 1;
			is_ban_bits &= ODD_BIT_MASK;
			// if all(same), clear
			uint64_t mask = is_ban_bits | (is_ban_bits << 1);
			buckets[bucket_pos] &= mask;

			if constexpr(use_counter) {
				buckets[bucket_pos] &= ~(kCellMask << (kCellBits * cell_pos));
				buckets[bucket_pos] |= uint64_t(now_tag) << (kCellBits * cell_pos);

				counters[bucket_pos] &= mask;
				int cnt = counters[bucket_pos] >> (kCellBits * cell_pos) & kCellMask;
				max_cnt = max(max_cnt, cnt);
				min_cnt = min(min_cnt, cnt);
				if (cnt != kCellMask) {
					counters[bucket_pos] ^= uint64_t(cnt + 1 ^ cnt) << (kCellBits * cell_pos);
				}
			} else {
				int old_tag = (buckets[bucket_pos] >> (kCellBits * cell_pos)) & kCellMask;
				if (old_tag == 0)
					min_cnt = 0;
				buckets[bucket_pos] ^= uint64_t(now_tag ^ old_tag) << (kCellBits * cell_pos);
			}
		}
	}
	return min_cnt;
}

template <size_t CellBits>
bool HyperBloomFilter::insert(int key, double time) {
	return insert_cnt<CellBits>(key, time) == 0;
}

#endif // _HYPERBLOOMFILTER_H_
