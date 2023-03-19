#ifndef _HYPERBLOOMFILTER_H_
#define _HYPERBLOOMFILTER_H_

#include <immintrin.h>
#include <random>

#include "../lib/param.h"

namespace {

static constexpr uint64_t ODD_BIT_MASK = 0x5555555555555555;
enum Bits : uint64_t {
	_00 = 0x0000000000000000,
	_01 = 0x5555555555555555,
	_10 = 0xaaaaaaaaaaaaaaaa,
	_11 = 0xffffffffffffffff,
};
static constexpr uint64_t STATE_MASKS[] = { Bits::_01, Bits::_10, Bits::_11 };
static constexpr uint64_t MASK[] = { ~Bits::_01, ~Bits::_10, ~Bits::_11 };	

}

#define CELL_PER_BUCKET 32
#define TABLE_NUM 8
class HyperBloomFilter {
public:
	uint64_t* buckets;
	double time_threshold;
	uint32_t bucket_num;
	uint32_t seeds[TABLE_NUM + 1];
#ifdef SIMD
	static constexpr bool use_simd = true;
#else
	static constexpr bool use_simd = false;
#endif

	HyperBloomFilter(uint32_t memory, double time_threshold_, int seed = 123) {
		bucket_num = memory / TABLE_NUM / sizeof(uint64_t) * TABLE_NUM;
		time_threshold = time_threshold_;
		buckets = new (align_val_t { 64 }) uint64_t[bucket_num];
		memset(buckets, 0, bucket_num * sizeof(*buckets));
		mt19937 rng(seed);
		for (int i = 0; i <= TABLE_NUM; ++i) {
			seeds[i] = rng();
		}
        printf("d = %d\t (Number of arrays in HyperBF)\n",bucket_num);
	}

	~HyperBloomFilter() {
		delete[] buckets;
	}

	bool insert(int key, double time);

private:
	inline uint32_t CalculatePos(uint32_t key, int i) {
		return CalculateBucketPos(key, seeds[i]);
	}
};


bool HyperBloomFilter::insert(int key, double time) {
	int first_bucket_pos = CalculatePos(key, TABLE_NUM) % bucket_num & ~(TABLE_NUM - 1);
	bool ans = 0;
	if constexpr(use_simd) {
		__m512i* x = (__m512i*)(buckets + first_bucket_pos);
		uint64_t b[8];
		for (int i = 0; i < 8; ++i) {
			int now_tag = int(time / time_threshold + 1.0 * i / TABLE_NUM) % 3 + 1;
			int ban_tag_m1 = now_tag % 3;
			b[7 - i] = MASK[ban_tag_m1];
		}
		__m512i ban = _mm512_set_epi64(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
		ban = _mm512_xor_epi64(*x, ban);
		ban = _mm512_and_epi64(_mm512_rol_epi64(ban, 63), ban);
		ban = _mm512_and_epi64(ban, _mm512_set1_epi64(ODD_BIT_MASK));
		ban = _mm512_or_epi64(_mm512_rol_epi64(ban, 1), ban);
		*x = _mm512_andnot_epi64(ban, *x);
		for (int i = 0; i < TABLE_NUM; ++i) {
			int pos = CalculatePos(key, i) % 32;
			int bucket_pos = (first_bucket_pos + i);

			int now_tag = int(time / time_threshold + 1.0 * i / TABLE_NUM) % 3 + 1;

			uint64_t& x = buckets[bucket_pos];
			int old_tag = (x >> (2 * pos)) & 3;
			if (old_tag == 0)
				ans = 1;
			x += uint64_t(now_tag - old_tag) << (2 * pos);
		}
	} else {
		for (int i = 0; i < TABLE_NUM; ++i) {
			int pos = CalculatePos(key, i) % 32;
			int bucket_pos = (first_bucket_pos + i);

			int now_tag = int(time / time_threshold + 1.0 * i / TABLE_NUM) % 3 + 1;
			int ban_tag_m1 = now_tag % 3;

			uint64_t is_ban_bits = buckets[bucket_pos] ^ MASK[ban_tag_m1];
			is_ban_bits &= is_ban_bits >> 1;
			is_ban_bits &= ODD_BIT_MASK;
			uint64_t mask = is_ban_bits | (is_ban_bits << 1);
			buckets[bucket_pos] &= ~mask;

			int old_tag = (buckets[bucket_pos] >> (2 * pos)) & 3;
			if (old_tag == 0)
				ans = 1;
			buckets[bucket_pos] += uint64_t(now_tag - old_tag) << (2 * pos);
		}
	}
	return ans;
}


#undef CELL_PER_BUCKET
#undef TABLE_NUM


#endif // _HYPERBLOOMFILTER_H_
