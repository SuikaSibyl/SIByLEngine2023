#ifndef _SRENDERER_COMMON_BIT_OPERATION_HEADER_
#define _SRENDERER_COMMON_BIT_OPERATION_HEADER_

uint nth_bit_set_pop(uint value, uint n) {
    const uint pop2 = (value & 0x55555555u) + ((value >> 1) & 0x55555555u);
    const uint pop4 = (pop2 & 0x33333333u) + ((pop2 >> 2) & 0x33333333u);
    const uint pop8 = (pop4 & 0x0f0f0f0fu) + ((pop4 >> 4) & 0x0f0f0f0fu);
    const uint pop16 = (pop8 & 0x00ff00ffu) + ((pop8 >> 8) & 0x00ff00ffu);
    const uint pop32 = (pop16 & 0x000000ffu) + ((pop16 >> 16) & 0x000000ffu);
    uint rank = 0; uint temp;
    // -------------------------------------------
    if (n++ >= pop32) return 32;
    // -------------------------------------------
    temp = pop16 & 0xffu;
    /* if (n > temp) { n -= temp; rank += 16; } */
    rank += ((temp - n) & 256) >> 4;
    n -= temp & ((temp - n) >> 8);
    // -------------------------------------------
    temp = (pop8 >> rank) & 0xffu;
    /* if (n > temp) { n -= temp; rank += 8; } */
    rank += ((temp - n) & 256) >> 5;
    n -= temp & ((temp - n) >> 8);
    // -------------------------------------------
    temp = (pop4 >> rank) & 0x0fu;
    /* if (n > temp) { n -= temp; rank += 4; } */
    rank += ((temp - n) & 256) >> 6;
    n -= temp & ((temp - n) >> 8);
    // -------------------------------------------
    temp = (pop2 >> rank) & 0x03u;
    /* if (n > temp) { n -= temp; rank += 2; } */
    rank += ((temp - n) & 256) >> 7;
    n -= temp & ((temp - n) >> 8);
    // -------------------------------------------
    temp = (value >> rank) & 0x01u;
    /* if (n > temp) rank += 1; */
    rank += ((temp - n) & 256) >> 8;
    // -------------------------------------------
    return rank;
}

/** Set the rightmost n bits of a uint32_t */
uint set_rightmost_n_bits(uint n) {
    if (n >= 32) return 0xFFFFFFFFu;
    return (1u << n) - 1u;
}
uint64_t set_rightmost_n_bits_64(uint n) {
    if (n >= 64) return 0xFFFFFFFFFFFFFFFFull;
    return (1ull << n) - 1ull;
}
/** Set the lestmost n bits of a uint32_t */
uint set_leftmost_n_bits(uint n) {
    if (n >= 32) return 0xFFFFFFFFu;
    return ~set_rightmost_n_bits(32 - n);
}
uint64_t set_leftmost_n_bits_64(uint n) {
    if (n >= 64) return 0xFFFFFFFFFFFFFFFFull;
    return ~set_rightmost_n_bits_64(64 - n);
}

/** Counts the number of elemenets in S less than or equal to m. */
uint rank(uint S, uint m) {
    const uint mask = set_leftmost_n_bits(m + 1);
    return countbits(S & mask);
}

/** Counts the number of elemenets in S less than or equal to m. */
uint rank(uint64_t S, uint m) {
    uint additional_rank = 0;
    uint section = uint(S >> 32);
    if (m >= 32) {
        additional_rank = countbits(section);
        section = uint(S & 0xFFFFFFFFu);
        m -= 32;
    }
    return rank(section, m) + additional_rank;
}

/** Finds the mth smallest element in S. */
uint flip(uint x, uint m) {
    uint32_t mask = 1u << (31 - m);
    return x ^ mask;
}
/** Finds the mth smallest element in S. */
uint64_t flip(uint64_t x, uint m) {
    uint64_t mask = 1ull << (63 - m);
    return x ^ mask;
}

#endif // !_SRENDERER_COMMON_BIT_OPERATION_HEADER_