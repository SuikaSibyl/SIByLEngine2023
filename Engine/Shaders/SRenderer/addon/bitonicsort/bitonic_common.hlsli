#ifndef _SRENDERER_ADDON_BITONIC_SORT_INTERFACE_
#define _SRENDERER_ADDON_BITONIC_SORT_INTERFACE_

// Key Type
// 0: uint32_t
// 1: uint64_t

static const uint32_t null_element_32 = 0xFFFFFFFF;
static const uint64_t null_element_64 = 0xFFFFFFFFFFFFFFFF;

#ifndef KEY_TYPE_ENUM
#define KEY_TYPE_ENUM 1
#endif

#if KEY_TYPE_ENUM == 0
#define KEY_TYPE uint32_t
#define NULL_KEY null_element_32

#elif KEY_TYPE_ENUM == 1
#define KEY_TYPE uint64_t
#define NULL_KEY null_element_64
#endif

// Takes Value and widens it by one bit at the location of the bit
// in the mask.  A one is inserted in the space.  OneBitMask must
// have one and only one bit set.
uint InsertOneBit(uint Value, uint OneBitMask) {
    uint Mask = OneBitMask - 1;
    return (Value & ~Mask) << 1 | (Value & Mask) | OneBitMask;
}

bool ShouldSwap(KEY_TYPE A, KEY_TYPE B) {
    return A > B;
}

#endif // _SRENDERER_ADDON_BITONIC_SORT_INTERFACE_