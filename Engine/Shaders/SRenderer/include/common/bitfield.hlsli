#ifndef _SRENDERER_BITFIELD_HEADER_
#define _SRENDERER_BITFIELD_HEADER_

/**
 * Set a bit in a bitfield to a given state
 * @param value The bitfield to modify
 * @param bitIndex The index of the bit to modify
 * @param state The state to set the bit to
 */
uint SetBit(uint value, uint bitIndex, bool state) {
    return state ?
        (value | (1 << bitIndex)) : // Set the bit to true (1)
        (value & ~(1 << bitIndex)); // Set the bit to false (0)
}

/**
 * Get the state of a bit in a bitfield
 * @param value The bitfield to read
 * @param bitIndex The index of the bit to read
 */
bool GetBit(uint value, uint bitIndex) {
    const uint mask = (1 << bitIndex);
    return (value & mask) != 0;
}

#endif // !_SRENDERER_BITFIELD_HEADER_