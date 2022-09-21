#pragma once

/**
* Alloca could allocate memory from system stack, which is useful for scoped new objects.
* The memory allocated by Alloca will be automatically freed after scope is end.
*/

#define Alloca(T) ((T*)alloca(sizeof(T)))
#define Alloca(T, count) ((T*)alloca(count * sizeof(T)))
