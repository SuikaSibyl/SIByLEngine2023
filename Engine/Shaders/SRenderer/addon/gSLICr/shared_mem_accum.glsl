#ifndef SM_MEM_SIZE
#define SM_MEM_SIZE 32
#endif

shared float SM_MEM[SM_MEM_SIZE];

void interlocked_add_sm_mem(float v, int index) {
    atomicAdd(SM_MEM[index], v);
}

void set_sm_mem(float v, int index) {
    SM_MEM[index] = v;
}

float load_sm_mem(int index) {
    return SM_MEM[index];
}
