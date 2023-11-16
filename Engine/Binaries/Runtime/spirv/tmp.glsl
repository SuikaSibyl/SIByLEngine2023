#version 450
layout(column_major) uniform;
layout(column_major) buffer;
#extension GL_NV_cooperative_matrix : enable
#include "matmul.glsli"

#line 1 0
layout(std430, binding = 0) buffer StructuredBuffer_float_t_0 {
    float _data[];
} u_outmat_0;



shared float  shared_inputs_buffer_0[1024];


#line 8
shared float  shared_weights_buffer_0[1024];


#line 9
shared float  output_buffer_0[1024];


#line 25
void moveWeightsToSharedMem_0(int index_0)
{

#line 25
    uint j_0;


    int _S1 = index_0 * 32;

#line 26
    for(;;)
    {

#line 26
        j_0 = 0U;

#line 26
        for(;;)
        {
            int _S2 = _S1 + int(j_0);

#line 28
            shared_inputs_buffer_0[_S2] = 2.0;
            shared_weights_buffer_0[_S2] = 1.0;

#line 27
            uint j_1 = j_0 + 1U;

#line 27
            if(int(j_1) < 32)
            {
            }
            else
            {

#line 27
                break;
            }

#line 27
            j_0 = j_1;

#line 27
        }

#line 26
        break;
    }

#line 32
    wmma_inline_matmul((shared_inputs_buffer_0), (shared_weights_buffer_0), (output_buffer_0));

#line 37
    for(;;)
    {

#line 37
        j_0 = 0U;

#line 37
        for(;;)
        {
            int _S3 = _S1 + int(j_0);

#line 39
            u_outmat_0._data[uint(_S3)] = output_buffer_0[_S3];

#line 38
            uint j_2 = j_0 + 1U;

#line 38
            if(int(j_2) < 32)
            {
            }
            else
            {

#line 38
                break;
            }

#line 38
            j_0 = j_2;

#line 38
        }

#line 37
        break;
    }



    return;
}


layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
void main()
{

#line 47
    moveWeightsToSharedMem_0(int(gl_GlobalInvocationID.x));
    return;
}

