#ifndef _SRENDERER_MMLT_CONFIG_HEADER_
#define _SRENDERER_MMLT_CONFIG_HEADER_

// tracing bounce setting
const int max_events         = 5;
const int possible_depth_num = max_events-1;    // 2, 3, ..., max_events
const int max_depth          = max_events-2;    // 2, 3, ..., max_events

// mutation setting
const float sigma = 0.02;
const float large_step_probability = 0.0;

// boostrap setting
const int boostrap_buffer_size  = 512;
const int boostrap_buffer_size_sq  = (boostrap_buffer_size*boostrap_buffer_size);
const int boostrap_grid_width   = boostrap_buffer_size / possible_depth_num;
const int boostrap_grid_height  = boostrap_buffer_size;

const int num_vec4_light_subpath    = max_events + 1;
const int num_vec4_camera_subpath   = max_events;
const int num_vec4_extra_items      = 1;

const int num_rngs_per_event = 4;   // Let's say every vertex consumes at most 4 random variables
const int num_states_vec4    = (num_vec4_light_subpath + num_vec4_camera_subpath + num_vec4_extra_items);

const int offset_camera_subpath         = 0;
const int offset_light_subpath          = num_vec4_camera_subpath;
const int offset_strategy_selection     = offset_light_subpath + num_vec4_light_subpath;

// const int offset_light_subpath          = 0;
// const int offset_camera_subpath         = num_vec4_light_subpath;
// const int offset_strategy_selection     = offset_camera_subpath + num_vec4_camera_subpath;

const int metroplis_buffer_width = 256;
const int metroplis_buffer_height = 256;

const int maximum_lod = 9;

#endif