typedef struct
{
	int r;
	int g;
	int b;
} rgb;

__kernel void matchingCostComputation(__global rgb* img1, __global rgb* img2, __global int*output_costs, __global int*output_disparities, int w, int h, int i)
{
	int gid = get_global_id(0);
	int gid2 = get_global_id(1);
	int window_range = 3; 
	int count = 0;
	int c = 0;

	int gid1_pos; for (gid1_pos = (1 - window_range); gid1_pos < window_range; gid1_pos++) {
		int gid2_pos; for (gid2_pos = (1 - window_range); gid2_pos < window_range; gid2_pos++) {
			int iterator = ((gid + gid1_pos) + w*(gid2 + gid2_pos));
			int iterator2 = ((i + gid1_pos) + w*(gid2 + gid2_pos));
			if ((iterator >= 0) && (iterator < w*h) && (iterator2 >= 0) && (iterator2 < w*h)) {
				c += ((	abs(img1[iterator].r - img2[iterator2].r) + 
						abs(img1[iterator].g - img2[iterator2].g) + 
						abs(img1[iterator].b - img2[iterator2].b) ) / 3);
				count++;
			}
		}
	}
		
	output_costs[w*gid + w*w*gid2 + i] = c/count;
	output_disparities[w*gid + w*w*gid2 + i] = abs(gid - i);
}


__kernel void costAggregation(__global int*input, __global int*output, int w, int h, int i)
{
	int gid = get_global_id(0);
	int gid2 = get_global_id(1);
	int window_range = 3; 
	int count = 0;
	int final_value = 0;

	int gid1_pos; for (gid1_pos = (1 - window_range); gid1_pos < window_range; gid1_pos++) {
		int gid2_pos; for (gid2_pos = (1 - window_range); gid2_pos < window_range; gid2_pos++) {
			int iterator = ((gid + gid1_pos)*w + w*w*(gid2 + gid2_pos) + i);
			if ((iterator >= 0) && (iterator < w*h*w)) {
				final_value += input[iterator];
				count++;
			}
		}
	}
	
	output[w*gid + w*w*gid2 + i] = final_value/count;
}

__kernel void disparitySelection(__global int*costs, __global int*disparities, __global int* disparity_selected, int w)
{
	int gid = get_global_id(0);
	int gid2 = get_global_id(1);
	int best_cost = 99999;
	int best_disparity = 0;
	int max_disparity = 50;

	int i; for (i = 0; i < w; i++) {
		int disparity = disparities[gid*w + w*w*gid2 + i];
		int cost = costs[gid*w + w*w*gid2 + i];
		if ((cost < best_cost) && (disparity < max_disparity)) {
			best_cost = cost;
			best_disparity = disparity;
		}
	}

	disparity_selected[gid + w*gid2] = best_disparity;
}

__kernel void disparityRefinement(__global int*input, __global int*output, int w, int h)
{
	int gid = get_global_id(0);
	int gid2 = get_global_id(1);
	int window_range = 2; 
	int count = 0;
	int final_value = 0;

	int gid1_pos; for (gid1_pos = (1 - window_range); gid1_pos < window_range; gid1_pos++) {
		int gid2_pos; for (gid2_pos = (1 - window_range); gid2_pos < window_range; gid2_pos++) {
			int iterator = ((gid + gid1_pos) + w*(gid2 + gid2_pos));
			if ((iterator >= 0) && (iterator < w*h)) {
				final_value += input[iterator];
				count++;
			}
		}
	}
	
	output[gid + w*gid2] = final_value/count;
}
