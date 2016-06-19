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
	
	int gid_pos[8] =  {0,  0, 1,  1, 1, -1, -1, -1};
	int gid2_pos[8] = {1, -1, 1, -1, 0,  1, -1,  0};
	
		int count = 1;
		int c = ((	abs(img1[gid+w*gid2].r - img2[i+w*gid2].r) + 
				abs(img1[gid+w*gid2].g - img2[i+w*gid2].g) + 
				abs(img1[gid+w*gid2].b - img2[i+w*gid2].b) ) / 3);
		
		int x; for (x = 0; x < 8; x++) {
			int iterator = ((gid + gid_pos[x]) + w*(gid2 + gid2_pos[x]));
			int iterator2 = ((i + gid_pos[x]) + w*(gid2 + gid2_pos[x]));
			if ((iterator >= 0) && (iterator < w*h) && (iterator2 >= 0) && (iterator2 < w*h)) {
				c += ((	abs(img1[iterator].r - img2[iterator2].r) + 
						abs(img1[iterator].g - img2[iterator2].g) + 
						abs(img1[iterator].b - img2[iterator2].b) ) / 3);
				count++;
			}
		}

		output_costs[w*gid + w*w*gid2 + i] = c/count;
		output_disparities[w*gid + w*w*gid2 + i] = abs(gid - i);
}


__kernel void costAggregation(__global int*input, __global int*output, int w, int h, int i)
{
	int gid = get_global_id(0);
	int gid2 = get_global_id(1);

	int gid_pos[8] =  {0,  0, 1,  1, 1, -1, -1, -1};
	int gid2_pos[8] = {1, -1, 1, -1, 0,  1, -1,  0};
	
	int final_value = input[gid*w + w*w*gid2 + i];
	int count = 1;
		
	int x; for (x = 0; x < 8; x++) { 
		int iterator = ((gid + gid_pos[x])*w + w*w*(gid2 + gid2_pos[x]) + i);
		if ((iterator >= 0) && (iterator < w*h*w)) {
			final_value += input[iterator];
			count++;
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

	int gid_pos[8] =  {0,  0, 1,  1, 1, -1, -1, -1};
	int gid2_pos[8] = {1, -1, 1, -1, 0,  1, -1,  0};
	
	int final_value = input[gid + w*gid2];
	int count = 1;
	
	int x; for (x = 0; x < 8; x++) { 
		int iterator = ((gid + gid_pos[x]) + w*(gid2 + gid2_pos[x]));
		if ((iterator >= 0) && (iterator < w*h)) {
			final_value += input[iterator];
			count++;
		}
	}
		
	output[gid + w*gid2] = final_value/count;
}
