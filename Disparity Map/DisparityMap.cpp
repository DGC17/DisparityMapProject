#include <stdio.h>
#include <stdlib.h>

#include "CImg.h"
using namespace cimg_library;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

typedef struct
{
	int r;
	int g;
	int b;
} rgb;

void main()
{
	/* Necessary declarations to identify the GPU device. */
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem image1mem = NULL;
	cl_mem image2mem = NULL;
	cl_mem outputcostmem = NULL;
	cl_mem outputdispmem = NULL;
	cl_mem outputcost2mem = NULL;
	cl_mem outputmem = NULL;
	cl_mem refinedoutputmem = NULL;
	cl_program program = NULL;
	cl_kernel matchcost = NULL;
	cl_kernel costagg = NULL;
	cl_kernel dispsel = NULL;
	cl_kernel dispref = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	/* Necessary delcarations to open the Kernel File. */
	FILE *fp;
	const char fileName[] = "./DisparityMap.cl";
	size_t source_size;
	char *source_str;

	//INTIALIZATION OF THE IMAGES... 
	CImg<unsigned char> image1("./img/teddy/im2.ppm");
	CImg<unsigned char> image2("./img/teddy/im6.ppm");
	CImg<unsigned char> groundtruth("./img/teddy/disp2.pgm");
	//CImg<unsigned char> image1("./img/cones/im2.ppm");
	//CImg<unsigned char> image2("./img/cones/im6.ppm");
	//CImg<unsigned char> groundtruth("./img/cones/disp2.pgm");
	CImg<unsigned char> result = image1.get_RGBtoYCbCr().get_channel(0).fill(0);
	
	CImgDisplay main_disp(image1, "Image 1");
	CImgDisplay main_disp2(image2, "Image 2");

	int w = image1.width();
	int h = image1.height();
	int s = (int) image1.size();
	int i = 0;

	printf("Loaded images with: \n - Width: %d\n - Height: %d\n - Size: %d\n", w, h, s);

	rgb* image1_raster = (rgb*)malloc(s*sizeof(int));
	rgb* image2_raster = (rgb*)malloc(s*sizeof(int));
	
	int* output_costs = (int*)malloc((s / 3)*w*sizeof(int));
	int* output_disparities = (int*)malloc((s / 3)*w*sizeof(int*));
	int* output_costs2 = (int*)malloc((s / 3)*w*sizeof(int));
	int* output = (int*)malloc((s / 3)*sizeof(int));
	int* refinedoutput = (int*)malloc((s / 3)*sizeof(int));

	cimg_forXY(image1, x, y){
		image1_raster[x + w*y].r = (int)image1(x, y, 0, 0);
		image1_raster[x + w*y].g = (int)image1(x, y, 0, 1);
		image1_raster[x + w*y].b = (int)image1(x, y, 0, 2);
		image2_raster[x + w*y].r = (int)image2(x, y, 0, 0);
		image2_raster[x + w*y].g = (int)image2(x, y, 0, 1);
		image2_raster[x + w*y].b = (int)image2(x, y, 0, 2);
		output[x + w*y] = 0;
		refinedoutput[x + w*y] = 0;
		int z;
		for (z = 0; z < w; z++) {
			output_costs[w*x + w*w*y + z] = 0;
			output_disparities[w*x + w*w*y + z] = 0;
			output_costs2[w*x + w*w*y + z] = 0;
		}
	}

	/* Load the source code containing the kernel*/
	fopen_s(&fp, fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Get platform/device information */
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
	/* Create OpenCL Context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	/* Create memory buffer*/
	image1mem = clCreateBuffer(context, CL_MEM_READ_WRITE, s * sizeof(int), NULL, &ret);
	image2mem = clCreateBuffer(context, CL_MEM_READ_WRITE, s * sizeof(int), NULL, &ret);
	outputcostmem = clCreateBuffer(context, CL_MEM_READ_WRITE, (s / 3)* w * sizeof(int), NULL, &ret);
	outputdispmem = clCreateBuffer(context, CL_MEM_READ_WRITE, (s / 3)* w * sizeof(int), NULL, &ret);
	outputcost2mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (s / 3)* w * sizeof(int), NULL, &ret);
	outputmem = clCreateBuffer(context, CL_MEM_READ_WRITE, (s / 3)* sizeof(int), NULL, &ret);
	refinedoutputmem = clCreateBuffer(context, CL_MEM_READ_WRITE, (s / 3)* sizeof(int), NULL, &ret);

	/* Transfer data to memory buffer */
	ret = clEnqueueWriteBuffer(command_queue, image1mem, CL_TRUE, 0, s * sizeof(int), image1_raster, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, image2mem, CL_TRUE, 0, s * sizeof(int), image2_raster, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, outputcostmem, CL_TRUE, 0, (s / 3)* w * sizeof(int), output_costs, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, outputdispmem, CL_TRUE, 0, (s / 3)* w * sizeof(int), output_disparities, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, outputcost2mem, CL_TRUE, 0, (s / 3)* w * sizeof(int), output_costs2, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, outputmem, CL_TRUE, 0, (s / 3)* sizeof(int), output, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, refinedoutputmem, CL_TRUE, 0, (s / 3)* sizeof(int), refinedoutput, 0, NULL, NULL);

	/* Create Kernel program from the read in source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	
	/* Create OpenCL Kernels */
	matchcost = clCreateKernel(program, "matchingCostComputation", &ret);
	costagg = clCreateKernel(program, "costAggregation", &ret);
	dispsel = clCreateKernel(program, "disparitySelection", &ret);
	dispref = clCreateKernel(program, "disparityRefinement", &ret);

	/* Set OpenCL kernel arguments */
	ret = clSetKernelArg(matchcost, 0, sizeof(cl_mem), (void *)&image1mem);
	ret = clSetKernelArg(matchcost, 1, sizeof(cl_mem), (void *)&image2mem);
	ret = clSetKernelArg(matchcost, 2, sizeof(cl_mem), (void *)&outputcostmem);
	ret = clSetKernelArg(matchcost, 3, sizeof(cl_mem), (void *)&outputdispmem);
	ret = clSetKernelArg(matchcost, 4, sizeof(cl_int), (void *)&w);
	ret = clSetKernelArg(matchcost, 5, sizeof(cl_int), (void *)&h);
	ret = clSetKernelArg(matchcost, 6, sizeof(cl_int), (void *)&i);

	ret = clSetKernelArg(costagg, 0, sizeof(cl_mem), (void *)&outputcostmem);
	ret = clSetKernelArg(costagg, 1, sizeof(cl_mem), (void *)&outputcost2mem);
	ret = clSetKernelArg(costagg, 2, sizeof(cl_int), (void *)&w);
	ret = clSetKernelArg(costagg, 3, sizeof(cl_int), (void *)&h);
	ret = clSetKernelArg(costagg, 4, sizeof(cl_int), (void *)&i);

	ret = clSetKernelArg(dispsel, 0, sizeof(cl_mem), (void *)&outputcost2mem);
	ret = clSetKernelArg(dispsel, 1, sizeof(cl_mem), (void *)&outputdispmem);
	ret = clSetKernelArg(dispsel, 2, sizeof(cl_mem), (void *)&outputmem);
	ret = clSetKernelArg(dispsel, 3, sizeof(cl_int), (void *)&w);

	ret = clSetKernelArg(dispref, 0, sizeof(cl_mem), (void *)&outputmem);
	ret = clSetKernelArg(dispref, 1, sizeof(cl_mem), (void *)&refinedoutputmem);
	ret = clSetKernelArg(dispref, 2, sizeof(cl_int), (void *)&w);
	ret = clSetKernelArg(dispref, 3, sizeof(cl_int), (void *)&h);

	printf("Kernels created succefully. \n");

	size_t global_work_size[2] = {image1.width(), image1.height()};
	size_t local_work_size[2] = {1,1};

	printf("Step 1: Matching Cost...");

	/* Execute OpenCL kernel: Matching Cost */
	for (i = 0; i < w; i++) {
		ret = clSetKernelArg(matchcost, 6, sizeof(cl_int), (void *)&i);
		ret = clEnqueueNDRangeKernel(command_queue, matchcost, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		ret = clFinish(command_queue);
	}

	printf(" Finished!\n");
	
	printf("Step 2: Aggregate Cost...");

	/* Execute OpenCL kernel: Aggregate Cost */
	for (i = 0; i < w; i++) {
		ret = clSetKernelArg(costagg, 4, sizeof(cl_int), (void *)&i);
		ret = clEnqueueNDRangeKernel(command_queue, costagg, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		ret = clFinish(command_queue);
	}

	printf(" Finished!\n");

	printf("Step 3: Disparity Selection...");

	/*Execute OpenCL kernel: Disparity Selection */
	ret = clEnqueueNDRangeKernel(command_queue, dispsel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	ret = clFinish(command_queue);
	
	printf(" Finished!\n");

	printf("Step 4: Disparity Refinement...");

	/*Execute OpenCL kernel: Disparity Refinement */
	ret = clEnqueueNDRangeKernel(command_queue, dispref, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	ret = clFinish(command_queue);

	printf(" Finished!\n");

	/*Transfer result from the memory buffer*/
	ret = clEnqueueReadBuffer(command_queue, refinedoutputmem, CL_TRUE, 0, (s / 3) * sizeof(int), refinedoutput, 0, NULL, NULL);
	ret = clFinish(command_queue);

	/* Display result */
	cimg_forXY(image1, x, y) {
		
		//int classified_value = 0;
		//int output_value = refinedoutput[x + y*w];
		//int i;
		//for (i = 10; i < w; i = i + 10) if (output_value >= i) classified_value += 20;
		//result(x, y) = (unsigned char)classified_value;
		
		result(x, y) = (unsigned char)refinedoutput[x + y*w];
	}

	CImgDisplay main_disp3(result, "Result");
	CImgDisplay main_disp4(groundtruth, "Ground Truth");
	
	printf("Computation finished! \nPress any key to finish...");
	getchar();

	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(matchcost);
	ret = clReleaseKernel(costagg);
	ret = clReleaseKernel(dispsel);
	ret = clReleaseKernel(dispref);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(image1mem);
	ret = clReleaseMemObject(image2mem);
	ret = clReleaseMemObject(outputcostmem);
	ret = clReleaseMemObject(outputdispmem);
	ret = clReleaseMemObject(outputcost2mem);
	ret = clReleaseMemObject(outputmem);
	ret = clReleaseMemObject(refinedoutputmem);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(source_str);
	free(image1_raster);
	free(image2_raster);
	free(output_costs);
	free(output_disparities);
	free(output);
	free(refinedoutput);
}