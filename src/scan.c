#include "common.h"

cl_command_queue que;
cl_kernel scan_krn[3];
cl_event scan_evt[3];

/* Assumption: options.elements and nwg are both powers of two */
void do_scan(cl_mem d_output, cl_mem d_scan_aux, cl_mem d_input,
	size_t scan_lws, size_t aux_lws)
{
	cl_int error;

	size_t gws = options.groups*scan_lws;

	/* each workgroup processes chunks of options.vecsize*lws elements */
	const cl_uint chunk_size = options.vecsize*scan_lws;
	const cl_uint num_chunks = (options.elements + chunk_size - 1)/chunk_size;
	const cl_uint chunks_per_wg = (num_chunks + options.groups - 1)/options.groups;
	const cl_uint els_per_wg = chunks_per_wg*chunk_size;

	printf("%u wg, %zu wi/wg, %u e/wg\n", options.groups, scan_lws, els_per_wg);

	error = clSetKernelArg(scan_krn[0], 0, sizeof(d_output), &d_output);
	check_ocl_error(error, "scan#0 arg 0");
	error = clSetKernelArg(scan_krn[0], 1, sizeof(d_scan_aux), &d_scan_aux);
	check_ocl_error(error, "scan#0 arg 1");
	error = clSetKernelArg(scan_krn[0], 2, sizeof(d_input), &d_input);
	check_ocl_error(error, "scan#0 arg 2");
	error = clSetKernelArg(scan_krn[0], 3, 2*scan_lws*sizeof(TYPE), NULL);
	check_ocl_error(error, "scan#0 arg 3");
	error = clSetKernelArg(scan_krn[0], 4, sizeof(options.elements), &options.elements);
	check_ocl_error(error, "scan#0 arg 4");
	error = clSetKernelArg(scan_krn[0], 5, sizeof(els_per_wg), &els_per_wg);
	check_ocl_error(error, "scan#0 arg 5");

	printf("Enqueue scan #0\n");

	error = clEnqueueNDRangeKernel(que, scan_krn[0], 1,
		NULL, &gws, &scan_lws, 0, NULL, scan_evt);

	check_ocl_error(error, "scan#0 enqueue");

	// aux scan is done in-place with a single block, sized the largest power of
	// two not smaller than the number of elements we need to reduce (nwg),
	// with the constraint of being no larger than max_wg;
	gws = aux_lws;

	printf("Aux: %zu wi/wg\n", aux_lws);
	error = clSetKernelArg(scan_krn[1], 0, sizeof(d_scan_aux), &d_scan_aux);
	check_ocl_error(error, "scan#1 arg 0");
	error = clSetKernelArg(scan_krn[1], 1, 2*aux_lws*sizeof(cl_uint), NULL);
	check_ocl_error(error, "scan#0 arg 1");
	error = clSetKernelArg(scan_krn[1], 2, sizeof(options.groups), &options.groups);
	check_ocl_error(error, "scan#0 arg 2");

	printf("Enqueue scan #1\n");

	error = clEnqueueNDRangeKernel(que, scan_krn[1], 1,
		NULL, &gws, &aux_lws, 1, scan_evt, scan_evt + 1);

	check_ocl_error(error, "scan#1 enqueue");

	gws = ROUND_MUL(options.elements, scan_lws);

	error = clSetKernelArg(scan_krn[2], 0, sizeof(d_output), &d_output);
	check_ocl_error(error, "scan#2 arg 0");
	error = clSetKernelArg(scan_krn[2], 1, sizeof(d_scan_aux), &d_scan_aux);
	check_ocl_error(error, "scan#2 arg 1");
	error = clSetKernelArg(scan_krn[2], 2, sizeof(options.elements), &options.elements);
	check_ocl_error(error, "scan#2 arg 2");
	error = clSetKernelArg(scan_krn[2], 3, sizeof(els_per_wg), &els_per_wg);
	check_ocl_error(error, "scan#2 arg 3");

	printf("Enqueue scan #2\n");

	error = clEnqueueNDRangeKernel(que, scan_krn[2], 1,
		NULL, &gws, &scan_lws, 1, scan_evt + 1, scan_evt + 2);
	check_ocl_error(error, "scan#2 enqueue");

}

int main(int argc, char *argv[])
{
	/* defaults */
	options.platform = 0;
	options.device = 0;
	options.elements = 0;
	options.vecsize = 1;
	options.groups = 0;
	options.reduction_style = -1;
	options.unroll = 1;

	parse_options(argc, argv);

	printf("Will use device %u on platform %u\n",
			options.device, options.platform);

	cl_int error;

	cl_platform_id platform = select_platform();
	cl_device_id dev = select_device(platform);

	/* If the user has not specified the number of groups, pick one based on
	 * device type and number of compute units */
	if (!options.groups) {
		if (options.cugroups)
			options.groups = options.cugroups*dev_info.compute_units;
		else if (dev_info.dev_type == CL_DEVICE_TYPE_GPU)
			options.groups = 6*dev_info.compute_units;
		else
			options.groups = dev_info.compute_units;
	}
	if (options.vecsize > 1)
		options.groups = ROUND_MUL(options.groups, options.vecsize);

	/* Round number of groups to power of two */
	if (!is_po2(options.groups)) {
		/* not power-of-two, fix */
		const size_t oldval = options.groups;
		const size_t newval = fix_po2(oldval);
		fprintf(stderr, "number of groups %zu is not a power of two,"
			" rounding to %zu.\n", oldval, newval);
		options.groups = newval;
	}

	printf("Scan will use %u groups (%g groups/CU)\n",
		options.groups, (double)options.groups/dev_info.compute_units);

	clbuild_add("-Isrc/");
	clbuild_printf("-DVECSIZE=%u -DUNROLL=%u", options.vecsize, options.unroll);
	if (options.vecsize > 1) {
		printf("Vector size: %u\n", options.vecsize);
	}
	if (options.unroll > 1) {
		printf("Unroll size: %u\n", options.unroll);
	}

	cl_context ctx = create_context(platform, dev);
	que = create_queue(ctx, dev);

	/* initialize test data set */
	if (options.elements == 0) {
		cl_ulong amt = dev_info.max_alloc;
		amt /= 2*sizeof(TYPE);
		if (amt > CL_UINT_MAX)
			options.elements = CL_UINT_MAX;
		else
			options.elements = amt;
		/* check that our elements, plus the auxiliary allocation, doesn't overflow
		 * the device memory
		 */
		if ((2*options.elements + options.groups)*sizeof(TYPE) > dev_info.mem_size) {
			/* clamp */
			options.elements = (dev_info.mem_size/sizeof(TYPE) - options.groups)/2;
		}
	}
	if (options.vecsize > 1) {
		/* round down, to avoid overflowing max_alloc */
		options.elements = (options.elements/options.vecsize)*options.vecsize;
	}
	printf("will process %u (%uM) elements\n", options.elements,
		options.elements>>20);

	TYPE *data = (TYPE*) calloc(options.elements, sizeof(TYPE));
	TYPE *host_res = (TYPE*) calloc(options.elements, sizeof(TYPE));
	if (!data || ! host_res) {
		fprintf(stderr, "unable to allocate host memory during init");
		exit(-CL_OUT_OF_HOST_MEMORY);
	}
	size_t data_size = options.elements * sizeof(TYPE);

	for (cl_uint i = 0; i < options.elements; ++i) {
		data[i] = i;
		host_res[i] = data[i];
		if (i > 0)
#if TEST_MIN
			if (host_res[i] < host_res[i-1])
				host_res[i] = host_res[i-1];
#else
			host_res[i] += host_res[i-1];
#endif
	}

	printf("%u elements generated, data size %zu (%zuMB)\n",
			options.elements, data_size, data_size>>20);


	/* device buffers */
	cl_mem d_input, d_output, d_scan_aux;
	cl_event mem_evt[1];

	cl_mem_flags host_flag = 0;
	void *host_ptr = NULL;
	if (dev_info.host_mem)
		host_flag |= CL_MEM_USE_HOST_PTR;

	if (dev_info.host_mem)
		host_ptr = data;

	d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | host_flag,
		data_size, host_ptr, &error);
	check_ocl_error(error, "allocating source memory buffer");
	if (!dev_info.host_mem) {
		error = clEnqueueWriteBuffer(que, d_input, false, 0,
				data_size, data,
				0, NULL, mem_evt);
		check_ocl_error(error, "copying source memory buffer");
	}

	d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		data_size, NULL, &error);
	check_ocl_error(error, "allocating output memory buffer");

	d_scan_aux = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
		options.groups*sizeof(TYPE), NULL, &error);
	check_ocl_error(error, "allocating auxiliary memory buffer");

	clbuild_add(KDEFS);

	if (clbuild_next > OPTBUFSIZE) {
		fprintf(stderr, "failed to assemble CL compiler options\n");
		exit(1);
	} else {
		printf("compiler options: %s\n", clbuild);
	}

	/* load and build program */
	cl_program program = create_program("scan_kernels.ocl", ctx, dev);


	scan_krn[0] = clCreateKernel(program, "first_scan_step", &error);
	check_ocl_error(error, "creating scan#0 kernel");
	scan_krn[1] = clCreateKernel(program, "aux_scan_step", &error);
	check_ocl_error(error, "creating scan#1 kernel");
	scan_krn[2] = clCreateKernel(program, "last_scan_step", &error);
	check_ocl_error(error, "creating scan#2 kernel");

	/* Selecting the kernel work-group size(s) */
	/* TODO per-kernel selection */
	clGetKernelWorkGroupInfo(scan_krn[0], dev, CL_KERNEL_WORK_GROUP_SIZE,
			sizeof(max_wg_size), &max_wg_size, NULL);
	error = clGetKernelWorkGroupInfo(scan_krn[0], dev,
			CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			sizeof(ws_multiple), &ws_multiple, NULL);
	check_ocl_error(error, "checking device properties");
	if (ws_multiple == 0) {
		fprintf(stderr, "device reported a null workgroup size multple; assuming 1\n");
		ws_multiple = 1;
	}
	if (!is_po2(ws_multiple)) {
		fprintf(stderr, "device has a non-power-of-two workgroup size multiple (%zu); expect oddities\n",
			ws_multiple);
		ws_multiple = fix_po2(ws_multiple);
		if (ws_multiple > max_wg_size)
			ws_multiple /= 2;
		fprintf(stderr, "\tusing %zu instead\n", ws_multiple);
	}
	printf("kernel prefers multiples of %zu up to %zu\n",
			ws_multiple, max_wg_size);

	if (!dev_info.host_mem) {
		clWaitForEvents(1, mem_evt);
		GET_RUNTIME(*mem_evt, "memory upload");
		clReleaseEvent(*mem_evt);
	}

	const size_t ws_first = options.groupsize ? options.groupsize : ws_multiple ;
	const size_t ws_last = options.groupsize ? options.groupsize : max_wg_size ;

	/* For the first kernel launch, we issue options.groups work-groups; each
	 * work-group produces one auxiliary result (the last value in the scan).
	 * We scan this auxiliary array with a kernel that operates with a single
	 * group, which is sized the largest power of two not smaller than the
	 * number of auxiliary values produced, divided by the vector size,
	 * with the contraint of being no larger than the maximum work-group size.
	 */
	size_t ws_second = options.groupsize && options.groupsize < ws_multiple
		? options.groupsize : ws_multiple;
	while (ws_second < options.groups/options.vecsize)
		ws_second *= 2;
	if (ws_second > max_wg_size)
		ws_second = max_wg_size;

	for (size_t ws = ws_first ; ws <= ws_last; ws *= 2) {
		do_scan(d_output, d_scan_aux, d_input, ws, ws_second);

		TYPE *dev_res = (TYPE *)clEnqueueMapBuffer(que, d_output, CL_TRUE,
			CL_MAP_READ, 0, data_size,
			1, scan_evt + 2, NULL, &error);
		check_ocl_error(error, "map buffer total");

		clFinish(que);

		for (cl_uint i = 0 ; i < options.elements; ++i) {
			if (dev_res[i] != host_res[i]) {
				fprintf(stderr, "error @ %u: got " PTYPE " , expected " PTYPE "\n",
					i, dev_res[i], host_res[i]);
				break;
			}
		}

		clEnqueueUnmapMemObject(que, d_output, dev_res, 0, NULL, NULL);

		printf("Group size: %zu, %zu\n", ws, ws_second);
		GET_RUNTIME(scan_evt[0], "Kernel scan #0");
		GET_RUNTIME(scan_evt[1], "Kernel scan #1");
		GET_RUNTIME(scan_evt[2], "Kernel scan #2");
		GET_RUNTIME_DELTA(scan_evt[0], scan_evt[2], "Total");

		/* count the intermediate reads and writes too in the effective bandwidth usage */
		const size_t scan_data_size = 4*data_size + 3*options.groups*sizeof(TYPE);
		printf("Bandwidth: %.4g GB/s\n", (double)scan_data_size/(endTime-startTime));
		printf("Scan performance: %.4g GE/s\n", (double)options.elements/(endTime-startTime));
		printf("SUMMARY: %6zu × %u + %4zu × 1 + %6zu × %9zu => %11.2fms | %11.2f GB/s | %11.2f GE/s\n",
			ws, options.groups, ws_second, ws, ROUND_MUL(options.elements, ws)/ws,
			(endTime - startTime)/1000000.0,
			(double)scan_data_size/(endTime - startTime),
			(double)options.elements/(endTime-startTime));

		clReleaseEvent(scan_evt[2]);
		clReleaseEvent(scan_evt[1]);
		clReleaseEvent(scan_evt[0]);
	}

	clReleaseMemObject(d_input);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_scan_aux);

	return 0;
}
