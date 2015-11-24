#include "common.h"

int main(int argc, char **argv) {

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

	printf("Reduction will use %u groups (%g groups/CU)\n",
		options.groups, (double)options.groups/dev_info.compute_units);

	/* Pick default reduction style unless specified; default to
	 * block reduction on CPU, interleaved otherwise */
	if (options.reduction_style < 0) {
		if (dev_info.dev_type == CL_DEVICE_TYPE_CPU)
			options.reduction_style = 1;
		else
			options.reduction_style = 0;
	}

	clbuild_add("-Isrc/");
	clbuild_printf("-DREDUCTION_STYLE=%u -DVECSIZE=%u -DUNROLL=%u",
		options.reduction_style, options.vecsize, options.unroll);
	printf("Reduction style: %s\n", options.reduction_style == 0 ? "interleaved" :
		options.reduction_style == 1 ? "blocked" : "<?>");
	if (options.vecsize > 1) {
		printf("Vector size: %u\n", options.vecsize);
	}
	if (options.unroll > 1) {
		printf("Unroll size: %u\n", options.unroll);
	}

	cl_context ctx = create_context(platform, dev);
	cl_command_queue queue = create_queue(ctx, dev);

	/* initialize test data set */
	if (options.elements == 0) {
		cl_ulong amt = dev_info.max_alloc;
		amt /= sizeof(TYPE);
		if (amt > CL_UINT_MAX)
			options.elements = CL_UINT_MAX;
		else
			options.elements = amt;
		/* check that our elements, plus the auxiliary allocation, doesn't overflow
		 * the device memory
		 */
		if ((options.elements + options.groups)*sizeof(TYPE) > dev_info.mem_size) {
			/* clamp */
			options.elements = dev_info.mem_size/sizeof(TYPE) - options.groups;
		}
	}
	if (options.vecsize > 1) {
		/* round down, to avoid overflowing max_alloc */
		options.elements = (options.elements/options.vecsize)*options.vecsize;
	}
	printf("will process %u (%uM) elements\n", options.elements,
		options.elements>>20);

	TYPE *data = (TYPE*) calloc(options.elements, sizeof(TYPE));
	if (!data) {
		fprintf(stderr, "unable to allocate host memory during init");
		exit(-CL_OUT_OF_HOST_MEMORY);
	}
	size_t data_size = options.elements * sizeof(TYPE);
	TYPE host_res = OP_NULL, dev_res = OP_NULL;

	for (cl_uint i = 0; i < options.elements; ++i) {
		// data[i] = (2*TYPE(rand())/RAND_MAX - 1)*1024;
		data[i] = options.elements - i;
#if TEST_MIN
		if (data[i] < host_res)
			host_res = data[i];
#else
		host_res += data[i];
#endif
	}
	printf("%u elements generated, " OP_NAME " " PTYPE ", data size %zu (%zuMB)\n",
			options.elements, host_res, data_size, data_size>>20);

	/* device buffers */

	cl_mem d_input, d_output;

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
		error = clEnqueueWriteBuffer(queue, d_input, false, 0,
				data_size, data,
				0, NULL, mem_evt);
		check_ocl_error(error, "copying source memory buffer");
	}

	d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			options.groups*sizeof(TYPE), NULL, &error);
	check_ocl_error(error, "allocating output memory buffer");

	/* load and build program */

	char *prog_source = read_file("reduction_kernels.ocl");
	if (prog_source == NULL)
		exit(1);

	cl_program program = clCreateProgramWithSource(ctx, 1,
			(const char **)&prog_source, NULL, &error);
	check_ocl_error(error, "creating program");

	clbuild_add(KDEFS);

	if (clbuild_next > OPTBUFSIZE) {
		fprintf(stderr, "failed to assemble CL compiler options\n");
		exit(1);
	} else {
		printf("compiler options: %s\n", clbuild);
	}

	cl_int build_error = clBuildProgram(program,
			1, &dev, // device(s)
			clbuild,
			NULL, // callback
			NULL);
	size_t logSize = 0;
	char *log;
	error = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &logSize);
	check_ocl_error(error, "getting program build info size");
	/* the log is NULL-terminated, so if it's empty it has size 1 */
	if (logSize > 1) {
		log = (char *)malloc(logSize);
		error = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
				logSize, log, NULL);
		check_ocl_error(error, "getting program build info");
		/* some platforms return a \n-terminated log, others don't.
		   equalize by stripping out all the final \n's */
		while (logSize > 1) {
			if (log[logSize - 2] == '\n') {
				log[logSize - 2] = 0;
				--logSize;
			} else break;
		}
		/* if we still have something, print the build info */
		if (logSize > 1)
			fprintf(stderr, "Program build info:\n%s\n", log);
	}

	if (build_error == CL_BUILD_PROGRAM_FAILURE)
		exit(1);

	char kname[] = "reduce";

	/* loading the kernel */
	cl_kernel reduceKernel = clCreateKernel(program, kname, &error);
	check_ocl_error(error, "creating kernel");

	/* Selecting the kernel work-group size(s) */
	clGetKernelWorkGroupInfo(reduceKernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
			sizeof(max_wg_size), &max_wg_size, NULL);
	error = clGetKernelWorkGroupInfo(reduceKernel, dev,
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
	 * work-group reduces some of the data, and produces one reduced value,
	 * so the first kernel launch returns one element per work-group. On the second
	 * kernel launch, we do a further reduction of all the previous partial reduction.
	 * The group size in this case will be the smallest multiple of ws_multiple
	 * not smaller than options.groups, obviously clamped to max_wg_size.
	 * If the user specified a groupsize smaller than ws_multiple, we'll go
	 * for multiples of options.groupsize instead */
	size_t ws_second = options.groupsize && options.groupsize < ws_multiple
		? options.groupsize : ws_multiple;
	while (ws_second < options.groups)
		ws_second *= 2;
	if (ws_second > max_wg_size)
		ws_second = max_wg_size;

	/* number of vector elements */
	const cl_uint vecelements = options.elements/options.vecsize;
	const cl_uint vecgroups = options.groups/options.vecsize;

	for (size_t ws = ws_first ; ws <= ws_last; ws *= 2) {
		/* First run: options.groups ws-sized work-groups */
		group_size[0] = ws;
		work_size[0] = group_size[0]*options.groups;

		int argnum = 0;
		KERNEL_ARG(d_input);
		error = clSetKernelArg(reduceKernel, argnum++,
				group_size[0]*sizeof(TYPE), NULL);
		check_ocl_error(error, "setting kernel param");
		KERNEL_ARG(vecelements);
		KERNEL_ARG(d_output);

		/* launch kernel, with an event to collect profiling info */

		clEnqueueNDRangeKernel(queue, reduceKernel,
				1,
				NULL, work_size, group_size,
				0, NULL,
				pass_evt);

		/* Second run: single work-group sized as mentioned above */
		group_size[0] = work_size[0] = ws_second;

		argnum = 0;
		KERNEL_ARG(d_output);
		error = clSetKernelArg(reduceKernel, argnum++,
				group_size[0]*sizeof(TYPE), NULL);
		check_ocl_error(error, "setting kernel param");
		KERNEL_ARG(vecgroups);
		KERNEL_ARG(d_output);

		clEnqueueNDRangeKernel(queue, reduceKernel,
				1,
				NULL, work_size, group_size,
				1, pass_evt,
				pass_evt + 1);

		error = clFinish(queue); // sync on queue
		check_ocl_error(error, "finishing queue");

		printf("Group size: %zu, %zu\n", ws, group_size[0]);
		GET_RUNTIME(pass_evt[0], "Kernel pass #1");
		GET_RUNTIME(pass_evt[1], "Kernel pass #2");
		GET_RUNTIME_DELTA(pass_evt[0], pass_evt[1], "Total");

		/* count the intermediate reads and writes too in the effective bandwidth usage */
		const size_t reduction_data_size = data_size + (2*options.groups+1)*sizeof(TYPE);
		printf("Bandwidth: %.4g GB/s\n", (double)reduction_data_size/(endTime-startTime));
		printf("Reduction performance: %.4g GE/s\n", (double)options.elements/(endTime-startTime));
		printf("SUMMARY: %6zu × %u + %4zu × 1 => %11.2fms | %11.2f GB/s | %11.2f GE/s\n",
			ws, options.groups, ws_second, (endTime - startTime)/1000000.0,
			(double)reduction_data_size/(endTime - startTime),
			(double)options.elements/(endTime-startTime));

	}

	/* copy memory down */
	error = clEnqueueReadBuffer(queue, d_output, true, 0,
			sizeof(TYPE), &dev_res,
			0, NULL, mem_evt);
	check_ocl_error(error, "getting results");

	clFinish(queue);
	GET_RUNTIME(*mem_evt, "memory download");
	clReleaseEvent(*mem_evt);

	data_size = sizeof(TYPE);
	printf("total download runtime: %gms for %gMB (%g GB/s)\n",
			(double)(endTime-startTime)/1000000,
			data_size/(1024*1024.0),
			/* the magic factor is 10^9/2^30, i.e.
			   decimal billions over binary gigas */
			data_size/(double)(endTime-startTime)*0.931322575);

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);

	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);

	fflush(stdout);
	fflush(stderr);

	free(data);

	TYPE expected = TEST_MIN ? 1 : (options.elements & 1 ?
		(options.elements + 1)/2*options.elements : options.elements/2*(options.elements + 1));
	printf("Parallel " OP_NAME ": " PTYPE " vs " PTYPE " (expected: " PTYPE ")\n",
		dev_res, host_res, expected);
	printf("Deltas: " PTYPE " vs " PTYPE "\n", dev_res - expected, host_res - expected);
}
