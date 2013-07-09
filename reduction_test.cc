/* usual C/C++ includes */
#include <stdio.h>
#include <string.h>
#include <stdlib.h> // for rand()
#include <errno.h>

#include <stdarg.h> // variadic arguments for clbuild_printf

/* OpenCL includes */
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define TEST_MIN 0 // set to 1 to test min instead of +
#if TEST_MIN
#define OP_NAME "min"
#define OP_NULL CL_INFINITY
#define TYPE cl_float
#define PTYPE "%g"
#define KDEFS "-DTYPE=float -DOP=FMIN_OP -DOP_NULL=FMIN_OP_NULL"
#else
#define OP_NAME "sum"
#define OP_NULL 0
#define TYPE cl_int
#define PTYPE "%u"
#define KDEFS "-DTYPE=int -DOP=ADD_OP -DOP_NULL=ADD_OP_NULL"
#endif

struct options_t {
	cl_uint platform;
	cl_uint device;
	cl_uint elements;
	cl_uint groups;
	cl_uint groupsize;
} options;

cl_ulong startTime, endTime;
cl_event pass_evt[2];

/* macro to get and display the runtime associated with an event */
#define GET_RUNTIME(evt, text) do {\
	clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, \
			sizeof(cl_ulong), &startTime, NULL);\
	error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,\
			sizeof(cl_ulong), &endTime, NULL); \
	check_ocl_error(error, "getting profiling info"); \
	printf(text " runtime: %gms\n", \
			double(endTime-startTime)/1000000); \
	} while (0);

#define GET_RUNTIME_DELTA(evt1, evt2, text) do {\
	clGetEventProfilingInfo(evt1, CL_PROFILING_COMMAND_START, \
			sizeof(cl_ulong), &startTime, NULL);\
	error = clGetEventProfilingInfo(evt2, CL_PROFILING_COMMAND_END,\
			sizeof(cl_ulong), &endTime, NULL); \
	check_ocl_error(error, "getting profiling info"); \
	printf(text " runtime: %gms\n", \
			double(endTime-startTime)/1000000); \
	} while (0);

/* macro to set the next kernel argument */
#define KERNEL_ARG(what) do {\
	error = clSetKernelArg(reduceKernel, argnum++, sizeof(what), &(what)); \
	check_ocl_error(error, "setting kernel param"); \
	} while(0)

/* initialize to 1 just so that the ROUND_MUL will not do anything */
size_t group_size[1] = {1};
size_t work_size[1] = {0};

/* maximum kernel group size */
size_t max_wg_size = 0;
/* preferred workgroup size multiple */
size_t ws_multiple = 0;

typedef struct dev_info_t {
	/* number of compute units */
	cl_uint compute_units;
	/* max alloc*/
	cl_ulong mem_size;
	/* max alloc*/
	cl_ulong max_alloc;
	/* local memory size */
	cl_ulong local_mem_size;
	/* does the device have host unified memory? */
	cl_bool host_mem;
	/* maximum work item size */
	size_t max_wi_size[3];
} dev_info_t;

dev_info_t dev_info;

/* macro used to round size up to the smallest multiple of base */
#define ROUND_MUL(size, base) \
	((size + base - 1)/base)*base

/* kernel build options */
#define OPTBUFSIZE 8192
char clbuild[OPTBUFSIZE];

/* offset in the clbuild array to where to insert the next parameter */
size_t clbuild_next;

void clbuild_add(char const* str)
{
	size_t len = strlen(str);
	if (clbuild_next) {
		clbuild[clbuild_next] = ' ';
		++clbuild_next;
	}
	if (clbuild_next + len >= OPTBUFSIZE) {
		fprintf(stderr, "cannot add build option %s\n", str);
		exit(1);
	}
	strcpy(clbuild + clbuild_next, str);
	clbuild_next += len;
}

void clbuild_printf(char const* fmt, ...)
{
	va_list args;
	va_start(args, fmt);

	if (clbuild_next) {
		clbuild[clbuild_next] = ' ';
		++clbuild_next;
	}
	clbuild_next += vsnprintf(clbuild + clbuild_next,
			OPTBUFSIZE - clbuild_next,
			fmt, args);
	va_end(args);
	if (clbuild_next >= OPTBUFSIZE) {
		fprintf(stderr, "cannot add build option %s\n", fmt);
		exit(1);
	}
}

void check_ocl_error(const cl_int &error, const char *message) {
	if (error != CL_SUCCESS) {
		fprintf(stderr, "error %d %s\n", error, message);
		exit(1);
	}
}

char *read_file(const char *fname) {
	size_t fsize, readsize;
	char *buff;

	FILE *fd = fopen(fname, "rb");
	if (!fd) {
		fprintf(stderr, "%s not found\n", fname);
		return NULL;
	}

	fseek(fd, 0, SEEK_END);
	fsize = ftell(fd);

	buff = (char *)malloc(fsize + 1);
	rewind(fd);
	readsize = fread(buff, 1, fsize, fd);
	if (fsize != readsize) {
		fprintf(stderr, "could only read %lu/%lu bytes from %s\n",
				readsize, fsize, fname);
		free(buff);
		return NULL;
	}
	buff[fsize] = '\0';

	printf("read %lu bytes from %s\n", fsize, fname);

	return buff;
}

void parse_options(int argc, char **argv)
{
	/* skip argv[0] */
	++argv; --argc;
	while (argc) {
		char *arg = *argv;
		++argv; --argc;
		if (!strcmp(arg, "--platform")) {
			sscanf(*argv, "%u", &(options.platform));
			++argv; --argc;
		} else if (!strcmp(arg, "--device")) {
			sscanf(*argv, "%u", &(options.device));
			++argv; --argc;
		} else if (!strcmp(arg, "--elements")) {
			sscanf(*argv, "%u", &(options.elements));
			++argv; --argc;
		} else if (!strcmp(arg, "--groups")) {
			sscanf(*argv, "%u", &(options.groups));
			++argv; --argc;
		} else if (!strcmp(arg, "--groupsize")) {
			sscanf(*argv, "%zu", &(options.groupsize));
			++argv; --argc;
		} else if (!strncmp(arg, "-D", 2)) {
			clbuild_add(arg);
		} else if (!strncmp(arg, "-", 1)) {
			fprintf(stderr, "unrecognized option %s\n", arg);
		} else {
			fprintf(stderr, "too many filenames: %s\n", arg);
		}
	}
}

int main(int argc, char **argv) {

	/* defaults */
	options.platform = 0;
	options.device = 0;
	options.elements = 0;
	options.groups = 4;

	parse_options(argc, argv);

	printf("Will use device %u on platform %u\n",
			options.device, options.platform);

	/* auxiliary buffer to read platform and device info */
	char buffer[1024];

	/* platform selection */        // variable declaration
	cl_uint num_platforms = 0;
	cl_platform_id *platform_list = NULL;
	cl_platform_id platform = NULL;

	clGetPlatformIDs(0, NULL, &num_platforms); //retrieve number of platform IDs
	platform_list = (cl_platform_id *)calloc(num_platforms,
			sizeof(cl_platform_id));
	cl_int error = clGetPlatformIDs(num_platforms, platform_list, NULL); // retrieve the actual platform IDs

	check_ocl_error(error, "getting platform IDs");

	printf("%d OpenCL platforms found:\n", num_platforms);

	for (cl_uint i = 0; i < num_platforms; ++i) {
		/* last param: actual size of the query result */
		error = clGetPlatformInfo(platform_list[i], CL_PLATFORM_NAME,
				sizeof(buffer), buffer, NULL);
		check_ocl_error(error, "getting platform name");
		printf("\tplatform %u: %s ", i, buffer);
		error = clGetPlatformInfo(platform_list[i], CL_PLATFORM_VENDOR,
				sizeof(buffer), buffer, NULL);
		check_ocl_error(error, "getting platform vendor");
		printf(" (%s)\n", buffer);
	}

	cl_uint platnum = 0;
	platform = platform_list[options.platform];
	printf("using platform %u\n", options.platform);

	/* device selection */

	cl_uint num_devs = 0;
	cl_device_id *dev_list = NULL;
	cl_device_id dev = NULL;

	/* possible types: CPU, GPU, ACCELERATOR, DEFAULT, ALL */
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devs);
	dev_list = (cl_device_id *)calloc(num_devs, sizeof(cl_device_id));
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devs,
			dev_list, NULL);

	check_ocl_error(error, "getting device IDs");

	printf("%d devs found:\n", num_devs);

	for (cl_uint i = 0; i < num_devs; ++i) {
		/* last param: actual size of the query result */
		error = clGetDeviceInfo(dev_list[i], CL_DEVICE_NAME,
				sizeof(buffer), buffer, NULL);
		check_ocl_error(error, "getting device name");
		printf("\tdev %u: %s\n", i, buffer);
	}

	cl_uint devnum = 0;
	dev = dev_list[options.device];
	printf("using device %u\n", options.device);

	error = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(dev_info.compute_units), &dev_info.compute_units,
			NULL);
	error = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,
			sizeof(dev_info.mem_size), &dev_info.mem_size,
			NULL);
	error = clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
			sizeof(dev_info.max_alloc), &dev_info.max_alloc,
			NULL);
	error = clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE,
			sizeof(dev_info.local_mem_size), &dev_info.local_mem_size,
			NULL);
	error = clGetDeviceInfo(dev, CL_DEVICE_HOST_UNIFIED_MEMORY,
			sizeof(dev_info.host_mem), &dev_info.host_mem, NULL);
	error = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
			3*sizeof(*dev_info.max_wi_size), dev_info.max_wi_size, NULL);
	check_ocl_error(error, "checking device properties");

	printf("Device has %u compute units, %lu local memory, %s memory\n",
			dev_info.compute_units, dev_info.local_mem_size,
			dev_info.host_mem ? "unified host" : "separate");

	/* creating a context for one dev */

	cl_context_properties ctx_prop[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
		0
	};

	cl_context ctx = clCreateContext(ctx_prop, 1, &dev, NULL, NULL, &error);
	check_ocl_error(error, "creating context");

	/* and a command queue to go with it */
	cl_command_queue queue = clCreateCommandQueue(ctx, dev,
			CL_QUEUE_PROFILING_ENABLE, &error);
	check_ocl_error(error, "creating command queue");

	/* initialize test data set */
	if (options.elements == 0) {
		cl_ulong amt = dev_info.mem_size;
		/*
		if (dev_info.max_alloc < amt)
			amt = dev_info.max_alloc;
			*/
		amt /= 1.3*sizeof(TYPE);
		if (amt > CL_UINT_MAX)
			options.elements = CL_UINT_MAX;
		else
			options.elements = amt;
		printf("will process %u (%uM) elements\n", options.elements,
				options.elements>>20);
	}

	TYPE *data = (TYPE*) calloc(options.elements, sizeof(TYPE));
	size_t data_size = options.elements * sizeof(TYPE);
	TYPE host_res = OP_NULL, dev_res = OP_NULL;

	for (cl_uint i = 0; i < options.elements; ++i) {
		// data[i] = (2*TYPE(rand())/RAND_MAX - 1)*1024;
#if TEST_MIN
		data[i] = options.elements - i;
		if (data[i] < host_res)
			host_res = data[i];
#else
		data[i] = 1;
		host_res += data[i];
#endif
	}
	printf("%u elements generated, " OP_NAME " " PTYPE ", data size %zu (%zuMB)\n",
			options.elements, host_res, data_size, data_size>>20);

	/* device buffers */

	cl_mem d_input, d_output;

	cl_event mem_evt[1];
	size_t mem_evt_count = 0;

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
				0, NULL, mem_evt + mem_evt_count++);
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

	clGetKernelWorkGroupInfo(reduceKernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
			sizeof(max_wg_size), &max_wg_size, NULL);
	error = clGetKernelWorkGroupInfo(reduceKernel, dev,
			CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
			sizeof(ws_multiple), &ws_multiple, NULL);
	check_ocl_error(error, "checking device properties");
	if (ws_multiple == 0) {
		printf("reported a null workgroup size multple! Assume 1\n");
		ws_multiple = 1;
	}
	printf("kernel prefers multiples of %zu up to %zu\n",
			ws_multiple, max_wg_size);

	if (!dev_info.host_mem) {
		clWaitForEvents(1, mem_evt);
		GET_RUNTIME(mem_evt[0], "memory upload");
	}

	size_t ws_first = options.groupsize ? options.groupsize : ws_multiple ;
	size_t ws_last = options.groupsize ? options.groupsize: max_wg_size ;
	for (size_t ws = ws_first ; ws <= ws_last; ws *= 2) {
		group_size[0] = ws;
		work_size[0] = group_size[0]*options.groups;

		int argnum = 0;
		KERNEL_ARG(d_input);
		error = clSetKernelArg(reduceKernel, argnum++,
				group_size[0]*sizeof(TYPE), NULL); \
			check_ocl_error(error, "setting kernel param"); \
			KERNEL_ARG(options.elements);
		KERNEL_ARG(d_output);

		/* launch kernel, with an event to collect profiling info */

		const cl_event *wait_evts = dev_info.host_mem ? NULL : mem_evt;

		clEnqueueNDRangeKernel(queue, reduceKernel,
				1,
				NULL, work_size, group_size,
				0, NULL,
				pass_evt);

		group_size[0] = ROUND_MUL(options.groups, ws_multiple);
		work_size[0] = group_size[0];

		argnum = 0;
		KERNEL_ARG(d_output);
		error = clSetKernelArg(reduceKernel, argnum++,
				options.groups*sizeof(TYPE), NULL); \
			check_ocl_error(error, "setting kernel param"); \
			KERNEL_ARG(options.groups);
		KERNEL_ARG(d_output);

		clEnqueueNDRangeKernel(queue, reduceKernel,
				1,
				NULL, work_size, group_size,
				1, pass_evt,
				pass_evt + 1);

		error = clFinish(queue); // sync on queue
		check_ocl_error(error, "finishing queue");

		printf("Group size: %zu\n", ws);
		GET_RUNTIME(pass_evt[0], "Kernel pass #1");
		GET_RUNTIME(pass_evt[1], "Kernel pass #2");
		GET_RUNTIME_DELTA(pass_evt[0], pass_evt[1], "Total");
		printf("Bandwidth: %.4g GB/s\n", double(data_size)/(endTime-startTime));
	}

	/* copy memory down */
	error = clEnqueueReadBuffer(queue, d_output, true, 0,
			sizeof(TYPE), &dev_res,
			0, NULL, mem_evt);
	check_ocl_error(error, "getting results");

	clFinish(queue);
	GET_RUNTIME(mem_evt[0], "memory download");
	data_size = sizeof(TYPE);
	printf("total download runtime: %gms for %gMB (%g GB/s)\n",
			double(endTime-startTime)/1000000,
			data_size/(1024*1024.0),
			/* the magic factor is 10^9/2^30, i.e.
			   decimal billions over binary gigas */
			data_size/double(endTime-startTime)*0.931322575);

	clReleaseMemObject(d_output);
	clReleaseMemObject(d_input);

	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);

	fflush(stdout);
	fflush(stderr);

	free(data);

	TYPE expected = TEST_MIN ? 1 : options.elements;
	printf("Parallel " OP_NAME ": " PTYPE " vs " PTYPE " (expected: " PTYPE ")\n",
		dev_res, host_res, expected);
	printf("Deltas: " PTYPE " vs " PTYPE "\n", dev_res - expected, host_res - expected);
}
