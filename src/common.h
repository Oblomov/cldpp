/* usual C includes */
#include <stdio.h>
#include <string.h>
#include <stdbool.h> // bool, true, false
#include <stdlib.h> // for rand()
#include <errno.h>

#include <stdarg.h> // variadic arguments for clbuild_printf

/* OpenCL includes */

/* OpenCL 2.0 deprecated clCreateCommandQueue, replaced by
 * clCreateCommandQueueWithProperties, which allows additional properties
 * to be defined (namely, the queue size for device queues.
 * However, we can't just use the 2.0+ API call as-is, since we might still
 * be running against 1.x platforms, so we'll just have to enable the use
 * of the deprecated APIs (for now).
 */

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

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
	cl_uint cugroups; // groups per CU
	cl_uint groups; // total number of groups, overrides cugroups if both are specified
	cl_int unroll; // number of times to unroll the loading loop in the reduction
	cl_int reduction_style; // 0: interleaved; 1: blocks
	cl_uint vecsize; // vectorized width
	size_t groupsize;
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
			(double)(endTime-startTime)/1000000); \
	} while (0);

#define GET_RUNTIME_DELTA(evt1, evt2, text) do {\
	clGetEventProfilingInfo(evt1, CL_PROFILING_COMMAND_START, \
			sizeof(cl_ulong), &startTime, NULL);\
	error = clGetEventProfilingInfo(evt2, CL_PROFILING_COMMAND_END,\
			sizeof(cl_ulong), &endTime, NULL); \
	check_ocl_error(error, "getting profiling info"); \
	printf(text " runtime: %gms\n", \
			(double)(endTime-startTime)/1000000); \
	} while (0);

/* macro to set the next kernel argument */
#define KERNEL_ARG(krn, what) do {\
	error = clSetKernelArg(krn, argnum++, sizeof(what), &(what)); \
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
	/* device type */
	cl_device_type dev_type;
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

void check_ocl_error(const cl_int error, const char *message) {
	if (error != CL_SUCCESS) {
		fprintf(stderr, "error %d %s\n", error, message);
		exit(1);
	}
}

const char * const paths[] = {
	".",
	"src",
	NULL
};

const size_t paths_len = 4; /* maximum length of a paths element, +1 */

char *read_file(const char *fname) {
	size_t fsize, readsize;
	char *buff = NULL;
	FILE *fd = NULL;
	const char * const *path = paths;
	const size_t path_max = strlen(fname) + paths_len + 1;
	char *full_path = malloc(path_max);

	if (!full_path) {
		fprintf(stderr, "unable to allocate read_file path buffer\n");
		return NULL;
	}

	while (!fd && *path) {
		const size_t len = snprintf(full_path, path_max, "%s/%s", *path, fname);
		if (len >= path_max)
			break;
		fd = fopen(full_path, "rb");
		++path;
	}

	if (!fd) {
		fprintf(stderr, "%s not found\n", fname);
		return NULL;
	} else {
		printf("loading %s\n", full_path);
	}

	fseek(fd, 0, SEEK_END);
	fsize = ftell(fd);

	buff = malloc(fsize + 1);
	if (!buff) {
		fprintf(stderr, "unable to allocate buffer for reading\n");
		return NULL;
	}

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

/* Work-group sizes must be a power of two, so we'll need to check
 * and optionally user/device-provied values */

/* check if a value is a power of two */
static inline bool is_po2(size_t in)
{
	return !(in & (in-1));
}

/* find the closest power-of-two approximation of a number */
size_t fix_po2(size_t in)
{
	/* I'd go for popcount, but meh */
	size_t out = 1;
	while (out < in)
		out *= 2;
	if (in - out > in - 2*out)
		out *= 2;
	return out;
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
		} else if (!strcmp(arg, "--cugroups")) {
			sscanf(*argv, "%u", &(options.cugroups));
			++argv; --argc;
		} else if (!strcmp(arg, "--groups")) {
			sscanf(*argv, "%u", &(options.groups));
			++argv; --argc;
		} else if (!strcmp(arg, "--vecsize")) {
			sscanf(*argv, "%u", &(options.vecsize));
			++argv; --argc;
		} else if (!strcmp(arg, "--groupsize")) {
			sscanf(*argv, "%zu", &(options.groupsize));
			++argv; --argc;
		} else if (!strcmp(arg, "--reduction-style")) {
			sscanf(*argv, "%d", &(options.reduction_style));
			++argv; --argc;
		} else if (!strcmp(arg, "--unroll")) {
			sscanf(*argv, "%d", &(options.unroll));
			++argv; --argc;
		} else if (!strncmp(arg, "-D", 2)) {
			clbuild_add(arg);
		} else if (!strncmp(arg, "-", 1)) {
			fprintf(stderr, "unrecognized option %s\n", arg);
		} else {
			fprintf(stderr, "too many filenames: %s\n", arg);
		}
	}

	if (options.cugroups && options.groups) {
		printf("number of groups %u overrides number of groups/CU %u\n",
			options.groups, options.cugroups);
	}

	/* Check that options.groupsize is a power-of-two */
	if (options.groupsize && !is_po2(options.groupsize)) {
		/* not power-of-two, fix */
		const size_t oldval = options.groupsize;
		const size_t newval = fix_po2(oldval);
		fprintf(stderr, "selected groupsize %zu is not a power of two,"
			" rounding to %zu.\n", oldval, newval);
		options.groupsize = newval;
	}

	if (options.vecsize == 0)
		options.vecsize = 1;
	else if (!is_po2(options.vecsize)) {
		fprintf(stderr, "vector size %u not supported\n", options.vecsize);
		exit(1);
	}

	if (options.unroll < 1 || options.unroll > 16) {
		fprintf(stderr, "only unrolls of 1 to 8 are supported\n");
		exit(1);
	}

}

/* auxiliary buffer to read platform and device info */
char buffer[1024];


cl_platform_id select_platform(void)
{
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

	platform = platform_list[options.platform];
	printf("using platform %u\n", options.platform);

	return platform;
}

cl_device_id select_device(cl_platform_id platform)
{
	/* device selection */

	cl_uint num_devs = 0;
	cl_device_id *dev_list = NULL;
	cl_device_id dev = NULL;

	/* possible types: CPU, GPU, ACCELERATOR, DEFAULT, ALL */
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devs);
	dev_list = (cl_device_id *)calloc(num_devs, sizeof(cl_device_id));
	cl_int error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devs,
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

	dev = dev_list[options.device];
	printf("using device %u\n", options.device);

	error = clGetDeviceInfo(dev, CL_DEVICE_TYPE,
			sizeof(dev_info.dev_type), &dev_info.dev_type,
			NULL);
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

	printf("Device has %u compute units, %lu local memory, %lu (%luMB) %s memory with max alloc of %lu (%luMB)\n",
			dev_info.compute_units, dev_info.local_mem_size,
			dev_info.mem_size, dev_info.mem_size >> 20,
			dev_info.host_mem ? "unified host" : "separate",
			dev_info.max_alloc, dev_info.max_alloc >> 20);

	return dev;
}

cl_context create_context(cl_platform_id pid, cl_device_id dev) {
	/* creating a context for one dev */

	cl_context_properties props[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)pid,
		0
	};
	cl_int err;

	cl_context ctx = clCreateContext(props, 1, &dev, NULL, NULL, &err);
	check_ocl_error(err, "create context");
	return ctx;
}

cl_command_queue create_queue(cl_context ctx, cl_device_id dev) {
	cl_int err;
	cl_command_queue que = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	check_ocl_error(err, "create command queue");
	return que;
}

cl_program create_program(const char *fname, cl_context ctx, cl_device_id dev)
{
	char *prog_source = read_file(fname);
	if (prog_source == NULL)
		exit(1);

	cl_int error;

	cl_program program = clCreateProgramWithSource(ctx, 1,
			(const char **)&prog_source, NULL, &error);
	check_ocl_error(error, "creating program");

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

	return program;
}

