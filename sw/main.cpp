
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }



#include "darknet.h"
#include <sys/time.h>
#include <assert.h>
#include <dirent.h>
#include <string.h>


int gpu_index = 0;
#ifndef SW_EMU
	int kernel_flag = 5;
#endif



int A_sizes[ARRAYS_SZS_SZ] = { 0 };
int B_sizes[ARRAYS_SZS_SZ] = { 0 };
int C_sizes[ARRAYS_SZS_SZ] = { 0 };
int counter_sz = 0;

int krnl_M = 0;
int krnl_K = 0;
int krnl_N = 0;
int krnl_lda = 0;
int krnl_ldb = 0;
int krnl_ldc = 0;
float krnl_var = 0;



swm::XilinxOcl xocl;
cl::CommandQueue q ;
cl::Kernel krnl;


int main(int argc, char* argv[]) {
    // TARGET_DEVICE macro needs to be passed from gcc command line
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    printf ("New Implementation \n");
	printf ("Welcome to  Vitis \n");
#ifndef SW_EMU
//	init_kernel();
	 EventTimer et;

	 printf("-- Parallelizing the Data Path --\n\n");
	 // Initialize the runtime (including a command queue) and load the
	 // FPGA image
    std::string xclbinFilename = argv[1];


    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary

    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx") {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size()) {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false) {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl = cl::Kernel(program, "wide_vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

/*	    std::cout << "-- Parallelizing the Data Path --" << std::endl
	              << std::endl;

	    // Initialize the runtime (including a command queue) and load the
	    // FPGA image
	    std::cout << "Loading binary_container_1.xclbin to program the board" << std::endl
	              << std::endl;
	    et.add("OpenCL Initialization");

	    // This application will use the first Xilinx device found in the system
	    swm::XilinxOcl xocl;
	    xocl.initialize("binary_container_1.xclbin");

	    cl::CommandQueue q = xocl.get_command_queue();
	    cl::Kernel krnl    = xocl.get_kernel("wide_vadd");
	    et.finish();

	 std::cout << "Running kernel test XRT-allocated buffers and wide data path:" << std::endl
	           << std::endl;

	 et.add("Allocate contiguous OpenCL buffers");

	 cl_mem_ext_ptr_t bank_ext;
	 bank_ext.flags = 2 | XCL_MEM_TOPOLOGY;
	 bank_ext.obj   = NULL;
	 bank_ext.param = 0;
	 */
	 cl::Buffer a_buf(context, //xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					  BUFSIZE_A * sizeof(float),
					  NULL,
					  NULL);
	 cl::Buffer b_buf(context, //xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					  BUFSIZE_B * sizeof(float),
					  NULL,
					  NULL);
	 cl::Buffer c_buf(context, //xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					  BUFSIZE_C * sizeof(float),
					  NULL,
					  NULL);
	 cl::Buffer d_buf(context, //xocl.get_context(),
					  static_cast<cl_mem_flags>(CL_MEM_READ_ONLY),
					  BUFSIZE_C * sizeof(float),
					  NULL,
					  NULL);
	 et.finish();





	 std::cout << "--------------- Key Initialiazation times ---------------" << std::endl;

	et.print();


#endif
	et.clear();
	 et.add("predict_unet_segmenter");
	predict_unet_segmenter(a_buf, b_buf, c_buf, d_buf);
	et.finish();
	printf ("End of UNET \n");
	std::cout << "--------------- predict_unet_segmenter times ---------------" << std::endl;
	et.print();


	int maximumA=0;
	int maximumB=0;
	int maximumC=0;
	for (int i=0;i<counter_sz;i++)
	{
		if(A_sizes[i]>maximumA)
		{
			maximumA=A_sizes[i];
		}
		if(B_sizes[i]>maximumB)
		{
			maximumB=B_sizes[i];
		}
		if(C_sizes[i]>maximumC)
		{
			maximumC=C_sizes[i];
		}
//	    printf("A size is: %d\t", A_sizes[i]);
//	    printf("B size is: %d\t", B_sizes[i]);
//	    printf("C size is: %d\n", C_sizes[i]);
	}




	printf("A Max size is %d \n", maximumA);
	printf("B Max size is %d \n", maximumB);
	printf("C Max size is %d \n", maximumC);

    OCL_CHECK(err, err = q.finish());

	return 1;
}
