#include <iostream>
#include <fstream>
#include <sstream>
#include "Processor.h"
#include "Debug.hpp"

Processor::Processor(std::string const & kernelPath, DeviceType deviceType, std::string const & kernelArgs)
  : _kernelPath(kernelPath), _kernelArgs(kernelArgs),
    _currentPlatform(nullptr), _currentDevice(nullptr), _context(nullptr), _program(nullptr), _queue(nullptr)
{
  _deviceType = LookupDevice(deviceType);
  init(0, 1);
}

Processor::~Processor()
{
  if (_queue != nullptr)
    clReleaseCommandQueue(_queue);
  if (_program != nullptr)
    clReleaseProgram(_program);
  if (_context != nullptr)
    clReleaseContext(_context);
}

void Processor::init(int selectedPlatform, int selectedDevice)
{
  _platforms = loadPlateforms();
  _currentPlatform = _platforms[selectedPlatform];

  _devices = loadDevices(_currentPlatform, _deviceType);
  _currentDevice = _devices[selectedDevice];

  _context = createContext(_currentPlatform);
  _program = createProgram(_context, _kernelPath, _kernelArgs);

  _queue = createCommandQueue(_currentDevice, _context);
}

std::vector<cl_platform_id> Processor::loadPlateforms()
{
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);

	if (platformIdCount == 0)
		throwError("No OpenCL platform found");
  else
    log(std::string("Found ") + std::to_string(platformIdCount) + " platform(s)");

	std::vector<cl_platform_id> platformIds(platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

	for (cl_platform_id platformId : platformIds)
		log(std::string("\t" + GetPlatformName(platformId)));

  return platformIds;
}

std::vector<cl_device_id> Processor::loadDevices(cl_platform_id platformId, cl_device_type deviceType)
{
	cl_uint deviceIdCount = 0;
	clGetDeviceIDs(platformId, deviceType, 0, nullptr, &deviceIdCount);

	if (deviceIdCount == 0)
		throwError("No OpenCL devices found for given device type ");
  else
    log(std::string("Found ") + std::to_string(deviceIdCount) + " device(s) for platform " + GetPlatformName(platformId));

	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs(platformId, deviceType, deviceIdCount, deviceIds.data(), nullptr);

	for (cl_device_id deviceId : deviceIds)
		log(std::string("\t" + GetDeviceName(deviceId)));

  return deviceIds;
}

cl_context Processor::createContext(cl_platform_id platformId)
{
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platformId), 0
	};

	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext(contextProperties, _devices.size(), _devices.data(), nullptr, nullptr, &error);
	checkError(error);

	return context;
}

std::string Processor::loadKernel(std::string const & path)
{
	std::ifstream in(path);

  if (!in.is_open())
    throwError(std::string("Cannot find kernel file '") + path + "'");

	return std::string((std::istreambuf_iterator<char> (in)), std::istreambuf_iterator<char>());
}

cl_program Processor::createProgram(cl_context context, std::string const & kernelPath, std::string const & kernelArgs)
{
  std::string source(loadKernel(kernelPath));

	size_t lengths[1] = { source.size() };
	char const * sources[1] = { source.data() };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
	checkError(error);

  error = clBuildProgram(program, _devices.size(), _devices.data(), kernelArgs.c_str(), nullptr, nullptr);
  if (error != CL_SUCCESS)
  {
    for (cl_device_id device : _devices)
      log(std::string("Build error: ") + GetProgramBuildLog(device, program));
  }
  checkError(error);

	return program;
}

cl_command_queue Processor::createCommandQueue(cl_device_id deviceId, cl_context context)
{
	cl_int error = 0;
	cl_command_queue queue = clCreateCommandQueue(context, deviceId, 0, &error);
	checkError(error);

  return queue;
}

void Processor::prepareArguments(cl_kernel kernel, std::list<KernelArg> const & args, InputArg& input, OutputArg& output, std::list<InternalArg>& internalArgs)
{
	cl_int error = 0;

  unsigned int index = 0;
  for (KernelArg arg : args)
  {
    cl_mem buffer;
    size_t size = arg.size;

    InternalArg iarg(nullptr);

    if (arg.type != KernelArg::RAW)
    {
      int flags = arg.direction == KernelArg::STATIC || arg.direction == KernelArg::INPUT ? CL_MEM_READ_ONLY : CL_MEM_WRITE_ONLY;
      flags |= arg.copy ? CL_MEM_COPY_HOST_PTR : 0;

      if (arg.type == KernelArg::BUFFER)
      {
        buffer = clCreateBuffer(_context, flags, arg.size, arg.copy ? arg.data : nullptr, &error);
        if (!arg.copy)
        {
          checkError(error);
          error = clEnqueueWriteBuffer(_queue, buffer, CL_TRUE, 0, arg.size, arg.data, 0, nullptr, nullptr);
        }
        if (arg.direction == KernelArg::INPUT)
        {
          input.dim = 1;
          input.sizes[0] = arg.size;
          input.sizes[1] = 0;
        }
      }
      else if (arg.type == KernelArg::IMAGE)
      {
        cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };
        Image image(0, 0);
        if (arg.direction == KernelArg::OUTPUT)
        {
          flags = flags & ~CL_MEM_COPY_HOST_PTR;
          image = Image(input.sizes[0], input.sizes[1]);
        }
        else
          image = RGBtoRGBA(loadImage(std::string(static_cast<char*>(arg.data))));

        void *imgData = arg.direction == KernelArg::OUTPUT ? nullptr : const_cast<char*>(image.pixel.data());

        buffer = clCreateImage2D(_context, flags, &format, image.width, image.height, 0, arg.copy ? imgData : nullptr, &error);
        if (!arg.copy && arg.direction != KernelArg::OUTPUT)
        {
          checkError(error);
        	std::size_t origin[3] = { 0, 0, 0 };
        	std::size_t region[3] = { image.width, image.height, 1 };
          error = clEnqueueWriteImage(_queue, buffer, CL_TRUE, origin, region, 0, 0, imgData, 0, nullptr, nullptr);
        }
        if (arg.direction == KernelArg::INPUT)
        {
          input.dim = 2;
          input.sizes[0] = image.width;
          input.sizes[1] = image.height;
          input.sizes[2] = 0;
        }
      }
      checkError(error);
      size = sizeof(cl_mem);

      if (arg.direction == KernelArg::OUTPUT)
        output = OutputArg(arg.type, buffer, arg.data, arg.size);

      iarg = InternalArg(buffer);
      internalArgs.push_back(iarg);
    }

    error = clSetKernelArg(kernel, index++, size, arg.type != KernelArg::RAW ? reinterpret_cast<void**>(&iarg.buffer) : arg.data);
    checkError(error);
  }

  if (input.dim == 0)
    throwError("No input parameter specified");
  if (output.buffer == nullptr)
    throwError("No output parameter specified");
}

void Processor::execute(std::string const & kernelFunction, std::list<KernelArg> const & args)
{
	cl_int error = 0;

	cl_kernel kernel = clCreateKernel(_program, kernelFunction.c_str(), &error);
	checkError(error);

  std::list<InternalArg> internalArgs;
  InputArg input(0);
  OutputArg output(KernelArg::RAW, nullptr, nullptr, 0);
  prepareArguments(kernel, args, input, output, internalArgs);

	checkError(clEnqueueNDRangeKernel(_queue, kernel, input.dim, nullptr, input.sizes, nullptr, 0, nullptr, nullptr));

  if (output.type == KernelArg::BUFFER)
    error = clEnqueueReadBuffer(_queue, output.buffer, CL_TRUE, 0, output.size, output.data, 0, nullptr, nullptr);
  else if (output.type == KernelArg::IMAGE)
  {
  	Image result(input.sizes[0], input.sizes[1], std::vector<char>(input.sizes[0] * input.sizes[1] * 4));
  	std::fill(result.pixel.begin(), result.pixel.end(), 0);

  	std::size_t origin[3] = { 0, 0, 0 };
  	std::size_t region[3] = { result.width, result.height, 1 };
  	error = clEnqueueReadImage(_queue, output.buffer, CL_TRUE, origin, region, 0, 0, result.pixel.data(), 0, nullptr, nullptr);

    saveImage(RGBAtoRGB(result), std::string(static_cast<char*>(output.data)));
  }
  checkError(error);

  for (InternalArg arg : internalArgs)
    clReleaseMemObject(arg.buffer);

	clReleaseKernel(kernel);
}

void Processor::throwError(std::string const & message)
{
  log(message);
  printStacktrace();
  throw std::runtime_error(message);
}

void Processor::checkError(cl_int error)
{
	if (error != CL_SUCCESS) {
    std::string message("OpenCL call failed with error ");
    message += GetErrorString(error);
    throwError(message);
	}
}

void Processor::log(std::string const & message)
{
  std::cout << "Processor: " << message << std::endl;
}

Processor::Image Processor::loadImage(std::string const & path)
{
	std::ifstream in(path, std::ios::binary);

  if (!in.is_open())
    throwError(std::string("Cannot open image '") + path + "'");

	std::string s;
	in >> s;

	if (s != "P6") {
    throwError("Bad image format for '" + path + "', only PPM supported");
	}

	// Skip comments
	for (;;) {
		getline(in, s);
		if (s.empty ()) {
			continue;
		}
		if (s[0] != '#') {
			break;
		}
	}

	std::stringstream str(s);
	unsigned int width, height, maxColor;
	str >> width >> height;
	in >> maxColor;

	if (maxColor != 255) {
    throwError("Bad max color for '" + path + "', should be 255");
	}

	{
		// Skip until end of line
		std::string tmp;
		getline(in, tmp);
	}

	std::vector<char> data(width * height * 3);
	in.read(reinterpret_cast<char*>(data.data()), data.size());

	return Image(width, height, data);
}

void Processor::saveImage(Image const & img, std::string const & path)
{
	std::ofstream out(path, std::ios::binary | std::ios::trunc);

  if (!out.is_open())
    throwError(std::string("Cannot save image '") + path + "'");

	out << "P6\n";
	out << img.width << " " << img.height << "\n";
	out << "255\n";
	out.write(img.pixel.data(), img.pixel.size());
}

Processor::Image Processor::RGBtoRGBA(Processor::Image const & input)
{
	Image result(input.width, input.height);

	for (std::size_t i = 0; i < input.pixel.size(); i += 3) {
		result.pixel.push_back(input.pixel [i + 0]);
		result.pixel.push_back(input.pixel [i + 1]);
		result.pixel.push_back(input.pixel [i + 2]);
		result.pixel.push_back(0);
	}

	return result;
}

Processor::Image Processor::RGBAtoRGB(Processor::Image const & input)
{
	Image result(input.width, input.height);

	for (std::size_t i = 0; i < input.pixel.size(); i += 4) {
		result.pixel.push_back(input.pixel [i + 0]);
		result.pixel.push_back(input.pixel [i + 1]);
		result.pixel.push_back(input.pixel [i + 2]);
	}

	return result;
}

cl_device_type Processor::LookupDevice(DeviceType deviceType)
{
  if (deviceType == All_Devices)
    return CL_DEVICE_TYPE_ALL;
  else if (deviceType == CPU_Devices)
    return CL_DEVICE_TYPE_CPU;
  else if (deviceType == GPU_Devices)
    return CL_DEVICE_TYPE_GPU;
  return CL_DEVICE_TYPE_ALL;
}

std::string Processor::GetPlatformName(cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetPlatformInfo(id, CL_PLATFORM_NAME, size, const_cast<char*>(result.data()), nullptr);

	return result;
}

std::string Processor::GetDeviceName(cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize(size);
	clGetDeviceInfo(id, CL_DEVICE_NAME, size, const_cast<char*>(result.data()), nullptr);

	return result;
}

std::string Processor::GetProgramBuildLog(cl_device_id deviceId, cl_program program)
{
  size_t size = 0;
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);

	std::string result;
	result.resize(size);
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, size, const_cast<char*>(result.data()), nullptr);

  return result;
}

std::string Processor::GetErrorString(cl_int error)
{
  switch (error) {
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
  }
}
