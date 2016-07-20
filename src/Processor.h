#ifndef PROCESSOR_H
# define PROCESSOR_H

#include <vector>
#include <list>
#include <string>

#ifdef __APPLE__
# include "OpenCL/opencl.h"
#else
# include "CL/cl.h"
#endif

class Processor
{
public:
  struct KernelArg
  {
    enum Type { RAW, BUFFER, IMAGE };
    enum Direction { STATIC, INPUT, OUTPUT };

    KernelArg(Type _type, void const *_data, size_t _size = 0, bool _copy = false, Direction _direction = STATIC)
      : data(const_cast<void*>(_data)), size(_size), type(_type), copy(_copy), direction(_direction)
    {}

    void* data;
    size_t size;
    Type type;
    bool copy;
    Direction direction;
  };

  enum DeviceType { All_Devices, CPU_Devices, GPU_Devices };

  Processor(std::string const & kernelPath, DeviceType deviceType = All_Devices, std::string const & kernelArgs = "");
  ~Processor();

  void execute(std::string const & kernelFunction, std::list<KernelArg> const & kernelArgs);

private:
  struct InternalArg
  {
    InternalArg(cl_mem _buffer) : buffer(_buffer) {}

    cl_mem buffer;
  };

  #define MAX_DIM 9
  struct InputArg
  {
    InputArg(size_t _dim) : dim(_dim) {}

    size_t dim;
    size_t sizes[MAX_DIM];
  };

  struct OutputArg
  {
    OutputArg(KernelArg::Type _type, cl_mem _buffer, void *_data, size_t _size) : type(_type), buffer(_buffer), data(_data), size(_size) {}

    KernelArg::Type type;
    cl_mem buffer;
    void *data;
    size_t size;
  };

  struct Image
  {
    Image(unsigned int _width, unsigned int _height) : width(_width), height(_height) {}
    Image(unsigned int _width, unsigned int _height, std::vector<char> const & _pixel)
      : width(_width), height(_height), pixel(_pixel)
    {}

  	std::vector<char> pixel;
  	unsigned int width;
    unsigned int height;
  };

  void init(int selectedPlatform, int selectedDevice);

  void prepareArguments(cl_kernel kernel, std::list<KernelArg> const & args, InputArg& input, OutputArg& output, std::list<InternalArg>& internalArgs);

  std::vector<cl_platform_id> loadPlateforms();
  std::vector<cl_device_id> loadDevices(cl_platform_id platformId, cl_device_type deviceType);
  cl_context createContext(cl_platform_id platformId);
  std::string loadKernel(std::string const & name);
  cl_program createProgram(cl_context context, std::string const & kernelPath, std::string const & kernelArgs);
  cl_command_queue createCommandQueue(cl_device_id deviceId, cl_context context);

  void throwError(std::string const & message);
  void checkError(cl_int error);
  void log(std::string const & message);

  Image loadImage(std::string const & path);
  void saveImage(Image const & img, std::string const & path);

  static Image RGBtoRGBA(Processor::Image const & input);
  static Image RGBAtoRGB(Processor::Image const & input);

  static cl_device_type LookupDevice(DeviceType deviceType);
  static std::string GetPlatformName(cl_platform_id id);
  static std::string GetDeviceName(cl_device_id id);
  static std::string GetProgramBuildLog(cl_device_id id, cl_program program);
  static std::string GetErrorString(cl_int error);

  std::string _kernelPath;
  std::string _kernelArgs;
  cl_device_type _deviceType;

  std::vector<cl_platform_id> _platforms;
  cl_platform_id _currentPlatform;

  std::vector<cl_device_id> _devices;
  cl_device_id _currentDevice;

  cl_context _context;
  cl_program _program;
  cl_command_queue _queue;
};

#endif
