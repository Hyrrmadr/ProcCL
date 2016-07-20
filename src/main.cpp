#include "Processor.h"

#include <iostream>

#include <cmath>
template <typename T>
float sq(T n)
{
  return n * n;
}

static std::vector<float> getGaussianKernel(float sigma, int radius)
{
  float twoSigmaSquared = 2 * sq(sigma);
  float twoPiSigmaSquared = twoSigmaSquared * M_PI;
  size_t size = radius * 2 + 1;

  std::vector<float> kernel(size * size);
  for (int i = -radius; i <= radius; ++i)
    for (int j = -radius; j <= radius; ++j)
      kernel[(i + radius) * size + (j + radius)] = twoPiSigmaSquared * exp(-(sq(i) + sq(j)) / twoSigmaSquared);
  float sum = 0;
  for (float n : kernel)
    sum += n;
  std::transform(kernel.begin(), kernel.end(), kernel.begin(), [sum] (float n) -> float { return n / sum; });
  return kernel;
}

#define USE_BLUR true

int main()
{
  size_t dataSize = 5;
  float factor = 2;
	std::vector<float> input(dataSize), output(dataSize);

	for (size_t i = 0; i < dataSize; ++i)
  {
		input[i] = static_cast<float>(23 ^ i);
    output[i] = 0.0f;
  }

  int kernelRadius = 5;
  std::vector<float> filter = getGaussianKernel(1.5, kernelRadius);

  try
  {
    std::string program(USE_BLUR ? "blur" : "saxpy");

    std::cout << "# Launching '" << program << "'" << std::endl;

    Processor p("src/kernels/" + program + ".cl", Processor::All_Devices);

    std::list<Processor::KernelArg> args;

    if (program == "blur")
    {
      args.push_back(Processor::KernelArg(Processor::KernelArg::IMAGE, "res/input.ppm", 0, false, Processor::KernelArg::INPUT));
      args.push_back(Processor::KernelArg(Processor::KernelArg::BUFFER, filter.data(), sizeof(float) * filter.size()));
      args.push_back(Processor::KernelArg(Processor::KernelArg::RAW, &kernelRadius, sizeof(kernelRadius)));
      args.push_back(Processor::KernelArg(Processor::KernelArg::IMAGE, "res/output.ppm", 0, false, Processor::KernelArg::OUTPUT));
    }
    else
    {
      args.push_back(Processor::KernelArg(Processor::KernelArg::BUFFER, input.data(), sizeof(float) * input.size(), true, Processor::KernelArg::INPUT));
      args.push_back(Processor::KernelArg(Processor::KernelArg::BUFFER, output.data(), sizeof(float) * output.size(), true, Processor::KernelArg::OUTPUT));
      args.push_back(Processor::KernelArg(Processor::KernelArg::RAW, &factor, sizeof(float)));
    }
    p.execute(program, args);
  }
  catch (std::exception const & e)
  {
    std::cerr << "Processor failed: " << e.what() << std::endl;
    return 1;
  }

#if USE_BLUR
  size_t kernelSize = kernelRadius * 2 + 1;
  for (size_t i = 0; i < kernelSize; ++i)
  {
    for (size_t j = 0; j < kernelSize; ++j)
      std::cout << (j == 0 ? "" : ", ") << filter[i * kernelSize + j];
    std::cout << std::endl;
  }
#else
  for (size_t i = 0; i < dataSize; ++i)
    std::cout << output[i] << " = " << factor << " * " << input[i] << std::endl;
#endif

	return 0;
}
