__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;

float FilterValue(__constant const float* filterWeights, size_t kernelRadius, const int x, const int y)
{
  return filterWeights[(x + kernelRadius) + (y + kernelRadius) * (kernelRadius * 2 + 1)];
}

__kernel void blur(__read_only image2d_t input, __constant float* filterWeights, int kernelRadius, __write_only image2d_t output)
{
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 sum = (float4)(0.0f);

  for (int y = -kernelRadius; y <= kernelRadius; ++y) {
      for (int x = -kernelRadius; x <= kernelRadius; ++x) {
          sum += FilterValue(filterWeights, kernelRadius, x, y) * read_imagef(input, sampler, pos + (int2)(x, y));
      }
  }

  write_imagef(output, (int2)(pos.x, pos.y), sum);
}
