#include <iostream>
#include <string>
#include <chrono>
#include <tuple>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


template <class T = float>
class Image
{
public:
    int width, height, nchannel;
    T* data;

    Image(int _w, int _h, int _n = 3) : width(_w), height(_h), nchannel(_n)
    {
        checkCudaErrors(cudaMallocManaged(&data, width * height * nchannel * sizeof(T)));
    }

    __host__ __device__ void set(int y, int x, T r, T g, T b)
    {
        int idx = y * width * nchannel + x * nchannel;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
    }

    void save(const std::string& filename)
    {
        stbi_write_jpg(filename.c_str(), width, height, nchannel, data, 0);
    }

};


__global__ void kernel(int w, int h, Image<> img)
{
    // int ww = w / gridDim.x, hh = h / blockDim.x;

    // int x0 = blockIdx.x * ww, y0 = threadIdx.x * hh;
    // for(int i = 0; i < ww; i++)
    // {
    //     for(int j = 0; j < hh; j++)
    //     {
    //         int x = x0 + i, y = y0 + j;
    //         img.set(y, x, (float)x / w, (float)y / h, 0.0f);
    //     }
    // }

    for(int i = 0; i < w; i++)
    {
        for(int j = 0; j < h; j++)
        {
            // img.set(j, i, (float)i / w, (float)j / h, 0.0f);
            img.set(j, i, 1.0f, 0.0f, 0.0f);
        }
    }
}


int main()
{
    auto start = std::chrono::steady_clock::now();

    int w = 800, h = 600;
    Image<> img(w, h);
    kernel<<<1, 1>>>(w, h, img);
    cudaDeviceSynchronize();

    img.save("test.jpg");

    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count() << "s" << std::endl;
    return 0;
}