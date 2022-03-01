#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main()
{
    const int w = 800, h = 600, n = 3;

    unsigned char data[w * h * n];
    for(int j = 0; j < h; j++)
    {
        for(int i = 0; i < w; i++)
        {
            int idx = j * w * n + i * n;
            if(j == 0 && i < 10) std::cout << idx << std::endl;
            data[idx] = 255; //(float)i / 800;
            data[idx + 1] = 0; //(float)j / 600;
            data[idx + 2] = 0;
        }
    }

    stbi_write_jpg("test.jpg", w, h, n, data, 0);

    return 0;
}