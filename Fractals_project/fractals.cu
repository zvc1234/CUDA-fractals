#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tinycthread.h"
#include "util.h"



__device__ void set_pixel(unsigned char* image, int width, int x, int y, unsigned char* c) {
    image[4 * width * y + 4 * x + 0] = c[0];
    image[4 * width * y + 4 * x + 1] = c[1];
    image[4 * width * y + 4 * x + 2] = c[2];
    image[4 * width * y + 4 * x + 3] = 255;
}

/* This should be conveted into a GPU kernel */
__global__ void generate_image(unsigned char* image, unsigned char* colormap) {
    int row, col, index, iteration;
    double c_re, c_im, x, y, x_new;

    int width = WIDTH;
    int height = HEIGHT;
    int max = MAX_ITERATION;

    index = blockIdx.x * blockDim.x + threadIdx.x;
    row = index / height;
    col = index % height;
    c_re = (col - width / 2.0) * 4.0 / width;
    c_im = (row - height / 2.0) * 4.0 / width;
    x = 0, y = 0;
    iteration = 0;
    while (x * x + y * y <= 4 && iteration < max) {
        x_new = x * x - y * y + c_re;
        y = 2 * x * y + c_im;
        x = x_new;
        iteration++;
    }
    if (iteration > max) {
        iteration = max;
    }
    set_pixel(image, width, col, row, &colormap[iteration * 3]);

}

int main(int argc, char** argv) {
    struct arg a;
    double times[REPEAT];
    struct timeb start, end;
    int i, r;
    char path[255];
    cudaError_t cudaStatus;

    int blockSize = 1024;
    int gridSize = (WIDTH * HEIGHT + blockSize - 1) / blockSize;
   

    unsigned char* colormap = (unsigned char*)malloc((MAX_ITERATION + 1) * 3);
    unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * 4);

    init_colormap(MAX_ITERATION, colormap);

    a.image = image;
    a.colormap = colormap;
    a.width = WIDTH;
    a.height = HEIGHT;
    a.max = MAX_ITERATION;

    unsigned char* dev_colormap;
    unsigned char* dev_image;

    cudaStatus = cudaMalloc((void**)&dev_colormap, (MAX_ITERATION + 1) * 3 * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for dev_colormap failed");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_image, WIDTH * HEIGHT * 4 * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for dev_image failed");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_colormap, colormap, (MAX_ITERATION + 1) * 3 * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcopy for dev_colormap failed");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_image, image, HEIGHT * WIDTH * 4 *  sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcopy for dev_image failed");
        goto Error;
    }

    for (r = 0; r < REPEAT; r++) {
        memset(image, 0, WIDTH * HEIGHT * 4);

        ftime(&start);
       
        generate_image << <gridSize, blockSize >> > (dev_image, dev_colormap);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        cudaStatus = cudaMemcpy(colormap, dev_colormap, (MAX_ITERATION + 1) * 3 * sizeof(char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cuda memcopy of colormap from device failed");
            goto Error;
        }

        cudaStatus = cudaMemcpy(image, dev_image, HEIGHT * WIDTH * 4 * sizeof(char), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cuda memcopy of image from device failed");
            goto Error;
        }

        ftime(&end);
        times[r] = end.time - start.time + ((double)end.millitm - (double)start.millitm) / 1000.0;

        sprintf(path, IMAGE, "gpu", r);
        save_image(path, image, WIDTH, HEIGHT);
        progress("gpu", r, times[r], gridSize, blockSize);
    }
    report("gpu", times, gridSize, blockSize);


Error:
    cudaFree(dev_colormap);
    cudaFree(dev_image);
    free(image);
    free(colormap);

    return 0;
}

