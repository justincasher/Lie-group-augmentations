#define PY_SSIZE_T_CLEAN
#include "...\include\Python.h" // set this to the Python.h file path

/* 
 * Squares the input number
 * 
 * @param double num -> The number to be squared 
 *
 * @returns double -> The result of squaring num
*/
double square(double num) {
    return num * num;
}

/*
 * Applies a given PGL(C, 2) transformation to a given image
 *
 * @param double *image -> The image to be augmented, given as a vector
 * @param int image_size -> The width/height of the square image 
 * @param double a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i -> Matrix elements
 *
 * @returns void
*/
void complexPGL2TransformImage(double *image, int image_size, double a_r, double a_i, double b_r, double b_i, double c_r, double c_i, double d_r, double d_i) {

    // create copy of array 
    int array_size = 3 * image_size * image_size;
    double temp_image[array_size];
    memcpy(temp_image, image, sizeof(double) * array_size);
    
    // perform PGL_2(C) transformations
    for (int x = 0; x < image_size; x++) {
        for (int y = 0; y < image_size; y++) {
            double z_r = (double) (x - ((image_size / 2) - 1)) / (image_size / 2);
            double z_i = (double) (y - ((image_size / 2) - 1)) / (image_size / 2);

            double N = square(c_r * c_r - c_i * z_i + d_r) + square(c_i * z_r + c_r * z_i + d_i);

            double real = (a_r * z_r - a_i * z_i + b_r) * (c_r * z_r - c_i * z_i + d_r);
            real -= (a_r * z_i + a_i * z_r + b_i) * (c_i * z_r + c_r * z_i + d_i);

            double imag = (a_r * z_r - a_i * z_i + b_r) * (c_i * z_r + c_r * z_i + d_i);
            imag += (a_r * z_i + a_i * z_r + b_i) * (c_r * z_r - c_i * z_i + d_r);

            if (N != 0) {
                real = real / N;
                imag = imag / N;
            }
            else {
                real = -1;
                imag = -1;
            }

            real = (image_size / 2) * real + ((image_size / 2) - 1);
            imag = (image_size / 2) * imag + ((image_size / 2) - 1);

            // add 0.5 to avoid rounding issues
            int real_int = (int) real + 0.5;
            int imag_int = (int) imag + 0.5;

            for (int channel = 0; channel < 3; channel++) {
                int coordinate = channel * (image_size * image_size) + y * image_size + x;
                int old_coordinate = channel * (image_size * image_size) + imag_int * image_size + real_int;

                if (0 <= real_int && real_int < image_size && 0 <= imag_int && imag_int < image_size) {
                    image[coordinate] = temp_image[old_coordinate];
                }
                else {
                    image[coordinate] = 0;
                }
            }
        }
    }
}

/*
 * Applies a given PGL(R, 2)^2 transformation to a given image
 *
 * @param double *image -> The image to be augmented, given as a vector
 * @param int image_size -> The width/height of the square image 
 * @param double a_1, b_1, c_1, d_1, a_2, b_2, c_2, d_2 -> Matrix elements
 *
 * @returns void
*/
void realPGL2SqrTransformImage(double *image, int image_size, double a_1, double b_1, double c_1, double d_1, double a_2, double b_2, double c_2, double d_2) {

    // create copy of array 
    int array_size = 3 * image_size * image_size;
    double temp_image[array_size];
    memcpy(temp_image, image, sizeof(double) * array_size);
    
    // perform PGL_2(C) transformations
    for (int x = 0; x < image_size; x++) {
        for (int y = 0; y < image_size; y++) {
            double x_normalized = (double) (x - ((image_size / 2) - 1)) / (image_size / 2);
            double y_normalized = (double) (y - ((image_size / 2) - 1)) / (image_size / 2);

            double old_x_normalized, old_y_normalized;

            if (c_1 * x_normalized + d_1 != 0) {
                 old_x_normalized = (a_1 * x_normalized + b_1) / (c_1 * x_normalized + d_1);
            }
            else {
                old_x_normalized = -1;
            }

            if (c_2 * y_normalized + d_2 != 0) {
                old_y_normalized = (a_2 * y_normalized + b_2) / (c_2 * y_normalized + d_2);
            }
            else {
                old_y_normalized = -1;
            }

            double old_x = (image_size / 2) * old_x_normalized + ((image_size / 2) - 1);
            double old_y = (image_size / 2) * old_y_normalized + ((image_size / 2) - 1);

            // add 0.5 to avoid rounding issues
            int old_x_int = (int) old_x + 0.5;
            int old_y_int = (int) old_y + 0.5;

            for (int channel = 0; channel < 3; channel++) {
                int coordinate = channel * (image_size * image_size) + y * image_size + x;
                int old_coordinate = channel * (image_size * image_size) + old_y_int * image_size + old_x_int;

                if (0 <= old_x_int && old_x_int < image_size && 0 <= old_y_int && old_y_int < image_size) {
                    image[coordinate] = temp_image[old_coordinate];
                }
                else {
                    image[coordinate] = 0;
                }
            }
        }
    }
}

/*
 * Applies a given PGL(R, 3) transformation to a given image
 *
 * @param double *image -> The image to be augmented, given as a vector
 * @param int image_size -> The width/height of the square image 
 * @param double a_1, b_1, c_1, a_2, b_2, c_2, a_3, b_3, c_3 -> Matrix elements
 *
 * @returns void
*/
void realPGL3TransformImage(double *image, int image_size, double a_1, double b_1, double c_1, double a_2, double b_2, double c_2, double a_3, double b_3, double c_3) {

    // create copy of array 
    int array_size = 3 * image_size * image_size;
    double temp_image[array_size];
    memcpy(temp_image, image, sizeof(double) * array_size);
    
    // perform PGL_2(C) transformations
    for (int x = 0; x < image_size; x++) {
        for (int y = 0; y < image_size; y++) {
            // normalize
            double x_normalized = (double) (x - ((image_size / 2) - 1)) / (image_size / 2);
            double y_normalized = (double) (y - ((image_size / 2) - 1)) / (image_size / 2);

            // transform
            double div = a_3 * x_normalized + b_3 * y_normalized + c_3;

            double old_x_normalized, old_y_normalized;

            if (div != 0) {
                old_x_normalized = (a_1 * x_normalized + b_1 * y_normalized + c_1) / div;
                old_y_normalized = (a_2 * x_normalized + b_2 * y_normalized + c_2) / div;
            }
            else {
                old_x_normalized = -1;
                old_y_normalized = -1;
            }

            // unnormalize
            double old_x = (image_size / 2) * old_x_normalized + ((image_size / 2) - 1);
            double old_y = (image_size / 2) * old_y_normalized + ((image_size / 2) - 1);

            // round; add 0.5 to avoid rounding issues
            int old_x_int = (int) old_x + 0.5;
            int old_y_int = (int) old_y + 0.5;

            for (int channel = 0; channel < 3; channel++) {
                int coordinate = channel * (image_size * image_size) + y * image_size + x;
                int old_coordinate = channel * (image_size * image_size) + old_y_int * image_size + old_x_int;

                if (0 <= old_x_int && old_x_int < image_size && 0 <= old_y_int && old_y_int < image_size) {
                    image[coordinate] = temp_image[old_coordinate];
                }
                else {
                    image[coordinate] = 0;
                }
            }
        }
    }
}
