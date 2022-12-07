#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstring>

namespace py = pybind11;

float *matrix_mul(const float *M, const float *N, size_t m, size_t n, size_t k) {  // M: mxn, N: nxk
    float *result = new float[m * k]();
    for(size_t ii = 0; ii < m; ii++) {
        for(size_t jj = 0; jj < k; jj++) {
            for(size_t kk = 0; kk < n; kk++) {
                result[ii * k + jj] += M[ii * n + kk] * N[kk * k + jj];
            }
        }
    }
    return result;
}

float *matrix_sub(const float *M, const float *N, size_t m, size_t n) {
    float *result = new float[m * n]();
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            result[i * n + j] = M[i * n + j] - N[i * n + j];
        }
    }
    return result;
}

float *matrix_transpose(const float *M, size_t m, size_t n) {
    float *result = new float[n * m]();
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            result[j * m + i] = M[i * n + j];
        }
    }
    return result;
}

float *softmax(const float *M, size_t m, size_t n) {
    float *exp_M = new float[m * n]();
    float *row_sum = new float[m]();
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            exp_M[i * n + j] = exp(M[i * n + j]);
            row_sum[i] += exp_M[i * n + j];
        }
    }
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            exp_M[i * n + j] /= row_sum[i];
        }
    }
    return exp_M;
}

float *one_hot_encoding(const unsigned char *y, size_t m, size_t k) {
    float *result = new float[m * k]();
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < k; j++) {
            if(y[i] == j) {
                result[i * k + j] = 1;
            }
        }
    }
    return result;
}

// void matrix_print(float* M, size_t m, size_t n) {
//     for(size_t i = 0; i < m; i++) {
//         for(size_t j = 0; j < n; j++) {
//             std::cout << M[i * n + j] << "\t";
//         }
//         std::cout << std::endl;
//     }
// }


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for(size_t i = 0; i < m; i+=batch) {
        float *X_batch = new float[batch * n];
        memcpy(X_batch, X + i * n, sizeof(float) * batch * n);
        unsigned char *y_batch = new unsigned char[batch];
        memcpy(y_batch, y + i, sizeof(unsigned char) * batch);
        // gradiant descent
        float *I = one_hot_encoding(y_batch, batch, k);  // batch x k
        float *X_T = matrix_transpose(X_batch, batch, n);  // n x batch
        float *Z = softmax(matrix_mul(X_batch, theta, batch, n, k), batch, k);  // batch x k
        float *G = matrix_mul(X_T, matrix_sub(Z, I, batch, k), n, batch, k);  // n x k
        
        for(size_t ii = 0; ii < n; ii++) {
            for(size_t jj = 0; jj < k; jj++) {
                theta[ii * k + jj] -= lr / batch * G[ii * k + jj];
            }
        }
        delete I;
        delete X_T;
        delete Z;
        delete G;
        delete X_batch;
        delete y_batch;
    }
    
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
