/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of
 *    Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "tsne.h"

#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <utility>
#include <vector>

#include "sptree.h"
#include "vptree.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

using std::abs;
using std::array;
using std::exp;
using std::fprintf;
using std::log;
using std::move;
using std::sqrt;
using std::unique_ptr;
using std::vector;

namespace {

struct TSNEState {
 public:
  TSNEState(int N,
            double* const Y,
            int no_dims,
            double theta,
            int max_iter,
            int stop_lying_iter,
            int mom_switch_iter,
            int iter,
            float total_time,
            float residual_time,
            vector<double>&& P,
            vector<unsigned int>&& row_P,
            vector<unsigned int>&& col_P,
            vector<double>&& val_P,
            vector<double>&& dY,
            vector<double>&& uY,
            vector<double>&& gains)
      : N{N},
        Y{Y},
        no_dims{no_dims},
        theta{theta},
        max_iter{max_iter},
        stop_lying_iter{stop_lying_iter},
        mom_switch_iter{mom_switch_iter},
        iter{iter},
        total_time{total_time},
        residual_time{residual_time},
        P{P},
        row_P{row_P},
        col_P{col_P},
        val_P{val_P},
        dY{dY},
        uY{uY},
        gains{gains} {
  }

  bool step_by(int step);

 private:
  template <int NDIMS>
  bool step_by_impl(int step);

  const int N;
  double* const Y;
  const int no_dims;
  const double theta;
  const int max_iter;
  const int stop_lying_iter;
  const int mom_switch_iter;
  int iter;
  float total_time;
  float residual_time;
  vector<double> P;
  vector<unsigned int> row_P;
  vector<unsigned int> col_P;
  vector<double> val_P;
  vector<double> dY;
  vector<double> uY;
  vector<double> gains;
};

template <int NDIMS>
void run(double* X,
         int N,
         int D,
         double* Y,
         double perplexity,
         double theta,
         int rand_seed,
         bool skip_random_init,
         double* init,
         bool use_init,
         int max_iter,
         int stop_lying_iter,
         int mom_switch_iter);

template <int NDIMS>
void computeGradient(const vector<unsigned int>& inp_row_P,
                     const vector<unsigned int>& inp_col_P,
                     const vector<double>& inp_val_P,
                     const double* Y,
                     int N,
                     vector<double>* _dC,
                     double theta);
template <int D>
void computeExactGradient(const vector<double>& P,
                          const double* Y,
                          int N,
                          vector<double>* _dC);
template <int D>
double evaluateError(const vector<double>& P, const double* Y, int N);
template <int NDIMS>
double evaluateError(const vector<unsigned int>& row_P,
                     const vector<unsigned int>& col_P,
                     const vector<double>& val_P,
                     const double* Y,
                     int N,
                     double theta);
void zeroMean(double* X, int N, int D);
template <int D>
void zeroMean(double* X, int N);
void computeGaussianPerplexity(
    const double* X, int N, int D, double* P, double perplexity);
void computeGaussianPerplexity(const double* X,
                               int N,
                               int D,
                               vector<unsigned int>* _row_P,
                               vector<unsigned int>* _col_P,
                               vector<double>* _val_P,
                               double perplexity,
                               int K);
vector<double> computeSquaredEuclideanDistance(const double* X, int N, int D);
double randn();
void symmetrizeMatrix(vector<unsigned int>* row_P,
                      vector<unsigned int>* col_P,
                      vector<double>* val_P,
                      int N);

inline double sign(const double x) {
  return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
template <int NDIMS>
void computeGradient(const vector<unsigned int>& inp_row_P,
                     const vector<unsigned int>& inp_col_P,
                     const vector<double>& inp_val_P,
                     const double* const Y,
                     const int N,
                     vector<double>* const _dC,
                     const double theta) {
  // Construct space-partitioning tree on current map
  SPTree<NDIMS> tree(Y, N);

  // Compute all terms required for t-SNE gradient
  double sum_Q = .0;
  vector<double> neg_f(N * NDIMS);
  auto pos_f = tree.computeEdgeForces(
      inp_row_P.data(), inp_col_P.data(), inp_val_P.data(), N);
  for (int n = 0; n < N; n++) {
    tree.computeNonEdgeForces(n, theta, neg_f.data() + n * NDIMS, &sum_Q);
  }

  // Compute final t-SNE gradient
  vector<double>& dC = *_dC;
  for (int i = 0; i < N * NDIMS; i++) {
    dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
  }
}

// Compute gradient of the t-SNE cost function (exact)
template <int D>
void computeExactGradient(const vector<double>& P,
                          double* const Y,
                          const int N,
                          vector<double>* const _dC) {
  // Make sure the current gradient contains zeros
  vector<double>& dC = *_dC;
  dC.assign(N * D, 0.0);

  // Compute the squared Euclidean distance matrix
  vector<double> DD = computeSquaredEuclideanDistance(Y, N, D);

  // Compute Q-matrix and normalization sum
  vector<double> Q(N * N);
  double sum_Q = .0;
  int nN = 0;
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < N; m++) {
      if (n != m) {
        Q[nN + m] = 1 / (1 + DD[nN + m]);
        sum_Q += Q[nN + m];
      }
    }
    nN += N;
  }

  // Perform the computation of the gradient
  nN = 0;
  int nD = 0;
  for (int n = 0; n < N; n++) {
    int mD = 0;
    for (int m = 0; m < N; m++) {
      if (n != m) {
        double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
        for (int d = 0; d < D; d++) {
          dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
        }
      }
      mD += D;
    }
    nN += N;
    nD += D;
  }
}

// Evaluate t-SNE cost function (exactly)
template <int D>
double evaluateError(const vector<double>& P,
                     const double* const Y,
                     const int N) {
  // Compute the squared Euclidean distance matrix
  vector<double> DD = computeSquaredEuclideanDistance(Y, N, D);
  vector<double> Q(N * N);

  // Compute Q-matrix and normalization sum
  int nN = 0;
  double sum_Q = DBL_MIN;
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < N; m++) {
      if (n != m) {
        Q[nN + m] = 1 / (1 + DD[nN + m]);
        sum_Q += Q[nN + m];
      } else {
        Q[nN + m] = DBL_MIN;
      }
    }
    nN += N;
  }
  for (int i = 0; i < N * N; i++) {
    Q[i] /= sum_Q;
  }

  // Sum t-SNE error
  double C = .0;
  for (int n = 0; n < N * N; n++) {
    C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
  }
  return C;
}

// Evaluate t-SNE cost function (approximately)
template <int NDIMS>
double evaluateError(const vector<unsigned int>& row_P,
                     const vector<unsigned int>& col_P,
                     const vector<double>& val_P,
                     const double* const Y,
                     const int N,
                     const double theta) {
  // Get estimate of normalization term
  array<double, NDIMS> buff;
  buff.fill(0);
  double sum_Q = .0;
  {
    SPTree<NDIMS> tree(Y, N);
    for (int n = 0; n < N; n++) {
      tree.computeNonEdgeForces(n, theta, buff.data(), &sum_Q);
    }
  }

  // Loop over all edges to compute t-SNE error
  int ind1;
  int ind2;
  double C = .0;
  double Q;
  for (int n = 0; n < N; n++) {
    ind1 = n * NDIMS;
    for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
      Q = .0;
      ind2 = col_P[i] * NDIMS;
      for (int d = 0; d < NDIMS; d++) {
        buff[d] = Y[ind1 + d];
      }
      for (int d = 0; d < NDIMS; d++) {
        buff[d] -= Y[ind2 + d];
      }
      for (int d = 0; d < NDIMS; d++) {
        Q += buff[d] * buff[d];
      }
      Q = (1.0 / (1.0 + Q)) / sum_Q;
      C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
    }
  }

  // Clean up memory
  return C;
}

// Compute input similarities with a fixed perplexity
void computeGaussianPerplexity(const double* const X,
                               const int N,
                               const int D,
                               double* const P,
                               const double perplexity) {
  // Compute the squared Euclidean distance matrix
  vector<double> DD = computeSquaredEuclideanDistance(X, N, D);

  // Compute the Gaussian kernel row by row
  int nN = 0;
  for (int n = 0; n < N; n++) {
    // Initialize some variables
    bool found = false;
    double beta = 1.0;
    double min_beta = -DBL_MAX;
    double max_beta = DBL_MAX;
    double tol = 1e-5;
    double sum_P;

    // Iterate until we found a good perplexity
    int iter = 0;
    while (!found && iter < 200) {
      // Compute Gaussian kernel row
      for (int m = 0; m < N; m++) {
        P[nN + m] = exp(-beta * DD[nN + m]);
      }
      P[nN + n] = DBL_MIN;

      // Compute entropy of current row
      sum_P = DBL_MIN;
      for (int m = 0; m < N; m++) {
        sum_P += P[nN + m];
      }
      double H = 0.0;
      for (int m = 0; m < N; m++) {
        H += beta * (DD[nN + m] * P[nN + m]);
      }
      H = (H / sum_P) + log(sum_P);

      // Evaluate whether the entropy is within the tolerance level
      double Hdiff = H - log(perplexity);
      if (Hdiff < tol && -Hdiff < tol) {
        found = true;
      } else {
        if (Hdiff > 0) {
          min_beta = beta;
          if (max_beta == DBL_MAX || max_beta == -DBL_MAX) {
            beta *= 2.0;
          } else {
            beta = (beta + max_beta) / 2.0;
          }
        } else {
          max_beta = beta;
          if (min_beta == -DBL_MAX || min_beta == DBL_MAX) {
            beta /= 2.0;
          } else {
            beta = (beta + min_beta) / 2.0;
          }
        }
      }

      // Update iteration counter
      iter++;
    }

    // Row normalize P
    for (int m = 0; m < N; m++) {
      P[nN + m] /= sum_P;
    }
    nN += N;
  }
}

// Compute input similarities with a fixed perplexity using ball trees.
void computeGaussianPerplexity(const double* const X,
                               const int N,
                               const int D,
                               vector<unsigned int>* _row_P,
                               vector<unsigned int>* _col_P,
                               vector<double>* _val_P,
                               const double perplexity,
                               const int K) {
  if (perplexity > K) {
    fprintf(stderr, "Perplexity should be lower than K!\n");
  }

  // Allocate the memory we need
  _row_P->resize(N + 1);
  _col_P->resize(N * K);
  _val_P->resize(N * K);
  vector<unsigned int>& row_P = *_row_P;
  vector<unsigned int>& col_P = *_col_P;
  vector<double>& val_P = *_val_P;
  vector<double> cur_P(N - 1);
  row_P[0] = 0;
  for (int n = 0; n < N; n++) {
    row_P[n + 1] = row_P[n] + static_cast<unsigned int>(K);
  }

  // Build ball tree on data set
  VpTree<unsigned int, euclidean_distance> tree(euclidean_distance(D, X));
  vector<unsigned int> obj_X(N);
  for (int n = 0; n < N; n++) {
    obj_X[n] = n;
  }
  tree.create(obj_X);

  // Loop over all points to find nearest neighbors
  fprintf(stderr, "Building tree...\n");
  vector<unsigned int> indices;
  vector<double> distances;
  indices.reserve(N);
  distances.reserve(N);
  for (int n = 0; n < N; n++) {
    if (n % 10000 == 0) {
      fprintf(stderr, " - point %d of %d\n", n, N);
    }

    // Find nearest neighbors
    indices.clear();
    distances.clear();
    tree.search(obj_X[n], K + 1, &indices, &distances);

    // Initialize some variables for binary search
    bool found = false;
    double beta = 1.0;
    double min_beta = -DBL_MAX;
    double max_beta = DBL_MAX;
    double tol = 1e-5;

    // Iterate until we found a good perplexity
    int iter = 0;
    double sum_P;
    while (!found && iter < 200) {
      // Compute Gaussian kernel row
      for (int m = 0; m < K; m++) {
        cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);
      }

      // Compute entropy of current row
      sum_P = DBL_MIN;
      for (int m = 0; m < K; m++) {
        sum_P += cur_P[m];
      }
      double H = .0;
      for (int m = 0; m < K; m++) {
        H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
      }
      H = (H / sum_P) + log(sum_P);

      // Evaluate whether the entropy is within the tolerance level
      double Hdiff = H - log(perplexity);
      if (Hdiff < tol && -Hdiff < tol) {
        found = true;
      } else {
        if (Hdiff > 0) {
          min_beta = beta;
          if (max_beta == DBL_MAX || max_beta == -DBL_MAX) {
            beta *= 2.0;
          } else {
            beta = (beta + max_beta) / 2.0;
          }
        } else {
          max_beta = beta;
          if (min_beta == -DBL_MAX || min_beta == DBL_MAX) {
            beta /= 2.0;
          } else {
            beta = (beta + min_beta) / 2.0;
          }
        }
      }

      // Update iteration counter
      iter++;
    }

    // Row-normalize current row of P and store in matrix
    for (unsigned int m = 0; m < K; m++) {
      cur_P[m] /= sum_P;
    }
    for (unsigned int m = 0; m < K; m++) {
      col_P[row_P[n] + m] = indices[m + 1];
      val_P[row_P[n] + m] = cur_P[m];
    }
  }
}

// Symmetrizes a sparse matrix
void symmetrizeMatrix(vector<unsigned int>* _row_P,
                      vector<unsigned int>* _col_P,
                      vector<double>* _val_P,
                      const int N) {
  // Get sparse matrix
  vector<unsigned int>& row_P = *_row_P;
  vector<unsigned int>& col_P = *_col_P;
  vector<double>& val_P = *_val_P;

  // Count number of elements and row counts of symmetric matrix
  vector<int> row_counts(N);
  for (int n = 0; n < N; n++) {
    for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
      // Check whether element (col_P[i], n) is present
      bool present = false;
      for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
        if (col_P[m] == n) {
          present = true;
        }
      }
      if (present) {
        row_counts[n]++;
      } else {
        row_counts[n]++;
        row_counts[col_P[i]]++;
      }
    }
  }
  int no_elem = 0;
  for (int n = 0; n < N; n++) {
    no_elem += row_counts[n];
  }

  // Allocate memory for symmetrized matrix
  vector<unsigned int> sym_row_P(N + 1);
  vector<unsigned int> sym_col_P(no_elem);
  vector<double> sym_val_P(no_elem);

  // Construct new row indices for symmetric matrix
  sym_row_P[0] = 0;
  for (int n = 0; n < N; n++) {
    sym_row_P[n + 1] = sym_row_P[n] + static_cast<unsigned int>(row_counts[n]);
  }

  // Fill the result matrix
  vector<int> offset(N);
  for (int n = 0; n < N; n++) {
    for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
      // considering element(n, col_P[i])

      // Check whether element (col_P[i], n) is present
      bool present = false;
      for (unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
        if (col_P[m] == n) {
          present = true;
          if (n <= col_P[i]) {  // make sure we do not add elements twice
            sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
            sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
            sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
            sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] =
                val_P[i] + val_P[m];
          }
        }
      }

      // If (col_P[i], n) is not present, there is no addition involved
      if (!present) {
        sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
        sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
      }

      // Update offsets
      if (!present || (present && n <= col_P[i])) {
        offset[n]++;
        if (col_P[i] != n) {
          offset[col_P[i]]++;
        }
      }
    }
  }

  // Divide the result by two
  for (int i = 0; i < no_elem; i++) {
    sym_val_P[i] /= 2.0;
  }

  // Return symmetrized matrices
  *_row_P = std::move(sym_row_P);
  *_col_P = std::move(sym_col_P);
  *_val_P = std::move(sym_val_P);
}

// Compute squared Euclidean distance matrix
vector<double> computeSquaredEuclideanDistance(const double* const X,
                                               const int N,
                                               const int D) {
  vector<double> DD(N * N);
  const double* XnD = X;
  for (int n = 0; n < N; ++n, XnD += D) {
    const double* XmD = XnD + D;
    double* curr_elem = &DD[n * N + n];
    *curr_elem = 0.0;
    double* curr_elem_sym = curr_elem + N;
    for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
      double total = 0.0;
      for (int d = 0; d < D; ++d) {
        double diff = XnD[d] - XmD[d];
        total += diff * diff;
      }
      *(++curr_elem) = total;
      *curr_elem_sym = *curr_elem;
    }
  }
  return DD;
}

// Makes data zero-mean
template <int D>
void zeroMean(double* const X, const int N) {
  // Compute data mean
  array<double, D> mean;
  mean.fill(0);
  int nD = 0;
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      mean[d] += X[nD + d];
    }
    nD += D;
  }
  for (int d = 0; d < D; d++) {
    mean[d] /= static_cast<double>(N);
  }

  // Subtract data mean
  nD = 0;
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      X[nD + d] -= mean[d];
    }
    nD += D;
  }
}

// Makes data zero-mean
void zeroMean(double* const X, const int N, const int D) {
  // Compute data mean
  vector<double> mean(D);
  int nD = 0;
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      mean[d] += X[nD + d];
    }
    nD += D;
  }
  for (int d = 0; d < D; d++) {
    mean[d] /= static_cast<double>(N);
  }

  // Subtract data mean
  nD = 0;
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      X[nD + d] -= mean[d];
    }
    nD += D;
  }
}

// Generates a Gaussian random number
double randn() {
  double x;
  double y;
  double radius;
  do {
    x = 2 * (rand() / (static_cast<double>(RAND_MAX) + 1)) - 1;
    y = 2 * (rand() / (static_cast<double>(RAND_MAX) + 1)) - 1;
    radius = (x * x) + (y * y);
  } while ((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  return x;
}

// Initialize t-SNE
template <int NDIMS>
unique_ptr<TSNEState> make_tsne_impl(double* const X,
                                     const int N,
                                     const int D,
                                     double* const Y,
                                     const double perplexity,
                                     const double theta,
                                     const int rand_seed,
                                     const bool skip_random_init,
                                     const double* const init,
                                     const bool use_init,
                                     const int max_iter,
                                     const int stop_lying_iter,
                                     const int mom_switch_iter) {
  // Set random seed
  if (!skip_random_init) {
    if (rand_seed >= 0) {
      fprintf(stderr, "Using random seed: %d\n", rand_seed);
      srand(static_cast<unsigned int>(rand_seed));
    } else {
      fprintf(stderr, "Using current time as random seed...\n");
      srand(time(nullptr));
    }
  }

  // Determine whether we are using an exact algorithm
  if (N - 1 < 3 * perplexity) {
    fprintf(stderr, "Perplexity too large for the number of data points!\n");
    exit(1);
  }
  fprintf(stderr,
          "Using no_dims = %d, perplexity = %f, and theta = %f\n",
          NDIMS,
          perplexity,
          theta);
  bool exact = theta == .0;

  fprintf(stderr,
          "Using max_iter = %d, stop_lying_iter = %d, mom_switch_iter = %d\n",
          max_iter,
          stop_lying_iter,
          mom_switch_iter);

  clock_t start;
  clock_t end;

  // Allocate some memory
  vector<double> dY(N * NDIMS);
  vector<double> uY(N * NDIMS);
  vector<double> gains(N * NDIMS, 1.0);

  // Normalize input data (to prevent numerical problems)
  fprintf(stderr, "Computing input similarities...\n");
  start = clock();
  zeroMean(X, N, D);
  double max_X = .0;
  for (int i = 0; i < N * D; i++) {
    auto ax = abs(X[i]);
    if (ax > max_X) {
      max_X = ax;
    }
  }
  for (int i = 0; i < N * D; i++) {
    X[i] /= max_X;
  }

  // Compute input similarities for exact t-SNE
  vector<double> P;
  vector<unsigned int> row_P;
  vector<unsigned int> col_P;
  vector<double> val_P;
  if (exact) {
    // Compute similarities
    fprintf(stderr, "Exact?");
    P.resize(N * N);
    computeGaussianPerplexity(X, N, D, P.data(), perplexity);

    // Symmetrize input similarities
    fprintf(stderr, "Symmetrizing...\n");
    int nN = 0;
    for (int n = 0; n < N; n++) {
      int mN = nN + N;
      for (int m = n + 1; m < N; m++) {
        P[nN + m] += P[mN + n];
        P[mN + n] = P[nN + m];
        mN += N;
      }
      nN += N;
    }
    double sum_P = .0;
    for (int i = 0; i < N * N; i++) {
      sum_P += P[i];
    }
    for (int i = 0; i < N * N; i++) {
      P[i] /= sum_P;
    }
  } else {
    // Compute input similarities for approximate t-SNE

    // Compute asymmetric pairwise input similarities
    computeGaussianPerplexity(X,
                              N,
                              D,
                              &row_P,
                              &col_P,
                              &val_P,
                              perplexity,
                              static_cast<int>(3 * perplexity));

    // Symmetrize input similarities
    symmetrizeMatrix(&row_P, &col_P, &val_P, N);
    double sum_P = .0;
    for (int i = 0; i < row_P[N]; i++) {
      sum_P += val_P[i];
    }
    for (int i = 0; i < row_P[N]; i++) {
      val_P[i] /= sum_P;
    }
  }
  end = clock();

  // Lie about the P-values
  if (exact) {
    for (int i = 0; i < N * N; i++) {
      P[i] *= 12.0;
    }
  } else {
    for (int i = 0; i < row_P[N]; i++) {
      val_P[i] *= 12.0;
    }
  }

  // Initialize solution (randomly or with given coordinates)
  if (!use_init && !skip_random_init) {
    for (int i = 0; i < N * NDIMS; i++) {
      Y[i] = randn() * .0001;
    }
  } else if (use_init) {
    for (int i = 0; i < N * NDIMS; i++) {
      Y[i] = init[i];
    }
  }

  // Perform main training loop
  if (exact) {
    fprintf(stderr,
            "Input similarities computed in %4.2f seconds!\nLearning "
            "embedding...\n",
            static_cast<float>(end - start) / CLOCKS_PER_SEC);
  } else {
    fprintf(stderr,
            "Input similarities computed in %4.2f seconds (sparsity = "
            "%f)!\nLearning embedding...\n",
            static_cast<float>(end - start) / CLOCKS_PER_SEC,
            static_cast<double>(row_P[N]) /
                (static_cast<double>(N) * static_cast<double>(N)));
  }

  return std::make_unique<TSNEState>(N,
                                     Y,
                                     NDIMS,
                                     theta,
                                     max_iter,
                                     stop_lying_iter,
                                     mom_switch_iter,
                                     0,
                                     .0,
                                     .0,
                                     std::move(P),
                                     std::move(row_P),
                                     std::move(col_P),
                                     std::move(val_P),
                                     std::move(dY),
                                     std::move(uY),
                                     std::move(gains));
}

// Optimize t-SNE
template <int NDIMS>
bool TSNEState::step_by_impl(const int step) {
  // Set learning parameters
  double momentum = .5;
  double final_momentum = .8;
  double eta = 200.0;

  // Extract state
  bool exact = theta == .0;
  clock_t start = clock();
  clock_t end;
  float elapsed;
  int iter_until = std::min(iter + step, max_iter);

  for (; iter < iter_until; iter++) {
    // Compute (approximate) gradient
    if (exact) {
      computeExactGradient<NDIMS>(P, Y, N, &dY);
    } else {
      computeGradient<NDIMS>(row_P, col_P, val_P, Y, N, &dY, theta);
    }

    // Update gains
    for (int i = 0; i < N * NDIMS; i++) {
      gains[i] =
          (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    }
    for (int i = 0; i < N * NDIMS; i++) {
      if (gains[i] < .01) {
        gains[i] = .01;
      }
    }

    // Perform gradient update (with momentum and gains)
    for (int i = 0; i < N * NDIMS; i++) {
      uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    }
    for (int i = 0; i < N * NDIMS; i++) {
      Y[i] = Y[i] + uY[i];
    }

    // Make solution zero-mean
    zeroMean<NDIMS>(Y, N);

    // Stop lying about the P-values after a while, and switch momentum
    if (iter == stop_lying_iter) {
      if (exact) {
        for (int i = 0; i < N * N; i++) {
          P[i] /= 12.0;
        }
      } else {
        for (int i = 0; i < row_P[N]; i++) {
          val_P[i] /= 12.0;
        }
      }
    }
    if (iter == mom_switch_iter) {
      momentum = final_momentum;
    }

    // Print out progress
    if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
      end = clock();
      double C = .0;
      if (exact) {
        C = evaluateError<NDIMS>(P, Y, N);
      } else {
        // doing approximate computation here!
        C = evaluateError<NDIMS>(row_P, col_P, val_P, Y, N, theta);
      }
      if (iter == 0) {
        fprintf(stderr, "Iteration %d: error is %f\n", iter + 1, C);
      } else {
        elapsed = static_cast<float>(end - start) / CLOCKS_PER_SEC;
        total_time += elapsed;
        residual_time += elapsed;
        fprintf(stderr,
                "Iteration %d: error is %f (50 iterations in %4.2f seconds)\n",
                iter,
                C,
                residual_time);
        residual_time = 0;
      }
      start = clock();
    }
  }
  end = clock();
  elapsed = static_cast<float>(end - start) / CLOCKS_PER_SEC;
  total_time += elapsed;
  residual_time += elapsed;

  if (iter >= max_iter) {
    fprintf(stderr, "Fitting performed in %4.2f seconds.\n", total_time);
    return true;
  }
  return false;
}

unique_ptr<TSNEState> make_tsne(double* const X,
                                const int N,
                                const int D,
                                double* const Y,
                                const int no_dims,
                                const double perplexity,
                                const double theta,
                                const int rand_seed,
                                const bool skip_random_init,
                                const double* const init,
                                const bool use_init,
                                const int max_iter,
                                const int stop_lying_iter,
                                const int mom_switch_iter) {
  switch (no_dims) {
    case 2:
      return make_tsne_impl<2>(X,
                               N,
                               D,
                               Y,
                               perplexity,
                               theta,
                               rand_seed,
                               skip_random_init,
                               init,
                               use_init,
                               max_iter,
                               stop_lying_iter,
                               mom_switch_iter);
    case 3:
      return make_tsne_impl<3>(X,
                               N,
                               D,
                               Y,
                               perplexity,
                               theta,
                               rand_seed,
                               skip_random_init,
                               init,
                               use_init,
                               max_iter,
                               stop_lying_iter,
                               mom_switch_iter);
    default:
      throw "unsupported dimension";
  }
}

bool TSNEState::step_by(int step) {
  switch (no_dims) {
    case 2:
      return step_by_impl<2>(step);
    case 3:
      return step_by_impl<3>(step);
    default:
      throw "unsupported dimension";
  }
}

}  // namespace

extern "C" {
DLL_PUBLIC struct TSNE* init_tsne(double* const X,
                                  const int N,
                                  const int D,
                                  double* const Y,
                                  const int no_dims,
                                  const double perplexity,
                                  const double theta,
                                  const int rand_seed,
                                  const bool skip_random_init,
                                  const double* const init,
                                  const bool use_init,
                                  const int max_iter,
                                  const int stop_lying_iter,
                                  const int mom_switch_iter) {
  assert(no_dims == 2 || no_dims == 3);
  auto tsne = make_tsne(X,
                        N,
                        D,
                        Y,
                        no_dims,
                        perplexity,
                        theta,
                        rand_seed,
                        skip_random_init,
                        init,
                        use_init,
                        max_iter,
                        stop_lying_iter,
                        mom_switch_iter);
  return reinterpret_cast<TSNE*>(tsne.release());
}

DLL_PUBLIC bool step_tsne_by(struct TSNE* const tsne, const int step) {
  return reinterpret_cast<TSNEState*>(tsne)->step_by(step);
}

DLL_PUBLIC void free_tsne(struct TSNE* const tsne) {
  if (tsne == nullptr) {
    return;
  }
  delete reinterpret_cast<TSNEState*>(tsne);
}

DLL_PUBLIC void run(double* const X,
                    const int N,
                    const int D,
                    double* const Y,
                    const int no_dims,
                    const double perplexity,
                    const double theta,
                    const int rand_seed,
                    const bool skip_random_init,
                    const double* const init,
                    const bool use_init,
                    const int max_iter,
                    const int stop_lying_iter,
                    const int mom_switch_iter) {
  auto tsne = make_tsne(X,
                        N,
                        D,
                        Y,
                        no_dims,
                        perplexity,
                        theta,
                        rand_seed,
                        skip_random_init,
                        init,
                        use_init,
                        max_iter,
                        stop_lying_iter,
                        mom_switch_iter);
  tsne->step_by(max_iter);
}

}  // extern "C"

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
