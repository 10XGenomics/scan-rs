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

#include "sptree.h"

#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

using std::array;
using std::make_unique;
using std::vector;

template <int NDims>
Cell<NDims>::Cell(const double* inp_corner, const double* inp_width) {
  for (int d = 0; d < NDims; d++) {
    setCorner(d, inp_corner[d]);
  }
  for (int d = 0; d < NDims; d++) {
    setWidth(d, inp_width[d]);
  }
}

template <int NDims>
double Cell<NDims>::getCorner(unsigned int d) const {
  return corner[d];
}

template <int NDims>
double Cell<NDims>::getWidth(unsigned int d) const {
  return width[d];
}

template <int NDims>
void Cell<NDims>::setCorner(unsigned int d, double val) {
  corner[d] = val;
}

template <int NDims>
void Cell<NDims>::setWidth(unsigned int d, double val) {
  width[d] = val;
}

// Checks whether a point lies in a cell
template <int NDims>
bool Cell<NDims>::containsPoint(const double point[]) const {
  assert(point);
  for (int d = 0; d < NDims; d++) {
    if (corner[d] - width[d] > point[d]) {
      return false;
    }
    if (corner[d] + width[d] < point[d]) {
      return false;
    }
  }
  return true;
}

// Default constructor for SPTree -- build tree, too!
template <int NDims>
SPTree<NDims>::SPTree(const double* inp_data, unsigned int N) : data(inp_data) {
  // Compute mean, width, and height of current map (boundaries of SPTree)
  int nD = 0;
  array<double, NDims> mean_Y;
  mean_Y.fill(0.0);
  array<double, NDims> min_Y;
  array<double, NDims> max_Y;
  min_Y.fill(DBL_MAX);
  max_Y.fill(-DBL_MAX);

  for (unsigned int n = 0; n < N; n++) {
    for (unsigned int d = 0; d < NDims; d++) {
      mean_Y[d] += inp_data[n * NDims + d];
      if (inp_data[nD + d] < min_Y[d]) {
        min_Y[d] = inp_data[nD + d];
      }
      if (inp_data[nD + d] > max_Y[d]) {
        max_Y[d] = inp_data[nD + d];
      }
    }
    nD += NDims;
  }

  for (int d = 0; d < NDims; d++) {
    mean_Y[d] /= static_cast<double>(N);
  }

  // Construct SPTree
  array<double, NDims> width;
  for (int d = 0; d < NDims; d++) {
    width[d] = fmax(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
  }
  node.init(mean_Y.data(), width.data());
  fill(N);
}

// Main initialization function
template <int NDims>
void SPTree<NDims>::Node::init(const double* inp_corner,
                               const double* inp_width) {
  cum_size = 0;

  for (unsigned int d = 0; d < NDims; d++) {
    boundary.setCorner(d, inp_corner[d]);
  }
  for (unsigned int d = 0; d < NDims; d++) {
    boundary.setWidth(d, inp_width[d]);
  }

  for (unsigned int d = 0; d < NDims; d++) {
    center_of_mass[d] = .0;
  }
}

// Insert a point into the SPTree
template <int NDims>
bool SPTree<NDims>::Node::insert(const double* data, unsigned int new_index) {
  assert(data);
  // Ignore objects which do not belong in this quad tree
  const double* point = data + new_index * NDims;
  if (!boundary.containsPoint(point)) {
    return false;
  }

  // Online update of cumulative size and center-of-mass
  ++cum_size;
  double mult1 =
      static_cast<double>(cum_size - 1) / static_cast<double>(cum_size);
  double mult2 = 1.0 / static_cast<double>(cum_size);

  for (unsigned int d = 0; d < NDims; d++) {
    center_of_mass[d] = center_of_mass[d] * mult1 + mult2 * point[d];
  }

  // If there is space in this quad tree and it is a leaf, add the object here
  if (cum_size == 1) {
    index = new_index;
    return true;
  }

  // Don't add duplicates for now (this is not very nice)
  if (!children) {
    bool duplicate = true;
    for (unsigned int d = 0; d < NDims; d++) {
      if (point[d] != data[index * NDims + d]) {
        duplicate = false;
        break;
      }
    }
    if (duplicate) {
      return true;
    }
  }

  // Otherwise, we need to subdivide the current cell
  if (!children) {
    subdivide(data);
  }

  // Find out where the point can be inserted
  for (auto& child : *children) {
    if (child.insert(data, new_index)) {
      return true;
    }
  }
  // Otherwise, the point cannot be inserted (this should never happen)
  return false;
}

// Create four children which fully divide this cell into four quads of equal
// area
template <int NDims>
void SPTree<NDims>::Node::subdivide(const double* data) {
  assert(!children);
  // Create new children
  double new_corner[NDims];
  double new_width[NDims];
  for (unsigned int d = 0; d < NDims; d++) {
    new_width[d] = .5 * boundary.getWidth(d);
  }
  children = make_unique<array<SPTree<NDims>::Node, no_children>>();
  for (unsigned int i = 0; i < no_children; i++) {
    unsigned int div = 1;
    for (unsigned int d = 0; d < NDims; d++) {
      if ((i / div) % 2 == 1) {
        new_corner[d] = boundary.getCorner(d) - .5 * boundary.getWidth(d);
      } else {
        new_corner[d] = boundary.getCorner(d) + .5 * boundary.getWidth(d);
      }
      div *= 2;
    }
    (*children)[i].init(new_corner, new_width);
  }

  // Move existing points to correct children
  for (auto& child : *children) {
    if (child.insert(data, index)) {
      break;
    }
  }
  index = -1;
}

// Build SPTree on dataset
template <int NDims>
void SPTree<NDims>::fill(unsigned int N) {
  for (unsigned int i = 0; i < N; i++) {
    node.insert(data, i);
  }
}

// Checks whether the specified tree is correct
template <int NDims>
bool SPTree<NDims>::isCorrect() const {
  return node.isCorrect(data);
}

template <int NDims>
bool SPTree<NDims>::Node::isCorrect(const double* data) const {
  if (!children && cum_size > 0) {
    const double* point = data + index * NDims;
    if (!boundary.containsPoint(point)) {
      return false;
    }
  }
  if (children) {
    for (const auto& child : *children) {
      if (!child.isCorrect(data)) {
        return false;
      }
    }
  }
  return true;
}

// Build a list of all indices in SPTree
template <int NDims>
void SPTree<NDims>::getAllIndices(unsigned int* indices) const {
  node.getAllIndices(indices, 0);
}

// Build a list of all indices in SPTree
template <int NDims>
unsigned int SPTree<NDims>::Node::getAllIndices(unsigned int* indices,
                                                unsigned int loc) const {
  // Gather indices in current quadrant
  if (!children && cum_size > 0) {
    indices[++loc] = index;
  }

  // Gather indices in children
  if (children) {
    for (auto& child : *children) {
      loc = child.getAllIndices(indices, loc);
    }
  }
  return loc;
}

// Compute non-edge forces using Barnes-Hut algorithm
template <int NDims>
void SPTree<NDims>::computeNonEdgeForces(unsigned int point_index,
                                         double theta,
                                         double neg_f[],
                                         double* sum_Q) const {
  node.computeNonEdgeForces(data, point_index, theta, neg_f, sum_Q);
}

template <int NDims>
[[gnu::hot]] void SPTree<NDims>::Node::computeNonEdgeForces(
    const double* data,
    unsigned int point_index,
    double theta,
    double neg_f[],
    double* sum_Q) const {
  // Make sure that we spend no time on empty nodes or self-interactions
  if (cum_size == 0 || (!children && cum_size >= 1 && index == point_index)) {
    return;
  }

  // Compute distance between point and center-of-mass
  double sqdist = .0;
  unsigned int ind = point_index * NDims;

  array<double, NDims> buff;
  for (unsigned int d = 0; d < NDims; d++) {
    buff[d] = data[ind + d] - center_of_mass[d];
    sqdist += buff[d] * buff[d];
  }

  // Check whether we can use this node as a "summary"
  double max_width = 0.0;
  for (unsigned int d = 0; d < NDims; d++) {
    double cur_width = boundary.getWidth(d);
    max_width = (max_width > cur_width) ? max_width : cur_width;
  }
  if (!children || max_width * max_width < sqdist * theta * theta) {
    // Compute and add t-SNE force between point and current node
    sqdist = 1.0 / (1.0 + sqdist);
    double mult = cum_size * sqdist;
    *sum_Q += mult;
    mult *= sqdist;
    for (unsigned int d = 0; d < NDims; d++) {
      neg_f[d] += mult * buff[d];
    }
  } else {
    // Recursively apply Barnes-Hut to children
    for (const auto& child : *children) {
      child.computeNonEdgeForces(data, point_index, theta, neg_f, sum_Q);
    }
  }
}

// Computes edge forces
template <int NDims>
vector<double> SPTree<NDims>::computeEdgeForces(const unsigned int* row_P,
                                                const unsigned int* col_P,
                                                const double* val_P,
                                                int N) const {
  return node.computeEdgeForces(data, row_P, col_P, val_P, N);
}

template <int NDims>
vector<double> SPTree<NDims>::Node::computeEdgeForces(const double* data,
                                                      const unsigned int* row_P,
                                                      const unsigned int* col_P,
                                                      const double* val_P,
                                                      int N) const {
  vector<double> pos_f(N * NDims);
  // Loop over all edges in the graph
  unsigned int ind1 = 0;
  unsigned int ind2 = 0;
  double sqdist;
  for (unsigned int n = 0; n < N; n++) {
    for (unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
      // Compute pairwise distance and Q-value
      sqdist = 1.0;
      ind2 = col_P[i] * NDims;

      array<double, NDims> buff;
      for (unsigned int d = 0; d < NDims; d++) {
        buff[d] = data[ind1 + d] - data[ind2 + d];
        sqdist += buff[d] * buff[d];
      }

      sqdist = val_P[i] / sqdist;

      // Sum positive force
      for (unsigned int d = 0; d < NDims; d++) {
        pos_f[ind1 + d] += sqdist * buff[d];
      }
    }
    ind1 += NDims;
  }
  return pos_f;
}

// Print out tree
template <int NDims>
void SPTree<NDims>::print() const {
  node.print(data);
}

template <int NDims>
void SPTree<NDims>::Node::print(const double* data) const {
  if (cum_size == 0) {
    fprintf(stderr, "Empty node\n");
    return;
  }

  if (!children) {
    fprintf(stderr, "Leaf node; data = [");
    if (!children && cum_size > 0) {
      const double* point = data + index * NDims;
      for (int d = 0; d < NDims; d++) {
        fprintf(stderr, "%f, ", point[d]);
      }
      fprintf(stderr, " (index = %d)", index);
    }
    fprintf(stderr, "]\n");
  } else {
    fprintf(stderr, "Intersection node with center-of-mass = [");
    for (int d = 0; d < NDims; d++) {
      fprintf(stderr, "%f, ", center_of_mass[d]);
    }
    fprintf(stderr, "]; children are:\n");
    for (const auto& child : *children) {
      child.print(data);
    }
  }
}

// declare templates explicitly
template class SPTree<2>;
template class SPTree<3>;

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
