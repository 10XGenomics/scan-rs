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

#ifndef SPTREE_H
#define SPTREE_H

#include <array>
#include <memory>
#include <vector>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

template <int NDims = 2>
class alignas(16) Cell {
  std::array<double, NDims> corner;
  std::array<double, NDims> width;

 public:
  Cell() = default;
  Cell(const double* inp_corner, const double* inp_width);
  Cell(const Cell&) = delete;
  Cell(Cell&&) noexcept = default;
  Cell& operator=(const Cell&) = delete;
  Cell& operator=(Cell&&) noexcept = default;
  ~Cell() = default;

  [[nodiscard]] double getCorner(unsigned int d) const;
  [[nodiscard]] double getWidth(unsigned int d) const;
  void setCorner(unsigned int d, double val);
  void setWidth(unsigned int d, double val);
  bool containsPoint(const double point[]) const;
};

template <int NDims = 2>
class SPTree {
 public:
  enum { no_children = 2 * SPTree<NDims - 1>::no_children };

 private:
  class alignas(16) Node {
   public:
    bool isCorrect(const double* data) const;
    bool insert(const double* data, unsigned int new_index);
    void subdivide(const double* data);
    void print(const double* data) const;
    void computeNonEdgeForces(const double* data,
                              unsigned int point_index,
                              double theta,
                              double neg_f[],
                              double* sum_Q) const;
    std::vector<double> computeEdgeForces(const double* data,
                                          const unsigned int* row_P,
                                          const unsigned int* col_P,
                                          const double* val_P,
                                          int N) const;
    unsigned int getAllIndices(unsigned int* indices, unsigned int loc) const;
    void init(const double* inp_corner, const double* inp_width);

   private:
    std::array<double, NDims> center_of_mass;

    // Axis-aligned bounding box stored as a center with half-dimensions to
    // represent the boundaries of this quad tree
    Cell<NDims> boundary;

    // Children
    std::unique_ptr<std::array<SPTree<NDims>::Node, no_children>> children;

    // Properties of this node in the tree
    unsigned int index;
    unsigned int cum_size;
  };

  // Indices in this space-partitioning tree node, corresponding center-of-mass,
  // and list of all children
  const double* data;

  Node node;

 public:
  SPTree() = default;
  SPTree(SPTree<NDims>&&) noexcept = default;
  SPTree(const SPTree<NDims>&) = delete;
  SPTree& operator=(SPTree<NDims>&&) noexcept = default;
  SPTree& operator=(const SPTree<NDims>&) = delete;
  SPTree(const double* inp_data, unsigned int N);
  ~SPTree() = default;
  [[nodiscard]] bool isCorrect() const;
  void getAllIndices(unsigned int* indices) const;
  void computeNonEdgeForces(unsigned int point_index,
                            double theta,
                            double neg_f[],
                            double* sum_Q) const;
  std::vector<double> computeEdgeForces(const unsigned int* row_P,
                                        const unsigned int* col_P,
                                        const double* val_P,
                                        int N) const;
  void print() const;

 private:
  void fill(unsigned int N);
};

template <>
struct SPTree<0> {
  enum { no_children = 1 };
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif  // !defined(SPTREE_H)
