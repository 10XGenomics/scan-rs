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

/* This code was adopted with minor modifications from Steve Hanov's great
 * tutorial at http://stevehanov.ca/blog/index.php?id=130 */

#ifndef VPTREE_H
#define VPTREE_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <vector>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class euclidean_distance final {
  const int D;
  const double* const data;

 public:
  explicit euclidean_distance(int D, const double* data) : D(D), data(data) {
  }
  euclidean_distance(const euclidean_distance&) = default;
  euclidean_distance(euclidean_distance&&) = delete;
  euclidean_distance& operator=(const euclidean_distance&) = delete;
  euclidean_distance& operator=(euclidean_distance&&) = delete;
  ~euclidean_distance() = default;
  double operator()(const unsigned int t1, const unsigned int t2) const {
    double dd = .0;
    const double* x1 = data + t1 * D;
    const double* x2 = data + t2 * D;
    for (int d = 0; d < D; d++) {
      double diff = (x1[d] - x2[d]);
      dd += diff * diff;
    }
    return sqrt(dd);
  }
};

template <typename T, class Distance>
class VpTree {
 public:
  // Default constructor
  explicit VpTree(Distance&& distance) : distance(distance){};

  VpTree(const VpTree<T, Distance>&) = delete;
  VpTree(VpTree<T, Distance>&&) noexcept = default;
  VpTree& operator=(const VpTree<T, Distance>&) = delete;
  VpTree& operator=(VpTree<T, Distance>&&) noexcept = default;

  // Destructor
  ~VpTree() = default;

  // Function to create a new VpTree from data
  void create(const std::vector<T>& items) {
    _items = items;
    _root = buildFromPoints(0, items.size());
  }

  // Function that uses the tree to find the k nearest neighbors of target
  void search(const T& target,
              int k,
              std::vector<T>* results,
              std::vector<double>* distances) {
    // Use a priority queue to store intermediate results on
    std::priority_queue<HeapItem> heap;

    // Variable that tracks the distance to the farthest point in our results
    _tau = DBL_MAX;

    // Perform the search
    search(_root.get(), target, k, heap);

    // Gather final results
    results->clear();
    distances->clear();
    while (!heap.empty()) {
      results->push_back(_items[heap.top().index]);
      distances->push_back(heap.top().dist);
      heap.pop();
    }

    // Results are in reverse order
    std::reverse(results->begin(), results->end());
    std::reverse(distances->begin(), distances->end());
  }

 private:
  const Distance distance;
  std::vector<T> _items;
  double _tau;

  // Single node of a VP tree (has a point and radius; left children are closer
  // to point than the radius)
  struct Node {
    int index = 0;                // index of point in node
    double threshold = 0;         // radius(?)
    std::unique_ptr<Node> left;   // points closer by than threshold
    std::unique_ptr<Node> right;  // points farther away than threshold

    Node() = default;
    Node(Node&&) noexcept = default;
    Node(const Node&) = delete;
    Node& operator=(Node&&) noexcept = default;
    Node& operator=(const Node&) = delete;

    ~Node() = default;
  };
  std::unique_ptr<Node> _root;

  // An item on the intermediate result queue
  struct HeapItem {
    HeapItem(int index, double dist) : index(index), dist(dist) {
    }
    int index;
    double dist;
    bool operator<(const HeapItem& o) const {
      return dist < o.dist;
    }
  };

  // Function that (recursively) fills the tree
  std::unique_ptr<Node> buildFromPoints(int lower, int upper) {
    if (upper == lower) {  // indicates that we're done here!
      return nullptr;
    }

    // Lower index is center of current node
    std::unique_ptr<Node> node = std::make_unique<Node>();
    node->index = lower;

    if (upper - lower > 1) {  // if we did not arrive at leaf yet

      // Choose an arbitrary point and move it to the start
      int i = static_cast<int>(static_cast<double>(rand()) / RAND_MAX *
                               (upper - lower - 1)) +
              lower;
      std::swap(_items[lower], _items[i]);

      // Partition around the median distance
      int median = (upper + lower) / 2;
      auto& lower_item = _items[lower];
      std::nth_element(_items.begin() + lower + 1,
                       _items.begin() + median,
                       _items.begin() + upper,
                       [this, lower_item](auto& x, auto& y) {
                         return distance(lower_item, x) <
                                distance(lower_item, y);
                       });

      // Threshold of the new node will be the distance to the median
      node->threshold = distance(lower_item, _items[median]);

      // Recursively build tree
      node->index = lower;
      node->left = buildFromPoints(lower + 1, median);
      node->right = buildFromPoints(median, upper);
    }

    // Return result
    return node;
  }

  // Helper function that searches the tree
  void search(const Node* node,
              const T& target,
              int k,
              std::priority_queue<HeapItem>& heap) {
    if (node == nullptr) {
      return;  // indicates that we're done here
    }

    // Compute distance between target and current node
    double dist = distance(_items[node->index], target);

    // If current node within radius tau
    if (dist < _tau) {
      if (heap.size() == k) {
        // remove furthest node from result list (if we already
        // have k results)
        heap.pop();
      }
      // add current node to result list
      heap.push(HeapItem(node->index, dist));
      if (heap.size() == k) {
        // update value of tau (farthest point in result list)
        _tau = heap.top().dist;
      }
    }

    // Return if we arrived at a leaf
    if (!node->left && !node->right) {
      return;
    }

    // If the target lies within the radius of ball
    if (dist < node->threshold) {
      if (dist - _tau <= node->threshold) {
        // if there can still be neighbors inside the
        // ball, recursively search left child first
        search(node->left.get(), target, k, heap);
      }

      if (dist + _tau >= node->threshold) {
        // if there can still be neighbors outside the
        // ball, recursively search right child
        search(node->right.get(), target, k, heap);
      }

      // If the target lies outsize the radius of the ball
    } else {
      if (dist + _tau >= node->threshold) {
        // if there can still be neighbors outside the
        // ball, recursively search right child first
        search(node->right.get(), target, k, heap);
      }

      if (dist - _tau <= node->threshold) {
        // if there can still be neighbors inside the
        // ball, recursively search left child
        search(node->left.get(), target, k, heap);
      }
    }
  }
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif  // !defined(VPTREE_H)
