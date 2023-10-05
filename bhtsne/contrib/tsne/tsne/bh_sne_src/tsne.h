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

#ifndef TSNE_H
#define TSNE_H

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_TSNE_DLL
#ifdef __GNUC__
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define DLL_PUBLIC __attribute__((dllexport))
#else
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define DLL_PUBLIC __declspec(dllexport)
#endif  // __GNUC__
#else
#ifdef __GNUC__
#define DLL_PUBLIC __attribute__ ((dllimport))
#else
#define DLL_PUBLIC __declspec(dllimport)
#endif  // __GNUC__
#endif  // BUILDING_TSNE_DLL
#else
#if __GNUC__ >= 4
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define DLL_PUBLIC __attribute__((visibility("default")))
#else
#define DLL_PUBLIC
#endif  // __GNUC__
#endif  // defined _WIN32 || defined __CYGWIN__

extern "C" {
// stateless t-SNE
DLL_PUBLIC void run(double* X,
                    int N,
                    int D,
                    double* Y,
                    int no_dims,
                    double perplexity,
                    double theta,
                    int rand_seed,
                    bool skip_random_init,
                    const double* init,
                    bool use_init,
                    int max_iter = 1000,
                    int stop_lying_iter = 250,
                    int mom_switch_iter = 250);
// stateful t-SNE
struct TSNE;
DLL_PUBLIC struct TSNE* init_tsne(double* X,
                                  int N,
                                  int D,
                                  double* Y,
                                  int no_dims,
                                  double perplexity,
                                  double theta,
                                  int rand_seed,
                                  bool skip_random_init,
                                  const double* init,
                                  bool use_init,
                                  int max_iter = 1000,
                                  int stop_lying_iter = 250,
                                  int mom_switch_iter = 250);
DLL_PUBLIC bool step_tsne_by(struct TSNE* tsne, int step);
DLL_PUBLIC void free_tsne(struct TSNE* tsne);
}

#endif  // !defined(TSNE_H)
