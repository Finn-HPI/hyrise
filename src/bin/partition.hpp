#pragma once
/**
 * @file   partition.h
 * @author Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
 * @date   Fri Jun 29 17:35:37 2012
 * 
 * @brief  Single-threaded Radix Partitioning Implementations
 * 
 * (c) 2014, ETH Zurich, Systems Group
 * 
 */
#include "operators/join_simd_sort_merge/util.hpp"

using Relation = hyrise::Relation;
/**
 * Partition a given input relation into fanout=2^radixbits partitions using
 * radix clustering algorithm with least significant `nbits`.
 *
 * @note uses software-managed buffers for partitioning
 *
 * @param partitions output partitions, an array of relation ptrs of size fanout
 * @param input input relation
 * @param output used for writing out partitioning results
 * @param nbits number of least significant bits to use for partitioning
 *
 * @return
 */
void partition_relation_optimized(std::span<Relation> partitions, Relation* input, Relation* output, uint32_t nbits,
                                  uint32_t shiftbits);
