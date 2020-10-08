/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015-2020 by Contributors
 * \file np_broadcast_reduce-inl.cuh
 * \brief GPU implementations for numpy binary broadcast ops
 * \author Zhaoqi Zhu
*/
#ifndef MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_INL_CUH_
#define MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_INL_CUH_

#include "../tensor/broadcast_reduce-inl.cuh"

using namespace mshadow::cuda;
using namespace mshadow;

template<typename Reducer, int NDim, typename DType, typename OType>
void NumpyArgMinMaxReduce(Stream<gpu> *s, const TBlob& in_data, const TBlob& out_data,
                          const Tensor<gpu, 1, char>& workspace) {
  std::cout << "dududu" << std::endl;
  
//   if (req == kNullOp) return;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  ReduceImplConfig config(out_date.shape_, in_data.shape_, nullptr, nullptr, sizeof(OType));
//   if (safe_acc) {
//     MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
//       typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
//       MSHADOW_TYPE_SWITCH(small.type_flag_, OType, {
//         typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
//         config = ReduceImplConfig(small.shape_, big.shape_, nullptr, nullptr,
//                                   sizeof(AccType));
//         ReduceImpl<Reducer, ndim, AccType, DataType, OutType, OP>(
//           stream, small, req, big, workspace, config);
//       });
//     });
//   } else {
//     ReduceImpl<Reducer, ndim, DType, DType, DType, OP>(stream, small, req, big, workspace, config);
//   }

/*
  if (config.M == 1) {
    reduce_kernel_M1<Reducer, ndim, AType, DType, OType, OP>
    <<< config.kernel_1.gridDim, config.kernel_1.blockDim, 0, stream >>>(
      config.N, false, big.dptr<DType>(), small.dptr<OType>(), big.shape_.get<ndim>(),
      small.shape_.get<ndim>());
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel_M1);
  } else {
    OType* small_dptr = reinterpret_cast<OType*>(in_data.dptr_)small.dptr<OType>();
    bool addto = (req == kAddTo);
    if (config.Mnext > 1) {
      // small_dptr[] is N*Mnext*sizeof(DType) bytes
      small_dptr = reinterpret_cast<OType*>(workspace.dptr_);
      addto = false;
      // Check that the workspace is contigiuous
      CHECK_EQ(workspace.CheckContiguous(), true);
      // Check that we have enough storage
      CHECK_GE(workspace.size(0), config.workspace_size);
    }

    const int by = (config.kernel_1.do_transpose) ?
      config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
    const bool do_unroll = ( config.M / (by*config.Mnext) >= unroll_reduce );
    KERNEL_UNROLL_SWITCH(do_unroll, unroll_reduce, UNROLL, {
      reduce_kernel<Reducer, ndim, AType, DType, OType, OP, UNROLL>
      <<< config.kernel_1.gridDim, config.kernel_1.blockDim, config.kernel_1.shMemSize, stream>>>(
        config.N, config.M, addto, big.dptr<DType>(), small_dptr, big.shape_.get<ndim>(),
        small.shape_.get<ndim>(), config.rshape.get<ndim>(), config.rstride.get<ndim>(),
        config.Mnext, config.kernel_1.do_transpose);
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_kernel);

    if (config.Mnext > 1) {
      reduce_lines_kernel<Reducer, OType>
      <<< config.kernel_2.gridSize, config.kernel_2.blockSize, 0, stream >>>
        (config.N, config.Mnext, req == kAddTo, config.N, small_dptr, small.dptr<OType>());
      MSHADOW_CUDA_POST_KERNEL_CHECK(reduce_lines_kernel);
    }
  }








*/


}




#endif // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_INL_CUH_