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

using namespace mshadow::cuda;


template<typename Reducer, int NDim, typename DType, typename OType>
void NumpyArgMinMaxReduce(mshadow::Stream<gpu> *s, const TBlob& small, const TBlob& big) {
    std::cout << "dududu" << std::endl;
//   if (req == kNullOp) return;
//   cudaStream_t stream = Stream<gpu>::GetStream(s);
//   ReduceImplConfig config(small.shape_, big.shape_, nullptr, nullptr, sizeof(DType));
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
}




#endif  //MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_INL_CUH_