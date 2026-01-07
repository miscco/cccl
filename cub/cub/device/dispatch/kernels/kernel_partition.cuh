// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if __cccl_ptx_isa >= 860

#  include <cub/detail/warpspeed/allocators/smem_allocator.h>
#  include <cub/detail/warpspeed/look_ahead.h>
#  include <cub/detail/warpspeed/resource/smem_ref.cuh>
#  include <cub/detail/warpspeed/resource/smem_resource.cuh>
#  include <cub/detail/warpspeed/special_registers.cuh>
#  include <cub/detail/warpspeed/squad/load_store.h>
#  include <cub/detail/warpspeed/squad/squad.h>
#  include <cub/detail/warpspeed/values.h>
#  include <cub/thread/thread_reduce.cuh>
#  include <cub/warp/warp_reduce.cuh>

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__memory/align_down.h>
#  include <cuda/__memory/align_up.h>
#  include <cuda/__ptx/instructions/clusterlaunchcontrol.h>
#  include <cuda/std/__algorithm/clamp.h>
#  include <cuda/std/__cccl/cuda_capabilities.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cassert>

CUB_NAMESPACE_BEGIN

namespace detail::partition
{
_CCCL_DEVICE_API inline void squadGetNextBlockIdx(const Squad& squad, SmemRef<uint4>& refDestSmem)
{
  if (squad.isLeaderThread())
  {
    ::cuda::ptx::clusterlaunchcontrol_try_cancel(&refDestSmem.data(), refDestSmem.ptrCurBarrierRelease());
  }
  refDestSmem.squadIncreaseTxCount(squad, refDestSmem.sizeBytes());
}

template <typename Tp, typename ScanOpT>
_CCCL_DEVICE_API inline Tp warpReduce(const Tp input, ScanOpT& predicate)
{
  using warp_reduce_t = WarpReduce<Tp>;

  // TODO (elstehle): Do proper temporary storage allocation in case WarpReduce may rely on it
  static_assert(sizeof(typename warp_reduce_t::TempStorage) <= 4,
                "WarpReduce with non-trivial temporary storage is not supported yet in this kernel.");

  typename warp_reduce_t::TempStorage temp_storage;
  return warp_reduce_t{temp_storage}.Reduce(input, predicate);
}

template <typename InputT, typename OutputT>
struct PartitionKernelParams
{
  InputT* ptrIn;
  OutputT* ptrOut;
  tile_state_t<uint64_t>* ptrTileStates;
  size_t numElem;
  int numStages;
};

// Struct holding all partition kernel resources
template <typename WarpspeedPolicy, typename InputT>
struct PartitionResources
{
  // Handle unaligned loads. We have 16 extra bytes of padding in every stage for squadLoadBulk.
  using InT       = InputT[WarpspeedPolicy::tile_size + 16 / sizeof(InputT)];
  using CopyMaskT = uint32_t[WarpspeedPolicy::squadCount().warpCount()];

  SmemResource<InT> smemInOut;
  SmemResource<uint4> smemNextBlockIdx;
  SmemResource<uint64_t> smemCountExclusiveCta;
  SmemResource<CopyMaskT> smemCopyMask;
};
// Function to allocate resources.

template <typename WarpspeedPolicy, typename InputT, typename OutputT>
[[nodiscard]] _CCCL_API constexpr CopyResources<WarpspeedPolicy, InputT, OutputT>
allocResources(SyncHandler& syncHandler, SmemAllocator& smemAllocator, int numStages)
{
  using CopyResourcesT = CopyResources<WarpspeedPolicy, InputT, OutputT>;
  using InT            = typename CopyResourcesT::InT;
  using CopyMaskT      = typename CopyResourcesT::CopyMaskT;

  // If numBlockIdxStages is one less than the number of stages, we find a small
  // speedup compared to setting it equal to num_stages. Not sure why.
  int numBlockIdxStages = numStages - 1;
  // Ensure we have at least 1 stage
  numBlockIdxStages = numBlockIdxStages < 1 ? 1 : numBlockIdxStages;

  // We do not need too many countExclusiveCta stages. The lookback warp is the
  // bottleneck. As soon as it produces a new value, it will be consumed by the
  // scanStore squad, releasing the stage.
  int numSumExclusiveCtaStages = 2;

  CopyResourcesT res = {
    SmemResource<InT>(syncHandler, smemAllocator, Stages{numStages}),
    SmemResource<uint4>(syncHandler, smemAllocator, Stages{numBlockIdxStages}),
    SmemResource<uint64_t>(syncHandler, smemAllocator, Stages{numSumExclusiveCtaStages}),
    SmemResource<CopyMaskT>(syncHandler, smemAllocator, Stages{numStages}),
  };

  constexpr SquadDesc scanSquads[WarpspeedPolicy::num_squads] = {
    WarpspeedPolicy::squadCount(),
    WarpspeedPolicy::squadCopyStore(),
    WarpspeedPolicy::squadLoad(),
    WarpspeedPolicy::squadSched(),
    WarpspeedPolicy::squadLookback(),
  };

  res.smemInOut.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadLoad());
  res.smemInOut.addPhase(syncHandler, smemAllocator, {WarpspeedPolicy::squadCount(), WarpspeedPolicy::squadCopyStore()});

  res.smemNextBlockIdx.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadSched());
  res.smemNextBlockIdx.addPhase(syncHandler, smemAllocator, scanSquads);

  res.smemCountExclusiveCta.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadLookback());
  res.smemCountExclusiveCta.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadCopyStore());

  res.smemCopyMask.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadCount());
  res.smemCopyMask.addPhase(syncHandler, smemAllocator, WarpspeedPolicy::squadCopyStore());

  return res;
}

// The kernelBody device function is a straight-line implementation of the
// warp-specialized kernel.
//
// It is called from the __global__ kernel body with a squad argument that is
// the active squad on the current thread.
//
// Using this structure, all code that is not executed by the current
// squad is DCE (dead-code-eliminated) by the compiler and all
// warp-specialization dispatch is performed once at the start of the kernel and
// not in any of the hot loops (even if that may seem the case from a first
// glance at the code).
template <typename WarpspeedPolicy, typename InputT, typename OutputT typename PredicateT>
_CCCL_DEVICE_API inline void kernelBody(
  Squad squad, SpecialRegisters specialRegisters, const copyKernelParams<InputT, OutputT>& params, PredicateT predicate)
{
  ////////////////////////////////////////////////////////////////////////////////
  // Tuning dependent variables
  ////////////////////////////////////////////////////////////////////////////////
  static constexpr SquadDesc squadCount     = WarpspeedPolicy::squadCount();
  static constexpr SquadDesc squadCopyStore = WarpspeedPolicy::squadCopyStore();
  static constexpr SquadDesc squadLoad      = WarpspeedPolicy::squadLoad();
  static constexpr SquadDesc squadSched     = WarpspeedPolicy::squadSched();
  static constexpr SquadDesc squadLookback  = WarpspeedPolicy::squadLookback();

  constexpr int tile_size          = WarpspeedPolicy::tile_size;
  constexpr int num_lookback_tiles = WarpspeedPolicy::num_lookback_tiles;

  constexpr int elemPerThread = tile_size / squadCount.threadCount();

  ////////////////////////////////////////////////////////////////////////////////
  // Resources
  ////////////////////////////////////////////////////////////////////////////////
  SyncHandler syncHandler{};
  SmemAllocator smemAllocator{};

  CopyResources<WarpspeedPolicy, InputT, OutputT> res =
    allocResources<WarpspeedPolicy, InputT, OutputT>(syncHandler, smemAllocator, params.numStages);
  syncHandler.clusterInitSync(specialRegisters);

  ////////////////////////////////////////////////////////////////////////////////
  // Pre-loop
  ////////////////////////////////////////////////////////////////////////////////

  // Start with the tile indicated by blockIdx.x
  int idxTile = specialRegisters.blockIdxX;
  // Lookback-specific variables:
  int idxTilePrev                = 0;
  uint64_t countExclusiveCtaPrev = 0;
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();

  ////////////////////////////////////////////////////////////////////////////////
  // Loop over tiles
  ////////////////////////////////////////////////////////////////////////////////
#  pragma unroll 1
  while (true)
  {
    // Get stages. When these objects go out of scope, the stage of the resource
    // is automatically incremented.
    SmemStage stageNextBlockIdx      = res.smemNextBlockIdx.popStage();
    SmemStage stageInOut             = res.smemInOut.popStage();
    SmemStage stageCopyMask          = res.smemCopyMask.popStage();
    SmemStage stageCountExclusiveCta = res.smemCountExclusiveCta.popStage();

    // Split the stages into phases. Each resource goes through phases where it
    // is writeable by a set of threads and readable by a set of threads. To
    // acquire and release a phase, we need to arrive and wait on certain
    // barriers. The selection of the barriers is handled under the hood.
    auto [phaseNextBlockIdxW, phaseNextBlockIdxR]           = bindPhases<2>(stageNextBlockIdx);
    auto [phaseInOutW, phaseInOutRW]                        = bindPhases<2>(stageInOut);
    auto [phaseCopyMaskW, phaseCopyMaskR]                   = bindPhases<2>(stageCopyMask);
    auto [phaseCountExclusiveCtaW, phaseCountExclusiveCtaR] = bindPhases<2>(stageCountExclusiveCta);

    if (squad == squadSched)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load next tile index
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refNextBlockIdxW = phaseNextBlockIdxW.acquireRef();
      squadGetNextBlockIdx(squad, refNextBlockIdxW);
    }

    const size_t idxTileBase = idxTile * size_t(tile_size);
    _CCCL_ASSERT(idxTileBase < params.numElem, "");
    const int valid_items   = static_cast<int>(cuda::std::min(params.numElem - idxTileBase, size_t(tile_size)));
    const bool is_last_tile = valid_items < tile_size;
    constexpr ::cuda::std::plus<> plus_op{};
    constexpr ::cuda::std::bit_or<uint32_t> or_op{};

    CpAsyncOobInfo loadInfo = prepareCpAsyncOob(params.ptrIn + idxTileBase, valid_items);
    if (squad == squadLoad)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Load current tile
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refInOutW = phaseInOutW.acquireRef();
      squadLoadBulk(squad, refInOutW, loadInfo);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Get next tile index from shared memory (all squads)
    ////////////////////////////////////////////////////////////////////////////////
    uint4 regNextBlockIdx{};
    {
      SmemRef refNextBlockIdxR = phaseNextBlockIdxR.acquireRef();
      regNextBlockIdx          = refNextBlockIdxR.data();
      refNextBlockIdxR.setFenceLdsToAsyncProxy();
    }
    bool nextIdxTileValid = ::cuda::ptx::clusterlaunchcontrol_query_cancel_is_canceled(regNextBlockIdx);

    if (squad == squadCount)
    {
      const int valid_items_this_thread =
        cuda::std::clamp(valid_items - squad.threadRank() * elemPerThread, 0, elemPerThread);
      const int valid_threads_this_warp =
        cuda::std::clamp(::cuda::ceil_div(valid_items, elemPerThread) - squad.warpRank() * 32, 0, 32);
      const int valid_warps = ::cuda::ceil_div(valid_items, elemPerThread * 32);
      _CCCL_ASSERT(0 < valid_warps && valid_warps <= squad.warpCount(), "");

      ////////////////////////////////////////////////////////////////////////////////
      // Load tile from shared memory
      ////////////////////////////////////////////////////////////////////////////////
      uint32_t regWarpCount = 0;

      // Load data into registers
      InputT regInput[elemPerThread];
      uint32_t regPredicates[elemPerThread];
      uint32_t regThreadSelected[elemPerThread];
      {
        // Acquire phaseInOutRW in this short scope
        SmemRef refInOutRW = phaseInOutRW.acquireRef();
        squadLoadSmem(squad, regInput, &refInOutRW.data()[0] + loadInfo.smemStartOffsetElem);

        ////////////////////////////////////////////////////////////////////////////////
        // Reduce across thread and warp
        ////////////////////////////////////////////////////////////////////////////////
        if (is_last_tile)
        {
          // Calculate the predicates
          for (int i = 0; i < elemPerThread; ++i)
          {
            const int elem_idx = squad.threadRank() * elemPerThread + i;
            regPredicates[i] =
              static_cast<uint32_t>((elem_idx < valid_items) ? static_cast<bool>(predicate(regInput[i])) : false);
          }
        }
        else
        {
          // Calculate the predicates
          for (int i = 0; i < elemPerThread; ++i)
          {
            regPredicates[i] = static_cast<uint32_t>(static_cast<bool>(predicate(regInput[i])));
          }
        }
        regThreadSelected[i] = ThreadReduce(regPredicates, plus_op);
        regWarpCount         = warpReduce(regThreadSelected[i], plus_op);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store predicate count to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      {
        SmemRef refCopyMaskW = phaseCopyMaskW.acquireRef();
        if (squad.isLeaderThreadOfWarp())
        {
          refCopyMaskW.data()[squad.warpRank()] = regWarpCount;
        }
      }
      squad.syncThreads();

      ////////////////////////////////////////////////////////////////////////////////
      // Store private sum for lookback
      ////////////////////////////////////////////////////////////////////////////////
      uint32_t regSquadCount[squadCount.warpCount() + 1];
      regSquadCount[0] = 0;
      { // Use all threads to ensure regSquadCount is available everywhere
        SmemRef refCopyMaskW = phaseCopyMaskW.acquireRef();
        // Exclusive sum over the individual regWarpCount
        for (int i = 0; i < squadCount.warpCount(); ++i)
        {
          regSquadCount[i + 1] =
            regSquadCount[i] + static_cast<uint32_t>(::cuda::std::popcount(refCopyMaskW.data()[i]));
        }
      }
      if (squad.isLeaderThread())
      {
        storeTileAggregate(params.ptrTileStates + idxTile, TILE_AGGREGATE, regSquadCount[squadCount.warpCount()]);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Hide data movement in shared memory in latency of tile aggregates
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Acquire phaseInOutRW in this short scope
        SmemRef refInOutRW = phaseInOutRW.acquireRef();

        const int laneIdx       = specialRegisters.laneIdx;
        const uint32_t lanemask = ::cuda::ptx::get_sreg_lanemask_eq();

        // Get a mask of all items in previous lanes
        const uint32_t previous_mask = lanemask - 1;

        for (int i = 0; i < elemPerThread; ++i)
        {
          // Get the mask of all selected elements in the first thread block
          const bool pred          = regPredicates[i];
          const uint32_t warp_mask = warpReduce(pred * lanemask, or_op);

          if (pred)
          { // The number of selected elements prior to this consists of
            // * The number of elements selected in the previous warps
            const uint32_t previous_warp_selected = regSquadCount[squadCount.warpRank()];

            // * The number of elements selected in the previous thread block
            const uint32_t previous_thread_block_selected =
              ::cuda::std::accumulate(regThreadSelected[0], regThreadSelected[i], uint32_t{0});

            // * The number of selected previous elements from this thread block
            const uint32_t previous_thread_selected = ::cuda::std::popcount(warp_mask & previous_mask);

            // Write back into the new position in smem
            const uint32_t new_position =
              previous_warp_selected + previous_thread_block_selected + previous_thread_selected;
            refInOutRW.data()[new_position] = regInput[i];
          }
          else
          { // The number of unselected elements prior to this consists of
            // * The number of all selected elements in this squad
            const uint32_t all_selected_squad = regSquadCount[squadCount.warpCount()];

            // * The number of elements not selected in the previous warps
            const uint32_t previous_warp_selected =
              (32u * elemPerThread * squadCount.warpRank() - regSquadCount[squadCount.warpRank()]);

            // * The number of elements selected in the previous thread block
            const uint32_t previous_thread_block_selected =
              (32u * static_cast<uint32_t>(i)
               - ::cuda::std::accumulate(regThreadSelected[0], regThreadSelected[i], 0u));

            // * The number of selected previous elements from this thread block
            const uint32_t previous_thread_selected = ::cuda::std::popcount(warp_mask & previous_mask);

            // Write back into the new position in smem
            const uint32_t new_position =
              previous_warp_selected + previous_thread_block_selected + previous_thread_selected;
            refInOutRW.data()[new_position] = regInput[i];
          }
        }
      }
    }

    if (squad == squadLookback)
    {
      ////////////////////////////////////////////////////////////////////////////////
      // Perform lookback
      ////////////////////////////////////////////////////////////////////////////////
      SmemRef refCountExclusiveCtaW = phaseCountExclusiveCtaW.acquireRef();

      if (!is_first_tile)
      {
        constexpr int numTileStatesPerThread = num_lookback_tiles / 32;
        static_assert(num_lookback_tiles % 32 == 0, "num_lookback_tiles must be a multiple of 32");

        uint64_t regCountExclusiveCta = warpIncrementalLookback<numTileStatesPerThread>(
          specialRegisters, params.ptrTileStates, idxTilePrev, countExclusiveCtaPrev, idxTile, plus_op);
        if (squad.isLeaderThread())
        {
          refCountExclusiveCtaW.data() = regCountExclusiveCta;
        }
        countExclusiveCtaPrev = regCountExclusiveCta;
        idxTilePrev           = idxTile - 1;
      }
    }

    if (squad == squadCopyStore)
    {
      static_assert(tile_size % squadCopyStore.threadCount() == 0);

      // Count of all threads up to but not including this one
      uint32_t countExclusive = 0;

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across warp and thread counts
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Acquire refCountExclusiveCtaW briefly
        SmemRef refCountThreadAndWarpR = phaseCopyMaskR.acquireRef();

        // Add the sums of the preceding warps in this CTA to the cumulative
        // sum. These sums have been calculated in squadCount(). We need
        // the reduce and partition squads to be the same size to do this.
        static_assert(squadCount.warpCount() == squadCopyStore.warpCount());

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < squadCopyStore.warpCount(); ++i)
        {
          // We want a predicated unrolled loop here.
          if (i < squad.warpRank())
          {
            // First warp has nothing to add
            if (i == 0)
            {
              countExclusive = refCountThreadAndWarpR.data()[squadCount.threadCount()];
            }
            else
            {
              countExclusive += refCountThreadAndWarpR.data()[squadCount.threadCount() + i];
            }
          }
        }

        // Add the sums of preceding threads in this warp to the cumulative sum.
        // Lane 0 reads invalid data.
        AccumT regSumThread = refCountThreadAndWarpR.data()[squad.threadRank()];

        // Perform partition of thread sums. If the warp contains partial data, we pass invalid elements to predicate,
        // and countExclusiveIntraWarp is invalid when the inputs were invalid and for warp_0/thread_0
        AccumT countExclusiveIntraWarp = warpScanExclusive(regSumThread, predicate);

        // warp_0 does not hold a valid value in countExclusive, so only include it in other warps
        countExclusive =
          squad.warpRank() == 0 ? countExclusiveIntraWarp : predicate(countExclusive, countExclusiveIntraWarp);
      }

      // countExclusive is valid except for warp_0/thread_0

      ////////////////////////////////////////////////////////////////////////////////
      // Include sum of previous CTAs
      ////////////////////////////////////////////////////////////////////////////////
      {
        // Briefly acquire refCountExclusiveCtaR (we have to do this for the first tile as well to prevent a hang)
        SmemRef refCountExclusiveCtaR = phaseCountExclusiveCtaR.acquireRef();

        if (!is_first_tile)
        {
          // Add the sums of preceding CTAs to the cumulative sum.
          AccumT regCountExclusiveCta = refCountExclusiveCtaR.data();
          // countExclusive is invalid in warp_0/thread_0, so only include it in other threads/warps
          countExclusive =
            squad.threadRank() == 0 ? regCountExclusiveCta : predicate(countExclusive, regCountExclusiveCta);
        }
      }

      // countExclusive is valid except for warp_0/thread_0 in the first tile

      // TODO(bgruber): consider merging the below branch into the next block of branches with `hasInit`
      if constexpr (hasInit)
      {
        if (is_first_tile)
        {
          // The first thread cannot use predicate because countExclusive holds garbage data
          if (squad.threadRank() == 0)
          {
            countExclusive = static_cast<AccumT>(real_init_value);
          }
          else
          {
            countExclusive = predicate(static_cast<AccumT>(real_init_value), countExclusive);
          }
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Scan across elements allocated to this thread
      ////////////////////////////////////////////////////////////////////////////////
      InputT regSumInclusive[elemPerThread] = {{}};

      // Acquire refInOut for remainder of scope.
      SmemRef refInOutRW = phaseInOutRW.acquireRef();

      // We are always loading a full tile even for the last one. That will call predicate on invalid data
      // loading partial tiles here regresses perf for about 10-15%
      squadLoadSmem(squad, regSumInclusive, &refInOutRW.data()[0] + loadInfo.smemStartOffsetElem);

      // Perform inclusive partition of register array in current thread.
      // warp_0/thread_0 in the first tile when there is no initial value, we MUST NOT use countExclusive
      const bool use_prefix = hasInit ? true : !(is_first_tile && squad.threadRank() == 0);
      if constexpr (isInclusive)
      {
        ThreadScanInclusive(regSumInclusive, regSumInclusive, predicate, countExclusive, use_prefix);
      }
      else
      {
        ThreadScanExclusive(regSumInclusive, regSumInclusive, predicate, countExclusive, use_prefix);
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Store result to shared memory
      ////////////////////////////////////////////////////////////////////////////////
      // Sync before storing to avoid data races on SMEM
      squad.syncThreads();

      // if the output types fit into the input type tile, we can alias it
      OutputT* smem_output_tile = reinterpret_cast<OutputT*>(refInOutRW.data());
      if constexpr (sizeof(OutputT) <= sizeof(InputT))
      {
        CpAsyncOobInfo storeInfo = prepareCpAsyncOob(params.ptrOut + idxTileBase, valid_items);

        squadStoreSmem(squad, smem_output_tile + storeInfo.smemStartOffsetElem, regSumInclusive);
        // We do *not* release refSmemInOut here, because we will issue a TMA
        // instruction below. Instead, we issue a squad-local syncthreads +
        // fence.proxy.async to sync the shared memory writes with the TMA store.
        squad.syncThreads();

        ////////////////////////////////////////////////////////////////////////////////
        // Store result to global memory using TMA
        ////////////////////////////////////////////////////////////////////////////////
        squadStoreBulkSync(squad, storeInfo, smem_output_tile);
      }
      else
      {
        // otherwise, issue multiple bulk copies in chunks of the input tile size
        // TODO(bgruber): I am sure this could be implemented a lot more efficiently
        static constexpr int elem_per_chunk = (WarpspeedPolicy::tile_size * sizeof(InputT)) / sizeof(OutputT);
        for (int chunk_offset = 0; chunk_offset < static_cast<int>(valid_items); chunk_offset += elem_per_chunk)
        {
          const int chunk_size     = ::cuda::std::min(static_cast<int>(valid_items) - chunk_offset, elem_per_chunk);
          CpAsyncOobInfo storeInfo = prepareCpAsyncOob(params.ptrOut + idxTileBase + chunk_offset, chunk_size);
          OutputT* smemBuf         = smem_output_tile + storeInfo.smemStartOffsetElem;

          // only stage elements of the current chunk to SMEM
          squadStoreSmemPartial(
            squad,
            smem_output_tile + storeInfo.smemStartOffsetElem,
            regSumInclusive,
            chunk_offset,
            chunk_offset + chunk_size);

          // We do *not* release refSmemInOut here, because we will issue a TMA
          // instruction below. Instead, we issue a squad-local syncthreads +
          // fence.proxy.async to sync the shared memory writes with the TMA store.
          squad.syncThreads();

          ////////////////////////////////////////////////////////////////////////////////
          // Store result to global memory using TMA
          ////////////////////////////////////////////////////////////////////////////////
          squadStoreBulkSync(squad, storeInfo, smem_output_tile);

          squad.syncThreads();
        }
      }

      // Release refInOut. No need to do any cross-proxy fencing here, because
      // the TMA store in this warp and the TMA load in the load warp are both
      // async proxy.
    }

    ////////////////////////////////////////////////////////////////////////////////
    // All squads: Check loop condition and update next tile index
    ////////////////////////////////////////////////////////////////////////////////
    if (!nextIdxTileValid)
    {
      break;
    }
    // Update idxTile
    idxTile = ::cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(regNextBlockIdx);
  }

  if (squad == squadLoad)
  {
    _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  }
}

template <typename ActivePolicy, class = void>
inline constexpr int get_scan_block_threads = 1;

template <typename ActivePolicy>
inline constexpr int get_scan_block_threads<ActivePolicy, ::cuda::std::void_t<typename ActivePolicy::WarpspeedPolicy>> =
  ActivePolicy::WarpspeedPolicy::num_total_threads;

template <typename MaxPolicy,
          typename InputT,
          typename OutputT,
          typename AccumT,
          typename ScanOpT,
          typename InitValueT,
          bool ForceInclusive>
__launch_bounds__(get_scan_block_threads<typename MaxPolicy::ActivePolicy>, 1) __global__ void partition(
  const __grid_constant__ scanKernelParams<InputT, OutputT, AccumT> params, ScanOpT predicate, InitValueT init_value)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_100, ({
      using ActivePolicy    = typename MaxPolicy::ActivePolicy;
      using WarpspeedPolicy = typename ActivePolicy::WarpspeedPolicy;

      // Cache special registers at start of kernel
      SpecialRegisters specialRegisters = getSpecialRegisters();

      // Dispatch for warp-specialization
      static constexpr SquadDesc scanSquads[WarpspeedPolicy::num_squads] = {
        WarpspeedPolicy::squadCount(),
        WarpspeedPolicy::squadCopyStore(),
        WarpspeedPolicy::squadLoad(),
        WarpspeedPolicy::squadSched(),
        WarpspeedPolicy::squadLookback(),
      };

      using real_init_value_t = typename InitValueT::value_type;

      squadDispatch(specialRegisters, scanSquads, [&](Squad squad) {
        kernelBody<WarpspeedPolicy, InputT, OutputT, AccumT, ScanOpT, real_init_value_t, ForceInclusive>(
          squad, specialRegisters, params, ::cuda::std::move(predicate), static_cast<real_init_value_t>(init_value));
      });
    }))
}

template <typename AccumT>
__launch_bounds__(128) __global__ void initTileStates(tile_state_t<AccumT>* tile_states, const size_t num_temp_states)
{
  const int tile_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (tile_id >= num_temp_states)
  {
    return;
  }
  _CCCL_PDL_GRID_DEPENDENCY_SYNC();
  _CCCL_PDL_TRIGGER_NEXT_LAUNCH();
  tile_states[tile_id] = {EMPTY, AccumT{}};
}
} // namespace detail::partition

CUB_NAMESPACE_END

#endif // __cccl_ptx_isa >= 860
