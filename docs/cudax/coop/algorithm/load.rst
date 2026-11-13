.. _cudax-coop-algorithm-load:

``cuda::experimental::coop::load``
==================================

.. code:: cuda

  namespace cuda::experimental::coop {

  struct blocked_t {};
  template <cuda::std::size_t VecSize>
  struct striped_t {};

  inline constexpr blocked_t blocked{};
  inline constexpr striped_t<1> striped{};
  template <cuda::std::size_t VecSize>
  inline constexpr striped_t<VecSize> striped_vectors{};

  struct element_aligned_t {};
  struct vector_aligned_t {};
  template <cuda::std::size_t Alignment>
  struct overaligned_t {};

  inline constexpr element_aligned_t element_aligned{};
  inline constexpr vector_aligned_t vector_aligned{};
  template <cuda::std::size_t Alignment>
  inline constexpr overaligned_t<Alignment> overaligned{};

  template <group Group, typename Arrangement, typename InputIt, typename T, cuda::std::size_t N>
  __device__ void load(const Group& group, Arrangement arrangement, InputIt src, T (&items)[N]);

  template <group Group, typename Arrangement, typename InputIt, typename T, cuda::std::size_t N, typename Alignment>
  __device__ void load(const Group& group, Arrangement arrangement, InputIt src, T (&items)[N], Alignment alignment);

  template <group Group, typename Arrangement, typename InputIt, typename T, cuda::std::size_t N, /*integer*/ Size>
  __device__ void load(const Group& group, Arrangement arrangement, InputIt src, T (&items)[N], Size valid_items);

  template <group Group,
            typename Arrangement,
            typename InputIt,
            typename T,
            cuda::std::size_t N,
            /*integer*/ Size,
            typename Default>
  __device__ void
  load(const Group& group, Arrangement arrangement, InputIt src, T (&items)[N], Size valid_items, Default oob_default);

  template <typename Arrangement, typename InputIt, typename T, cuda::std::size_t N>
  __device__ bool is_vector_aligned(Arrangement arrangement, InputIt src, T (&items)[N]);

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::load`` loads a tile of :math:`P \cdot N` consecutive items from ``src`` into the :math:`P` threads of a group, :math:`N` items per thread. The thread rank in the group and the thread count of the group, static when the group knows it, define which items a thread owns:

- ``blocked``: thread :math:`r` owns the items :math:`r N` to :math:`r N + N - 1`.
- ``striped``: item :math:`i` of thread :math:`r` is item :math:`i P + r`.
- ``striped_vectors<V>``: vector :math:`i` of thread :math:`r` holds the :math:`V` items from item :math:`(i P + r) V` on. :math:`V` must divide :math:`N`.

Vectorization
-------------

The load reads the widest vectors that the arrangement, the items and the alignment of ``src`` allow, and the items one by one otherwise. Vectors are read from contiguous ranges of trivially copyable items that are not accessed through volatile references. They are at least 4 bytes wide and wider than the alignment of the items, at most as wide as the items of a vector (all :math:`N` items of a thread for ``blocked``), and at most 16 bytes wide, or 32 bytes on sm_100 and newer without a cache modifier or with ``LOAD_LDG``. The items are converted to ``T``.

A ``cub::CacheModifiedInputIterator`` loads its vectors with its cache modifier. With ``LOAD_CG``, ``LOAD_CS``, ``LOAD_CV`` and ``LOAD_VOLATILE``, only the widest vectors that the arrangement allows are used, and ``LOAD_VOLATILE`` vectors are at most 8 bytes wide. Items of a range that could be loaded with vectors are loaded without the cache modifier.

Alignment
---------

Without an alignment flag, the load tests the alignment of ``src`` at runtime and is correct for any ``src``. A load inside a loop over tiles should not test the alignment of every tile, because the kernel then contains every path of the load, which costs registers. Test the alignment once per range instead and pass a flag, which makes the load rely on it without testing anything:

- ``element_aligned``: ``src`` is only aligned to its items, which the load reads one by one.
- ``vector_aligned``: ``src`` satisfies ``is_vector_aligned``, the alignment of the widest vectors of the load on the current target.
- ``overaligned<A>``: ``src`` is aligned to ``A`` bytes, and the load reads the widest vectors that this alignment allows.

Every tile that starts a whole number of tiles after a tile that satisfies ``is_vector_aligned`` satisfies it as well.

Guarded loads
-------------

The overloads with ``valid_items`` load only the first ``valid_items`` items of the tile, in the same arrangement, item by item through ``src``. The other items keep their values, or are assigned ``oob_default``.

Example
-------

.. code:: cuda

  template <class Group>
  __device__ int sum_tiles(const Group& group, const int* range_begin, int num_tiles)
  {
    constexpr int items_per_thread = 8;
    const auto arrangement         = cuda::experimental::coop::striped_vectors<4>;
    const int tile_items           = cuda::gpu_thread.count_as<int>(group) * items_per_thread;

    int sum = 0;
    int items[items_per_thread];
    auto sum_all = [&](auto alignment) {
      for (int tile = 0; tile < num_tiles; ++tile)
      {
        cuda::experimental::coop::load(group, arrangement, range_begin + tile * tile_items, items, alignment);
        for (int i = 0; i < items_per_thread; ++i)
        {
          sum += items[i];
        }
      }
    };
    if (cuda::experimental::coop::is_vector_aligned(arrangement, range_begin, items))
    {
      sum_all(cuda::experimental::coop::vector_aligned);
    }
    else
    {
      sum_all(cuda::experimental::coop::element_aligned);
    }
    return sum;
  }
