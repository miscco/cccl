#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/memory.h>

#include <cuda/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <unittest/unittest.h>

struct ContiguousAdaptorPlusOne
{
  _CCCL_HOST_DEVICE int operator()(int value) const
  {
    return value + 1;
  }
};

TEST_CASE("contiguous iterator adaptor thrust storage", "[iterators]")
{
  static_assert(cuda::std::__has_contiguous_traversal<thrust::device_ptr<int>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_ptr<int>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::pointer<int, thrust::device_system_tag, int&>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::pointer<int, thrust::device_system_tag, int&>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::detail::normal_iterator<thrust::device_ptr<int>>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::detail::normal_iterator<int*>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::detail::normal_iterator<int*>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::host_vector<int>::iterator>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_vector<int>::iterator>);

  using Transformed       = thrust::transform_iterator<ContiguousAdaptorPlusOne, thrust::device_ptr<int>>;
  using Nested            = thrust::transform_iterator<ContiguousAdaptorPlusOne, Transformed>;
  using Normal            = thrust::detail::normal_iterator<int*>;
  using TransformedNormal = thrust::transform_iterator<ContiguousAdaptorPlusOne, Normal>;
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Transformed>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Transformed>
                == cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_ptr<int>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Nested>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<TransformedNormal>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_iterator<ContiguousAdaptorPlusOne, cuda::counting_iterator<int>>>);

  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<thrust::device_ptr<int>>())),
                         int*>);
  static_assert(cuda::std::is_same_v<
                decltype(cuda::__unwrap_iterator_adaptor(
                  ::cuda::std::declval<thrust::pointer<int, thrust::device_system_tag, int&>>())),
                int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Normal>())), int*>);
  static_assert(
    cuda::std::is_same_v<
      decltype(cuda::__unwrap_iterator_adaptor(
        ::cuda::std::declval<thrust::detail::normal_iterator<thrust::device_ptr<int>>>())),
      int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Transformed>())),
                                     int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Nested>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<TransformedNormal>())), int*>);

  int input[]  = {1, 2, 3, 4};
  int staged[] = {10, 20, 30, 40};

  {
    auto iter      = thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{});
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    REQUIRE(unwrapped == input);

    auto rebased = iter;
    rebased.__rebase(staged);
    REQUIRE(*iter == 2);
    REQUIRE(*rebased == 11);
    REQUIRE(rebased.base() == staged);
  }

  {
    auto inner = thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{});
    auto outer = thrust::make_transform_iterator(inner, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(outer) == input);
    auto rebased = outer;
    rebased.__rebase(staged);
    REQUIRE(*outer == 3);
    REQUIRE(*rebased == 12);
  }

  {
    Normal iter{input};
    auto rebased = iter;
    rebased.__rebase(staged);
    REQUIRE(iter.base() == input);
    REQUIRE(rebased.base() == staged);
    REQUIRE(*rebased == 10);

    auto fancy = thrust::make_transform_iterator(iter, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == input);
    auto fancy_rebased = fancy;
    fancy_rebased.__rebase(staged);
    REQUIRE(*fancy == 2);
    REQUIRE(*fancy_rebased == 11);
  }

  {
    thrust::pointer<int, thrust::device_system_tag, int&> stored{input};
    auto stored_rebased = stored;
    stored_rebased.__rebase(staged);
    REQUIRE(stored.get() == input);
    REQUIRE(stored_rebased.get() == staged);

    thrust::device_ptr<int> ptr{input};
    auto ptr_rebased = ptr;
    ptr_rebased.__rebase(staged);
    REQUIRE(ptr.get() == input);
    REQUIRE(ptr_rebased.get() == staged);

    auto fancy = thrust::make_transform_iterator(ptr, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == input);
    auto fancy_rebased = fancy;
    fancy_rebased.__rebase(staged);
    REQUIRE(fancy.base().get() == input);
    REQUIRE(fancy_rebased.base().get() == staged);
  }

  {
    int alternative = 7;
    thrust::device_vector<int> values(1);
    auto rebased = values.begin();
    rebased.__rebase(&alternative);
    REQUIRE(rebased.base().get() == &alternative);

    auto fancy = thrust::make_transform_iterator(values.begin(), ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == values.begin().base().get());
    auto fancy_rebased = fancy;
    fancy_rebased.__rebase(&alternative);
    REQUIRE(fancy_rebased.base().base().get() == &alternative);
  }
}
