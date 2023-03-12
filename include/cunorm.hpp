#pragma once
#include <cstdint>

namespace cunorm {
template <class IN_T, class OUT_T>
void norm(
		OUT_T* const device_out_ptr,
		const std::size_t m, const std::size_t n,
		const IN_T* const in_ptr,
		const std::size_t ld,
		cudaStream_t cuda_stream = 0
		);

template <class IN_T, class OUT_T = IN_T>
OUT_T norm(
		const std::size_t m, const std::size_t n,
		const IN_T* const in_ptr,
		const std::size_t ld,
		cudaStream_t cuda_stream = 0
		);
} // namespace cunorm
