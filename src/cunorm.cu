#include <cunorm.hpp>

namespace {
template <class OUT_T>
__global__ void cunorm_preprocess_kernel(
		OUT_T* const norm2_ptr
		) {
	*norm2_ptr = 0;
}

template <class OUT_T>
__global__ void cunorm_postprocess_kernel(
		OUT_T* const res_ptr
		) {
	*res_ptr = sqrt(*res_ptr);
}

template <class IN_T, class OUT_T>
__global__ void norm_kernel(
		OUT_T* const norm2_ptr,
		const std::size_t m, const std::size_t n,
		const IN_T* const in_ptr,
		const std::size_t ld
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	OUT_T norm2 = 0;
	for (std::size_t i = tid; i < m * n; i += gridDim.x * blockDim.x) {
		const auto mi = i % m;
		const auto ni = i / m;
		const auto mem_index = mi + ld * ni;

		const OUT_T v = in_ptr[mem_index];
		norm2 += v * v;
	}

	constexpr unsigned warp_size = 32;
	for(auto mask = (warp_size >> 1); mask > 0; mask >>= 1) {
		norm2 += __shfl_xor_sync(0xffffffff, norm2, mask);
	}

	__shared__ OUT_T smem[32];
	if ((threadIdx.x & 0x1f) == 0) {
		smem[threadIdx.x / warp_size] = norm2;
	}
	__syncthreads();

	if (threadIdx.x < warp_size) {
		norm2 = 0;

		if (threadIdx.x < blockDim.x / warp_size) {
			norm2 = smem[threadIdx.x];
		}

		for(auto mask = (warp_size >> 1); mask > 0; mask >>= 1) {
			norm2 += __shfl_xor_sync(0xffffffff, norm2, mask);
		}

		if (threadIdx.x == 0) {
			atomicAdd(norm2_ptr, norm2);
		}
	}
}
} // noname namespace

template <class IN_T, class OUT_T>
void cunorm::norm(
		OUT_T* const out_ptr,
		const std::size_t m, const std::size_t n,
		const IN_T* const in_ptr,
		const std::size_t ld,
		cudaStream_t cuda_stream
		) {
	cunorm_preprocess_kernel<<<1, 1, 0, cuda_stream>>>(
			out_ptr
			);

	constexpr std::size_t block_size = 256;
	norm_kernel<<<100, block_size, 0, cuda_stream>>>(
			out_ptr,
			m, n,
			in_ptr, ld
			);

	cunorm_postprocess_kernel<<<1, 1, 0, cuda_stream>>>(
			out_ptr
			);
}

template <class IN_T, class OUT_T>
OUT_T cunorm::norm(
		const std::size_t m, const std::size_t n,
		const IN_T* const in_ptr,
		const std::size_t ld,
		cudaStream_t cuda_stream
		) {
	OUT_T* out_ptr;
	cudaMallocManaged(&out_ptr, sizeof(OUT_T));
	*out_ptr = 0;

	cunorm::norm(
			out_ptr,
			m, n,
			in_ptr, ld,
			cuda_stream
			);

	cudaStreamSynchronize(cuda_stream);

	const auto res = *out_ptr;

	cudaFree(out_ptr);

	return res;
}

#define INSTANCE(in_t, out_t) \
template out_t cunorm::norm<in_t, out_t>(const std::size_t, const std::size_t, const in_t* const, const std::size_t, cudaStream_t)

INSTANCE(double, double);
INSTANCE(float , float );
