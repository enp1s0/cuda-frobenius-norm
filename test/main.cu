#include <iostream>
#include <cunorm.hpp>

template <class T>
constexpr T threshold = 0;
template <> constexpr double threshold<double> = 1e-14 * 1e-3;
template <> constexpr float  threshold<float > = 1e-5  * 1e-3;

template <class T>
std::string get_dtype_string();
template <> std::string get_dtype_string<double>() {return "double";};
template <> std::string get_dtype_string<float >() {return "float";};

template <class T>
void eval(const unsigned m, const unsigned n, const unsigned ld) {
	T *mat_ptr;
	cudaMallocHost(&mat_ptr, sizeof(T) * n * ld);

	const T v = 1.234;
	for (unsigned mi = 0; mi < m; mi++) {
		for (unsigned ni = 0; ni < n; ni++) {
			const auto mem_index = mi + ni * ld;

			mat_ptr[mem_index] = v;
		}
	}
	const auto ref = v * sqrt(m * n);

	const auto norm = cunorm::norm(m, n, mat_ptr, ld);

	const auto relative_error = std::abs((norm - ref) / ref);

	std::printf("%s,%u,%u,%u,%e,%s\n",
			get_dtype_string<T>().c_str(),
			m, n, ld,
			relative_error,
			((relative_error < threshold<T> * sqrt(m * n)) ? "OK" : "NG")
			);

	cudaFree(mat_ptr);
}

int main() {
	std::printf("dtype,m,n,ld,error,check\n");
	eval<double>(1u << 14, 1u << 14, 1u << 14);
	eval<double>(1u << 14, 1u << 14, 1u << 15);
	eval<float >(1u << 14, 1u << 14, 1u << 14);
	eval<float >(1u << 14, 1u << 14, 1u << 15);
}
