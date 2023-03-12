# CUDA Frobenius norm

## Usage
- Store the result in device memory

```cpp
#include <cunorm.hpp>

T *norm;
cudaMalloc(&norm, sizeorf(T));
cunorm::norm(norm, m, n, mat_ptr, ld);
```

- Store the result in host memory

```cpp
#include <cunorm.hpp>

const auto norm = cunorm::norm(m, n, mat_ptr, ld);
```

## License

MIT
