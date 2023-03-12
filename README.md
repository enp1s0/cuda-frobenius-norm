# CUDA Frobenius norm

## Usage

### Source code
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


### Compile
Compile `src/cunorm.cu` and link the resulting object file.

## License

MIT
