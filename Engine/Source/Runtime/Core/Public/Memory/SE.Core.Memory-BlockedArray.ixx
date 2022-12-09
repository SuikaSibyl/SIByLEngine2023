export module SE.Core.Memory:BlockedArray;
import :Memory;

namespace SIByL::Core 
{
	/**
	* 2D array with better spatial memory locality.
	* The 2D array is represented by square blocks of a small fixed size that is a power of 2.
	*/
	export template<class T, int logBlockSize = 2>
	struct BlockedArray
	{
		/** @para d: a standard c++ array to initialize, which is optional */
		BlockedArray(int uRes, int vRes, T const* d = nullptr);

		/** access each element in blocked array */
		auto operator()(int u, int v)->T&;

		/** size of a single block */
		constexpr auto blockSize() const noexcept -> int;
		/** round both dimensions up to be a multiple of the block size */
		auto roundUp(int x) const noexcept -> int;

		auto uSize() const noexcept -> int { return uRes; }
		auto vSize() const noexcept -> int { return vRes; }

		auto block(int a) const noexcept -> int { return a >> logBlockSize; }
		auto offset(int a) const noexcept -> int { return (a & (blockSize() - 1)); }

	private:
		T* data;
		int const uRes, vRes, uBlocks;
	};

	template<class T, int logBlockSize>
	BlockedArray<T, logBlockSize>::BlockedArray(int uRes, int vRes, T const* d)
		: uRes(uRes), vRes(vRes), uBlocks(roundUp(uRes) >> logBlockSize) {
		int nAlloc = roundUp(uRes) * roundUp(vRes);
		data = AllocAligned<T>(nAlloc);
		for (int i = 0; i < nAlloc; ++i)
			new (&data[i]) T();
		if (d)
			for (int v = 0; v < vRes; v++)
				for (int u = 0; u < uRes; u++)
					(*this)(u, v) = d[v * uRes + u];
	}

	template<class T, int logBlockSize>
	constexpr auto BlockedArray<T, logBlockSize>::blockSize() const noexcept -> int {
		return 1 << logBlockSize;
	}

	template<class T, int logBlockSize>
	auto BlockedArray<T, logBlockSize>::roundUp(int x) const noexcept -> int {
		return (x + blockSize() - 1) & ~(blockSize() - 1);
	}

	template<class T, int logBlockSize>
	auto BlockedArray<T, logBlockSize>::operator()(int u, int v) -> T& {
		int bu = block(u), bv = block(v);
		int ou = offset(u), ov = offset(v);
		int offset = blockSize() * blockSize() * (uBlocks * bv + bu);
		offset += blockSize() * ov + ou;
		return data[offset];
	}
}